
import os
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.utils import (
    set_seed,
    get_device,
    load_config,
    setup_logger,
    ensure_dirs,
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
)
from src.data_loader import download_dataset, prepare_splits, get_dataloaders
from src.model import build_model
from src.baseline_model import BaselineCNN

logger = setup_logger("trainer")


def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_clip=1.0, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    use_amp = scaler is not None

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        acc = (outputs.argmax(1) == labels).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = (outputs.argmax(1) == labels).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


def run_phase(tag, model, train_loader, val_loader, criterion, optimizer,
              scheduler, device, config, writer, epochs, early_stopper,
              scaler=None):
    grad_clip = config["training"].get("gradient_clip", 1.0)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        logger.info(f"[{tag}] Epoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip,
            scaler=scaler,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        step = epoch
        writer.add_scalars(f"{tag}/loss", {"train": train_loss, "val": val_loss}, step)
        writer.add_scalars(f"{tag}/accuracy", {"train": train_acc, "val": val_acc}, step)

        if HAS_MLFLOW:
            mlflow.log_metrics(
                {
                    f"{tag}_train_loss": train_loss,
                    f"{tag}_train_acc": train_acc,
                    f"{tag}_val_loss": val_loss,
                    f"{tag}_val_acc": val_acc,
                },
                step=step,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch,
                {"val_acc": val_acc, "val_loss": val_loss},
                os.path.join(config["paths"]["models"], f"best_{tag}.pth"),
            )
            logger.info(f"  New best val_acc: {val_acc:.4f} — checkpoint saved")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            logger.info(f"  Early stopping triggered at epoch {epoch}")
            break

    return history


def train_transfer_learning(config):
    device = get_device(config["project"]["device"])
    logger.info(f"Using device: {device}")

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, _ = get_dataloaders(config)

    writer = SummaryWriter(log_dir=config["paths"]["tensorboard"])

    use_amp = config["training"].get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Mixed-precision training (AMP) enabled")

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        mlflow.start_run(run_name="transfer_learning")

    p1 = config["training"]["phase1"]
    p2 = config["training"]["phase2"]
    sched_cfg = config["training"]["scheduler"]
    es_cfg = config["training"]["early_stopping"]

    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"=== Phase 1: Feature Extraction === "
        f"(trainable: {trainable:,} / {total:,} params)"
    )

    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=p1["learning_rate"], weight_decay=p1["weight_decay"],
    )
    sched1 = ReduceLROnPlateau(opt1, factor=sched_cfg["factor"], patience=sched_cfg["patience"])
    es1 = EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"])

    h1 = run_phase("phase1", model, train_loader, val_loader, criterion,
                   opt1, sched1, device, config, writer, p1["epochs"], es1,
                   scaler=scaler)

    model.unfreeze_backbone(from_layer=p2.get("unfreeze_from", -20))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"=== Phase 2: Fine-Tuning === "
        f"(trainable: {trainable:,} / {total:,} params)"
    )

    opt2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=p2["learning_rate"], weight_decay=p2["weight_decay"],
    )
    sched2 = ReduceLROnPlateau(opt2, factor=sched_cfg["factor"], patience=sched_cfg["patience"])
    es2 = EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"])

    h2 = run_phase("phase2", model, train_loader, val_loader, criterion,
                   opt2, sched2, device, config, writer, p2["epochs"], es2,
                   scaler=scaler)

    writer.close()
    if HAS_MLFLOW:
        mlflow.end_run()

    history = {k: h1[k] + h2[k] for k in h1}
    return model, history


def train_baseline(config):
    device = get_device(config["project"]["device"])
    model = BaselineCNN(num_classes=config["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, _ = get_dataloaders(config)

    bcfg = config["baseline"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=bcfg["learning_rate"], weight_decay=bcfg["weight_decay"]
    )
    sched_cfg = config["training"]["scheduler"]
    scheduler = ReduceLROnPlateau(optimizer, factor=sched_cfg["factor"], patience=sched_cfg["patience"])
    es_cfg = config["training"]["early_stopping"]
    early_stopper = EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"])

    writer = SummaryWriter(log_dir=os.path.join(config["paths"]["tensorboard"], "baseline"))

    total = sum(p.numel() for p in model.parameters())
    logger.info(f"=== Training Baseline CNN === ({total:,} params, all trainable)")

    history = run_phase(
        "baseline", model, train_loader, val_loader, criterion,
        optimizer, scheduler, device, config, writer, bcfg["epochs"], early_stopper,
    )

    save_checkpoint(
        model, optimizer, bcfg["epochs"],
        {"final": True},
        os.path.join(config["paths"]["models"], "baseline_final.pth"),
    )
    writer.close()
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train transfer-learning and baseline image classifiers"
    )
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download if data/ already exists")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Train only the baseline CNN")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    ensure_dirs(config)

    if not args.skip_download:
        src = download_dataset(config)
        prepare_splits(src, config["data"]["data_dir"], config)

    if args.baseline_only:
        train_baseline(config)
    else:
        train_transfer_learning(config)
        train_baseline(config)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
