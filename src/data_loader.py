
import os
import shutil
import random
from collections import Counter

import kagglehub
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.augmentations import get_train_transforms, get_val_transforms
from src.utils import setup_logger

logger = setup_logger("data_loader")


def download_dataset(config: dict) -> str:
    dataset_name = config["data"]["dataset_name"]
    logger.info(f"Downloading dataset: {dataset_name}")
    path = kagglehub.dataset_download(dataset_name)
    logger.info(f"Dataset downloaded to: {path}")
    return path


def prepare_splits(source_dir: str, dest_dir: str, config: dict) -> dict:
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]
    seed = config["project"]["seed"]
    random.seed(seed)

    merged_dir = os.path.join(dest_dir, "_merged")

    seg_train = os.path.join(source_dir, "seg_train", "seg_train")
    seg_test = os.path.join(source_dir, "seg_test", "seg_test")

    source_dirs = [d for d in [seg_train, seg_test] if os.path.isdir(d)]
    if not source_dirs:
        source_dirs = [source_dir]

    class_names = set()
    for sd in source_dirs:
        for entry in os.listdir(sd):
            if os.path.isdir(os.path.join(sd, entry)):
                class_names.add(entry)

    for cls in class_names:
        os.makedirs(os.path.join(merged_dir, cls), exist_ok=True)

    idx = 0
    for sd in source_dirs:
        for cls in class_names:
            cls_dir = os.path.join(sd, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                src = os.path.join(cls_dir, fname)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(merged_dir, cls, f"{idx}_{fname}")
                shutil.copy2(src, dst)
                idx += 1

    split_names = ["train", "val", "test"]
    for split in split_names:
        for cls in class_names:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

    stats = {s: Counter() for s in split_names}

    for cls in sorted(class_names):
        files = sorted(os.listdir(os.path.join(merged_dir, cls)))
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split, split_files in splits.items():
            for fname in split_files:
                shutil.move(
                    os.path.join(merged_dir, cls, fname),
                    os.path.join(dest_dir, split, cls, fname),
                )
            stats[split][cls] = len(split_files)

    shutil.rmtree(merged_dir, ignore_errors=True)

    logger.info("Dataset split statistics:")
    for split in split_names:
        total = sum(stats[split].values())
        logger.info(f"  {split}: {total} images — {dict(stats[split])}")

    return stats


def get_dataloaders(config: dict) -> tuple:
    data_cfg = config["data"]
    data_dir = data_cfg["data_dir"]

    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, "test"), transform=val_transform)

    loader_kwargs = dict(
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    logger.info(
        f"Loaders ready — train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
