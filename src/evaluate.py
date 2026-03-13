
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.utils import setup_logger

logger = setup_logger("evaluator")


@torch.no_grad()
def evaluate_model(model, test_loader, device, class_names=None) -> dict:
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"Test Accuracy : {accuracy:.4f}")
    logger.info(f"Precision     : {precision:.4f}")
    logger.info(f"Recall        : {recall:.4f}")
    logger.info(f"F1-Score      : {f1:.4f}")
    logger.info(f"\n{report}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
    }


def plot_confusion_matrix(cm, class_names,
                          save_path="outputs/plots/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved -> {save_path}")


def plot_training_history(history, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss",
                 linewidth=2, color="#2196F3")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss",
                 linewidth=2, color="#FF5722")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy",
                 linewidth=2, color="#4CAF50")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy",
                 linewidth=2, color="#9C27B0")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved -> {path}")


def error_analysis(model, test_loader, device, class_names,
                   save_dir="outputs/reports") -> str:
    os.makedirs(save_dir, exist_ok=True)
    results = evaluate_model(model, test_loader, device, class_names)
    cm = results["confusion_matrix"].copy()

    np.fill_diagonal(cm, 0)
    n = len(class_names)
    pairs = []
    for i in range(n):
        for j in range(n):
            if cm[i][j] > 0:
                pairs.append((class_names[i], class_names[j], cm[i][j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    lines = [
        "=" * 60,
        "  ERROR ANALYSIS — Most Confused Class Pairs",
        "=" * 60, "",
    ]
    for true_cls, pred_cls, count in pairs[:10]:
        lines.append(f"  {true_cls:>12s}  ->  {pred_cls:<12s}  ({count} errors)")
        lines.append(
            f"      Likely cause: visual similarity between '{true_cls}' and "
            f"'{pred_cls}' — shared textures, colours, or spatial patterns.\n"
        )

    summary = "\n".join(lines)
    report_path = os.path.join(save_dir, "error_analysis.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Error analysis saved -> {report_path}")
    return summary


def save_evaluation_report(results, class_names,
                           save_dir="outputs/reports"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "evaluation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("       MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy : {results['accuracy']:.4f}\n")
        f.write(f"Precision     : {results['precision']:.4f}\n")
        f.write(f"Recall        : {results['recall']:.4f}\n")
        f.write(f"F1-Score      : {results['f1']:.4f}\n\n")
        f.write("Classification Report\n")
        f.write("-" * 60 + "\n")
        f.write(results["report"] + "\n")
    logger.info(f"Evaluation report saved -> {path}")
