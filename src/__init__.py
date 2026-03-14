# Lazy imports so that `from src.utils import ...` or `from src.data_loader import ...`
# does not require timm/matplotlib until model or evaluation code is used.

__all__ = [
    "build_model",
    "TransferLearningClassifier",
    "BaselineCNN",
    "download_dataset",
    "prepare_splits",
    "get_dataloaders",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_training_history",
    "save_evaluation_report",
    "error_analysis",
    "GradCAM",
    "visualize_gradcam_batch",
    "set_seed",
    "get_device",
    "load_config",
    "setup_logger",
    "ensure_dirs",
]


def __getattr__(name):
    if name in ("build_model", "TransferLearningClassifier"):
        from src.model import build_model, TransferLearningClassifier
        return build_model if name == "build_model" else TransferLearningClassifier
    if name == "BaselineCNN":
        from src.baseline_model import BaselineCNN
        return BaselineCNN
    if name in ("download_dataset", "prepare_splits", "get_dataloaders"):
        from src.data_loader import download_dataset, prepare_splits, get_dataloaders
        return {"download_dataset": download_dataset, "prepare_splits": prepare_splits, "get_dataloaders": get_dataloaders}[name]
    if name in ("evaluate_model", "plot_confusion_matrix", "plot_training_history", "save_evaluation_report", "error_analysis"):
        from src.evaluate import evaluate_model, plot_confusion_matrix, plot_training_history, save_evaluation_report, error_analysis
        return {"evaluate_model": evaluate_model, "plot_confusion_matrix": plot_confusion_matrix, "plot_training_history": plot_training_history, "save_evaluation_report": save_evaluation_report, "error_analysis": error_analysis}[name]
    if name in ("GradCAM", "visualize_gradcam_batch"):
        from src.gradcam import GradCAM, visualize_gradcam_batch
        return GradCAM if name == "GradCAM" else visualize_gradcam_batch
    if name in ("set_seed", "get_device", "load_config", "setup_logger", "ensure_dirs"):
        from src.utils import set_seed, get_device, load_config, setup_logger, ensure_dirs
        return {"set_seed": set_seed, "get_device": get_device, "load_config": load_config, "setup_logger": setup_logger, "ensure_dirs": ensure_dirs}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
