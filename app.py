
import os
import sys

import numpy as np
import torch
from PIL import Image

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, get_device, load_checkpoint
from src.model import build_model
from src.augmentations import get_val_transforms
from src.gradcam import GradCAM, get_target_layer, overlay_heatmap, denormalize


# --- Load model once at startup ---
CONFIG_PATH = "configs/config.yaml"
CHECKPOINT_PATH = "outputs/models/best_phase2.pth"

config = load_config(CONFIG_PATH)
device = get_device(config["project"]["device"])
model = build_model(config).to(device)

if os.path.isfile(CHECKPOINT_PATH):
    load_checkpoint(model, CHECKPOINT_PATH, optimizer=None, device=device)
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
else:
    print(f"Warning: No checkpoint at {CHECKPOINT_PATH}. Run training first. Predictions will use random weights.")

class_names = config["model"]["class_names"]
transform = get_val_transforms(config)

# Grad-CAM (lazy init when first requested)
_gradcam = None

def get_gradcam():
    global _gradcam
    if _gradcam is None:
        target_layer = get_target_layer(model)
        _gradcam = GradCAM(model, target_layer)
    return _gradcam


def preprocess(image: np.ndarray) -> torch.Tensor:
    """Convert numpy image (0–255, HWC) to batch tensor for the model."""
    if image is None:
        return None
    pil = Image.fromarray(image.astype("uint8"))
    tensor = transform(pil).unsqueeze(0)
    return tensor.to(device)


def predict(image: np.ndarray, show_gradcam: bool = True):
    """Run inference and optionally Grad-CAM. Returns dict and optional overlay image."""
    if image is None:
        return "Upload an image.", None

    x = preprocess(image)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    pred_conf = float(probs[pred_idx])

    # Text result: top class + all scores
    lines = [f"**Prediction: {pred_label}** ({pred_conf:.1%})", ""]
    for name, p in zip(class_names, probs):
        bar = "█" * int(round(p * 20)) + "░" * (20 - int(round(p * 20)))
        lines.append(f"{name:12} {bar} {p:.2%}")
    text_out = "\n".join(lines)

    # Grad-CAM overlay image
    overlay_img = None
    if show_gradcam:
        try:
            gradcam = get_gradcam()
            heatmap = gradcam.generate(x, target_class=pred_idx)
            img_np = denormalize(x.squeeze(0))
            overlay_np = overlay_heatmap(img_np, heatmap)
            overlay_img = (overlay_np * 255).astype(np.uint8)
        except Exception as e:
            overlay_img = None  # e.g. no conv layer found

    return text_out, overlay_img


def main():
    try:
        import gradio as gr
    except ImportError:
        print("Install Gradio first: pip install gradio")
        sys.exit(1)

    with gr.Blocks(title="Scene Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Natural scene classifier\nUpload an image to get a prediction and Grad-CAM heatmap.")
        with gr.Row():
            inp = gr.Image(label="Upload image", type="numpy")
            with gr.Column():
                out_text = gr.Markdown(label="Prediction")
                out_gradcam = gr.Image(label="Grad-CAM overlay (where the model looked)")
        with gr.Row():
            run_btn = gr.Button("Classify", variant="primary")
            show_cam = gr.Checkbox(value=True, label="Show Grad-CAM overlay")

        def run(img, show_cam_flag):
            text, overlay = predict(img, show_gradcam=show_cam_flag)
            return text, overlay

        run_btn.click(
            fn=lambda img, cam: run(img, cam),
            inputs=[inp, show_cam],
            outputs=[out_text, out_gradcam],
        )
        inp.upload(
            fn=lambda img, cam: run(img, cam),
            inputs=[inp, show_cam],
            outputs=[out_text, out_gradcam],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
