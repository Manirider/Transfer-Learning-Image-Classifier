
import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import setup_logger

logger = setup_logger("gradcam")


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def overlay_heatmap(image_np, heatmap, alpha=0.5):
    h, w = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colour = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB) / 255.0
    overlay = heatmap_colour * alpha + image_np * (1 - alpha)
    return np.clip(overlay, 0, 1)


def denormalize(tensor, mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)):
    img = tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def get_target_layer(model):
    last_conv = None
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Could not find a Conv2d layer in the backbone.")
    return last_conv


def visualize_gradcam_batch(model, dataloader, device, class_names,
                            num_images=8, save_dir="outputs/gradcam"):
    os.makedirs(save_dir, exist_ok=True)

    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    model.eval()
    images_done = 0

    fig, axes = plt.subplots(num_images, 3, figsize=(14, 4 * num_images))
    if num_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for images, labels in dataloader:
        for i in range(images.size(0)):
            if images_done >= num_images:
                break

            img_tensor = images[i].unsqueeze(0).to(device)
            true_label = labels[i].item()

            heatmap = gradcam.generate(img_tensor)
            pred_class = model(img_tensor).argmax(1).item()

            img_np = denormalize(images[i])

            axes[images_done, 0].imshow(img_np)
            axes[images_done, 0].set_ylabel(
                f"True: {class_names[true_label]}", fontsize=10, fontweight="bold"
            )
            axes[images_done, 0].set_xticks([])
            axes[images_done, 0].set_yticks([])

            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            axes[images_done, 1].imshow(heatmap_resized, cmap="jet")
            axes[images_done, 1].axis("off")

            overlay = overlay_heatmap(img_np, heatmap)
            axes[images_done, 2].imshow(overlay)
            status = "+" if pred_class == true_label else "x"
            axes[images_done, 2].set_title(
                f"Pred: {class_names[pred_class]} {status}", fontsize=10
            )
            axes[images_done, 2].axis("off")

            images_done += 1

        if images_done >= num_images:
            break

    plt.tight_layout()
    path = os.path.join(save_dir, "gradcam_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grad-CAM visualisations saved -> {path}")
