"""
Grad-CAM explainability runner for SkinFLNet++.

Uses pytorch-grad-cam library to generate heatmaps for:
  - 10 correctly classified test images per class
  - 10 misclassified test images per class

Saves overlays to: figures/xai/<class>/{correct,incorrect}/gradcam_<id>.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_AVAILABLE = True
except ImportError:
    logger.warning("pytorch-grad-cam not installed. Run: pip install grad-cam")
    _GRADCAM_AVAILABLE = False


def _get_target_layer(model: torch.nn.Module) -> list:
    """
    Heuristically select the last convolutional layer for Grad-CAM.
    Supports the backbones in this project.
    """
    backbone = model.backbone

    # timm models expose `model.backbone.features` (VGG, DenseNet)
    # or `model.backbone.layer4` (ResNet-style) etc.
    # We'll look for the last conv layer in order of preference.

    candidates = [
        # DenseNet
        getattr(backbone, "features", None),
        # EfficientNet / ConvNeXt
        getattr(backbone, "blocks", None),
        # ViT — uses attention, not conv; fall back to norm layer
        getattr(backbone, "norm", None),
    ]

    # Try to get the last BatchNorm/Conv from features
    for candidate in candidates:
        if candidate is not None:
            # Return the last child module that is a Conv2d
            last_conv = None
            for m in candidate.modules():
                if isinstance(m, torch.nn.Conv2d):
                    last_conv = m
            if last_conv is not None:
                return [last_conv]

    # Fallback: iterate all backbone modules
    last_conv = None
    for m in backbone.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        return [last_conv]

    raise RuntimeError(
        "Could not identify a target convolutional layer for Grad-CAM. "
        "Please specify it manually."
    )


def run_gradcam_for_dataset(
    model: torch.nn.Module,
    test_loader: Any,
    class_names: list[str],
    device: torch.device,
    figures_dir: str | Path,
    run_name: str,
    n_per_class: int = 10,
) -> None:
    """
    Generate Grad-CAM heatmaps for correct and incorrect test predictions.

    Args:
        model:       Trained SkinFLNet model.
        test_loader: DataLoader for the test split.
        class_names: Ordered list of class name strings.
        device:      Torch device.
        figures_dir: Root of figures directory.
        run_name:    Experiment run name.
        n_per_class: Number of images to explain per class per bucket.
    """
    if not _GRADCAM_AVAILABLE:
        logger.error("pytorch-grad-cam not available. Skipping Grad-CAM.")
        return

    model.eval()
    model.to(device)

    target_layers = _get_target_layer(model)
    cam = GradCAM(model=model, target_layers=target_layers)

    out_root = Path(figures_dir) / run_name / "xai"
    correct_imgs: dict[int, list] = {c: [] for c in range(len(class_names))}
    incorrect_imgs: dict[int, list] = {c: [] for c in range(len(class_names))}

    from src.xai.lime_runner import _denorm_to_uint8

    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device)
            logits = model(images_dev)
            preds = logits.argmax(1).cpu()

            for img_t, true_lbl, pred_lbl in zip(images, labels, preds):
                true_lbl = int(true_lbl)
                pred_lbl = int(pred_lbl)
                bucket = correct_imgs if pred_lbl == true_lbl else incorrect_imgs
                if len(bucket[true_lbl]) < n_per_class:
                    bucket[true_lbl].append((img_t, true_lbl, pred_lbl))

            if all(
                len(correct_imgs[c]) >= n_per_class and
                len(incorrect_imgs[c]) >= n_per_class
                for c in range(len(class_names))
            ):
                break

    # Generate and save overlays
    for cls_idx, cls_name in enumerate(class_names):
        for bucket_name, bucket in [("correct", correct_imgs), ("incorrect", incorrect_imgs)]:
            save_dir = out_root / cls_name / bucket_name
            save_dir.mkdir(parents=True, exist_ok=True)

            for i, (img_t, true_lbl, pred_lbl) in enumerate(bucket[cls_idx]):
                try:
                    input_tensor = img_t.unsqueeze(0).to(device)
                    targets = [ClassifierOutputTarget(cls_idx)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]  # (H, W)

                    rgb_img = _denorm_to_uint8(img_t).astype(np.float32) / 255.0
                    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    out_path = save_dir / f"gradcam_{i:03d}.png"
                    Image.fromarray(overlay).save(out_path)
                    logger.debug("GradCAM: %s/%s/%d saved", cls_name, bucket_name, i)
                except Exception as e:
                    logger.warning(
                        "GradCAM failed for %s/%s/%d: %s", cls_name, bucket_name, i, e
                    )

    logger.info("Grad-CAM overlays saved to %s", out_root)
