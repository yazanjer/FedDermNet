"""
LIME explainability runner for SkinFLNet++.

Generates LIME image explanations for 10 correctly-classified and
10 misclassified test images per class.

Saves overlays to:
  figures/xai/<class>/{correct,incorrect}/lime_<image_id>.png
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
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    _LIME_AVAILABLE = True
except ImportError:
    logger.warning("lime not installed. Run: pip install lime")
    _LIME_AVAILABLE = False


def _model_predict_fn(model: torch.nn.Module, device: torch.device):
    """Return a prediction function compatible with LIME (numpy in → numpy out)."""
    from src.data.transforms import get_val_transforms

    transform = get_val_transforms(224)

    def predict(images: np.ndarray) -> np.ndarray:
        """images: (N, H, W, 3) uint8 numpy array → (N, C) probabilities."""
        model.eval()
        batch = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8))
            batch.append(transform(pil))
        tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            logits = model(tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()

    return predict


def run_lime_for_dataset(
    model: torch.nn.Module,
    test_loader: Any,
    class_names: list[str],
    device: torch.device,
    figures_dir: str | Path,
    run_name: str,
    n_per_class: int = 10,
    num_samples: int = 1000,
) -> None:
    """
    Run LIME on n_per_class correct + n_per_class incorrect test images per class.

    Args:
        model:       Trained SkinFLNet model.
        test_loader: DataLoader for the test set.
        class_names: List of class name strings.
        device:      Compute device.
        figures_dir: Root figures directory.
        run_name:    Experiment run name.
        n_per_class: Number of images per class per bucket (correct/incorrect).
        num_samples: LIME perturbation samples (higher = better but slower).
    """
    if not _LIME_AVAILABLE:
        logger.error("LIME not available. Skipping.")
        return

    out_root = Path(figures_dir) / run_name / "xai"
    predict_fn = _model_predict_fn(model, device)
    explainer = lime_image.LimeImageExplainer()

    # Gather predictions
    model.eval()
    correct_imgs: dict[int, list] = {c: [] for c in range(len(class_names))}
    incorrect_imgs: dict[int, list] = {c: [] for c in range(len(class_names))}

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            preds = logits.argmax(1).cpu()
            for img_tensor, true_lbl, pred_lbl in zip(images, labels, preds):
                true_lbl = int(true_lbl)
                pred_lbl = int(pred_lbl)
                # Denormalize to uint8 for LIME
                img_np = _denorm_to_uint8(img_tensor)
                bucket = correct_imgs if pred_lbl == true_lbl else incorrect_imgs
                if len(bucket[true_lbl]) < n_per_class:
                    bucket[true_lbl].append((img_np, true_lbl, pred_lbl))

            if all(
                len(correct_imgs[c]) >= n_per_class and
                len(incorrect_imgs[c]) >= n_per_class
                for c in range(len(class_names))
            ):
                break

    # Generate LIME explanations
    for cls_idx, cls_name in enumerate(class_names):
        for bucket_name, bucket in [("correct", correct_imgs), ("incorrect", incorrect_imgs)]:
            save_dir = out_root / cls_name / bucket_name
            save_dir.mkdir(parents=True, exist_ok=True)

            for i, (img_np, true_lbl, pred_lbl) in enumerate(bucket[cls_idx]):
                try:
                    explanation = explainer.explain_instance(
                        img_np,
                        predict_fn,
                        top_labels=1,
                        hide_color=0,
                        num_samples=num_samples,
                    )
                    from lime.lime_image import LimeImageExplainer
                    temp, mask = explanation.get_image_and_mask(
                        cls_idx,
                        positive_only=True,
                        num_features=5,
                        hide_rest=False,
                    )
                    _save_lime_overlay(temp, mask, save_dir / f"lime_{i:03d}.png", cls_name)
                    logger.debug("LIME: %s/%s image %d done", cls_name, bucket_name, i)
                except Exception as e:
                    logger.warning("LIME failed for %s/%s/%d: %s", cls_name, bucket_name, i, e)

    logger.info("LIME overlays saved to %s", out_root)


def _denorm_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised image tensor → (H,W,3) uint8 numpy for LIME."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


def _save_lime_overlay(
    temp: np.ndarray,
    mask: np.ndarray,
    save_path: Path,
    label: str,
) -> None:
    """Save a LIME explanation overlay image."""
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.set_title(f"LIME — {label}", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
