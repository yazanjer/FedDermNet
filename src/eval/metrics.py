"""
Evaluation metrics for SkinFLNet++.

Uses torchmetrics and scikit-learn — no hand-rolled implementations.

Computes per-round: accuracy, macro-F1/precision/recall/specificity,
balanced accuracy, AUROC, AUPRC, per-class F1, and confusion matrix.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """
    Compute the full metric suite for a classification run.

    Args:
        y_true:      (N,)   integer ground-truth labels.
        y_pred:      (N,)   integer predicted labels.
        y_prob:      (N, C) predicted probabilities (softmax output).
        num_classes: Number of classes C.

    Returns:
        Dictionary with all metrics as Python scalars or lists.
    """
    # Basic metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    # Macro specificity = mean(per-class specificity)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    macro_specificity = float(_macro_specificity(cm))

    # AUROC and AUPRC (require probability scores)
    auroc, auprc = _compute_auroc_auprc(y_true, y_prob, num_classes)

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_specificity": macro_specificity,
        "auroc": auroc,
        "auprc": auprc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def _macro_specificity(cm: np.ndarray) -> float:
    """Compute macro-averaged specificity from a confusion matrix."""
    specificities = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        denom = tn + fp
        spec = float(tn / denom) if denom > 0 else 0.0
        specificities.append(spec)
    return float(np.mean(specificities))


def _compute_auroc_auprc(
    y_true: np.ndarray, y_prob: np.ndarray, num_classes: int
) -> tuple[float, float]:
    """
    Compute macro OVR AUROC and macro AUPRC.
    Falls back gracefully if a class has no positive examples.
    """
    try:
        if num_classes == 2:
            auroc = float(roc_auc_score(y_true, y_prob[:, 1]))
            auprc = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            auroc = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
            # Per-class AUPRC then macro-average
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auprc_per_class = [
                average_precision_score(y_bin[:, c], y_prob[:, c])
                for c in range(num_classes)
            ]
            auprc = float(np.mean(auprc_per_class))
    except ValueError as e:
        logger.warning("AUROC/AUPRC computation failed: %s", e)
        auroc, auprc = float("nan"), float("nan")
    return auroc, auprc


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """
    Run inference on a DataLoader and return the full metric dict.

    Returns metrics dict with an additional 'loss' key (cross-entropy).
    """
    import torch.nn.functional as F

    model.eval()
    all_logits, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    loss = float(F.cross_entropy(logits, labels).item())
    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    y_true = labels.numpy()

    metrics = compute_all_metrics(y_true, preds, probs, num_classes)
    metrics["loss"] = loss
    return metrics
