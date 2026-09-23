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

    per_class = per_class_metrics(y_true, y_pred, y_prob, num_classes)

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
        **per_class,
    }
    return metrics


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int
) -> dict[str, list]:
    """One-vs-rest sensitivity, specificity, AUROC and AUPRC for every class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sens, spec, auc, ap, support = [], [], [], [], []
    for c in range(num_classes):
        tp = cm[c, c]; fn = cm[c, :].sum() - tp; fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens.append(float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"))
        spec.append(float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"))
        yb = (y_true == c).astype(int)
        support.append(int(yb.sum()))
        if 0 < yb.sum() < len(yb):
            auc.append(float(roc_auc_score(yb, y_prob[:, c])))
            ap.append(float(average_precision_score(yb, y_prob[:, c])))
        else:
            auc.append(float("nan")); ap.append(float("nan"))
    return {
        "per_class_sensitivity": sens,
        "per_class_specificity": spec,
        "per_class_auroc": auc,
        "per_class_auprc": ap,
        "per_class_support": support,
    }


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
def predict_proba(
    model: torch.nn.Module, dataloader: Any, device: torch.device, amp: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities and labels over a loader (fixed order)."""
    model.eval()
    all_logits, all_labels = [], []
    use_amp = bool(amp) and device.type == "cuda"
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return torch.softmax(logits, dim=1).numpy(), labels.numpy()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    num_classes: int,
    amp: bool = False,
) -> dict[str, Any]:
    """Run inference on a DataLoader and return the full metric dict (+ 'loss')."""
    probs, y_true = predict_proba(model, dataloader, device, amp=amp)
    eps = 1e-12
    loss = float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps)))
    preds = probs.argmax(axis=1)
    metrics = compute_all_metrics(y_true, preds, probs, num_classes)
    metrics["loss"] = loss
    metrics["n"] = int(len(y_true))
    return metrics
