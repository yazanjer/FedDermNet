"""
Expected Calibration Error (ECE) and reliability diagram.

ECE measures how well predicted probabilities match observed frequencies.
Lower ECE = better calibrated model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def compute_ece(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        y_prob: (N,) max predicted probability (confidence).
        y_pred: (N,) predicted class labels.
        y_true: (N,) ground-truth class labels.
        n_bins: Number of equal-width bins.

    Returns:
        ECE as a float in [0, 1].
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        if mask.sum() == 0:
            continue
        bin_acc = (y_pred[mask] == y_true[mask]).mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def plot_reliability_diagram(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[str | Path] = None,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
) -> float:
    """
    Plot a reliability diagram and return the ECE.

    Args:
        y_prob:     (N,) max predicted probability (confidence score).
        y_pred:     (N,) predicted class labels.
        y_true:     (N,) ground-truth class labels.
        save_path:  If given, save the figure to this path.
        n_bins:     Number of bins.
        title:      Figure title.

    Returns:
        ECE value.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    bin_accs, bin_confs, bin_counts = [], [], []

    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append((low + high) / 2)
            bin_counts.append(0)
        else:
            bin_accs.append(float((y_pred[mask] == y_true[mask]).mean()))
            bin_confs.append(float(y_prob[mask].mean()))
            bin_counts.append(int(mask.sum()))

    ece = compute_ece(y_prob, y_pred, y_true, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(
        bin_centers,
        bin_accs,
        width=1.0 / n_bins,
        align="center",
        alpha=0.7,
        label="Model",
        color="steelblue",
        edgecolor="black",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    plt.close(fig)
    return ece
