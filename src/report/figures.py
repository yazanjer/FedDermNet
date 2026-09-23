"""Report curves and confusion matrices from ``results/<run>/``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_ROUND_JSON = re.compile(r"^round_\d{3}\.json$")


def load_history(run_dir: Path) -> list[dict]:
    """Load server ``round_NNN.json`` only (exclude ``round_NNN_clients.json``)."""
    rows: list[tuple[int, dict]] = []
    for f in sorted(run_dir.glob("round_*.json")):
        if not _ROUND_JSON.match(f.name):
            continue
        idx = int(f.stem.split("_")[1])
        with open(f) as fp:
            rows.append((idx, json.load(fp)))
    rows.sort(key=lambda x: x[0])
    return [h for _, h in rows]


def plot_training_curve(history: list[dict], run_name: str, save_dir: Path) -> None:
    rounds = [h.get("round", i + 1) for i, h in enumerate(history)]
    accuracies = [h.get("accuracy", float("nan")) for h in history]
    f1s = [h.get("macro_f1", float("nan")) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(rounds, [a * 100 for a in accuracies], "b-o", ms=3)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy vs Round")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, [f * 100 for f in f1s], "g-o", ms=3)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].set_title("Macro-F1 vs Round")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(run_name, fontsize=11)
    plt.tight_layout()
    out = save_dir / f"curve_{run_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    run_name: str,
    save_dir: Path,
) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {run_name}")
    plt.tight_layout()
    out = save_dir / f"confusion_{run_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def generate_report_figures(
    results_dir: Path | str,
    figures_dir: Path | str,
    dataset: Literal["isic2019", "isic2018"],
) -> None:
    """Plot curves + confusion matrices for each run under ``results_dir``."""
    rd = Path(results_dir)
    fd = Path(figures_dir)
    fd.mkdir(parents=True, exist_ok=True)

    _class_names = {
        "isic2019": ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
        "isic2018": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
    }
    class_names = _class_names[dataset]

    for run_dir in sorted(rd.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        run_figures_dir = fd / run_name
        run_figures_dir.mkdir(parents=True, exist_ok=True)

        history = load_history(run_dir)
        if not history:
            print(f"No round files found in {run_dir} — skipping.")
            continue

        plot_training_curve(history, run_name, run_figures_dir)

        summary_path = run_dir / "summary.json"
        cm_data = None
        if summary_path.exists():
            with open(summary_path) as f:
                cm_data = json.load(f).get("confusion_matrix")
        elif history:
            cm_data = history[-1].get("confusion_matrix")

        if cm_data is not None:
            plot_confusion_matrix(cm_data, class_names, run_name, run_figures_dir)
