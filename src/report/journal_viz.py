"""Publication-oriented figures for SkinFL runs (journal SVG + PDF exports).

Used by ``notebooks/skinfl_colab_visualizations.ipynb``. Prefer ``summary.json``
``experiment`` / ``stopped`` blocks when present (see ``MetricsTracker.save_summary``).
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 3,
    'axes.linewidth': 1.5,
    'pdf.fonttype': 42,
})

ROUND_FILE = re.compile(r"^round_(\d{3})\.json$")
CLIENT_FILE = re.compile(r"^round_(\d{3})_clients\.json$")

COMPARE_GROUPS: dict[str, list[str]] = {
    "partition": [
        "ablation_partition_dirichlet_01",
        "ablation_partition_dirichlet_05",
        "ablation_partition_dirichlet_10",
        "isic2019_dirichlet",
    ],
    "nclients": [
        "ablation_nclients_5",
        "ablation_nclients_10",
        "ablation_nclients_20",
    ],
    "strategy": [
        "isic2019_dirichlet",
        "ablation_strategy_fedprox",
        "ablation_strategy_fedadam",
    ],
    "localepochs": [
        "ablation_localepochs_1",
        "ablation_localepochs_5",
        "ablation_localepochs_10",
    ],
    "primary": [
        "isic2019_dirichlet",
        "centralized_upperbound",
    ],
}


# Human-readable figure title fragments (no raw folder names).
_COMPARISON_GROUP_TITLES: dict[str, str] = {
    "partition": r"Dirichlet heterogeneity ($\alpha$)",
    "nclients": "Client count",
    "strategy": "Aggregation strategy",
    "localepochs": r"Local epochs ($E_{\mathrm{loc}}$)",
    "primary": "Federated vs. centralized upper bound",
}


def comparison_group_figure_title(group_key: str) -> str:
    """Short fragment for prose / LaTeX captions (figures intentionally have no matplotlib title)."""
    return _COMPARISON_GROUP_TITLES.get(
        group_key, group_key.replace("_", " ").title()
    )


# Static labels shared across contexts (primary ``isic2019_dirichlet`` / ``isic2018_dirichlet`` handled separately).
_CLEAN_RUN_STATIC: dict[str, str] = {
    "ablation_strategy_fedprox": "FedProx",
    "ablation_strategy_fedadam": "FedAdam",
    "centralized_upperbound": "Centralized Upper Bound",
    "ablation_partition_dirichlet_01": r"$\alpha=0.1$",
    "ablation_partition_dirichlet_05": r"$\alpha=0.5$",
    "ablation_partition_dirichlet_10": r"$\alpha=10$",
    "ablation_nclients_5": "5 Clients",
    "ablation_nclients_10": "10 Clients",
    "ablation_nclients_20": "20 Clients",
    "ablation_localepochs_1": r"$E_{\mathrm{loc}}=1$",
    "ablation_localepochs_5": r"$E_{\mathrm{loc}}=5$",
    "ablation_localepochs_10": r"$E_{\mathrm{loc}}=10$",
}


def _pretty_unknown_run(run_name: str) -> str:
    s = run_name.strip()
    m = re.match(r"^ablation_nclients_(\d+)$", s)
    if m:
        return f"{int(m.group(1))} Clients"
    m = re.match(r"^ablation_localepochs_(\d+)$", s)
    if m:
        return rf"$E_{{\mathrm{{loc}}}}={int(m.group(1))}$"
    m = re.match(r"^ablation_partition_dirichlet_(\d+)$", s)
    if m:
        tail = m.group(1)
        if len(tail) == 1:
            alpha_s = "0." + tail
        else:
            alpha_s = f"{float(int(tail)):.0f}".rstrip("0").rstrip(".")
        return rf"$\alpha={alpha_s}$"
    return s.replace("ablation_", "").replace("_", " ").title()


def get_clean_run_name(run_name: str, comparison_group_name: str) -> str:
    """Map a results-folder name to a publication-ready legend / tick label.

    Primary benchmark folders ``isic2019_dirichlet`` / ``isic2018_dirichlet`` are
    rewritten using ``comparison_group_name`` so the baseline reads correctly in
    each ablation context (strategy vs. Dirichlet skew vs. client count, etc.).

    Run folders may use an ``isic2018_`` / ``isic2019_`` prefix; labels reuse the
    same static map as the short names (suffix after the prefix).
    """
    g = comparison_group_name.strip().lower()
    rn = run_name.strip()

    if rn in ("isic2019_dirichlet", "isic2018_dirichlet"):
        if g == "strategy":
            return "FedAvg (Baseline)"
        if g == "partition":
            return r"$\alpha=100$ (Near-IID)"
        if g == "nclients":
            return "10 Clients (Baseline)"
        if g == "localepochs":
            return r"$E_{\mathrm{loc}}=2$ (Baseline)"
        if g == "primary":
            return r"Federated ($\alpha=100$, near-IID)"
        return r"Federated ($\alpha=100$, near-IID)"

    suffix = rn
    for prefix in ("isic2018_", "isic2019_"):
        if rn.startswith(prefix):
            suffix = rn[len(prefix) :]
            break
    if suffix in _CLEAN_RUN_STATIC:
        return _CLEAN_RUN_STATIC[suffix]
    return _pretty_unknown_run(suffix)


def multi_run_palette(n: int) -> list:
    """Colorblind-friendly palette for overlays (matplotlib color specs)."""
    base = sns.color_palette("colorblind", max(n, 8))
    return [base[i % len(base)] for i in range(n)]


def apply_journal_style() -> None:
    """Matplotlib rcParams for camera-ready exports (serif, larger type, thick lines)."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "figure.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 3,
            "axes.linewidth": 1.5,
            "pdf.fonttype": 42,
            # Export / layout (not in user snippet; kept for SVG quality)
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )
    sns.set_theme(style="whitegrid", font_scale=1.0)


def save_figure(fig: mpl.figure.Figure, path_without_ext: Path) -> None:
    """Save SVG and PDF with tight bbox."""
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    stem = path_without_ext
    for fmt in ("svg", "pdf"):
        p = stem.with_suffix(f".{fmt}")
        fig.savefig(p, format=fmt, bbox_inches="tight")
        print("Saved:", p)
    plt.close(fig)


def load_yaml_experiment(project_root: Path, run_name: str) -> dict[str, Any]:
    """Load ``experiment`` block from ``configs/<run_name>.yaml`` if present."""
    path = project_root / "configs" / f"{run_name}.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        block = raw.get("experiment") if isinstance(raw, dict) else None
        return dict(block) if isinstance(block, dict) else {}
    except Exception as e:
        warnings.warn(f"Could not load YAML for {run_name}: {e}")
        return {}


def load_summary_sidecar(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(experiment_meta, stopped_meta)`` from summary.json when present."""
    p = run_dir / "summary.json"
    if not p.is_file():
        return {}, {}
    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        exp = data.get("experiment")
        st = data.get("stopped")
        return (
            dict(exp) if isinstance(exp, dict) else {},
            dict(st) if isinstance(st, dict) else {},
        )
    except Exception as e:
        warnings.warn(f"Could not parse summary.json in {run_dir}: {e}")
        return {}, {}


def merge_run_meta(
    run_dir: Path,
    project_root: Path,
    run_name: str,
    history: list[dict],
) -> dict[str, Any]:
    """Flatten hyperparameters used by comparisons (YAML/summary defaults)."""
    exp_yaml = load_yaml_experiment(project_root, run_name)
    exp_sum, stopped_sum = load_summary_sidecar(run_dir)

    def _coalesce(key: str, default: Any = None) -> Any:
        if key in exp_sum and exp_sum[key] is not None:
            return exp_sum[key]
        if key in exp_yaml and exp_yaml[key] is not None:
            return exp_yaml[key]
        return default

    xr = x_rounds(history)
    last_hist = int(round(max(xr))) if xr else -1

    last_round = stopped_sum.get("last_round")
    if last_round is None:
        last_round = last_hist
    else:
        last_round = int(last_round)

    early_stop = stopped_sum.get("early_stop")
    if early_stop is None:
        num_rounds = _coalesce("num_rounds", None)
        early_stop = bool(
            num_rounds is not None and last_hist >= 0 and last_hist < int(num_rounds)
        )
    else:
        early_stop = bool(early_stop)

    meta = {
        "mode": str(_coalesce("mode", "federated")),
        "num_rounds": int(_coalesce("num_rounds", max(last_hist + 1, 1))),
        "local_epochs": int(_coalesce("local_epochs", 2)),
        "num_clients": int(_coalesce("num_clients", 10)),
        "fraction_fit": float(_coalesce("fraction_fit", 0.5)),
        "strategy": str(_coalesce("strategy", "fedavg")),
        "partition": str(_coalesce("partition", "dirichlet")),
        "alpha": float(_coalesce("alpha", 0.5)),
        "early_stop": early_stop,
        "last_round": last_round,
    }
    return meta


def uploads_per_round(num_clients: int, fraction_fit: float) -> int:
    """Matches ``simulate.py``: participants each FL training round."""
    return max(1, int(num_clients * fraction_fit))


def cumulative_communication_rounds(
    rounds: Iterable[float],
    *,
    num_clients: int,
    fraction_fit: float,
    mode: str,
) -> np.ndarray:
    """Map each global round index to cumulative client-slot uploads (FL only).

    Round 0 contributes 0 uploads (initial server eval). Each integer round r>=1 adds
    ``uploads_per_round(...)``.
    """
    rounds_arr = np.asarray(list(rounds), dtype=float)
    out = np.zeros_like(rounds_arr, dtype=float)
    step = uploads_per_round(num_clients, fraction_fit)
    if mode.lower() != "federated":
        return out  # centralized: caller should use alternate x-axis if needed

    def cum_at_round(r: float) -> float:
        ri = int(math.floor(float(r) + 1e-9))
        if ri <= 0:
            return 0.0
        return float(ri * step)

    for i, r in enumerate(rounds_arr):
        out[i] = cum_at_round(r)
    return out


def load_round_history(run_dir: Path) -> list[dict]:
    summ = run_dir / "summary.json"
    if summ.is_file():
        with summ.open(encoding="utf-8") as f:
            data = json.load(f)
        hist = data.get("history")
        if isinstance(hist, list) and len(hist) > 0:
            return hist
    rows: list[tuple[int, dict]] = []
    for f in run_dir.iterdir():
        if not f.is_file():
            continue
        m = ROUND_FILE.match(f.name)
        if m:
            with f.open(encoding="utf-8") as fp:
                rows.append((int(m.group(1)), json.load(fp)))
    rows.sort(key=lambda x: x[0])
    return [h for _, h in rows]


def x_rounds(history: list[dict]) -> list[float]:
    xs: list[float] = []
    for i, h in enumerate(history):
        x = h.get("round")
        if x is None:
            x = h.get("virtual_round")
        if x is None:
            x = h.get("epoch")
        if x is None:
            x = i
        xs.append(float(x))
    return xs


def infer_class_names(history: list[dict]) -> list[str]:
    for h in history:
        cm = h.get("confusion_matrix")
        if cm and len(cm) > 0:
            n = len(cm)
            if n == 8:
                return ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
            if n == 2:
                return ["benign", "malignant"]
            return [f"C{i}" for i in range(n)]
    return []


def best_macro_f1_idx(history: list[dict]) -> int:
    best_i, best_v = 0, -1.0
    for i, h in enumerate(history):
        v = float(h.get("macro_f1", float("nan")))
        if not math.isnan(v) and v >= best_v - 1e-9:
            best_v, best_i = v, i
    return best_i


def safe_get(h: dict, k: str) -> float:
    v = h.get(k)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def series_for_metric(history: list[dict], mkey: str) -> tuple[np.ndarray, np.ndarray]:
    """Observed (round_x, y) with finite y."""
    xr = np.asarray(x_rounds(history), dtype=float)
    y = np.array([safe_get(h, mkey) for h in history], dtype=float)
    mask = np.isfinite(y)
    return xr[mask], y[mask]


def plot_aligned_metric_lines(
    ax: plt.Axes,
    loaded: list[tuple[str, list[dict]]],
    metas: list[dict[str, Any]],
    mkey: str,
    *,
    line_labels: list[str],
    palette: list,
    x_mode: str,
    global_max_round: int,
) -> bool:
    """Solid observed curves + dashed ffill tail + ES markers. Returns True if any ES."""

    # pylint: disable=too-many-locals
    any_es_marker = False

    if len(line_labels) != len(loaded):
        raise ValueError("line_labels must align with loaded runs")

    for i, ((rn, hist), meta) in enumerate(zip(loaded, metas, strict=True)):
        ox, oy = series_for_metric(hist, mkey)
        if ox.size == 0:
            warnings.warn(f"compare: skipping {rn} for metric {mkey} (no finite values)")
            continue

        order = np.argsort(ox)
        ox = ox[order]
        oy = oy[order]
        hi = int(round(float(ox[-1])))

        local_epochs = int(meta["local_epochs"])

        def x_from_round(r_arr: np.ndarray) -> np.ndarray:
            if x_mode == "cum_local_epochs":
                return r_arr.astype(float) * float(local_epochs)
            return r_arr.astype(float)

        lab = line_labels[i]
        xs_obs = x_from_round(np.asarray([int(round(float(v))) for v in ox]))
        ys_obs = oy.copy()

        ax.plot(xs_obs, ys_obs, "-", color=palette[i % len(palette)], label=lab)

        if hi < global_max_round:
            pad_r = np.arange(hi + 1, global_max_round + 1, dtype=int)
            tail_y = np.full_like(pad_r, float(oy[-1]), dtype=float)
            xs_tail = x_from_round(pad_r)
            ax.plot(
                xs_tail,
                tail_y,
                linestyle=(0, (4, 4)),
                alpha=0.5,
                color=palette[i % len(palette)],
            )

        early_stop = bool(meta.get("early_stop"))
        last_r = int(meta.get("last_round", hi))
        if early_stop:
            lx = float(last_r)
            if x_mode == "cum_local_epochs":
                lx *= float(local_epochs)
            ly = float(oy[-1])
            ax.scatter(
                [lx],
                [ly],
                marker="*",
                s=140,
                color=palette[i % len(palette)],
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
            any_es_marker = True

    return any_es_marker


def compare_group(
    group_key: str,
    *,
    results_dir: Path,
    journal_out: Path,
    project_root: Path,
    compare_groups: dict[str, list[str]] | None = None,
    x_mode: str = "round",
) -> None:
    """Overlay curves (single axis per metric) with early-stop-aware styling."""

    groups = compare_groups if compare_groups is not None else COMPARE_GROUPS
    if group_key not in groups:
        raise KeyError(f"Unknown group {group_key}. Keys: {list(groups)}")
    names = groups[group_key]
    out = journal_out / f"compare_{group_key}"
    out.mkdir(parents=True, exist_ok=True)
    apply_journal_style()

    loaded: list[tuple[str, list[dict]]] = []
    metas: list[dict[str, Any]] = []
    for rn in names:
        run_dir = results_dir / rn
        if not run_dir.is_dir():
            print(f"MISSING run dir: {run_dir}")
            continue
        hist = load_round_history(run_dir)
        if not hist:
            print(f"SKIP compare entry {rn}: no history")
            continue
        meta = merge_run_meta(run_dir, project_root, rn, hist)
        loaded.append((rn, hist))
        metas.append(meta)

    if len(loaded) < 2:
        print(f"SKIP compare_{group_key}: need >= 2 runs with data, got {len(loaded)}")
        return

    line_labels = [get_clean_run_name(rn, group_key) for rn, _ in loaded]
    palette = multi_run_palette(len(loaded))

    last_rounds = [int(m["last_round"]) for m in metas]
    global_max_round = max(last_rounds)

    xlab = (
        r"Cumulative local epochs ($r \times E_{\mathrm{loc}}$)"
        if x_mode == "cum_local_epochs"
        else "Round"
    )

    # Per-metric overlays and one stacked figure (macro-F1, accuracy, training loss).
    metrics_plot = [
        ("macro_f1", "Macro-F1"),
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
    ]

    for mkey, mlabel in metrics_plot:
        skip_all = True
        for _, hist in loaded:
            ox, _oy = series_for_metric(hist, mkey)
            if ox.size > 0:
                skip_all = False
                break
        if skip_all:
            warnings.warn(f"compare_{group_key}: skipping overlay for {mkey} (all NaN)")
            continue

        fig, ax = plt.subplots(figsize=(6.0, 3.9))
        any_es = plot_aligned_metric_lines(
            ax,
            loaded,
            metas,
            mkey,
            line_labels=line_labels,
            palette=palette,
            x_mode=x_mode,
            global_max_round=global_max_round,
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel(mlabel)
        if any_es:
            ax.scatter(
                [],
                [],
                marker="*",
                s=140,
                c="gray",
                edgecolors="black",
                linewidths=0.6,
                label="Early stopping point",
            )
        ax.legend(loc="lower right", frameon=True)
        save_figure(fig, out / f"compare_{group_key}__overlay_{mkey}")

    # Single figure: three metrics (shared x-axis), one legend on the bottom panel.
    triple_specs = [
        ("macro_f1", "Macro-F1"),
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
    ]
    fig_t, axes_t = plt.subplots(3, 1, figsize=(6.0, 9.0), sharex=True)
    es_triple = False
    for ax_i, (mkey, mlabel) in zip(axes_t, triple_specs, strict=True):
        skip_all = True
        for _, hist in loaded:
            ox, _oy = series_for_metric(hist, mkey)
            if ox.size > 0:
                skip_all = False
                break
        if skip_all:
            ax_i.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_i.transAxes)
            ax_i.set_ylabel(mlabel)
            continue
        es_triple = (
            plot_aligned_metric_lines(
                ax_i,
                loaded,
                metas,
                mkey,
                line_labels=line_labels,
                palette=palette,
                x_mode=x_mode,
                global_max_round=global_max_round,
            )
            or es_triple
        )
        ax_i.set_ylabel(mlabel)
    axes_t[-1].set_xlabel(xlab)
    if es_triple:
        axes_t[-1].scatter(
            [],
            [],
            marker="*",
            s=140,
            c="gray",
            edgecolors="black",
            linewidths=0.6,
            label="Early stopping point",
        )
    axes_t[-1].legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig_t.align_ylabels(axes_t)
    save_figure(fig_t, out / f"compare_{group_key}__panel_macro_acc_loss")

    bar_labels_all = [
        "macro_f1",
        "accuracy",
        "loss",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "ece",
    ]
    bar_titles_all = [
        "Macro-F1",
        "Accuracy",
        "Loss",
        "AUROC",
        "AUPRC",
        "Bal. acc.",
        "ECE",
    ]

    def _bars(which: str) -> None:
        rows = []
        labels = []
        for (rn, hist), clean in zip(loaded, line_labels, strict=True):
            if which == "best":
                j = best_macro_f1_idx(hist)
            else:
                j = len(hist) - 1
            row = [safe_get(hist[j], k) for k in bar_labels_all]
            rows.append(row)
            labels.append(clean)
        arr = np.asarray(rows, dtype=float)

        usable_idx = [
            bi
            for bi in range(len(bar_labels_all))
            if np.any(np.isfinite(arr[:, bi]))
        ]
        if not usable_idx:
            warnings.warn(f"compare_{group_key}: skipping bars_{which} (no finite metrics)")
            return

        bar_labels = [bar_labels_all[i] for i in usable_idx]
        bar_titles = [bar_titles_all[i] for i in usable_idx]
        arr_f = arr[:, usable_idx]

        x = np.arange(len(labels))
        n_b = len(bar_labels)
        w = 0.8 / n_b
        fig_b, ax_b = plt.subplots(figsize=(max(7.0, len(labels) * 0.72), 4.2))
        colors_bar = multi_run_palette(n_b)
        for bi, key in enumerate(bar_labels):
            offs = (bi - (n_b - 1) / 2) * w
            ys_col = np.nan_to_num(arr_f[:, bi], nan=0.0)
            ax_b.bar(
                x + offs,
                ys_col,
                width=w * 0.95,
                label=bar_titles[bi],
                color=colors_bar[bi % len(colors_bar)],
                edgecolor="black",
                linewidth=0.3,
            )
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(labels, rotation=25, ha="right")
        ax_b.set_ylabel("Value")
        ax_b.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.28))
        plt.tight_layout()
        save_figure(fig_b, out / f"compare_{group_key}__bars_{which}")

    _bars("best")
    _bars("last")


def compare_strategy_nclients_communication(
    *,
    results_dir: Path,
    journal_out: Path,
    project_root: Path,
    compare_groups: dict[str, list[str]] | None = None,
) -> None:
    """Strategy vs. client-count panels: macro-F1, accuracy, and loss vs uploads."""
    groups = compare_groups if compare_groups is not None else COMPARE_GROUPS
    strat_names = groups["strategy"]
    ncli_names = groups["nclients"]

    out_dir = journal_out / "compare_communication"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_journal_style()

    palette_s = multi_run_palette(len(strat_names))
    palette_n = multi_run_palette(len(ncli_names))

    def panel(
        ax: plt.Axes,
        names: list[str],
        palette: list,
        group_key: str,
        mkey: str,
        ylabel: str,
        *,
        show_xlabel: bool,
    ) -> None:
        loaded_p: list[tuple[str, list[dict]]] = []
        metas_p: list[dict[str, Any]] = []
        for rn in names:
            rd = results_dir / rn
            if not rd.is_dir():
                continue
            hist = load_round_history(rd)
            if not hist:
                continue
            loaded_p.append((rn, hist))
            metas_p.append(merge_run_meta(rd, project_root, rn, hist))

        if len(loaded_p) < 2:
            ax.set_visible(False)
            return

        line_labels = [get_clean_run_name(rn, group_key) for rn, _ in loaded_p]

        max_round = max(int(m["last_round"]) for m in metas_p)
        es_any = False
        plot_idx = 0

        for (rn, hist), meta, lab in zip(loaded_p, metas_p, line_labels, strict=True):
            ox, oy = series_for_metric(hist, mkey)
            if ox.size == 0:
                warnings.warn(f"communication panel: skip {rn} (no {mkey})")
                continue
            color = palette[plot_idx % len(palette)]
            plot_idx += 1
            order = np.argsort(ox)
            ox = ox[order]
            oy = oy[order]

            xc = cumulative_communication_rounds(
                ox,
                num_clients=int(meta["num_clients"]),
                fraction_fit=float(meta["fraction_fit"]),
                mode=str(meta["mode"]),
            )

            ax.plot(xc, oy, "-", color=color, label=lab)

            hi_r = int(round(float(ox[-1])))
            last_y = float(oy[-1])
            if hi_r < max_round:
                pad_r = np.arange(hi_r + 1, max_round + 1, dtype=float)
                xc_tail = cumulative_communication_rounds(
                    pad_r,
                    num_clients=int(meta["num_clients"]),
                    fraction_fit=float(meta["fraction_fit"]),
                    mode=str(meta["mode"]),
                )
                ax.plot(
                    xc_tail,
                    np.full_like(xc_tail, last_y),
                    linestyle=(0, (4, 4)),
                    alpha=0.5,
                    color=color,
                )

            if meta.get("early_stop"):
                lr = float(meta["last_round"])
                xc_es = cumulative_communication_rounds(
                    np.array([lr]),
                    num_clients=int(meta["num_clients"]),
                    fraction_fit=float(meta["fraction_fit"]),
                    mode=str(meta["mode"]),
                )[0]
                ax.scatter([xc_es], [last_y], marker="*", s=140, color=color, edgecolors="k")
                es_any = True

        if show_xlabel:
            ax.set_xlabel("Cumulative client uploads (approx.)")
        ax.set_ylabel(ylabel)
        if es_any:
            ax.scatter(
                [],
                [],
                marker="*",
                s=140,
                c="gray",
                edgecolors="k",
                linewidths=0.6,
                label="Early stopping point",
            )
        ax.legend(loc="lower right", frameon=True)

    metric_rows = [
        ("macro_f1", "Macro-F1"),
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 11.5), sharex=False)
    for ri, (mkey, ytitle) in enumerate(metric_rows):
        show_x = ri == len(metric_rows) - 1
        panel(axes[ri, 0], strat_names, palette_s, "strategy", mkey, ytitle, show_xlabel=show_x)
        panel(axes[ri, 1], ncli_names, palette_n, "nclients", mkey, ytitle, show_xlabel=show_x)
    axes[0, 0].set_title("Aggregation strategy", fontsize=14, pad=6)
    axes[0, 1].set_title("Federation size ($K$)", fontsize=14, pad=6)
    plt.tight_layout()
    save_figure(fig, out_dir / "compare_comm_strategy_nclients__macro_acc_loss_vs_uploads")


def plot_client_data_distribution(
    counts: np.ndarray | dict[Any, Iterable[int]],
    alpha_val: float | None,
    *,
    class_names: list[str] | None = None,
    out_path_without_ext: Path,
    title_suffix: str = "",
) -> None:
    """Heatmap: clients × classes (counts).

    ``counts`` may be shape ``(n_clients, n_classes)`` or mapping ``client_id -> class counts``.

    ``title_suffix`` is accepted for API compatibility; figures omit matplotlib titles
    (caption in prose). ``alpha_val`` is not rendered on the figure; mention $\\alpha$
    in the caption if helpful.
    """
    apply_journal_style()
    _ = title_suffix
    _ = alpha_val

    if isinstance(counts, dict):
        rows = sorted(counts.keys())
        mat_list = []
        for cid in rows:
            row = counts[cid]
            mat_list.append(np.asarray(list(row), dtype=float))
        arr = np.stack(mat_list, axis=0)
        ytick = [str(r) for r in rows]
    else:
        arr = np.asarray(counts, dtype=float)
        ytick = [str(i) for i in range(arr.shape[0])]

    if class_names is None:
        class_names = [str(j) for j in range(arr.shape[1])]

    fig, ax = plt.subplots(figsize=(max(6.5, arr.shape[1] * 0.65), max(4.0, arr.shape[0] * 0.35)))
    annot = arr.shape[0] * arr.shape[1] <= 120
    sns.heatmap(
        arr,
        ax=ax,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=ytick,
        annot=annot,
        fmt=".0f" if annot else "",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Train samples"},
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    plt.tight_layout()
    save_figure(fig, out_path_without_ext)


def client_class_counts_from_manifest(manifest_csv: Path) -> tuple[np.ndarray, list[str]]:
    """Pivot train split: rows=client_id (sorted), columns=sorted label codes.

    Note: ``manifest.csv`` is overwritten when re-partitioning; save snapshots for
    multiple $\\alpha$ comparisons.
    """
    df = pd.read_csv(manifest_csv)
    train = df[df["split"].astype(str).str.lower() == "train"].copy()
    if "client_id" not in train.columns or "label" not in train.columns:
        raise ValueError("manifest must contain client_id and label columns")

    pivot = (
        train.groupby(["client_id", "label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    classes = [str(c) for c in pivot.columns.tolist()]
    mat = pivot.to_numpy(dtype=float)
    return mat, classes


def _metric_matrix_from_clients(
    run_dir: Path, path: tuple[str, ...]
) -> tuple[list[float], list[int], np.ndarray] | None:
    rounds: list[int] = []
    by_round: list[dict[int, float]] = []
    for f in sorted(run_dir.iterdir()):
        if not f.is_file():
            continue
        m = CLIENT_FILE.match(f.name)
        if not m:
            continue
        srv_round = int(m.group(1))
        with f.open(encoding="utf-8") as fp:
            payload = json.load(fp)
        clients = payload.get("clients") or []
        row: dict[int, float] = {}
        for c in clients:
            cid = int(c.get("client_id", -1))
            cur: Any = c
            for key in path:
                if cur is None:
                    break
                cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is None:
                continue
            try:
                fv = float(cur)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            row[cid] = fv
        if row:
            rounds.append(srv_round)
            by_round.append(row)
    if not rounds:
        return None
    all_ids = sorted({k for d in by_round for k in d})
    mat = np.full((len(rounds), len(all_ids)), np.nan, dtype=float)
    for ri, d in enumerate(by_round):
        for ci, cid in enumerate(all_ids):
            if cid in d:
                mat[ri, ci] = d[cid]
    xr = [float(r) for r in rounds]
    return xr, all_ids, mat


def _plot_confusion(
    cm: list[list[float]],
    class_names: list[str],
    *,
    normalize: str | None,
) -> mpl.figure.Figure:
    arr_raw = np.asarray(cm, dtype=float)
    if normalize == "row":
        s = arr_raw.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        arr = arr_raw / s
        fmt = ".2f"
    elif normalize == "col":
        s = arr_raw.sum(axis=0, keepdims=True)
        s[s == 0] = 1
        arr = arr_raw / s
        fmt = ".2f"
    else:
        arr = np.rint(arr_raw).astype(np.int64)
        fmt = "d"
    ncls = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5.0, ncls * 0.55), max(4.5, ncls * 0.45)))
    sns.heatmap(
        arr,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.35,
        linecolor="white",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    return fig


def visualize_single_run(
    run_name: str,
    *,
    results_dir: Path,
    journal_out: Path,
) -> None:
    """Single-run journal exports (SVG + PDF)."""
    run_dir = results_dir / run_name
    out = journal_out / run_name
    out.mkdir(parents=True, exist_ok=True)
    history = load_round_history(run_dir)
    if not history:
        print(f"SKIP {run_name}: no round_*.json or summary history")
        return

    apply_journal_style()
    colors = multi_run_palette(8)

    xr = x_rounds(history)
    class_names = infer_class_names(history)
    last_i = len(history) - 1
    best_i = best_macro_f1_idx(history)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax2 = ax.twinx()
    l1, = ax.plot(
        xr,
        [safe_get(h, "accuracy") for h in history],
        color=colors[0],
        label="Accuracy",
    )
    l2, = ax2.plot(
        xr,
        [safe_get(h, "balanced_accuracy") for h in history],
        color=colors[1],
        label="Balanced acc.",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy", color=colors[0])
    ax2.set_ylabel("Balanced accuracy", color=colors[1])
    ax.tick_params(axis="y", labelcolor=colors[0])
    ax2.tick_params(axis="y", labelcolor=colors[1])
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best", frameon=True)
    save_figure(fig, out / "01_curves_accuracy_balanced_acc")

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(xr, [safe_get(h, "macro_f1") for h in history], color=colors[2])
    ax.set_xlabel("Round")
    ax.set_ylabel("Macro-F1")
    save_figure(fig, out / "02_curves_macro_f1")

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(xr, [safe_get(h, "loss") for h in history], color=colors[3])
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    save_figure(fig, out / "03_curves_loss")

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax2 = ax.twinx()
    l1, = ax.plot(xr, [safe_get(h, "auroc") for h in history], color=colors[4], label="AUROC")
    l2, = ax2.plot(xr, [safe_get(h, "auprc") for h in history], color=colors[5], label="AUPRC")
    ax.set_xlabel("Round")
    ax.set_ylabel("AUROC", color=colors[4])
    ax2.set_ylabel("AUPRC", color=colors[5])
    ax.tick_params(axis="y", labelcolor=colors[4])
    ax2.tick_params(axis="y", labelcolor=colors[5])
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best", frameon=True)
    save_figure(fig, out / "04_curves_auroc_auprc")

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(
        xr,
        [safe_get(h, "macro_precision") for h in history],
        color=colors[0],
        label="Macro-precision",
    )
    ax.plot(
        xr,
        [safe_get(h, "macro_recall") for h in history],
        color=colors[1],
        label="Macro-recall",
    )
    ax.plot(
        xr,
        [safe_get(h, "macro_specificity") for h in history],
        color=colors[2],
        label="Macro-specificity",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.legend(loc="best", frameon=True)
    save_figure(fig, out / "05_curves_macro_pr_sp")

    eces = [safe_get(h, "ece") for h in history]
    if any(math.isfinite(e) for e in eces):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ex, ey = [], []
        for x, e in zip(xr, eces, strict=False):
            if math.isfinite(e):
                ex.append(x)
                ey.append(e)
        if ex:
            ax.plot(ex, ey, color=colors[6])
            ax.set_xlabel("Round")
            ax.set_ylabel("ECE")
            save_figure(fig, out / "06_curves_ece")

    scalar_keys = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Bal. acc."),
        ("macro_f1", "Macro-F1"),
        ("loss", "Loss"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("macro_precision", "Macro-prec."),
        ("macro_recall", "Macro-recall"),
        ("macro_specificity", "Macro-spec."),
        ("ece", "ECE"),
    ]
    n_p = len(scalar_keys)
    n_cols = 3
    n_rows = int(math.ceil(n_p / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5, 1.2 + 2.0 * n_rows), sharex=False)
    axes_list = np.atleast_1d(axes).ravel()
    for ax_i, (key, label) in zip(axes_list, scalar_keys, strict=False):
        ys = [safe_get(h, key) for h in history]
        ax_i.plot(xr, ys, color=colors[0])
        ax_i.set_ylabel(label)
        ax_i.set_xlabel("Round")
    for j in range(len(scalar_keys), len(axes_list)):
        axes_list[j].set_visible(False)
    plt.tight_layout()
    save_figure(fig, out / "07_curves_all_scalars_small_multiples")

    h0 = history[0]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    da = [safe_get(h, "accuracy") - safe_get(h0, "accuracy") for h in history]
    df = [safe_get(h, "macro_f1") - safe_get(h0, "macro_f1") for h in history]
    ax.plot(xr, da, color=colors[0], label=r"$\Delta$ accuracy")
    ax.plot(xr, df, color=colors[2], label=r"$\Delta$ macro-F1")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Round")
    ax.set_ylabel("Delta vs round 0")
    ax.legend(loc="best", frameon=True)
    save_figure(fig, out / "08_delta_accuracy_macro_f1")

    if class_names:
        cm_last = history[last_i].get("confusion_matrix")
        cm_best = history[best_i].get("confusion_matrix")
        if cm_last is not None:
            fig = _plot_confusion(cm_last, class_names, normalize=None)
            save_figure(fig, out / "09_confusion_counts_last")
            fig = _plot_confusion(cm_last, class_names, normalize="row")
            save_figure(fig, out / "10_confusion_row_norm_last")
            fig = _plot_confusion(cm_last, class_names, normalize="col")
            save_figure(fig, out / "11_confusion_col_norm_last")
        if cm_best is not None:
            fig = _plot_confusion(cm_best, class_names, normalize=None)
            save_figure(fig, out / "12_confusion_counts_best_macro_f1")
            fig = _plot_confusion(cm_best, class_names, normalize="row")
            save_figure(fig, out / "13_confusion_row_norm_best_macro_f1")

        def _bars_per_class(h: dict, stem: str) -> None:
            pcf = h.get("per_class_f1")
            if not pcf:
                return
            fig_b, ax_b = plt.subplots(figsize=(max(5.2, len(pcf) * 0.48), 3.2))
            x_b = np.arange(len(pcf))
            ax_b.bar(x_b, pcf, color=colors[2], edgecolor="black", linewidth=0.35)
            ax_b.set_xticks(x_b)
            ax_b.set_xticklabels(class_names[: len(pcf)], rotation=35, ha="right")
            ax_b.set_ylabel("Per-class F1")
            plt.tight_layout()
            save_figure(fig_b, out / stem)

        _bars_per_class(history[last_i], "14_per_class_f1_bars_last")
        _bars_per_class(history[best_i], "15_per_class_f1_bars_best_macro_f1")

        pcf_all = [h.get("per_class_f1") for h in history]
        n_c = len(pcf_all[-1] or [])
        if n_c > 0 and all(pcf_all[i] is not None and len(pcf_all[i]) == n_c for i in range(len(history))):
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            for ci in range(n_c):
                ys = [float(pcf_all[i][ci]) for i in range(len(history))]
                ax.plot(
                    xr,
                    ys,
                    color=colors[ci % len(colors)],
                    label=class_names[ci] if ci < len(class_names) else f"C{ci}",
                )
            ax.set_xlabel("Round")
            ax.set_ylabel("Per-class F1")
            ax.legend(ncol=2, loc="best", frameon=True)
            save_figure(fig, out / "16_per_class_f1_lines_vs_round")

    def _lines_client_metric(path: tuple[str, ...], stem: str, ylabel: str) -> None:
        tri = _metric_matrix_from_clients(run_dir, path)
        if tri is None:
            return
        rx, cids, mat = tri
        fig_w = max(5.0, min(11.0, 0.42 * len(cids) + 3.0))
        fig, ax = plt.subplots(figsize=(fig_w, 3.5))
        for j, cid in enumerate(cids):
            ys = mat[:, j]
            ax.plot(rx, ys, color=colors[j % len(colors)], label=f"C{cid}")
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ncol = min(4, max(1, len(cids)))
        ax.legend(ncol=ncol, loc="upper right", fontsize=8)
        plt.tight_layout()
        save_figure(fig, out / stem)

    _lines_client_metric(("train_loss",), "18_lines_client_train_loss", "Train loss")
    _lines_client_metric(
        ("local_test", "macro_f1"),
        "19_lines_client_local_macro_f1",
        "Local macro-F1",
    )
    _lines_client_metric(
        ("local_test", "auroc"),
        "20_lines_client_local_auroc",
        "Local AUROC",
    )


def list_calibration_pngs(run_name: str, *, figures_dir: Path) -> None:
    p = figures_dir / run_name
    if not p.is_dir():
        print(f"No figures dir for {run_name}")
        return
    files = sorted(p.glob("calibration_round_*.png"))
    if not files:
        print(f"No calibration_round_*.png under {p}")
        return
    print(f"Calibration PNGs ({len(files)}):")
    for f in files:
        print(" ", f)
