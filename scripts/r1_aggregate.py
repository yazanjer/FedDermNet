"""
Aggregate the R1 grid into manuscript tables, statistics and figures.

  python scripts/r1_aggregate.py --results results_r1 --out paper_r1

Outputs (under --out)
  summary_runs.csv          one row per run: test metrics at the selected round
  summary_configs.csv       mean / sd over seeds per configuration
  stats.json                bootstrap CIs and paired comparisons used in the text
  tables/*.tex              booktabs tables
  figs/*.pdf|png            journal-style figures (300 dpi PNG + vector PDF)

Statistics
  * mean +- sd across the three training seeds;
  * 95% bootstrap CI of the seed-averaged macro-F1 (test images resampled jointly
    across seeds, 2000 replicates);
  * paired comparison against the reference setting: seed-averaged difference in
    macro-F1 with a 95% paired bootstrap CI over test images; a difference is called
    statistically distinguishable only when that CI excludes zero.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLASSES = {
    "isic2019": ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
    "isic2018": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
}
NAME_RE = re.compile(r"^(isic20\d\d)_(.+)_s(\d+)$")
REF_TAG = "fedavg_a0.5_K10_E2"
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "auroc", "auprc", "loss"]
B = 2000


# ── loading ───────────────────────────────────────────────────────────────────
def load_runs(res: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(res.glob("*/summary.json")):
        m = NAME_RE.match(f.parent.name)
        if not m:
            continue
        ds, tag, seed = m.group(1), m.group(2), int(m.group(3))
        d = json.loads(f.read_text())
        sel = d.get("selected")
        if sel is None:
            continue
        t, v = sel["test"], sel["val"]
        hist = d.get("history", [])
        rows.append({
            "dataset": ds, "tag": tag, "seed": seed, "run": f.parent.name,
            **{m_: t.get(m_) for m_ in METRICS},
            "val_macro_f1": v.get("macro_f1"),
            "best_round": sel.get("best_round"),
            "last_round": max((h.get("round", 0) for h in hist), default=0),
            "n_test": t.get("n"),
        })
    return pd.DataFrame(rows)


def load_preds(res: Path, run: str):
    z = np.load(res / run / "test_predictions_best.npz", allow_pickle=True)
    return z["probs"], z["labels"].astype(int), z["image_id"]


def macro_f1(y, p, k):
    return f1_score(y, p, average="macro", labels=list(range(k)), zero_division=0)


# ── bootstrap ─────────────────────────────────────────────────────────────────
def boot_mean_f1(preds: list[tuple[np.ndarray, np.ndarray]], k: int, rng) -> tuple[float, float, float]:
    """CI of the seed-averaged macro-F1; images resampled jointly across seeds."""
    y = preds[0][1]
    n = len(y)
    point = float(np.mean([macro_f1(yy, pp, k) for pp, yy in preds]))
    idx = rng.randint(0, n, size=(B, n))
    reps = np.empty(B)
    for b in range(B):
        ii = idx[b]
        reps[b] = np.mean([macro_f1(yy[ii], pp[ii], k) for pp, yy in preds])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def boot_paired_diff(a, b_, k: int, rng) -> dict:
    """Seed-averaged macro-F1 difference (a - b) with a paired image bootstrap."""
    y = a[0][1]
    for (_, ya), (_, yb) in zip(a, b_):
        assert np.array_equal(ya, y) and np.array_equal(yb, y), "test sets differ"
    n = len(y)
    d_seed = [macro_f1(y, pa, k) - macro_f1(y, pb, k) for (pa, _), (pb, _) in zip(a, b_)]
    idx = rng.randint(0, n, size=(B, n))
    reps = np.empty(B)
    for bb in range(B):
        ii = idx[bb]
        reps[bb] = np.mean([macro_f1(y[ii], pa[ii], k) - macro_f1(y[ii], pb[ii], k)
                            for (pa, _), (pb, _) in zip(a, b_)])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"diff": float(np.mean(d_seed)), "ci": [float(lo), float(hi)],
            "per_seed": [float(x) for x in d_seed],
            "distinguishable": bool(lo > 0 or hi < 0)}


def argmax_preds(res, runs):
    out = []
    for r in runs:
        p, y, _ = load_preds(res, r)
        out.append((p.argmax(1), y))
    return out


# ── figures ───────────────────────────────────────────────────────────────────
def mpl_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return plt


PALETTE = ["#2F4B7C", "#A05195", "#D45087", "#F95D6A", "#FF7C43", "#665191", "#4C7A5A", "#7F7F7F"]
MUTED = ["#1F3A5F", "#8C4A2F", "#3E7C59", "#7A6A9A", "#9E8F3D", "#5C5C5C"]


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)


def curve(res: Path, run: str, split: str, key="macro_f1"):
    d = json.loads((res / run / "summary.json").read_text())
    h = d["history"]
    r = [x["round"] for x in h if split in x]
    v = [x[split][key] for x in h if split in x]
    return np.array(r), np.array(v)


def fig_sweep(plt, res, runs_df, ds, tags, labels, name, out, title):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        for c, (tag, lab) in enumerate(zip(tags, labels)):
            sub = runs_df[(runs_df.dataset == ds) & (runs_df.tag == tag)].sort_values("seed")
            if sub.empty:
                continue
            cs = [curve(res, r, split) for r in sub.run]
            L = min(len(x[0]) for x in cs)
            rr = cs[0][0][:L]
            M = np.stack([x[1][:L] for x in cs]) * 100
            ax.plot(rr, M.mean(0), color=MUTED[c % len(MUTED)], label=lab)
            if len(cs) > 1:
                ax.fill_between(rr, M.min(0), M.max(0), color=MUTED[c % len(MUTED)], alpha=0.15, lw=0)
        ax.set_xlabel("Communication round")
        ax.set_title(f"{'Validation' if split == 'val' else 'Test'} macro-F1", loc="left")
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Macro-F1 (%)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=8)
    save(fig, out, name)
    plt.close(fig)


def fig_confusion(plt, res, runs_df, ds, tags, labels, name, out):
    k = len(CLASSES[ds])
    fig, axes = plt.subplots(1, len(tags), figsize=(3.4 * len(tags), 3.0))
    axes = np.atleast_1d(axes)
    for ax, tag, lab in zip(axes, tags, labels):
        sub = runs_df[(runs_df.dataset == ds) & (runs_df.tag == tag)]
        cm = np.zeros((k, k))
        for r in sub.run:
            p, y, _ = load_preds(res, r)
            yp = p.argmax(1)
            for a, b in zip(y, yp):
                cm[a, b] += 1
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.55 else "black")
        ax.set_xticks(range(k)); ax.set_xticklabels(CLASSES[ds], rotation=45)
        ax.set_yticks(range(k)); ax.set_yticklabels(CLASSES[ds])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(lab, loc="left")
        ax.spines[:].set_visible(True)
    save(fig, out, name)
    plt.close(fig)


def fig_partition(plt, res, ds, out):
    k = len(CLASSES[ds])
    alphas = [("0.1", r"$\alpha=0.1$"), ("0.5", r"$\alpha=0.5$"), ("1", r"$\alpha=1.0$"), ("100", r"$\alpha=100$")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), sharey=True)
    cmap = plt.get_cmap("tab10") if False else None
    colors = ["#2F4B7C", "#6A7FA8", "#A05195", "#C98BB9", "#4C7A5A", "#8FB59A", "#9E8F3D", "#5C5C5C"]
    for ax, (a, lab) in zip(axes, alphas):
        f = res / f"{ds}_fedavg_a{a}_K10_E2_s42" / "client_assignment.csv.gz"
        if not f.is_file():
            continue
        df = pd.read_csv(f)
        tab = pd.crosstab(df.client_id, df.label).reindex(columns=range(k), fill_value=0)
        bottom = np.zeros(len(tab))
        for c in range(k):
            ax.bar(tab.index, tab[c].values, bottom=bottom, color=colors[c], width=0.8, lw=0, label=CLASSES[ds][c])
            bottom += tab[c].values
        ax.set_title(lab, loc="left"); ax.set_xlabel("Client")
        ax.set_xticks(range(len(tab)))
    axes[0].set_ylabel("Training images")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=6)
    save(fig, out, f"partition_{ds}")
    plt.close(fig)


# ── tables ────────────────────────────────────────────────────────────────────
def pm(m, s, scale=100.0, nd=2):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    if s is None or np.isnan(s):
        return f"{m*scale:.{nd}f}"
    return f"{m*scale:.{nd}f} $\\pm$ {s*scale:.{nd}f}"


def config_table(cfg: pd.DataFrame, ds: str, tags: list[str], labels: list[str], caption: str, label: str,
                 extra_ci: dict | None = None) -> str:
    lines = [r"\begin{table*}[!t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Setting & Acc.\ (\%) & Bal.-acc.\ (\%) & Macro-F1 (\%) & AUROC (\%) & AUPRC (\%) & Loss & Round \\",
             r"\midrule"]
    for tag, lab in zip(tags, labels):
        r = cfg[(cfg.dataset == ds) & (cfg.tag == tag)]
        if r.empty:
            continue
        r = r.iloc[0]
        f1 = pm(r.macro_f1_mean, r.macro_f1_sd)
        if extra_ci and tag in extra_ci:
            lo, hi = extra_ci[tag]
            f1 += f" [{lo*100:.1f}, {hi*100:.1f}]"
        lines.append(
            f"{lab} & {pm(r.accuracy_mean, r.accuracy_sd)} & {pm(r.balanced_accuracy_mean, r.balanced_accuracy_sd)} & "
            f"{f1} & {pm(r.auroc_mean, r.auroc_sd)} & {pm(r.auprc_mean, r.auprc_sd)} & "
            f"{pm(r.loss_mean, r.loss_sd, 1.0, 3)} & {r.best_round_mean:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def per_class_table(res, runs_df, ds, tag_fed, tag_cen, caption, label) -> tuple[str, dict]:
    names = CLASSES[ds]
    keys = [("per_class_sensitivity", "Sens."), ("per_class_specificity", "Spec."), ("per_class_f1", "F1"),
            ("per_class_auroc", "AUROC"), ("per_class_auprc", "AUPRC")]
    data = {}
    for tag in (tag_fed, tag_cen):
        sub = runs_df[(runs_df.dataset == ds) & (runs_df.tag == tag)]
        vals = {k: [] for k, _ in keys}
        sup = None
        for r in sub.run:
            t = json.loads((res / r / "summary.json").read_text())["selected"]["test"]
            for k, _ in keys:
                vals[k].append(t[k])
            sup = t.get("per_class_support")
        data[tag] = {k: (np.nanmean(v, 0), np.nanstd(v, 0, ddof=1) if len(v) > 1 else np.full(len(names), np.nan))
                     for k, v in vals.items()}
        data[tag]["support"] = sup
    lines = [r"\begin{table*}[!t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3.2pt}",
             r"\begin{tabular}{lr" + "c" * 5 + "c" * 5 + "}", r"\toprule",
             r" & & \multicolumn{5}{c}{Federated ($\alpha{=}0.5$)} & \multicolumn{5}{c}{Centralized reference} \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
             "Class & $n$ & " + " & ".join(h for _, h in keys) + " & " + " & ".join(h for _, h in keys) + r" \\",
             r"\midrule"]
    for i, c in enumerate(names):
        cells = []
        for tag in (tag_fed, tag_cen):
            for k, _ in keys:
                m, s = data[tag][k][0][i], data[tag][k][1][i]
                cells.append(f"{m*100:.1f}" + ("" if np.isnan(s) else f"$\\pm${s*100:.1f}"))
        sup = data[tag_fed]["support"][i] if data[tag_fed]["support"] else ""
        lines.append(f"{c} & {sup} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines), {t: {k: [list(map(float, v[0])), list(map(float, v[1]))] for k, v in d.items() if k != "support"}
                              for t, d in data.items()}


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_r1")
    ap.add_argument("--out", default="paper_r1")
    ap.add_argument("--ref", default=REF_TAG)
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(12345)

    runs = load_runs(res)
    runs.to_csv(out / "summary_runs.csv", index=False)
    agg = {}
    for m in METRICS + ["best_round", "last_round", "val_macro_f1"]:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_sd"] = (m, lambda x: x.std(ddof=1) if len(x) > 1 else np.nan)
    cfg = runs.groupby(["dataset", "tag"]).agg(n_seeds=("seed", "nunique"), **agg).reset_index()
    cfg.to_csv(out / "summary_configs.csv", index=False)

    ref = args.ref
    tuning = {}
    tsel = res / "tuning_selection.json"
    if tsel.is_file():
        tuning = json.loads(tsel.read_text())

    stats: dict = {"tuning_selection": tuning, "configs": {}, "paired_vs_reference": {}, "ci_macro_f1": {}}
    for _, r in cfg.iterrows():
        stats["configs"][f"{r.dataset}|{r.tag}"] = {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v))
                                                    for k, v in r.items()}

    def seeds_runs(ds, tag):
        sub = runs[(runs.dataset == ds) & (runs.tag == tag)].sort_values("seed")
        return list(sub.run), list(sub.seed)

    for ds in CLASSES:
        k = len(CLASSES[ds])
        ref_runs, ref_seeds = seeds_runs(ds, ref)
        if not ref_runs:
            continue
        for tag in sorted(runs[runs.dataset == ds].tag.unique()):
            rr, ss = seeds_runs(ds, tag)
            if not rr:
                continue
            pr = argmax_preds(res, rr)
            stats["ci_macro_f1"][f"{ds}|{tag}"] = boot_mean_f1(pr, k, rng)
            if tag == ref:
                continue
            common = [s for s in ss if s in ref_seeds]
            if not common:
                continue
            a = argmax_preds(res, [r for r, s in zip(rr, ss) if s in common])
            b = argmax_preds(res, [r for r, s in zip(ref_runs, ref_seeds) if s in common])
            stats["paired_vs_reference"][f"{ds}|{tag}"] = boot_paired_diff(a, b, k, rng)

    # Strategy comparison: pairwise among tuned rules.
    stats["strategy_pairs"] = {}
    for ds in CLASSES:
        k = len(CLASSES[ds])
        chosen = {s: None for s in ("fedavg", "fedprox", "fedadam")}
        for s in chosen:
            key = f"{ds}|{s}"
            if key in tuning:
                ch = tuning[key]
                chosen[s] = re.sub(r"^isic20\d\d_|_s\d+$", "", ch["run"])
        tags = [t for t in chosen.values() if t]
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                ra, sa = seeds_runs(ds, tags[i]); rb, sb = seeds_runs(ds, tags[j])
                common = sorted(set(sa) & set(sb))
                if not common:
                    continue
                a = argmax_preds(res, [r for r, s in zip(ra, sa) if s in common])
                b = argmax_preds(res, [r for r, s in zip(rb, sb) if s in common])
                stats["strategy_pairs"][f"{ds}|{tags[i]} - {tags[j]}"] = boot_paired_diff(a, b, k, rng)
        stats.setdefault("strategy_tags", {})[ds] = chosen

    # Federated vs centralized
    stats["fed_vs_central"] = {}
    for ds in CLASSES:
        k = len(CLASSES[ds])
        for tag in (ref, "fedavg_a100_K10_E2"):
            ra, sa = seeds_runs(ds, tag); rb, sb = seeds_runs(ds, "centralized")
            common = sorted(set(sa) & set(sb))
            if common:
                a = argmax_preds(res, [r for r, s in zip(ra, sa) if s in common])
                b = argmax_preds(res, [r for r, s in zip(rb, sb) if s in common])
                stats["fed_vs_central"][f"{ds}|{tag}"] = boot_paired_diff(a, b, k, rng)

    # Cross-release transfer
    cross = {}
    for f in sorted((res / "cross_eval").glob("cross_s*.json")):
        s = int(re.search(r"s(\d+)", f.stem).group(1))
        d = json.loads(f.read_text())
        for key, v in d.items():
            cross.setdefault(key, []).append({"seed": s, **{m: v["metrics"].get(m) for m in METRICS}, **v["meta"]})
    cstats = {}
    for key, rows in cross.items():
        df = pd.DataFrame(rows)
        cstats[key] = {m: [float(df[m].mean()), float(df[m].std(ddof=1)) if len(df) > 1 else None] for m in METRICS}
        cstats[key]["n_test"] = int(df["n_test"].iloc[0])
        cstats[key]["meta"] = {k: rows[0][k] for k in rows[0] if k.startswith("n_")}
        preds = []
        for s in df.seed:
            z = np.load(res / "cross_eval" / f"{key}_s{s}.npz", allow_pickle=True)
            preds.append((z["probs"].argmax(1), z["labels"].astype(int)))
        cstats[key]["macro_f1_ci"] = boot_mean_f1(preds, 7, rng)
    stats["cross"] = cstats

    # Split audits
    for ds in CLASSES:
        f = res / f"split_audit_{ds}.json"
        if f.is_file():
            stats[f"split_audit_{ds}"] = json.loads(f.read_text())

    # Tables
    T = out / "tables"
    ci = {ds: {t.split("|")[1]: (v[1], v[2]) for t, v in stats["ci_macro_f1"].items() if t.startswith(ds)} for ds in CLASSES}
    for ds, nice in (("isic2019", "ISIC~2019"), ("isic2018", "ISIC~2018")):
        (T / f"{ds}_main.tex").write_text(config_table(
            cfg, ds, [ref, "fedavg_a100_K10_E2", "centralized"],
            [r"Federated, $\alpha{=}0.5$ (reference)", r"Federated, $\alpha{=}100$ (near-IID)", "Centralized reference"],
            f"{nice}: federated reference setting, near-IID federation and centralized reference (test split, "
            r"mean $\pm$ s.d. over three seeds; bracket: 95\% bootstrap CI of the seed-averaged macro-F1).",
            f"tab:{ds[-2:]}-main", ci[ds]))
        (T / f"{ds}_partition.tex").write_text(config_table(
            cfg, ds, ["fedavg_a0.1_K10_E2", ref, "fedavg_a1_K10_E2", "fedavg_a100_K10_E2"],
            [r"$\alpha=0.1$", r"$\alpha=0.5$", r"$\alpha=1.0$", r"$\alpha=100$"],
            f"{nice}: label-skew sweep ($K{{=}}10$, $E{{=}}2$, FedAvg).", f"tab:{ds[-2:]}-partition", ci[ds]))
        (T / f"{ds}_nclients.tex").write_text(config_table(
            cfg, ds, ["fedavg_a0.5_K5_E2", ref, "fedavg_a0.5_K20_E2"], ["$K=5$", "$K=10$", "$K=20$"],
            f"{nice}: federation size ($\\alpha{{=}}0.5$, $E{{=}}2$, FedAvg).", f"tab:{ds[-2:]}-nclients", ci[ds]))
        (T / f"{ds}_localepochs.tex").write_text(config_table(
            cfg, ds, ["fedavg_a0.5_K10_E1", ref, "fedavg_a0.5_K10_E5", "fedavg_a0.5_K10_E10"],
            ["$E=1$", "$E=2$", "$E=5$", "$E=10$"],
            f"{nice}: local epochs per round ($\\alpha{{=}}0.5$, $K{{=}}10$, FedAvg).", f"tab:{ds[-2:]}-localepochs", ci[ds]))
        st = stats.get("strategy_tags", {}).get(ds, {})
        tags = [st.get(s) for s in ("fedavg", "fedprox", "fedadam") if st.get(s)]
        labs = []
        for s in ("fedavg", "fedprox", "fedadam"):
            if not st.get(s):
                continue
            ch = tuning.get(f"{ds}|{s}", {})
            p = {"lr": r"client lr", "fedprox_mu": r"$\mu$", "fedadam_eta": r"$\eta$"}.get(ch.get("param"), "")
            labs.append(f"{ {'fedavg':'FedAvg','fedprox':'FedProx','fedadam':'FedAdam'}[s] } ({p}$=${ch.get('value'):g})")
        if tags:
            (T / f"{ds}_strategy.tex").write_text(config_table(
                cfg, ds, tags, labs,
                f"{nice}: aggregation rules, each with its validation-selected setting from an equal three-point "
                r"grid ($\alpha{=}0.5$, $K{=}10$, $E{=}2$).", f"tab:{ds[-2:]}-strategy", ci[ds]))
        txt, pcd = per_class_table(res, runs, ds, ref, "centralized",
                                   f"{nice}: per-class test sensitivity, specificity, F1, AUROC and AUPRC (\\%, mean $\\pm$ s.d. over three seeds).",
                                   f"tab:{ds[-2:]}-perclass")
        (T / f"{ds}_perclass.tex").write_text(txt)
        stats[f"per_class_{ds}"] = pcd

    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    # Figures
    plt = mpl_setup()
    F = out / "figs"
    for ds, nice in (("isic2019", "ISIC 2019"), ("isic2018", "ISIC 2018")):
        fig_sweep(plt, res, runs, ds, ["fedavg_a0.1_K10_E2", ref, "fedavg_a1_K10_E2", "fedavg_a100_K10_E2", "centralized"],
                  [r"$\alpha=0.1$", r"$\alpha=0.5$", r"$\alpha=1.0$", r"$\alpha=100$", "Centralized"], f"{ds}_partition_curves", F,
                  f"{nice}: label-skew sweep")
        fig_sweep(plt, res, runs, ds, ["fedavg_a0.5_K5_E2", ref, "fedavg_a0.5_K20_E2"], ["$K=5$", "$K=10$", "$K=20$"],
                  f"{ds}_nclients_curves", F, f"{nice}: federation size")
        fig_sweep(plt, res, runs, ds, ["fedavg_a0.5_K10_E1", ref, "fedavg_a0.5_K10_E5", "fedavg_a0.5_K10_E10"],
                  ["$E=1$", "$E=2$", "$E=5$", "$E=10$"], f"{ds}_localepochs_curves", F, f"{nice}: local epochs")
        st = stats.get("strategy_tags", {}).get(ds, {})
        tags = [st[s] for s in ("fedavg", "fedprox", "fedadam") if st.get(s)]
        if tags:
            fig_sweep(plt, res, runs, ds, tags, ["FedAvg", "FedProx", "FedAdam"][: len(tags)], f"{ds}_strategy_curves", F,
                      f"{nice}: aggregation rules (tuned)")
        fig_confusion(plt, res, runs, ds, [ref, "centralized"], [r"Federated ($\alpha=0.5$)", "Centralized"], f"{ds}_confusion", F)
        fig_partition(plt, res, ds, F)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
