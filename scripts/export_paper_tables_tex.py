"""
Write ``paper/tables/tab_t1``–``tab_t6`` from ``results/`` (best-macro-F1 round per federated row).

Columns: accuracy, macro-F1, loss at the selected round, plus Round for ablations.
``tab_t6`` lists per-class F1 for the primary federated run (same checkpoint as T1 FedAvg row).

Defaults to ``results/isic2019`` and ``paper/tables``. For ``results/isic2018``, manifests and
default output dir ``paper/tables/isic2018`` follow ``src/report/manuscript_runs.py``.

Usage (from repo root)::

    python scripts/export_paper_tables_tex.py
    python scripts/export_paper_tables_tex.py --results-dir results/isic2019 --selection best_macro_f1
    python scripts/export_paper_tables_tex.py --results-dir results/isic2018
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.report.manuscript_runs import (  # noqa: E402
    ISIC2018_CLASS_LABELS,
    ISIC2019_CLASS_LABELS,
    ManifestRow,
    _resolve_run_dir,
    escape_latex,
    infer_paper_dataset,
    manifest_isic2018_t1_primary_vs_centralized,
    manifest_isic2018_t2_partitioning,
    manifest_isic2018_t3_nclients,
    manifest_isic2018_t4_localepochs,
    manifest_isic2018_t5_aggregation,
    manifest_isic2018_t6_primary,
    manifest_t1_primary_vs_centralized,
    manifest_t2_partitioning,
    manifest_t3_nclients,
    manifest_t4_localepochs,
    manifest_t5_aggregation,
    manifest_t8_primary,
)
from src.report.metrics import RoundSelection, get_scalar, load_run_metrics  # noqa: E402


def _default_paper_tables(results_dir: Path) -> Path:
    if infer_paper_dataset(results_dir) == "isic2018":
        return Path("paper/tables/isic2018")
    return Path("paper/tables")


def _pct(m) -> str:
    v = get_scalar(m, "accuracy")
    if v is None or v != v:
        return "---"
    return f"{v * 100:.2f}\\%"


def _macro_f1_pct(m) -> str:
    v = get_scalar(m, "macro_f1")
    if v is None or v != v:
        return "---"
    return f"{v * 100:.2f}\\%"


def _scalar_pct(v: object) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "---"
    return f"{float(v) * 100:.2f}\\%"


def _loss(m) -> str:
    v = get_scalar(m, "loss")
    if v is None or v != v:
        return "---"
    return f"{v:.4f}"


def _round_cell(m) -> str:
    if m.missing or m.round_index is None:
        return "---"
    return str(m.round_index)


def _label_tex(row: ManifestRow) -> str:
    if row.latex_label is not None:
        return row.latex_label
    return escape_latex(row.label)


def t1_tex(results_dir: Path, selection: RoundSelection) -> str:
    ds = infer_paper_dataset(results_dir)
    if ds == "isic2018":
        rows = manifest_isic2018_t1_primary_vs_centralized()
        hdr = (
            r"% Table T1 — primary federated vs centralized (ISIC 2018); "
            r"best-macro-F1 round for FL, summary flat for centralized"
        )
    else:
        rows = manifest_t1_primary_vs_centralized()
        hdr = (
            r"% Table T1 — primary federated vs centralized (ISIC 2019); "
            r"best-macro-F1 round for FL, summary flat for centralized"
        )
    lines = [
        hdr,
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Setting & Acc.\ (\%) & Macro-F1 (\%) & Loss \\",
        r"\midrule",
    ]
    for spec in rows:
        m = load_run_metrics(_resolve_run_dir(results_dir, spec), selection)
        lines.append(f"{_label_tex(spec)} & {_pct(m)} & {_macro_f1_pct(m)} & {_loss(m)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def ablation_tex(
    results_dir: Path,
    selection: RoundSelection,
    rows: tuple[ManifestRow, ...],
    *,
    comment: str,
    header_run: str,
) -> str:
    out = [
        f"% {comment}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        header_run,
        r"\midrule",
    ]
    for spec in rows:
        m = load_run_metrics(_resolve_run_dir(results_dir, spec), selection)
        out.append(
            f"{_label_tex(spec)} & {_pct(m)} & {_macro_f1_pct(m)} & {_loss(m)} & {_round_cell(m)} \\\\"
        )
    out.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(out)


def t6_perclass_tex(results_dir: Path, selection: RoundSelection) -> str:
    ds = infer_paper_dataset(results_dir)
    if ds == "isic2018":
        primary = manifest_isic2018_t6_primary()[0]
        classes = ISIC2018_CLASS_LABELS
        year_tex = "2018"
    else:
        primary = manifest_t8_primary()[0]
        classes = ISIC2019_CLASS_LABELS
        year_tex = "2019"
    m = load_run_metrics(_resolve_run_dir(results_dir, primary), selection)
    lines = [
        rf"% Table T6 — per-class F1 (ISIC {year_tex} primary federated run; same checkpoint as Table T1 federated row)",
        r"\begin{tabular}{lc}",
        r"\toprule",
        rf"Class (ISIC {year_tex}) & F1 (\%) \\",
        r"\midrule",
    ]
    f1s = (m.data.get("per_class_f1") or []) if not m.missing else []
    for i, cls in enumerate(classes):
        v = f1s[i] if i < len(f1s) else None
        lines.append(rf"{escape_latex(cls)} & {_scalar_pct(v)} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/isic2019")
    parser.add_argument(
        "--selection",
        choices=("last", "best_macro_f1"),
        default="best_macro_f1",
    )
    parser.add_argument(
        "--paper-tables",
        default=None,
        help="Output directory for tab_t*.tex (default: paper/tables or paper/tables/isic2018)",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    selection: RoundSelection = args.selection  # type: ignore[assignment]
    tables = Path(args.paper_tables) if args.paper_tables else _default_paper_tables(results_dir)

    ds = infer_paper_dataset(results_dir)
    if ds == "isic2018":
        m2 = manifest_isic2018_t2_partitioning
        m3 = manifest_isic2018_t3_nclients
        m4 = manifest_isic2018_t4_localepochs
        m5 = manifest_isic2018_t5_aggregation
    else:
        m2 = manifest_t2_partitioning
        m3 = manifest_t3_nclients
        m4 = manifest_t4_localepochs
        m5 = manifest_t5_aggregation

    tables.mkdir(parents=True, exist_ok=True)
    (tables / "tab_t1_main.tex").write_text(t1_tex(results_dir, selection), encoding="utf-8")
    (tables / "tab_t2_partition.tex").write_text(
        ablation_tex(
            results_dir,
            selection,
            m2(),
            comment="Table T2 — partitioning ablation; best-macro-F1 round per row",
            header_run=r"Setting & Acc.\ (\%) & Macro-F1 (\%) & Loss & Round \\",
        ),
        encoding="utf-8",
    )
    (tables / "tab_t3_nclients.tex").write_text(
        ablation_tex(
            results_dir,
            selection,
            m3(),
            comment=r"Table T3 — federation size $K$; best-macro-F1 round per row",
            header_run=r"Setting & Acc.\ (\%) & Macro-F1 (\%) & Loss & Round \\",
        ),
        encoding="utf-8",
    )
    (tables / "tab_t4_localepochs.tex").write_text(
        ablation_tex(
            results_dir,
            selection,
            m4(),
            comment=r"Table T4 — local epochs per round; best-macro-F1 round per row",
            header_run=r"Setting & Acc.\ (\%) & Macro-F1 (\%) & Loss & Round \\",
        ),
        encoding="utf-8",
    )
    (tables / "tab_t5_aggregation.tex").write_text(
        ablation_tex(
            results_dir,
            selection,
            m5(),
            comment="Table T5 — server aggregation; best-macro-F1 round per row",
            header_run=r"Setting & Acc.\ (\%) & Macro-F1 (\%) & Loss & Round \\",
        ),
        encoding="utf-8",
    )
    (tables / "tab_t6_perclass.tex").write_text(t6_perclass_tex(results_dir, selection), encoding="utf-8")
    print(f"Wrote tab_t1_main.tex … tab_t6_perclass.tex under {tables.resolve()}")


if __name__ == "__main__":
    main()
