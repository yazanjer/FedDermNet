"""Manifest and Markdown helpers for results tables T1–T6 (Markdown report)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from src.report.metrics import RoundSelection, RunMetrics, get_scalar, load_run_metrics

ISIC2019_CLASS_LABELS = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
ISIC2018_CLASS_LABELS = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")

PaperDataset = Literal["isic2019", "isic2018"]


@dataclass(frozen=True)
class ManifestRow:
    """One logical experiment row in a result table."""

    label: str
    run_folder: str
    fallback_folder: str | None = None
    latex_label: str | None = None


def _resolve_run_dir(results_dir: Path, row: ManifestRow) -> Path:
    primary = results_dir / row.run_folder
    if primary.is_dir():
        return primary
    if row.fallback_folder:
        fb = results_dir / row.fallback_folder
        if fb.is_dir():
            return fb
    return primary


def row_metrics(results_dir: Path, row: ManifestRow, selection: RoundSelection) -> tuple[str, RunMetrics]:
    rd = _resolve_run_dir(results_dir, row)
    return row.label, load_run_metrics(rd, selection)


def manifest_t1_primary_vs_centralized() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow("Federated", "isic2019_dirichlet"),
        ManifestRow("Centralized", "centralized_upperbound"),
    )


def manifest_t2_partitioning() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow("IID split", "ablation_partition_iid"),
        ManifestRow(
            "Dirichlet α=0.1",
            "ablation_partition_dirichlet_01",
            latex_label=r"Dirichlet $\alpha=0.1$",
        ),
        ManifestRow(
            "Dirichlet α=0.5",
            "ablation_partition_dirichlet_05",
            latex_label=r"Dirichlet $\alpha=0.5$",
        ),
        ManifestRow(
            "Dirichlet α=1.0",
            "ablation_partition_dirichlet_10",
            latex_label=r"Dirichlet $\alpha=1.0$",
        ),
    )


def manifest_t3_nclients() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow("$K=5$", "ablation_nclients_5", latex_label=r"$K=5$"),
        ManifestRow(
            "$K=10$",
            "ablation_nclients_10",
            fallback_folder="isic2019_dirichlet",
            latex_label=r"$K=10$",
        ),
        ManifestRow("$K=20$", "ablation_nclients_20", latex_label=r"$K=20$"),
    )


def manifest_t4_localepochs() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow(
            "$E=1$ local epochs / round",
            "ablation_localepochs_1",
            latex_label=r"$E=1$ local epochs / round",
        ),
        ManifestRow(
            "$E=2$ local epochs / round",
            "isic2019_dirichlet",
            fallback_folder="ablation_localepochs_2",
            latex_label=r"$E=2$ local epochs / round",
        ),
        ManifestRow(
            "$E=5$ local epochs / round",
            "ablation_localepochs_5",
            latex_label=r"$E=5$ local epochs / round",
        ),
        ManifestRow(
            "$E=10$ local epochs / round",
            "ablation_localepochs_10",
            latex_label=r"$E=10$ local epochs / round",
        ),
    )


def manifest_t5_aggregation() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow("FedAvg", "isic2019_dirichlet", fallback_folder="ablation_strategy_fedavg"),
        ManifestRow("FedProx", "ablation_strategy_fedprox"),
        ManifestRow("FedAdam", "ablation_strategy_fedadam"),
    )


def manifest_t8_primary() -> tuple[ManifestRow, ...]:
    return (ManifestRow("Primary federated", "isic2019_dirichlet"),)


def infer_paper_dataset(results_dir: Path) -> PaperDataset:
    """Infer benchmark from ``results/<name>/`` folder (defaults to ISIC 2019)."""
    if results_dir.name.lower() == "isic2018":
        return "isic2018"
    return "isic2019"


def manifest_isic2018_t1_primary_vs_centralized() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow("Federated", "isic2018_dirichlet"),
        ManifestRow("Centralized", "isic2018_centralized_upperbound"),
    )


def manifest_isic2018_t2_partitioning() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow("IID split", "isic2018_ablation_partition_iid"),
        ManifestRow(
            "Dirichlet α=0.1",
            "isic2018_ablation_partition_dirichlet_01",
            latex_label=r"Dirichlet $\alpha=0.1$",
        ),
        ManifestRow(
            "Dirichlet α=0.5",
            "isic2018_ablation_partition_dirichlet_05",
            latex_label=r"Dirichlet $\alpha=0.5$",
        ),
        ManifestRow(
            "Dirichlet α=1.0",
            "isic2018_ablation_partition_dirichlet_10",
            latex_label=r"Dirichlet $\alpha=1.0$",
        ),
    )


def manifest_isic2018_t3_nclients() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow("$K=5$", "isic2018_ablation_nclients_5", latex_label=r"$K=5$"),
        ManifestRow(
            "$K=10$",
            "isic2018_ablation_nclients_10",
            fallback_folder="isic2018_dirichlet",
            latex_label=r"$K=10$",
        ),
        ManifestRow("$K=20$", "isic2018_ablation_nclients_20", latex_label=r"$K=20$"),
    )


def manifest_isic2018_t4_localepochs() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow(
            "$E=1$ local epochs / round",
            "isic2018_ablation_localepochs_1",
            latex_label=r"$E=1$ local epochs / round",
        ),
        ManifestRow(
            "$E=2$ local epochs / round",
            "isic2018_dirichlet",
            fallback_folder="isic2018_ablation_localepochs_2",
            latex_label=r"$E=2$ local epochs / round",
        ),
        ManifestRow(
            "$E=5$ local epochs / round",
            "isic2018_ablation_localepochs_5",
            latex_label=r"$E=5$ local epochs / round",
        ),
        ManifestRow(
            "$E=10$ local epochs / round",
            "isic2018_ablation_localepochs_10",
            latex_label=r"$E=10$ local epochs / round",
        ),
    )


def manifest_isic2018_t5_aggregation() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow(
            "FedAvg",
            "isic2018_dirichlet",
            fallback_folder="isic2018_ablation_strategy_fedavg",
        ),
        ManifestRow("FedProx", "isic2018_ablation_strategy_fedprox"),
        ManifestRow("FedAdam", "isic2018_ablation_strategy_fedadam"),
    )


def manifest_isic2018_t6_primary() -> tuple[ManifestRow, ...]:
    return (ManifestRow("Primary federated", "isic2018_dirichlet"),)


def list_run_coverage(results_dir: Path) -> str:
    """Human-readable report: which manifest folders exist on disk."""
    lines: list[str] = []
    seen: set[str] = set()
    if infer_paper_dataset(results_dir) == "isic2018":
        groups = (
            ("T1", manifest_isic2018_t1_primary_vs_centralized()),
            ("T2", manifest_isic2018_t2_partitioning()),
            ("T3", manifest_isic2018_t3_nclients()),
            ("T4", manifest_isic2018_t4_localepochs()),
            ("T5", manifest_isic2018_t5_aggregation()),
            ("T6", manifest_isic2018_t6_primary()),
        )
    else:
        groups = (
            ("T1", manifest_t1_primary_vs_centralized()),
            ("T2", manifest_t2_partitioning()),
            ("T3", manifest_t3_nclients()),
            ("T4", manifest_t4_localepochs()),
            ("T5", manifest_t5_aggregation()),
            ("T6", manifest_t8_primary()),
        )
    for gid, rows in groups:
        lines.append(f"[{gid}]")
        for r in rows:
            key = r.run_folder + (f"|{r.fallback_folder}" if r.fallback_folder else "")
            if key in seen:
                continue
            seen.add(key)
            p = results_dir / r.run_folder
            ok = p.is_dir()
            extra = ""
            if r.fallback_folder:
                fb = results_dir / r.fallback_folder
                extra = f"  fallback `{r.fallback_folder}`: {'OK' if fb.is_dir() else 'MISSING'}"
            status = "OK" if ok else "MISSING"
            lines.append(f"  {r.run_folder}: {status}{'  ' + extra if extra else ''}")
        lines.append("")
    return "\n".join(lines).rstrip()


def escape_latex(text: str) -> str:
    out: list[str] = []
    for c in text:
        if c == "&":
            out.append(r"\&")
        elif c == "%":
            out.append(r"\%")
        elif c == "$":
            out.append(r"\$")
        elif c == "#":
            out.append(r"\#")
        elif c == "_":
            out.append(r"\_")
        elif c == "{":
            out.append(r"\{")
        elif c == "}":
            out.append(r"\}")
        elif c == "~":
            out.append(r"\textasciitilde{}")
        elif c == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(c)
    return "".join(out)


def _md_pct(v: float | None) -> str:
    if v is None or v != v:
        return "—"
    return f"{v * 100:.2f}%"


def _md_prob(v: float | None) -> str:
    if v is None or v != v:
        return "—"
    return f"{v:.4f}"


def _md_loss(v: float | None) -> str:
    if v is None or v != v:
        return "—"
    return f"{v:.4f}"


def _core_cells_md(m: RunMetrics) -> str:
    return (
        f"{_md_pct(get_scalar(m, 'accuracy'))} | {_md_pct(get_scalar(m, 'macro_f1'))} | "
        f"{_md_loss(get_scalar(m, 'loss'))} | {_md_prob(get_scalar(m, 'auroc'))} | {_md_prob(get_scalar(m, 'auprc'))}"
    )


def build_tabular_t1_t6_markdown(
    results_dir: Path,
    rows: Iterable[ManifestRow],
    selection: RoundSelection,
    *,
    include_round_column: bool = True,
) -> str:
    body: list[str] = []
    for spec in rows:
        m = load_run_metrics(_resolve_run_dir(results_dir, spec), selection)
        rnd = "—" if m.missing or m.round_index is None else str(m.round_index)
        if not include_round_column:
            rnd = ""
        row = (
            f"| {spec.label} | {_core_cells_md(m)} | {rnd} |"
            if include_round_column
            else f"| {spec.label} | {_core_cells_md(m)} |"
        )
        body.append(row)

    header_line = (
        "| Setting | Acc. | Macro-F1 | Loss | AUROC | AUPRC | Round |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        if include_round_column
        else "| Setting | Acc. | Macro-F1 | Loss | AUROC | AUPRC |\n|---|---:|---:|---:|---:|---:|\n"
    )
    return header_line + "\n".join(body) + "\n"


def _journal_compare_dir(results_dir: Path) -> Path:
    """``figures/journal/<dataset>/`` under the repo root (``results_dir`` = ``.../results/isic2019``)."""
    slug = infer_paper_dataset(results_dir)
    return (results_dir.resolve().parent.parent / "figures" / "journal" / slug).resolve()


def build_journal_comparison_figures_markdown(results_dir: Path) -> str:
    """Markdown block with embedded journal cross-run comparisons (SVG under each ``compare_*``)."""
    journal_dir = _journal_compare_dir(results_dir)
    repo_root = journal_dir.parent.parent.parent
    ds = infer_paper_dataset(results_dir)
    label = "ISIC 2018" if ds == "isic2018" else "ISIC 2019"
    rel_journal = journal_dir.relative_to(repo_root).as_posix()
    heading = f"## Figures — {label} cross-run comparisons\n\n"
    hint = (
        f"_SVG panels under [`{rel_journal}/`]({rel_journal}/) "
        "(`compare_partition`, `compare_nclients`, `compare_localepochs`, `compare_strategy`, "
        "`compare_communication`). Regenerate with the journal viz module "
        "(see `paper/README.md` / Colab notebooks)._\n\n"
    )

    if not journal_dir.is_dir():
        return (
            heading
            + f"_Journal directory not found at `{journal_dir.as_posix()}` "
            f"(expected `[repo]/{rel_journal}/` when `--results-dir` is under `[repo]/results/`)._"
            "\n\n"
        )

    groups = sorted(d for d in journal_dir.iterdir() if d.is_dir() and d.name.startswith("compare_"))
    if not groups:
        return heading + hint + "_No `compare_*` subdirectories yet._\n\n"

    blocks: list[str] = [heading, hint]
    for group in groups:
        rel_group = group.relative_to(repo_root).as_posix()
        blocks.append(f"### `{rel_group}`\n\n")
        figures = sorted(group.glob("*.svg"))
        if not figures:
            blocks.append("_No SVG exports in this folder._\n\n")
            continue
        for fig_path in figures:
            rel_fig = fig_path.relative_to(repo_root).as_posix()
            cap = fig_path.stem.replace("__", ", ").replace("_", " ")
            blocks.append(f"![{cap}]({rel_fig})\n\n")

    return "".join(blocks)


def compose_report_tables_markdown(results_dir: Path, selection: RoundSelection) -> str:
    ds = infer_paper_dataset(results_dir)
    rel_results = results_dir.as_posix()
    if ds == "isic2018":
        m1 = manifest_isic2018_t1_primary_vs_centralized
        m2 = manifest_isic2018_t2_partitioning
        m3 = manifest_isic2018_t3_nclients
        m4 = manifest_isic2018_t4_localepochs
        m5 = manifest_isic2018_t5_aggregation
        m6_row = manifest_isic2018_t6_primary()[0]
        class_labels = ISIC2018_CLASS_LABELS
    else:
        m1 = manifest_t1_primary_vs_centralized
        m2 = manifest_t2_partitioning
        m3 = manifest_t3_nclients
        m4 = manifest_t4_localepochs
        m5 = manifest_t5_aggregation
        m6_row = manifest_t8_primary()[0]
        class_labels = ISIC2019_CLASS_LABELS

    parts: list[str] = [
        "# SkinFLNet++ — Results tables & figures\n",
        f"_Generated from `{rel_results}/` (selection: **{selection}**)._\n",
        "_For the LaTeX paper, run `python scripts/export_paper_tables_tex.py` or edit `paper/tables/`._\n\n",
    ]
    parts.append("## T1 — Primary vs centralized\n\n")
    parts.append(
        build_tabular_t1_t6_markdown(
            results_dir,
            m1(),
            selection,
            include_round_column=False,
        )
    )
    parts.append("\n")

    for title, man, incl_rnd in (
        ("T2 — Partitioning", m2, True),
        ("T3 — Number of clients", m3, True),
        ("T4 — Local epochs", m4, True),
        ("T5 — Aggregation", m5, True),
    ):
        parts.append(f"## {title}\n\n")
        parts.append(
            build_tabular_t1_t6_markdown(results_dir, man(), selection, include_round_column=incl_rnd)
        )
        parts.append("\n")

    parts.append("## T6 — Per-class F1 (primary)\n\n")
    _, m = row_metrics(results_dir, m6_row, selection)
    if m.missing:
        parts.append("_Missing primary run._\n")
    else:
        f1s = m.data.get("per_class_f1") or []
        parts.append("| Class | F1 |\n|---:|---:|\n")
        for i, cls in enumerate(class_labels):
            v = float(f1s[i]) if i < len(f1s) and isinstance(f1s[i], (int, float)) else None
            parts.append(f"| {cls} | {_md_pct(v)} |\n")

    parts.append("\n")
    parts.append(build_journal_comparison_figures_markdown(results_dir))
    return "".join(parts)
