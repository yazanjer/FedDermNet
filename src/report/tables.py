"""Markdown tables from ``results/`` (legacy entrypoint)."""

from __future__ import annotations

from pathlib import Path

from src.report.manuscript_runs import compose_report_tables_markdown
from src.report.metrics import RoundSelection


def write_report_tables(
    results_dir: Path | str,
    output_path: Path | str,
    *,
    selection: RoundSelection = "best_macro_f1",
) -> None:
    """Write ``REPORT_TABLES.md``-style markdown (Tables T1–T8 manifest)."""
    rd = Path(results_dir)
    out = Path(output_path)
    text = compose_report_tables_markdown(rd, selection)
    out.write_text(text, encoding="utf-8")
    print(f"Tables written to {out}")
