"""
Refresh ``REPORT_TABLES.md`` from ``results/…`` (Markdown tables plus embedded figures).

The appendix section lists SVG plots under ``figures/journal/<dataset>/compare_*`` (dataset is inferred from ``--results-dir``, e.g. ``results/isic2019`` or ``results/isic2018``).

The LaTeX paper uses static tables in ``paper/tables/`` and ``paper/tables/isic2018/`` --- regenerate with ``scripts/export_paper_tables_tex.py`` or edit those files when numbers change.

Usage:
    python scripts/make_report_tables.py
    python scripts/make_report_tables.py --results-dir results --output REPORT_TABLES.md
    python scripts/make_report_tables.py --list-runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report.metrics import RoundSelection  # noqa: E402
from src.report.manuscript_runs import (  # noqa: E402
    compose_report_tables_markdown,
    list_run_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate REPORT_TABLES.md from results/")
    parser.add_argument("--results-dir", default="results/isic2019")
    parser.add_argument("--output", default="REPORT_TABLES.md")
    parser.add_argument(
        "--selection",
        choices=("last", "best_macro_f1"),
        default="best_macro_f1",
        help="Round-selection policy for federated runs (ignored for flat summary rows).",
    )
    parser.add_argument("--list-runs", action="store_true")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    selection = cast(RoundSelection, args.selection)
    if args.list_runs:
        print(list_run_coverage(rd))
        return

    text = compose_report_tables_markdown(rd, selection)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"Markdown tables written to {args.output}")


if __name__ == "__main__":
    main()
