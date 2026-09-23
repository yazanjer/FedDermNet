"""
Generate figures from results/ (curves + confusion matrices).

Usage:
    python scripts/make_report_figures.py --results-dir results --figures-dir figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report.figures import generate_report_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures from results/")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument(
        "--dataset",
        default="isic2019",
        choices=["isic2019", "isic2018"],
    )
    args = parser.parse_args()

    generate_report_figures(Path(args.results_dir), Path(args.figures_dir), args.dataset)


if __name__ == "__main__":
    main()
