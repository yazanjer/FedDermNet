"""
Export stacked comparison figures from ``figures/journal/<dataset>/`` to ``paper/figures/<dataset>/``.

Maps journal ``*_panel_macro_acc_loss.svg`` (and the communication upload SVG) to the basenames
used in ``paper/sections/results.tex`` (``fig_compare_*_metrics``).

Usage (from repo root)::

    python scripts/export_paper_compare_figures.py --dataset isic2018
    python scripts/export_paper_compare_figures.py --dataset isic2019

Requires Inkscape on PATH, or ``svglib`` + ``reportlab`` (same approach as ``paper/README.md``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _svg_to_pdf_inkscape(svg: Path, pdf: Path) -> bool:
    inkscape = shutil.which("inkscape") or shutil.which("inkscape.exe")
    if not inkscape:
        return False
    pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [inkscape, str(svg), "--export-type=pdf", f"--export-filename={pdf}"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return True


def _svg_to_pdf_svglib(svg: Path, pdf: Path) -> None:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

    pdf.parent.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(svg2rlg(str(svg)), str(pdf))


def _export_pair(svg: Path, pdf: Path) -> None:
    if not svg.is_file():
        raise FileNotFoundError(f"Missing source SVG: {svg}")
    if _svg_to_pdf_inkscape(svg, pdf):
        return
    try:
        _svg_to_pdf_svglib(svg, pdf)
    except ImportError as e:
        raise SystemExit(
            "Neither Inkscape nor svglib+reportlab is available. "
            "Install Inkscape or `pip install svglib reportlab`.\n"
            f"Import error: {e}"
        ) from e


def _journal_dir(dataset: str) -> Path:
    return _repo_root() / "figures" / "journal" / dataset


def _paper_fig_dir(dataset: str) -> Path:
    return _repo_root() / "paper" / "figures" / dataset


def export_compare_figures(dataset: str) -> list[str]:
    """Write PDFs under ``paper/figures/<dataset>/``. Returns list of warnings."""
    root = _repo_root()
    jdir = _journal_dir(dataset)
    out_dir = _paper_fig_dir(dataset)
    warnings: list[str] = []

    specs: list[tuple[str, Path, str]] = [
        (
            "partition",
            jdir / "compare_partition" / "compare_partition__panel_macro_acc_loss.svg",
            "fig_compare_partition_metrics.pdf",
        ),
        (
            "nclients",
            jdir / "compare_nclients" / "compare_nclients__panel_macro_acc_loss.svg",
            "fig_compare_nclients_metrics.pdf",
        ),
        (
            "localepochs",
            jdir / "compare_localepochs" / "compare_localepochs__panel_macro_acc_loss.svg",
            "fig_compare_localepochs_metrics.pdf",
        ),
        (
            "strategy",
            jdir / "compare_strategy" / "compare_strategy__panel_macro_acc_loss.svg",
            "fig_compare_strategy_metrics.pdf",
        ),
    ]

    comm_dir = jdir / "compare_communication"
    comm_candidates = sorted(comm_dir.glob("*.svg")) if comm_dir.is_dir() else []
    if comm_candidates:
        # Prefer the combined strategy+clients upload figure when present.
        preferred = [p for p in comm_candidates if "macro_acc_loss" in p.name or "macro_f1" in p.name]
        comm_svg = preferred[0] if preferred else comm_candidates[0]
        specs.append(("communication", comm_svg, "fig_compare_communication_metrics.pdf"))
    else:
        fallback = root / "paper" / "figures" / "fig_compare_communication_macro_f1_vs_uploads.svg"
        if dataset == "isic2019" and fallback.is_file():
            specs.append(("communication", fallback, "fig_compare_communication_metrics.pdf"))
            warnings.append(
                f"Using legacy communication SVG {fallback.relative_to(root)} "
                "(no figures/journal/isic2019/compare_communication/ export)."
            )
        else:
            warnings.append(
                f"No SVG in {comm_dir} — skipped fig_compare_communication_metrics.pdf"
            )

    for name, svg, pdf_name in specs:
        pdf = out_dir / pdf_name
        _export_pair(svg, pdf)
        print(f"Wrote {pdf.relative_to(root)}")

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("isic2018", "isic2019"),
        required=True,
        help="Journal subfolder under figures/journal/",
    )
    args = parser.parse_args()
    try:
        warns = export_compare_figures(args.dataset)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    for w in warns:
        print(f"WARNING: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
