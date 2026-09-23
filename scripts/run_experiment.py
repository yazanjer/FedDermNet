"""
Main experiment runner for SkinFLNet++ (CLI).

In-process API: src.fl.experiment_runner.run_experiment_from_yaml

Usage:
    python scripts/run_experiment.py configs/isic2019_dirichlet.yaml
    python scripts/run_experiment.py configs/isic2019_dirichlet.yaml --wandb
    python scripts/run_experiment.py configs/centralized_upperbound.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fl.experiment_runner import run_experiment_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SkinFLNet++ experiment")
    parser.add_argument("config", help="Path to YAML experiment config")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--no-round-progress",
        action="store_true",
        help="Disable Rich federated-round progress bar during simulation",
    )
    parser.add_argument("--data-root", default="data", help="Override data root path")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint_latest.pt if valid (overrides YAML resume)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Never resume even if YAML has resume: true",
    )
    parser.add_argument(
        "--save-weights-archive",
        action="store_true",
        help="Also save model_round_NNN.pth each round (overrides YAML)",
    )
    parser.add_argument(
        "--ignore-completed-early-stop",
        action="store_true",
        help="After resume, clear early-stop gate so training can continue",
    )

    args = parser.parse_args()

    resume_kw = False if args.no_resume else (True if args.resume else None)

    run_experiment_from_yaml(
        args.config,
        data_root=args.data_root or "data",
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        wandb_cli=args.wandb,
        no_round_progress=args.no_round_progress,
        resume=resume_kw,
        save_weights_archive=True if args.save_weights_archive else None,
        ignore_completed_early_stop=True if args.ignore_completed_early_stop else None,
        force_rich_logging=True,
    )


if __name__ == "__main__":
    main()
