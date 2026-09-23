"""
Experiment configuration dataclass and global singleton.

All FL components (client, server, simulate) read config from get_config().
Call set_config() once in scripts/run_experiment.py before run_simulation().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = "isic2019"           # 'isic2019' | 'isic2018' | 'isbi2016'
    data_root: str = "data"            # root folder containing ISIC2019/ ISIC2018/ ISBI2016/

    # ── Partitioning ─────────────────────────────────────────────────────────
    partition: str = "dirichlet"       # 'iid' | 'dirichlet' | 'patient'
    alpha: float = 0.5                 # Dirichlet concentration
    num_clients: int = 10              # total number of FL clients

    # ── Model ────────────────────────────────────────────────────────────────
    backbone: str = "vgg16_bn"
    img_size: int = 224

    # ── FL Protocol ──────────────────────────────────────────────────────────
    num_rounds: int = 50
    local_epochs: int = 2
    fraction_fit: float = 0.5          # fraction of clients selected per round
    strategy: str = "fedavg"          # 'fedavg' | 'fedprox' | 'fedadam'
    fedprox_mu: float = 0.01
    fedadam_eta: float = 1e-3          # FedAdam server learning rate (tuned on validation)
    fedadam_tau: float = 1e-3          # FedAdam adaptivity / epsilon
    fedadam_beta1: float = 0.9
    fedadam_beta2: float = 0.99

    # ── Optimizer ────────────────────────────────────────────────────────────
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    freeze_rounds: int = 3            # unused: partial freeze applies every FL round / epoch in code

    # ── Early stopping ───────────────────────────────────────────────────────
    early_stop_patience: int = 5

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Logging ──────────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "skinflnet-plus"
    show_round_progress: bool = True   # Rich FL bar when Rich enabled (see simulate._use_rich_round_bar)

    # ── Paths ────────────────────────────────────────────────────────────────
    results_dir: str = "results"
    figures_dir: str = "figures"
    run_name: str = ""                 # set automatically by run_experiment.py

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    resume: bool = False
    save_weights_archive: bool = False
    ignore_completed_early_stop: bool = False

    # ── Client local evaluation (held-out fraction of client's train shard) ───
    # 0.0 (revision default) = clients train on their whole shard, so the union of
    # client training data equals the centralized pool exactly.
    client_local_eval_fraction: float = 0.0

    # ── Centralized only: which split drives best-checkpoint + early stopping ──
    # "val" (default) is the honest protocol. "test" matches federated logging
    # (test each round) but leaks the test set into model selection.
    centralized_best_on: str = "val"  # "val" | "test"

    # ── Worker threads ───────────────────────────────────────────────────────
    num_workers: int = 4

    # ── Revision R1 additions ────────────────────────────────────────────────
    amp: bool = True                   # bf16 autocast on CUDA (speed only)
    eval_test_every_round: bool = True # log test curves; SELECTION always uses val
    run_manifest: str = ""             # per-run partitioned manifest (set by the runner)
    save_best_global: bool = True      # keep weights of the best-validation round


# Global config singleton ─────────────────────────────────────────────────────

_config: Optional[ExperimentConfig] = None


def set_config(cfg: ExperimentConfig) -> None:
    global _config
    _config = cfg


def get_config(context=None) -> ExperimentConfig:
    global _config
    if _config is None:
        raise RuntimeError("ExperimentConfig not set. Call set_config() first.")
    return _config


def num_classes_for(dataset: str) -> int:
    """Return the number of output classes for a given dataset."""
    return {"isic2019": 8, "isic2018": 7, "isbi2016": 2}[dataset.lower()]


def manifest_path_for(cfg: "ExperimentConfig"):
    """Per-run partitioned manifest if set, else the dataset's base manifest."""
    from pathlib import Path

    if getattr(cfg, "run_manifest", ""):
        return Path(cfg.run_manifest)
    return Path(cfg.data_root) / dataset_folder_for(cfg.dataset) / "manifest.csv"


def dataset_folder_for(dataset: str) -> str:
    """Return the data subdirectory name for a dataset key."""
    return {"isic2019": "ISIC2019", "isic2018": "ISIC2018", "isbi2016": "ISBI2016"}[
        dataset.lower()
    ]
