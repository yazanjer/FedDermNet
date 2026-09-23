"""
In-process experiment driver: YAML -> manifests -> partition -> train/simulate.

CLI wraps run_experiment_from_yaml via scripts/run_experiment.py.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

if TYPE_CHECKING:
    from src.fl.config import ExperimentConfig

logger = logging.getLogger(__name__)


def load_config_yaml(yaml_path: str | Path) -> dict[str, Any]:
    path = Path(yaml_path)
    with path.open() as f:
        return yaml.safe_load(f)


def _in_ipython() -> bool:
    try:
        ip = get_ipython()  # type: ignore[name-defined]
        return ip is not None
    except NameError:
        return False


def _use_plain_logging(force_rich_logging: bool | None) -> bool:
    if force_rich_logging is True:
        return False
    if force_rich_logging is False:
        return True
    import sys

    if not sys.stderr.isatty():
        return True
    return bool(os.environ.get("JPY_PARENT_PID")) or _in_ipython()


def configure_experiment_logging(*, force_rich_logging: bool | None = None) -> None:
    """Attach handlers on root logger for one experiment run."""
    plain = _use_plain_logging(force_rich_logging)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    if plain:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        root.addHandler(h)
    else:
        from rich.logging import RichHandler

        h = RichHandler(rich_tracebacks=True, markup=True)
        h.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        root.addHandler(h)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("timm").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _ensure_manifests(cfg: ExperimentConfig) -> None:
    from src.data.manifest_build import ensure_manifest_for_dataset

    from src.fl.config import dataset_folder_for

    ds = cfg.dataset.lower()
    if ds not in ("isic2019", "isic2018", "isbi2016"):
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    manifest_path = Path(cfg.data_root) / dataset_folder_for(ds) / "manifest.csv"
    if not manifest_path.exists():
        logger.info("Manifest not found — building now...")
    ensure_manifest_for_dataset(cfg.data_root, ds)  # type: ignore[arg-type]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing after build. Expected: {manifest_path}")


def _apply_partition(cfg: ExperimentConfig) -> None:
    """Partition the base manifest into a PER-RUN manifest (safe for concurrent runs)."""
    from src.data.partition import partition
    from src.fl.config import dataset_folder_for

    base = Path(cfg.data_root) / dataset_folder_for(cfg.dataset) / "manifest.csv"
    df = pd.read_csv(base)
    patient_col = "lesion_key" if "lesion_key" in df.columns else (
        "lesion_id" if "lesion_id" in df.columns else None
    )
    df_partitioned = partition(
        df=df,
        scheme=cfg.partition,
        n_clients=cfg.num_clients,
        alpha=cfg.alpha,
        seed=cfg.seed,
        patient_col=patient_col,
    )
    out = base.parent / "partitions" / f"{cfg.run_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_partitioned.to_csv(out, index=False)
    cfg.run_manifest = str(out)
    # Archive the client assignment next to the results (reproducibility manifest).
    rdir = Path(cfg.results_dir) / cfg.run_name
    rdir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("image_id", "label", "split", "lesion_key", "client_id") if c in df_partitioned.columns]
    df_partitioned[df_partitioned["split"] == "train"][cols].to_csv(rdir / "client_assignment.csv.gz", index=False)
    logger.info(
        "Partition '%s' applied (alpha=%.2f, K=%d, seed=%d) -> %s",
        cfg.partition, cfg.alpha, cfg.num_clients, cfg.seed, out,
    )
    _save_partition_figure(df_partitioned, cfg)


def _save_partition_figure(df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    import matplotlib.pyplot as plt

    train_df = df[df["split"] == "train"]
    if "label" not in train_df.columns:
        return

    figures_dir = Path(cfg.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_clients = cfg.num_clients
    classes = sorted(train_df["label"].unique())

    fig, axes = plt.subplots(
        1, n_clients, figsize=(max(4 * n_clients, 12), 4), sharey=True
    )
    if n_clients == 1:
        axes = [axes]

    for cid, ax in enumerate(axes):
        client_df = train_df[train_df["client_id"] == cid]
        counts = [len(client_df[client_df["label"] == c]) for c in classes]
        ax.bar(classes, counts, color="steelblue", edgecolor="black")
        ax.set_title(f"Client {cid}\n(n={len(client_df)})", fontsize=8)
        ax.set_xlabel("Class")
        if cid == 0:
            ax.set_ylabel("Count")

    fig.suptitle(
        f"Partition: {cfg.partition} (α={cfg.alpha}, K={n_clients})", fontsize=11
    )
    plt.tight_layout()
    out_path = (
        figures_dir
        / f"partition_{cfg.dataset}_{cfg.partition}_K{n_clients}_a{cfg.alpha}.png"
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Partition figure saved: %s", out_path)


def run_experiment_from_yaml(
    config_path: str | Path,
    *,
    data_root: str | Path,
    results_dir: str | Path,
    figures_dir: str | Path,
    use_wandb: bool | None = None,
    show_round_progress: bool | None = None,
    wandb_cli: bool = False,
    no_round_progress: bool = False,
    resume: bool | None = None,
    no_resume: bool = False,
    save_weights_archive: bool | None = None,
    ignore_completed_early_stop: bool | None = None,
    force_rich_logging: bool | None = None,
    configure_logging: bool = True,
) -> None:
    """
    Run one experiment from a YAML file (FL or centralized).

    wandb_cli / no_round_progress mirror argparse flags from scripts/run_experiment.py.
    If use_wandb is not None, it overrides YAML for W&B. Otherwise use_wb =
    wandb_cli or YAML ``use_wandb``.

    If show_round_progress is not None, it wins; elif no_round_progress, False;
    else YAML ``show_round_progress``.

    Notebook-friendly plain logging when IPython/JPY_PARENT_PID unless force_rich_logging=True.

    Set configure_logging=False if the caller already configured root logging.
    """
    from src.fl.config import ExperimentConfig

    config_path = Path(config_path)
    raw = load_config_yaml(config_path)
    exp = raw.get("experiment", raw)

    if configure_logging:
        configure_experiment_logging(force_rich_logging=force_rich_logging)

    dr = str(data_root)
    res = str(results_dir)
    fig = str(figures_dir)

    if use_wandb is not None:
        use_wb = use_wandb
    else:
        use_wb = wandb_cli or bool(exp.get("use_wandb", False))

    if show_round_progress is not None:
        show_rp = show_round_progress
    elif no_round_progress:
        show_rp = False
    else:
        show_rp = bool(exp.get("show_round_progress", True))

    if resume is not None:
        do_resume = resume
    elif no_resume:
        do_resume = False
    else:
        do_resume = bool(exp.get("resume", False))

    if save_weights_archive is not None:
        save_wa = save_weights_archive
    else:
        save_wa = bool(exp.get("save_weights_archive", False))

    if ignore_completed_early_stop is not None:
        ignore_es = ignore_completed_early_stop
    else:
        ignore_es = bool(exp.get("ignore_completed_early_stop", False))

    _cbo = str(exp.get("centralized_best_on", "val")).lower().strip()
    if _cbo != "val":
        raise ValueError(
            f"experiment.centralized_best_on must be 'val' (test-set selection is not allowed), got {_cbo!r}"
        )

    cfg = ExperimentConfig(
        dataset=str(exp.get("dataset", "isic2019")).lower(),
        data_root=dr or exp.get("data_root", "data"),
        partition=exp.get("partition", "dirichlet"),
        alpha=float(exp.get("alpha", 0.5)),
        num_clients=int(exp.get("num_clients", 10)),
        backbone=exp.get("backbone", "vgg16_bn"),
        img_size=int(exp.get("img_size", 224)),
        num_rounds=int(exp.get("num_rounds", 50)),
        local_epochs=int(exp.get("local_epochs", 2)),
        fraction_fit=float(exp.get("fraction_fit", 0.5)),
        strategy=exp.get("strategy", "fedavg"),
        fedprox_mu=float(exp.get("fedprox_mu", 0.01)),
        fedadam_eta=float(exp.get("fedadam_eta", 1e-3)),
        fedadam_tau=float(exp.get("fedadam_tau", 1e-3)),
        fedadam_beta1=float(exp.get("fedadam_beta1", 0.9)),
        fedadam_beta2=float(exp.get("fedadam_beta2", 0.99)),
        amp=bool(exp.get("amp", True)),
        deterministic=bool(exp.get("deterministic", True)),
        eval_test_every_round=bool(exp.get("eval_test_every_round", True)),
        lr=float(exp.get("lr", 1e-4)),
        weight_decay=float(exp.get("weight_decay", 1e-4)),
        batch_size=int(exp.get("batch_size", 32)),
        freeze_rounds=int(exp.get("freeze_rounds", 3)),
        early_stop_patience=int(exp.get("early_stop_patience", 5)),
        seed=int(exp.get("seed", 42)),
        use_wandb=use_wb,
        show_round_progress=show_rp,
        results_dir=res,
        figures_dir=fig,
        resume=do_resume,
        save_weights_archive=save_wa,
        ignore_completed_early_stop=ignore_es,
        client_local_eval_fraction=float(exp.get("client_local_eval_fraction", 0.0)),
        centralized_best_on=_cbo,
        num_workers=int(exp.get("num_workers", 4)),
    )

    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = exp.get("name", "").replace(" ", "_") or (
        f"{cfg.dataset}_{cfg.backbone}_{cfg.partition}"
        f"_K{cfg.num_clients}_R{cfg.num_rounds}_{ts}"
    )
    cfg.run_name = run_name

    logger.info("Experiment: %s", run_name)

    _ensure_manifests(cfg)
    _apply_partition(cfg)

    if cfg.use_wandb:
        try:
            import wandb

            wandb.init(project=cfg.wandb_project, name=run_name, config=vars(cfg))
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb")

    is_centralized = exp.get("mode", "federated") == "centralized"

    if is_centralized:
        logger.info("Running CENTRALIZED upper-bound training")
        from src.train.centralized import train_centralized

        train_centralized(cfg)
    else:
        logger.info("Running FEDERATED simulation")
        from src.fl.simulate import run_federated_simulation

        run_federated_simulation(cfg)

    if cfg.use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    logger.info("Done. Results in %s/%s/", cfg.results_dir, run_name)
