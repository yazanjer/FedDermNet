"""
Resume checkpoints: federated global model + tracker + FedAdam, and centralized training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch

from src.fl.config import ExperimentConfig

logger = logging.getLogger(__name__)

CHECKPOINT_LATEST = "checkpoint_latest.pt"
CHECKPOINT_KIND_FEDERATED = "federated"
CHECKPOINT_KIND_CENTRALIZED = "centralized"


def experiment_config_digest(cfg: ExperimentConfig) -> str:
    key = "|".join(
        str(x)
        for x in (
            cfg.dataset,
            cfg.backbone,
            cfg.num_clients,
            cfg.num_rounds,
            cfg.strategy.lower(),
            cfg.seed,
            cfg.partition,
            f"{cfg.alpha:.6f}",
            cfg.local_epochs,
            cfg.fraction_fit,
            f"{cfg.client_local_eval_fraction:.6f}",
        )
    )
    # Centralized-only; omit when default so existing checkpoints keep matching.
    if getattr(cfg, "centralized_best_on", "val").lower() == "test":
        key += "|centralized_best_on=test"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def fl_checkpoint_path(cfg: ExperimentConfig) -> Path:
    return Path(cfg.results_dir) / cfg.run_name / CHECKPOINT_LATEST


def centralized_checkpoint_path(cfg: ExperimentConfig) -> Path:
    return Path(cfg.results_dir) / cfg.run_name / CHECKPOINT_LATEST


def save_fl_checkpoint(
    cfg: ExperimentConfig,
    *,
    global_state_dict: dict[str, torch.Tensor],
    last_completed_round: int,
    best_macro_f1: float,
    rounds_without_improvement: int,
    should_stop: bool,
    fedadam_state: dict[str, Any] | None,
) -> None:
    run_dir = Path(cfg.results_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "checkpoint_kind": CHECKPOINT_KIND_FEDERATED,
        "config_digest": experiment_config_digest(cfg),
        "last_completed_round": int(last_completed_round),
        "global_state_dict": {k: v.cpu().clone() for k, v in global_state_dict.items()},
        "tracker": {
            "best_macro_f1": float(best_macro_f1),
            "rounds_without_improvement": int(rounds_without_improvement),
            "should_stop": bool(should_stop),
        },
        "fedadam": fedadam_state,
    }

    path = run_dir / CHECKPOINT_LATEST
    torch.save(payload, path)

    if cfg.save_weights_archive:
        arch = run_dir / f"model_round_{last_completed_round:03d}.pth"
        torch.save({"global_state_dict": payload["global_state_dict"]}, arch)


def load_fl_checkpoint(path: Path, cfg: ExperimentConfig) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", path, e)
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("checkpoint_kind") != CHECKPOINT_KIND_FEDERATED:
        logger.warning("Checkpoint %s is not a federated checkpoint.", path)
        return None
    digest = experiment_config_digest(cfg)
    if payload.get("config_digest") != digest:
        logger.error(
            "Checkpoint config mismatch (digest %s vs expected %s). Refusing resume.",
            payload.get("config_digest"),
            digest,
        )
        return None
    return payload


def apply_fl_checkpoint_to_server(server: Any, payload: dict[str, Any], cfg: ExperimentConfig) -> None:
    """Mutate SkinFLServer weights and FedAdam from a loaded FL checkpoint payload."""
    server.global_state_dict = {
        k: v.clone() for k, v in payload["global_state_dict"].items()
    }

    fd = payload.get("fedadam")
    if cfg.strategy.lower() == "fedadam" and fd is not None:
        server.fedadam_optimizer.load_state_dict(fd)
    elif cfg.strategy.lower() == "fedadam" and fd is None:
        logger.warning("FedAdam strategy but checkpoint has no FedAdam state; optimizer re-initialized.")


ROUND_JSON_RE = re.compile(r"^round_(\d+)\.json$")


def rebuild_tracker_history_from_json(tracker: Any) -> None:
    """Repopulate tracker.history from round_*.json on disk (sorted by round index)."""
    rd = tracker.results_dir
    if not rd.is_dir():
        return
    entries: list[tuple[int, Path]] = []
    for p in rd.iterdir():
        m = ROUND_JSON_RE.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda x: x[0])
    history: list[dict] = []
    for _, path in entries:
        with open(path) as f:
            history.append(json.load(f))
    tracker.history = history


def hydrate_tracker_after_resume(tracker: Any, payload: dict[str, Any]) -> None:
    """Restore counters from checkpoint and align history JSON files on disk."""
    t = payload["tracker"]
    tracker.best_macro_f1 = float(t["best_macro_f1"])
    tracker.rounds_without_improvement = int(t["rounds_without_improvement"])
    tracker.should_stop = bool(t["should_stop"])
    rebuild_tracker_history_from_json(tracker)


def save_centralized_checkpoint(
    cfg: ExperimentConfig,
    *,
    model_state_dict: dict[str, torch.Tensor],
    last_completed_epoch: int,
    last_completed_virtual_round: int,
    best_val_f1: float,
    patience_counter: int,
    history: list[dict],
) -> None:
    run_dir = Path(cfg.results_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "checkpoint_kind": CHECKPOINT_KIND_CENTRALIZED,
        "config_digest": experiment_config_digest(cfg),
        "last_completed_epoch": int(last_completed_epoch),
        "last_completed_virtual_round": int(last_completed_virtual_round),
        "model_state_dict": {k: v.cpu().clone() for k, v in model_state_dict.items()},
        "best_val_f1": float(best_val_f1),
        "patience_counter": int(patience_counter),
        "history": history,
    }

    path = run_dir / CHECKPOINT_LATEST
    torch.save(payload, path)

    if cfg.save_weights_archive:
        arch = run_dir / f"model_round_{last_completed_virtual_round:03d}.pth"
        torch.save({"model_state_dict": payload["model_state_dict"]}, arch)


def load_centralized_checkpoint(path: Path, cfg: ExperimentConfig) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", path, e)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("checkpoint_kind") != CHECKPOINT_KIND_CENTRALIZED:
        logger.warning("Checkpoint %s is not a centralized checkpoint.", path)
        return None
    digest = experiment_config_digest(cfg)
    if payload.get("config_digest") != digest:
        logger.error(
            "Checkpoint config mismatch (digest %s vs expected %s). Refusing resume.",
            payload.get("config_digest"),
            digest,
        )
        return None
    return payload
