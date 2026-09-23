"""
Custom Federated Simulation Orchestrator for SkinFLNet++.

Runs the FL loop in-memory using custom Server and Client classes.
No Flower subprocesses or Ray dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.auto import tqdm

from src.fl.config import ExperimentConfig, set_config
from src.fl.checkpoint_io import (
    apply_fl_checkpoint_to_server,
    fl_checkpoint_path,
    hydrate_tracker_after_resume,
    load_fl_checkpoint,
)
from src.fl.client_app import SkinFLClient
from src.fl.server_app import SkinFLServer
from src.fl.strategies import MetricsTracker

logger = logging.getLogger(__name__)


def federated_round_client_pick_seed(global_seed: int, server_round: int, num_clients: int) -> int:
    """Isolated RNG seed for which FL clients are sampled each round (stable across runs)."""
    return int(global_seed) + int(server_round) * 1_000_003 + int(num_clients) * 97


def _save_round_clients_json(
    results_dir: str, run_name: str, server_round: int, entries: list[dict]
) -> None:
    out = Path(results_dir) / run_name / f"round_{server_round:03d}_clients.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"round": server_round, "clients": entries}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=MetricsTracker._json_default)

_SENTINEL_FL_LOOP = "__SKINFL_FL_LOOP_START__"
_SENTINEL_ROUND_PREFIX = "__SKINFL_ROUND__"


def _rich_stdout_is_notebook_shim() -> bool:
    """Win/vscode Jupyter often wraps stdout so isatty() is True though output is captured."""
    return bool(os.environ.get("JPY_PARENT_PID"))


def _force_plain_fl_console() -> bool:
    v = os.environ.get("SKINFL_FORCE_PLAIN_FL_LOG", "").strip().lower()
    return v in ("1", "true", "yes")


def _use_rich_round_bar(cfg: ExperimentConfig) -> bool:
    """Rich live progress needs a real console; notebooks and pipes corrupt streamed output."""
    if not cfg.show_round_progress:
        return False
    if _force_plain_fl_console():
        return False
    if _rich_stdout_is_notebook_shim():
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _clear_round_sentinels_enabled() -> bool:
    return os.environ.get("SKINFL_CLEAR_FL_ROUNDS") == "1"


def _emit_notebook_sentinel(marker: str) -> None:
    """Plain print so logging formatters never wrap it (for notebook line parsers)."""
    print(marker, flush=True)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds plus cudnn deterministic flags.

    Also disables CUDA TF32 where supported (helps match metrics across Ampere+ GPUs).
    For strictest bitwise reproducibility with dataloaders, use ``num_workers: 0``
    (multi-worker PIL/torchvision ops still consume RNG in worker processes).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic=False enables cuDNN autotuning (faster; results then vary in
        # the last digits between identical runs, which the multi-seed design absorbs).
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except AttributeError:
            pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_federated_simulation(cfg: ExperimentConfig) -> None:
    """
    Run one complete federated learning simulation sequentially.

    Args:
        cfg: Fully populated ExperimentConfig.
             Must have run_name set before calling.
    """
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    set_config(cfg)

    logger.info(
        "Starting custom simulation | Run: %s | Dataset: %s | Backbone: %s | "
        "Clients: %d | Rounds: %d | Strategy: %s | Partition: %s(α=%.2f)",
        cfg.run_name, cfg.dataset, cfg.backbone,
        cfg.num_clients, cfg.num_rounds, cfg.strategy,
        cfg.partition, cfg.alpha,
    )

    # ── Initialize Server ──────────────────────────────────────────────────
    server = SkinFLServer(cfg)

    last_done = -1
    if cfg.resume:
        ckpt_payload = load_fl_checkpoint(fl_checkpoint_path(cfg), cfg)
        if ckpt_payload is not None:
            apply_fl_checkpoint_to_server(server, ckpt_payload, cfg)
            hydrate_tracker_after_resume(server.tracker, ckpt_payload)
            if cfg.ignore_completed_early_stop:
                server.tracker.should_stop = False
            last_done = int(ckpt_payload["last_completed_round"])
            logger.info(
                "Resuming | last completed global round: %d / %d",
                last_done,
                cfg.num_rounds,
            )
        else:
            logger.info("resume=True but no valid checkpoint; starting from scratch.")

    # ── Initialize All Clients ─────────────────────────────────────────────
    clients = [
        SkinFLClient(partition_id=i, cfg=cfg)
        for i in range(cfg.num_clients)
    ]

    n_active = max(1, int(cfg.num_clients * cfg.fraction_fit))
    use_journal = not _use_rich_round_bar(cfg)

    round_task = None
    progress = None
    if _use_rich_round_bar(cfg):
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn(" ETA "),
            TimeRemainingColumn(),
            TextColumn("• [cyan]Acc: {task.fields[acc]}[/cyan]"),
            TextColumn("• [green]F1: {task.fields[f1]}[/green]"),
            expand=False,
        )
        round_task = progress.add_task(
            f"[magenta]FL {cfg.run_name}[/magenta]",
            total=cfg.num_rounds,
            acc="N/A",
            f1="N/A",
        )
        progress.start()
        if last_done > 0:
            progress.update(
                round_task,
                completed=min(last_done, cfg.num_rounds),
            )

    if last_done < 0:
        logger.info("Running initial centralized evaluation (Round 0)")
        server.evaluate(0)
        start_fl_round = 1
    else:
        logger.info(
            "Skipping Round 0 eval (resumed; last completed round=%d).",
            last_done,
        )
        start_fl_round = last_done + 1

    if _clear_round_sentinels_enabled():
        _emit_notebook_sentinel(_SENTINEL_FL_LOOP)

    run_fl_loop = True
    if start_fl_round > cfg.num_rounds:
        run_fl_loop = False
        logger.info(
            "Checkpoint already at or past num_rounds=%d; no FL rounds to run.",
            cfg.num_rounds,
        )
    elif (
        server.tracker.should_stop
        and not cfg.ignore_completed_early_stop
        and start_fl_round <= cfg.num_rounds
    ):
        run_fl_loop = False
        logger.info(
            "Early stop from checkpoint; not running rounds %d–%d. "
            "Set ignore_completed_early_stop to continue training.",
            start_fl_round,
            cfg.num_rounds,
        )

    # ── FL Loop ───────────────────────────────────────────────────────────
    journal_round_bar = None
    if run_fl_loop and use_journal:
        journal_round_bar = tqdm(
            range(start_fl_round, cfg.num_rounds + 1),
            desc=f"FL rounds | {cfg.run_name}",
            unit="round",
            dynamic_ncols=True,
            smoothing=0.08,
            mininterval=1.0,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )
    try:
        if run_fl_loop:
            round_iter = (
                journal_round_bar
                if journal_round_bar is not None
                else range(start_fl_round, cfg.num_rounds + 1)
            )

            for server_round in round_iter:
                if _clear_round_sentinels_enabled():
                    _emit_notebook_sentinel(f"{_SENTINEL_ROUND_PREFIX} {server_round}")

                rng_pick = random.Random(
                    federated_round_client_pick_seed(
                        cfg.seed, server_round, cfg.num_clients
                    )
                )
                picked_idx = rng_pick.sample(range(len(clients)), n_active)
                sampled_clients = [clients[i] for i in picked_idx]

                if use_journal:
                    ids = ", ".join(str(c.partition_id) for c in sampled_clients)
                    logger.info(
                        "Round %d/%d | sampled clients (%d/%d): %s",
                        server_round,
                        cfg.num_rounds,
                        len(sampled_clients),
                        cfg.num_clients,
                        ids,
                    )

                client_results = []
                round_client_entries: list[dict] = []
                for client in sampled_clients:
                    if use_journal:
                        logger.info("  Client %d training ...", client.partition_id)
                    new_state_dict, n_train, metrics = client.fit(
                        server.global_state_dict, server_round
                    )
                    lt = metrics.get("local_test") or {}
                    if use_journal:
                        logger.info(
                            "  Client %d done | train_loss=%.4f | n_train=%d | "
                            "local_test_macro_f1=%.4f",
                            client.partition_id,
                            float(metrics.get("train_loss", float("nan"))),
                            n_train,
                            float(lt.get("macro_f1", float("nan"))),
                        )
                    round_client_entries.append(
                        {"client_id": client.partition_id, "n_train": n_train, **metrics}
                    )
                    client_results.append((new_state_dict, n_train))

                _save_round_clients_json(
                    cfg.results_dir, cfg.run_name, server_round, round_client_entries
                )

                if use_journal:
                    logger.info("  Aggregating (%s)", cfg.strategy)

                server.aggregate(client_results)
                server.evaluate(server_round)

                if journal_round_bar is not None and server.tracker.history:
                    lm = server.tracker.history[-1]
                    journal_round_bar.set_postfix(
                        vF1=f"{lm.get('val', {}).get('macro_f1', float('nan')):.3f}",
                        tF1=f"{lm.get('test', {}).get('macro_f1', float('nan')):.3f}",
                        refresh=False,
                    )

                if progress and round_task is not None:
                    latest_metrics = server.tracker.history[-1]
                    progress.update(
                        round_task,
                        advance=1,
                        acc=f"{latest_metrics.get('val', {}).get('accuracy', float('nan')):.3f}",
                        f1=f"{latest_metrics.get('val', {}).get('macro_f1', float('nan')):.3f}",
                    )

                if server.tracker.should_stop:
                    logger.info("Early stopping triggered, breaking simulation loop.")
                    break
    finally:
        if journal_round_bar is not None:
            journal_round_bar.close()

    if progress:
        progress.stop()

    _hist = server.tracker.history
    _last_r = max(int(h.get("round", -1)) for h in _hist) if _hist else -1
    selected = server.finalize()
    logger.info(
        "Selected round %d (best validation macro-F1) | val F1=%.4f | test F1=%.4f acc=%.4f",
        selected["best_round"], selected["val"]["macro_f1"],
        selected["test"]["macro_f1"], selected["test"]["accuracy"],
    )
    server.tracker.save_summary(
        selected=selected,
        experiment={
            "mode": "federated",
            "dataset": cfg.dataset,
            "backbone": cfg.backbone,
            "num_rounds": cfg.num_rounds,
            "local_epochs": cfg.local_epochs,
            "num_clients": cfg.num_clients,
            "fraction_fit": float(cfg.fraction_fit),
            "strategy": cfg.strategy.lower(),
            "partition": cfg.partition,
            "alpha": float(cfg.alpha),
            "seed": cfg.seed,
            "fedprox_mu": float(cfg.fedprox_mu),
            "fedadam_eta": float(cfg.fedadam_eta),
            "fedadam_tau": float(cfg.fedadam_tau),
            "lr": float(cfg.lr),
            "batch_size": int(cfg.batch_size),
            "client_local_eval_fraction": float(cfg.client_local_eval_fraction),
            "n_train_per_client": [int(c.n_train) for c in clients],
        },
        stopped={
            "early_stop": bool(server.tracker.should_stop),
            "last_round": _last_r,
        },
    )

    logger.info(
        "Simulation complete. Results in %s/%s/",
        cfg.results_dir, cfg.run_name,
    )
