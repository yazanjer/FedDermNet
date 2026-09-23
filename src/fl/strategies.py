"""
FL strategy logic: FedAvg, FedProx, FedAdam.

Replaces Flower strategy classes with custom aggregation functions using
PyTorch state_dicts.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Metrics tracker with early stopping ──────────────────────────────────────


class MetricsTracker:
    """Per-round log + early stopping on VALIDATION macro-F1.

    ``update`` receives ``{"val": {...}, "test": {...}}``. Only ``val`` drives the
    best-round choice and the patience counter; ``test`` is logged for curves.
    """

    def __init__(self, results_dir: str, run_name: str, patience: int = 5) -> None:
        self.results_dir = Path(results_dir) / run_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.best_macro_f1: float = -1.0      # best VALIDATION macro-F1
        self.best_round: int = -1
        self.rounds_without_improvement: int = 0
        self.history: list[dict] = []
        self.should_stop: bool = False

    def update(self, server_round: int, record: dict[str, Any]) -> bool:
        record = dict(record)
        record["round"] = server_round
        record["timestamp"] = time.time()
        self.history.append(record)
        with open(self.results_dir / f"round_{server_round:03d}.json", "w") as f:
            json.dump(record, f, indent=2, default=self._json_default)

        val_f1 = float(record.get("val", {}).get("macro_f1", -1.0))
        improved = val_f1 > self.best_macro_f1 + 1e-4
        if improved:
            self.best_macro_f1 = val_f1
            self.best_round = server_round
            self.rounds_without_improvement = 0
        elif server_round > 0:
            self.rounds_without_improvement += 1
            if self.patience > 0 and self.rounds_without_improvement >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping at round %d (best val macro-F1=%.4f at round %d, patience=%d)",
                    server_round, self.best_macro_f1, self.best_round, self.patience,
                )
        return improved

    def save_summary(
        self,
        *,
        experiment: dict[str, Any] | None = None,
        stopped: dict[str, Any] | None = None,
        selected: dict[str, Any] | None = None,
    ) -> None:
        """Write all-rounds summary JSON.

        Optionally embeds ``experiment`` (hyperparameters for viz/reports) and
        ``stopped`` (early-stop flag + last round index). Older readers may ignore these keys.
        """
        summary_path = self.results_dir / "summary.json"
        payload: dict[str, Any] = {
            "history": self.history,
            "best_val_macro_f1": self.best_macro_f1,
            "best_round": self.best_round,
            "selection": "validation macro-F1 (test never used for selection)",
        }
        if experiment is not None:
            payload["experiment"] = experiment
        if stopped is not None:
            payload["stopped"] = stopped
        if selected is not None:
            payload["selected"] = selected
        with open(summary_path, "w") as f:
            json.dump(payload, f, indent=2, default=self._json_default)
        logger.info("Results saved to %s", summary_path)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)}")


# ── Custom Aggregation Functions ──────────────────────────────────────────────


def aggregate_fedavg(
    client_results: list[tuple[dict[str, torch.Tensor], int]],
    like: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Sample-weighted average of client state_dicts (float accumulation).

    Integer buffers (e.g. BatchNorm ``num_batches_tracked``) are cast back to their
    original dtype so the aggregate loads cleanly.
    """
    total = sum(n for _, n in client_results)
    if total == 0:
        return dict(like) if like is not None else {}
    out: dict[str, torch.Tensor] = {}
    for i, (sd, n) in enumerate(client_results):
        w = n / total
        for k, v in sd.items():
            vf = v.detach().float()
            out[k] = vf * w if i == 0 else out[k] + vf * w
    ref = like if like is not None else client_results[0][0]
    for k in out:
        if not ref[k].is_floating_point():
            out[k] = out[k].round().to(ref[k].dtype)
        else:
            out[k] = out[k].to(ref[k].dtype)
    return out


class FedAdamOptimizer:
    """Server-side Adam on the pseudo-gradient (Reddi et al., 2021).

    Revision R1: the adaptive step is applied to trainable PARAMETERS only.
    BatchNorm running statistics and counters are taken from the weighted client
    average; v1 passed them through the Adam step, which distorted the running
    variance and destabilised FedAdam.
    """

    def __init__(
        self,
        initial_state_dict: dict[str, torch.Tensor],
        param_names: list[str] | None = None,
        eta: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
    ):
        self.eta, self.beta_1, self.beta_2, self.tau = float(eta), float(beta_1), float(beta_2), float(tau)
        names = param_names if param_names is not None else [
            k for k, v in initial_state_dict.items() if v.is_floating_point()
        ]
        self.param_names = list(names)
        self.m = {k: torch.zeros_like(initial_state_dict[k], dtype=torch.float32) for k in self.param_names}
        self.v = {k: torch.zeros_like(initial_state_dict[k], dtype=torch.float32) for k in self.param_names}

    def state_dict(self) -> dict[str, Any]:
        return {
            "eta": self.eta, "beta_1": self.beta_1, "beta_2": self.beta_2, "tau": self.tau,
            "param_names": self.param_names,
            "m": {k: v.cpu().clone() for k, v in self.m.items()},
            "v": {k: v.cpu().clone() for k, v in self.v.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.eta = float(state["eta"]); self.beta_1 = float(state["beta_1"])
        self.beta_2 = float(state["beta_2"]); self.tau = float(state["tau"])
        self.param_names = list(state.get("param_names", list(state["m"].keys())))
        self.m = {k: v.clone() for k, v in state["m"].items()}
        self.v = {k: v.clone() for k, v in state["v"].items()}

    def step(self, global_sd: dict[str, torch.Tensor], averaged_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        new_sd = dict(averaged_sd)  # buffers: plain weighted average
        for k in self.param_names:
            g = averaged_sd[k].float() - global_sd[k].float()
            self.m[k] = self.beta_1 * self.m[k] + (1 - self.beta_1) * g
            self.v[k] = self.beta_2 * self.v[k] + (1 - self.beta_2) * g.pow(2)
            upd = global_sd[k].float() + self.eta * self.m[k] / (self.v[k].sqrt() + self.tau)
            new_sd[k] = upd.to(global_sd[k].dtype)
        return new_sd


def aggregate_fedadam(
    global_state_dict: dict[str, torch.Tensor],
    client_results: list[tuple[dict[str, torch.Tensor], int]],
    optimizer: FedAdamOptimizer,
) -> dict[str, torch.Tensor]:
    averaged = aggregate_fedavg(client_results, like=global_state_dict)
    return optimizer.step(global_state_dict, averaged)
