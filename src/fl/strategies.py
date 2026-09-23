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
    """Tracks per-round metrics and implements early stopping logic."""

    def __init__(self, results_dir: str, run_name: str, patience: int = 5) -> None:
        self.results_dir = Path(results_dir) / run_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.best_macro_f1: float = -1.0
        self.rounds_without_improvement: int = 0
        self.history: list[dict] = []
        self.should_stop: bool = False

    def update(self, server_round: int, metrics: dict[str, Any]) -> None:
        metrics["round"] = server_round
        metrics["timestamp"] = time.time()
        self.history.append(metrics)

        # Save per-round JSON
        round_path = self.results_dir / f"round_{server_round:03d}.json"
        with open(round_path, "w") as f:
            json.dump(metrics, f, indent=2, default=self._json_default)

        # Early stopping check on macro-F1
        macro_f1 = metrics.get("macro_f1", -1.0)
        if macro_f1 > self.best_macro_f1 + 1e-4:
            self.best_macro_f1 = macro_f1
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
            if self.rounds_without_improvement >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered at round %d "
                    "(best macro-F1=%.4f, patience=%d)",
                    server_round, self.best_macro_f1, self.patience,
                )

    def save_summary(
        self,
        *,
        experiment: dict[str, Any] | None = None,
        stopped: dict[str, Any] | None = None,
    ) -> None:
        """Write all-rounds summary JSON.

        Optionally embeds ``experiment`` (hyperparameters for viz/reports) and
        ``stopped`` (early-stop flag + last round index). Older readers may ignore these keys.
        """
        summary_path = self.results_dir / "summary.json"
        payload: dict[str, Any] = {
            "history": self.history,
            "best_macro_f1": self.best_macro_f1,
        }
        if experiment is not None:
            payload["experiment"] = experiment
        if stopped is not None:
            payload["stopped"] = stopped
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
    client_results: list[tuple[dict[str, torch.Tensor], int]]
) -> dict[str, torch.Tensor]:
    """Calculates weighted average of client weights."""
    total_samples = sum(num_samples for _, num_samples in client_results)
    if total_samples == 0:
        return {}

    aggregated_state_dict = {}
    
    for i, (state_dict, num_samples) in enumerate(client_results):
        weight = num_samples / total_samples
        for key, value in state_dict.items():
            if i == 0:
                aggregated_state_dict[key] = value.clone() * weight
            else:
                aggregated_state_dict[key] += value * weight
                
    return aggregated_state_dict


class FedAdamOptimizer:
    """Server-side optimizer for FedAdam strategy."""
    def __init__(
        self,
        initial_state_dict: dict[str, torch.Tensor],
        eta: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3
    ):
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau
        self.m = {k: torch.zeros_like(v) for k, v in initial_state_dict.items()}
        self.v = {k: torch.zeros_like(v) for k, v in initial_state_dict.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "eta": self.eta,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "tau": self.tau,
            "m": {k: v.cpu().clone() for k, v in self.m.items()},
            "v": {k: v.cpu().clone() for k, v in self.v.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.eta = float(state["eta"])
        self.beta_1 = float(state["beta_1"])
        self.beta_2 = float(state["beta_2"])
        self.tau = float(state["tau"])
        self.m = {k: v.clone() for k, v in state["m"].items()}
        self.v = {k: v.clone() for k, v in state["v"].items()}

    def step(
        self,
        global_state_dict: dict[str, torch.Tensor],
        pseudo_gradient: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for key in global_state_dict.keys():
            grad = pseudo_gradient[key]
            self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grad
            self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grad ** 2)
            
            new_weight = global_state_dict[key] + self.eta * self.m[key] / (
                torch.sqrt(self.v[key]) + self.tau
            )
            new_state_dict[key] = new_weight
        return new_state_dict


def aggregate_fedadam(
    global_state_dict: dict[str, torch.Tensor],
    client_results: list[tuple[dict[str, torch.Tensor], int]],
    optimizer: FedAdamOptimizer
) -> dict[str, torch.Tensor]:
    """Computes pseudo-gradient and applies FedAdam server-side step."""
    aggregated_client_weights = aggregate_fedavg(client_results)
    
    pseudo_gradient = {}
    for key in global_state_dict.keys():
        pseudo_gradient[key] = aggregated_client_weights[key] - global_state_dict[key]
        
    return optimizer.step(global_state_dict, pseudo_gradient)
