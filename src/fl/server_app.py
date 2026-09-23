"""
Custom Server for SkinFLNet++ (Standard Python, no Flower).

The server:
  1. Holds the global model state_dict.
  2. Selects a fraction of clients per round.
  3. Aggregates client updates via custom logic (FedAvg, FedAdam).
  4. Evaluates the aggregated model on the held-out global test set.
  5. Logs per-round metrics to results/<run_name>/round_XXX.json.
  6. Tracks early stopping.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.data.datasets import SkinDataset, make_dataloader, dataloader_rng_seed
from src.data.transforms import get_val_transforms
from src.eval.metrics import evaluate_model
from src.fl.config import ExperimentConfig, dataset_folder_for, num_classes_for
from src.fl.checkpoint_io import save_fl_checkpoint
from src.fl.strategies import (
    MetricsTracker,
    aggregate_fedavg,
    aggregate_fedadam,
    FedAdamOptimizer
)
from src.models.build import build_model

logger = logging.getLogger(__name__)


class SkinFLServer:
    """
    Standard Python implementation of the federated learning server.

    Attributes:
        cfg:               Global experiment config.
        device:            Torch compute device.
        num_classes:       Output dimension.
        test_loader:       DataLoader for the global test set.
        global_state_dict: Current global model parameters (on CPU).
        tracker:           MetricsTracker for logging and early stopping.
        fedadam_optimizer: Server-side optimizer (optional).
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes_for(cfg.dataset)

        # ── Global Test Set ──────────────────────────────────────────────────
        manifest_path = Path(cfg.data_root) / dataset_folder_for(cfg.dataset) / "manifest.csv"

        test_ds = SkinDataset(
            manifest_path=manifest_path,
            split="test",
            client_id=None,
            transform=get_val_transforms(cfg.img_size),
        )
        self.test_loader = make_dataloader(
            test_ds,
            batch_size=cfg.batch_size * 2,
            weighted_sampling=False,
            num_workers=cfg.num_workers,
            rng_seed=dataloader_rng_seed(cfg.seed, None, slot=10),
        )

        # ── Global Model ─────────────────────────────────────────────────────
        init_model = build_model(
            cfg.backbone,
            num_classes=self.num_classes,
            pretrained=True,
            device=self.device
        )
        # Store state_dict on CPU to save VRAM between rounds
        self.global_state_dict = {
            k: v.cpu() for k, v in init_model.state_dict().items()
        }
        del init_model

        # ── Metrics & Strategy ───────────────────────────────────────────────
        self.tracker = MetricsTracker(
            results_dir=cfg.results_dir,
            run_name=cfg.run_name,
            patience=cfg.early_stop_patience,
        )
        
        if self.cfg.strategy.lower() == "fedadam":
            self.fedadam_optimizer = FedAdamOptimizer(self.global_state_dict)

    def aggregate(
        self,
        client_results: list[tuple[dict[str, torch.Tensor], int]]
    ) -> None:
        """Aggregate client updates into the global state_dict."""
        s = self.cfg.strategy.lower()
        if s in ["fedavg", "fedprox"]:
            self.global_state_dict = aggregate_fedavg(client_results)
        elif s == "fedadam":
            self.global_state_dict = aggregate_fedadam(
                self.global_state_dict, client_results, self.fedadam_optimizer
            )
        else:
            raise ValueError(f"Unknown strategy: {s}")

    def evaluate(self, server_round: int) -> None:
        """Evaluate the current global model on the global test set."""
        model = build_model(
            self.cfg.backbone,
            num_classes=self.num_classes,
            pretrained=False,
            device=self.device
        )
        model.load_state_dict(self.global_state_dict)
        
        metrics = evaluate_model(model, self.test_loader, self.device, self.num_classes)

        # Calibration plot every 10 rounds and at the end
        if server_round % 10 == 0 or server_round == self.cfg.num_rounds:
            from src.eval.calibration import plot_reliability_diagram
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for imgs, lbls in self.test_loader:
                    logits = model(imgs.to(self.device))
                    all_logits.append(logits.cpu())
                    all_labels.append(lbls)
            
            logits_cat = torch.cat(all_logits)
            labels_cat = torch.cat(all_labels).numpy()
            probs = torch.softmax(logits_cat, dim=1).numpy()
            preds = logits_cat.argmax(1).numpy()
            confidence = probs.max(axis=1)
            
            cal_path = (
                Path(self.cfg.figures_dir) / self.cfg.run_name
                / f"calibration_round_{server_round:03d}.png"
            )
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            
            ece = plot_reliability_diagram(
                confidence, preds, labels_cat,
                save_path=cal_path,
                title=f"{self.cfg.run_name} – Round {server_round}",
            )
            metrics["ece"] = ece

        self.tracker.update(server_round, metrics)

        fed_opt = getattr(self, "fedadam_optimizer", None)
        save_fl_checkpoint(
            self.cfg,
            global_state_dict=self.global_state_dict,
            last_completed_round=server_round,
            best_macro_f1=self.tracker.best_macro_f1,
            rounds_without_improvement=self.tracker.rounds_without_improvement,
            should_stop=self.tracker.should_stop,
            fedadam_state=(
                fed_opt.state_dict()
                if fed_opt is not None
                else None
            ),
        )

        logger.info(
            "Round %3d | Loss=%.4f | Acc=%.4f | F1=%.4f | AUROC=%.4f%s",
            server_round,
            metrics.get("loss", float("nan")),
            metrics.get("accuracy", float("nan")),
            metrics.get("macro_f1", float("nan")),
            metrics.get("auroc", float("nan")),
            " [EARLY STOP TRIGGERED]" if self.tracker.should_stop else "",
        )
