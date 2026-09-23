"""
Federated server for FedDermNet (plain PyTorch simulation, no Flower runtime).

The server
  1. holds the global model state_dict;
  2. aggregates client updates (FedAvg / FedProx share the weighted average;
     FedAdam applies an adaptive server step to trainable parameters only);
  3. after every round evaluates the global model on the VALIDATION split, which is
     the only split used for model selection and early stopping;
  4. evaluates the same global model on the TEST split for reporting and curves only;
  5. keeps the weights of the best-validation round (``best_global.pt``) and, at the
     end, re-scores that checkpoint on the test split and archives its test-set
     probabilities for bootstrap confidence intervals.

Revision R1: the v1 code selected the reported round and triggered early stopping on
the TEST split. That was test-set selection and is corrected here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import SkinDataset, dataloader_rng_seed, make_dataloader
from src.data.transforms import get_val_transforms
from src.eval.metrics import evaluate_model, predict_proba
from src.fl.checkpoint_io import save_fl_checkpoint
from src.fl.config import ExperimentConfig, manifest_path_for, num_classes_for
from src.fl.strategies import (
    FedAdamOptimizer,
    MetricsTracker,
    aggregate_fedadam,
    aggregate_fedavg,
)
from src.models.build import build_model

logger = logging.getLogger(__name__)


class SkinFLServer:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes_for(cfg.dataset)
        manifest_path = manifest_path_for(cfg)

        def _loader(split: str, slot: int):
            ds = SkinDataset(
                manifest_path=manifest_path,
                split=split,
                client_id=None,
                transform=get_val_transforms(cfg.img_size),
            )
            return make_dataloader(
                ds,
                batch_size=cfg.batch_size * 4,
                weighted_sampling=False,
                shuffle=False,
                num_workers=cfg.num_workers,
                rng_seed=dataloader_rng_seed(cfg.seed, None, slot=slot),
            )

        self.val_loader = _loader("val", 11)
        self.test_loader = _loader("test", 10)

        init_model = build_model(
            cfg.backbone, num_classes=self.num_classes, pretrained=True, device=self.device
        )
        self.global_state_dict = {k: v.detach().cpu() for k, v in init_model.state_dict().items()}
        # Names of float trainable parameters (FedAdam acts on these only; BN buffers
        # and counters are averaged, never passed through the adaptive server step).
        self.param_names = [n for n, _ in init_model.named_parameters()]
        del init_model
        self._eval_model = build_model(
            cfg.backbone, num_classes=self.num_classes, pretrained=False, device=self.device
        )

        self.tracker = MetricsTracker(
            results_dir=cfg.results_dir, run_name=cfg.run_name, patience=cfg.early_stop_patience
        )
        self.best_state_path = Path(cfg.results_dir) / cfg.run_name / "best_global.pt"

        if self.cfg.strategy.lower() == "fedadam":
            self.fedadam_optimizer = FedAdamOptimizer(
                self.global_state_dict,
                param_names=self.param_names,
                eta=cfg.fedadam_eta,
                beta_1=cfg.fedadam_beta1,
                beta_2=cfg.fedadam_beta2,
                tau=cfg.fedadam_tau,
            )

    # ── aggregation ──────────────────────────────────────────────────────────
    def aggregate(self, client_results: list[tuple[dict[str, torch.Tensor], int]]) -> None:
        client_results = [(sd, n) for sd, n in client_results if n > 0]
        if not client_results:
            logger.warning("No client with data this round; global model unchanged.")
            return
        s = self.cfg.strategy.lower()
        if s in ("fedavg", "fedprox"):
            self.global_state_dict = aggregate_fedavg(client_results, like=self.global_state_dict)
        elif s == "fedadam":
            self.global_state_dict = aggregate_fedadam(
                self.global_state_dict, client_results, self.fedadam_optimizer
            )
        else:
            raise ValueError(f"Unknown strategy: {s}")

    # ── evaluation ───────────────────────────────────────────────────────────
    def _model(self) -> torch.nn.Module:
        self._eval_model.load_state_dict(self.global_state_dict)
        return self._eval_model

    def evaluate(self, server_round: int) -> None:
        model = self._model()
        amp = bool(self.cfg.amp)
        val = evaluate_model(model, self.val_loader, self.device, self.num_classes, amp=amp)
        record: dict = {"val": val}
        if self.cfg.eval_test_every_round:
            record["test"] = evaluate_model(
                model, self.test_loader, self.device, self.num_classes, amp=amp
            )
        improved = self.tracker.update(server_round, record)
        if improved and self.cfg.save_best_global:
            torch.save(
                {"global_state_dict": self.global_state_dict, "round": server_round},
                self.best_state_path,
            )

        fed_opt = getattr(self, "fedadam_optimizer", None)
        save_fl_checkpoint(
            self.cfg,
            global_state_dict=self.global_state_dict,
            last_completed_round=server_round,
            best_macro_f1=self.tracker.best_macro_f1,
            rounds_without_improvement=self.tracker.rounds_without_improvement,
            should_stop=self.tracker.should_stop,
            fedadam_state=(fed_opt.state_dict() if fed_opt is not None else None),
        )
        t = record.get("test", {})
        logger.info(
            "Round %3d | val F1=%.4f acc=%.4f | test F1=%.4f acc=%.4f%s%s",
            server_round, val["macro_f1"], val["accuracy"],
            t.get("macro_f1", float("nan")), t.get("accuracy", float("nan")),
            " [best-val]" if improved else "",
            " [EARLY STOP]" if self.tracker.should_stop else "",
        )

    def finalize(self) -> dict:
        """Score the best-validation checkpoint on val and test; archive test probabilities."""
        if self.best_state_path.is_file():
            payload = torch.load(self.best_state_path, map_location="cpu", weights_only=False)
            self.global_state_dict = payload["global_state_dict"]
            best_round = int(payload["round"])
        else:
            best_round = self.tracker.best_round
        model = self._model()
        amp = bool(self.cfg.amp)
        val = evaluate_model(model, self.val_loader, self.device, self.num_classes, amp=amp)
        test = evaluate_model(model, self.test_loader, self.device, self.num_classes, amp=amp)
        probs, labels = predict_proba(model, self.test_loader, self.device, amp=amp)
        np.savez_compressed(
            Path(self.cfg.results_dir) / self.cfg.run_name / "test_predictions_best.npz",
            probs=probs.astype(np.float32),
            labels=labels.astype(np.int16),
            image_id=np.asarray(self.test_loader.dataset.df["image_id"].astype(str)),
        )
        return {"best_round": best_round, "val": val, "test": test}
