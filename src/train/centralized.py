"""
Centralized upper-bound training for SkinFLNet++.

Trains the same backbone+head on the POOLED training set (all FL clients combined).
Used as the centralized upper bound in the ablation table (§10.2 row 6).

Identical optimizer, schedule, augmentation, and loss as FL clients —
so the FL gap is purely attributable to the federated setup.

``centralized_best_on`` (YAML / :class:`ExperimentConfig`): default ``"val"`` picks
the checkpoint by validation F1 (honest test report). Set ``"test"`` only to mirror
the federated server's per-round **test** evaluation (test-set leakage into model
selection — not valid for claiming generalization).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from src.data.datasets import SkinDataset, make_dataloader, dataloader_rng_seed
from src.data.transforms import get_train_transforms, get_val_transforms
from src.eval.calibration import plot_reliability_diagram
from src.eval.metrics import evaluate_model
from src.fl.checkpoint_io import (
    centralized_checkpoint_path,
    load_centralized_checkpoint,
    save_centralized_checkpoint,
)
from src.fl.config import ExperimentConfig, dataset_folder_for, num_classes_for
from src.fl.simulate import seed_everything
from src.models.build import build_model, freeze_backbone
from src.train.losses import get_loss_fn

logger = logging.getLogger(__name__)


def train_centralized(cfg: ExperimentConfig) -> dict:
    """
    Train one epoch per 'round' (matching the FL communication budget).

    Args:
        cfg: ExperimentConfig with run_name, backbone, dataset, etc.

    Returns:
        Final metrics dictionary.
    """
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = num_classes_for(cfg.dataset)

    # ── Data ─────────────────────────────────────────────────────────────────
    manifest_path = Path(cfg.data_root) / dataset_folder_for(cfg.dataset) / "manifest.csv"

    train_ds = SkinDataset(
        manifest_path=manifest_path,
        split="train",
        client_id=None,   # pool all clients
        transform=get_train_transforms(cfg.img_size),
    )
    val_ds = SkinDataset(
        manifest_path=manifest_path,
        split="val",
        transform=get_val_transforms(cfg.img_size),
    )
    test_ds = SkinDataset(
        manifest_path=manifest_path,
        split="test",
        transform=get_val_transforms(cfg.img_size),
    )

    train_loader = make_dataloader(
        train_ds,
        batch_size=cfg.batch_size,
        weighted_sampling=True,
        num_workers=cfg.num_workers,
        rng_seed=dataloader_rng_seed(cfg.seed, None, slot=21),
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        weighted_sampling=False,
        num_workers=cfg.num_workers,
        rng_seed=dataloader_rng_seed(cfg.seed, None, slot=22),
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        weighted_sampling=False,
        num_workers=cfg.num_workers,
        rng_seed=dataloader_rng_seed(cfg.seed, None, slot=23),
    )

    best_on = str(getattr(cfg, "centralized_best_on", "val")).lower().strip()
    if best_on not in ("val", "test"):
        raise ValueError(f"centralized_best_on must be 'val' or 'test', got {best_on!r}")
    selection_loader = test_loader if best_on == "test" else val_loader

    logger.info(
        "Centralized | %d train | %d val | %d test samples | best_checkpoint_on=%s",
        len(train_ds), len(val_ds), len(test_ds), best_on,
    )
    if best_on == "test":
        logger.warning(
            "centralized_best_on=test: early stopping and best_model.pth use the "
            "TEST set — biased vs real deployment; use for FL-protocol parity only.",
        )

    # ── Model & Loss ─────────────────────────────────────────────────────────
    model = build_model(cfg.backbone, num_classes=num_classes, pretrained=True, device=device)
    loss_fn = get_loss_fn(cfg.dataset).to(device)

    # ── Results dir ──────────────────────────────────────────────────────────
    results_dir = Path(cfg.results_dir) / cfg.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(cfg.figures_dir) / cfg.run_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_val_f1 = -1.0
    patience_counter = 0
    total_epochs = cfg.num_rounds * cfg.local_epochs  # match FL compute budget
    patience = cfg.early_stop_patience

    resume_payload: dict | None = None
    start_epoch = 1

    if cfg.resume:
        resume_payload = load_centralized_checkpoint(centralized_checkpoint_path(cfg), cfg)
        if resume_payload is not None:
            model.load_state_dict(resume_payload["model_state_dict"])
            history = list(resume_payload.get("history", []))
            best_val_f1 = float(resume_payload.get("best_val_f1", -1.0))
            patience_counter = int(resume_payload.get("patience_counter", 0))
            last_completed_epoch = int(resume_payload["last_completed_epoch"])
            start_epoch = last_completed_epoch + 1
            if cfg.ignore_completed_early_stop:
                patience_counter = 0

    skip_training = (start_epoch > total_epochs) or (
        resume_payload is not None
        and patience_counter >= patience
        and not cfg.ignore_completed_early_stop
    )

    stopped_early = False
    if skip_training:
        if start_epoch <= total_epochs and resume_payload is not None:
            if patience_counter >= patience and not cfg.ignore_completed_early_stop:
                stopped_early = True

    if skip_training:
        if start_epoch > total_epochs:
            logger.info(
                "Centralized: skipping training (resume past total_epochs=%d).",
                total_epochs,
            )
        else:
            logger.info(
                "Centralized: skipping training (checkpoint indicates early stop exhausted).",
            )

    if not skip_training:
        epoch_range = range(start_epoch, total_epochs + 1)
        epoch_pbar = tqdm(
            epoch_range,
            desc=f"Centralized | {cfg.run_name}",
            unit="epoch",
            dynamic_ncols=True,
            smoothing=0.08,
            mininterval=1.0,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )
        for epoch in epoch_pbar:
            # Partial backbone freeze every epoch (matches FL client: input-near half frozen).
            freeze_backbone(model)

            # Cosine LR (map epoch to round)
            virtual_round_lr = (epoch - 1) // cfg.local_epochs + 1
            lr = cfg.lr * 0.5 * (
                1.0
                + math.cos(
                    math.pi * (virtual_round_lr - 1) / max(cfg.num_rounds - 1, 1)
                )
            )

            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=cfg.weight_decay,
            )

            # ── Train one epoch ───────────────────────────────────────────────────
            model.train()
            ep_loss, n_batches = 0.0, 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
                n_batches += 1

            avg_loss = ep_loss / max(n_batches, 1)
            epoch_pbar.set_postfix(tLoss=f"{avg_loss:.4f}", refresh=False)

            # Evaluate on val every `local_epochs` epochs (= one FL round)
            if epoch % cfg.local_epochs == 0:
                virtual_round = epoch // cfg.local_epochs
                selection_metrics = evaluate_model(
                    model, selection_loader, device, num_classes
                )
                selection_metrics["epoch"] = epoch
                selection_metrics["virtual_round"] = virtual_round
                selection_metrics["train_loss"] = ep_loss / max(n_batches, 1)
                selection_metrics["centralized_best_on"] = best_on
                history.append(selection_metrics)

                _tag = "Test" if best_on == "test" else "Val"
                logger.info(
                    "Epoch %d/%d (round %d) | TrainLoss=%.4f | %sAcc=%.4f | %sF1=%.4f",
                    epoch, total_epochs, virtual_round,
                    selection_metrics["train_loss"],
                    _tag, selection_metrics["accuracy"],
                    _tag, selection_metrics["macro_f1"],
                )

                epoch_pbar.set_postfix(
                    vAcc=f"{selection_metrics['accuracy']:.3f}",
                    vF1=f"{selection_metrics['macro_f1']:.3f}",
                    refresh=True,
                )

                # Save round JSON
                with open(results_dir / f"round_{virtual_round:03d}.json", "w") as f:
                    json.dump(selection_metrics, f, indent=2)

                # Early stopping (macro-F1 on selection split: val or test)
                if selection_metrics["macro_f1"] > best_val_f1 + 1e-4:
                    best_val_f1 = selection_metrics["macro_f1"]
                    patience_counter = 0
                    torch.save(model.state_dict(), results_dir / "best_model.pth")
                else:
                    patience_counter += 1

                save_centralized_checkpoint(
                    cfg,
                    model_state_dict=model.state_dict(),
                    last_completed_epoch=epoch,
                    last_completed_virtual_round=virtual_round,
                    best_val_f1=best_val_f1,
                    patience_counter=patience_counter,
                    history=history,
                )

                if patience_counter >= cfg.early_stop_patience:
                    stopped_early = True
                    logger.info("Early stopping at epoch %d (round %d)", epoch, virtual_round)
                    break

    # ── Final test evaluation ─────────────────────────────────────────────────
    best_ckpt = results_dir / "best_model.pth"
    if best_ckpt.is_file():
        try:
            sd = torch.load(best_ckpt, map_location=device, weights_only=False)
        except TypeError:
            sd = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(sd)

    test_metrics = evaluate_model(model, test_loader, device, num_classes)
    test_metrics["history"] = history

    # Calibration plot
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            all_logits.append(model(imgs.to(device)).cpu())
            all_labels.append(lbls)
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits_cat, dim=1).numpy()
    preds = logits_cat.argmax(1).numpy()
    confidence = probs.max(axis=1)
    ece = plot_reliability_diagram(
        confidence, preds, labels_cat,
        save_path=figures_dir / "calibration_final.png",
        title=f"{cfg.run_name} – Final",
    )
    test_metrics["ece"] = ece

    _lr = -1
    if history:
        _lr = max(
            int(h.get("virtual_round", h.get("round", -1))) for h in history
        )

    test_metrics["experiment"] = {
        "mode": "centralized",
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
        "centralized_best_on": best_on,
    }
    test_metrics["stopped"] = {
        "early_stop": stopped_early,
        "last_round": _lr,
    }

    with open(results_dir / "summary.json", "w") as f:
        import numpy as np

        json.dump(
            test_metrics,
            f,
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o),
        )

    logger.info(
        "Centralized FINAL | Acc=%.4f | F1=%.4f | AUROC=%.4f | ECE=%.4f",
        test_metrics["accuracy"], test_metrics["macro_f1"],
        test_metrics.get("auroc", float("nan")), ece,
    )
    return test_metrics
