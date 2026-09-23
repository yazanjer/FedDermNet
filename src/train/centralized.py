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
from src.eval.metrics import evaluate_model, predict_proba
from src.fl.checkpoint_io import (
    centralized_checkpoint_path,
    load_centralized_checkpoint,
    save_centralized_checkpoint,
)
from src.fl.config import ExperimentConfig, dataset_folder_for, manifest_path_for, num_classes_for
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
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = num_classes_for(cfg.dataset)

    # ── Data ─────────────────────────────────────────────────────────────────
    manifest_path = manifest_path_for(cfg)

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
        persistent_workers=True,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        weighted_sampling=False,
        shuffle=False,
        num_workers=cfg.num_workers,
        rng_seed=dataloader_rng_seed(cfg.seed, None, slot=22),
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        weighted_sampling=False,
        shuffle=False,
        num_workers=cfg.num_workers,
        rng_seed=dataloader_rng_seed(cfg.seed, None, slot=23),
    )

    best_on = str(getattr(cfg, "centralized_best_on", "val")).lower().strip()
    if best_on != "val":
        raise ValueError(
            "Revision R1: centralized checkpoint selection must use the validation split "
            f"(centralized_best_on='val'); got {best_on!r}."
        )

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
    best_round = -1
    patience_counter = 0
    # Revision R1: match the federated SAMPLE budget. One FL round processes, in
    # expectation, fraction_fit * local_epochs passes over the pooled training data
    # (half of the clients x E local epochs), so one centralized "virtual round" is
    # that many epochs (1 epoch for the default recipe).
    epochs_per_round = max(1, int(round(cfg.fraction_fit * cfg.local_epochs)))
    total_epochs = cfg.num_rounds * epochs_per_round
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
            if history:
                best_round = int(max(history, key=lambda h: h["val"]["macro_f1"])["round"])
            last_completed_epoch = int(resume_payload["last_completed_epoch"])
            start_epoch = last_completed_epoch + 1
            if cfg.ignore_completed_early_stop:
                patience_counter = 0

    skip_training = (start_epoch > total_epochs) or (
        resume_payload is not None
        and patience > 0 and patience_counter >= patience
        and not cfg.ignore_completed_early_stop
    )

    stopped_early = False
    if skip_training:
        if start_epoch <= total_epochs and resume_payload is not None:
            if patience > 0 and patience_counter >= patience and not cfg.ignore_completed_early_stop:
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
            virtual_round_lr = (epoch - 1) // epochs_per_round + 1
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
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=bool(cfg.amp) and device.type == "cuda"):
                    logits = model(images)
                loss = loss_fn(logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
                n_batches += 1

            avg_loss = ep_loss / max(n_batches, 1)
            epoch_pbar.set_postfix(tLoss=f"{avg_loss:.4f}", refresh=False)

            # Evaluate once per virtual round: VALIDATION drives selection and early
            # stopping; TEST is logged for curves only.
            if epoch % epochs_per_round == 0:
                virtual_round = epoch // epochs_per_round
                val_m = evaluate_model(model, val_loader, device, num_classes, amp=cfg.amp)
                rec = {"val": val_m, "epoch": epoch, "round": virtual_round,
                       "train_loss": ep_loss / max(n_batches, 1)}
                if cfg.eval_test_every_round:
                    rec["test"] = evaluate_model(model, test_loader, device, num_classes, amp=cfg.amp)
                history.append(rec)
                logger.info(
                    "Epoch %d/%d (round %d) | TrainLoss=%.4f | val F1=%.4f | test F1=%.4f",
                    epoch, total_epochs, virtual_round, rec["train_loss"], val_m["macro_f1"],
                    rec.get("test", {}).get("macro_f1", float("nan")),
                )
                with open(results_dir / f"round_{virtual_round:03d}.json", "w") as f:
                    json.dump(rec, f, indent=2)
                if val_m["macro_f1"] > best_val_f1 + 1e-4:
                    best_val_f1 = val_m["macro_f1"]
                    best_round = virtual_round
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

                if cfg.early_stop_patience > 0 and patience_counter >= cfg.early_stop_patience:
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

    val_metrics = evaluate_model(model, val_loader, device, num_classes, amp=cfg.amp)
    test_metrics = evaluate_model(model, test_loader, device, num_classes, amp=cfg.amp)
    probs, labels = predict_proba(model, test_loader, device, amp=cfg.amp)
    import numpy as np
    np.savez_compressed(
        results_dir / "test_predictions_best.npz",
        probs=probs.astype(np.float32), labels=labels.astype(np.int16),
        image_id=np.asarray(test_ds.df["image_id"].astype(str)),
    )
    if best_round < 0 and history:
        best_round = max(history, key=lambda h: h["val"]["macro_f1"])["round"]
    test_metrics["history"] = history
    test_metrics["selected"] = {"best_round": best_round, "val": val_metrics, "test": dict(test_metrics)}
    test_metrics["selected"]["test"].pop("history", None)
    test_metrics["best_round"] = best_round
    test_metrics["selection"] = "validation macro-F1 (test never used for selection)"

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
        "epochs_per_round": epochs_per_round,
        "n_train": len(train_ds),
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
        "Centralized FINAL (round %d) | Acc=%.4f | F1=%.4f | AUROC=%.4f",
        best_round, test_metrics["accuracy"], test_metrics["macro_f1"],
        test_metrics.get("auroc", float("nan")),
    )
    return test_metrics
