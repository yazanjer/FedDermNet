"""
Custom Client for SkinFLNet++ (Standard Python, no Flower).

Each client:
  1. Receives global model parameters from the server.
  2. Trains locally for `local_epochs` epochs on its private partition (after an
     optional stratified train/local-test split per client).
  3. Uses WeightedRandomSampler (not SMOTE) for class imbalance.
  4. Partially freezes the input-near backbone every round (deeper half + head train).
  5. For strategy ``fedprox``, adds a $\mu/2$ proximal term toward the round's initial global weights on trainable parameters.
  6. Returns updated state_dict + sample count to the server (weighted by train subset).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.data.datasets import SkinDataset, dataloader_rng_seed, make_dataloader
from src.data.transforms import get_train_transforms, get_val_transforms
from src.eval.metrics import evaluate_model
from src.fl.config import ExperimentConfig, dataset_folder_for, manifest_path_for, num_classes_for
from src.models.build import build_model, freeze_backbone
from src.train.losses import get_loss_fn

logger = logging.getLogger(__name__)


def _client_split_seed(global_seed: int, partition_id: int) -> int:
    return int(global_seed) + int(partition_id) * 1009


class SkinFLClient:
    """
    Standard Python implementation for a single federated client.

    Attributes:
        partition_id: ID of the data partition this client uses.
        cfg:          Global experiment config.
        device:       Torch compute device.
        num_classes:  Output dimension.
        train_loader: DataLoader for this client's private training data (train subset).
        local_test_loader: DataLoader for held-out local evaluation after fit.
        model:        The SkinFLNet model.
        loss_fn:      Loss function.
    """

    def __init__(
        self,
        partition_id: int,
        cfg: ExperimentConfig,
    ) -> None:
        self.partition_id = partition_id
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes_for(cfg.dataset)

        frac = cfg.client_local_eval_fraction
        if not (0.0 <= frac < 1.0):
            raise ValueError(
                f"client_local_eval_fraction must be in [0, 1), got {frac}"
            )

        manifest_path = manifest_path_for(cfg)

        base_df = SkinDataset.load_filtered_manifest_df(
            manifest_path,
            split="train",
            client_id=partition_id,
        )

        if frac == 0.0 or len(base_df) == 0:
            # Revision default: the whole shard trains (union of shards = centralized pool).
            if len(base_df) == 0:
                logger.warning(
                    "Client %d has 0 training samples (possible at small alpha).",
                    partition_id,
                )
            train_df = base_df
            test_df = base_df.iloc[:0].copy()
        elif len(base_df) < 2:
            logger.warning(
                "Client %d has %d samples — cannot stratify/split; using all for train.",
                partition_id,
                len(base_df),
            )
            train_df = base_df
            test_df = base_df.iloc[:0].copy()
        else:
            rs = _client_split_seed(cfg.seed, partition_id)
            labels = base_df["label"]
            stratify_arg = labels if labels.nunique() > 1 else None
            try:
                train_df, test_df = train_test_split(
                    base_df,
                    test_size=frac,
                    stratify=stratify_arg,
                    random_state=rs,
                )
            except ValueError:
                logger.warning(
                    "Client %d: stratified split unavailable; using random split.",
                    partition_id,
                )
                train_df, test_df = train_test_split(
                    base_df,
                    test_size=frac,
                    random_state=rs,
                )
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

        train_ds = SkinDataset.from_dataframe(
            manifest_path,
            train_df,
            transform=get_train_transforms(cfg.img_size),
        )
        local_test_ds = SkinDataset.from_dataframe(
            manifest_path,
            test_df,
            transform=get_val_transforms(cfg.img_size),
        )

        self.train_loader = make_dataloader(
            train_ds,
            batch_size=cfg.batch_size,
            weighted_sampling=len(train_ds) > 0,
            num_workers=cfg.num_workers,
            rng_seed=dataloader_rng_seed(cfg.seed, partition_id, slot=1),
        )
        self.local_test_loader = make_dataloader(
            local_test_ds,
            batch_size=cfg.batch_size * 2,
            weighted_sampling=False,
            num_workers=cfg.num_workers,
            rng_seed=dataloader_rng_seed(cfg.seed, partition_id, slot=2),
        )
        self.n_train = len(train_ds)

        self.model = build_model(
            backbone_name=cfg.backbone,
            num_classes=self.num_classes,
            pretrained=True,
            device=self.device,
        )

        self.loss_fn = get_loss_fn(cfg.dataset).to(self.device)

        logger.info(
            "Client %d | train=%d local_test=%d (fraction=%.2f) | device=%s",
            partition_id,
            len(train_ds),
            len(local_test_ds),
            frac,
            self.device,
        )

    def fit(
        self,
        global_state_dict: dict[str, torch.Tensor],
        server_round: int
    ) -> tuple[dict[str, torch.Tensor], int, dict]:
        """Load global model, train locally, evaluate on local holdout, return updates."""
        if self.n_train == 0:
            return ({k: v.clone() for k, v in global_state_dict.items()}, 0,
                    {"train_loss": float("nan"), "n_local_train": 0, "n_local_test": 0, "local_test": {}})
        self.model.load_state_dict(global_state_dict)

        freeze_backbone(self.model)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self._cosine_lr(server_round),
            weight_decay=self.cfg.weight_decay,
        )

        train_loss = self._train_local(
            optimizer,
            self.cfg.local_epochs,
            global_anchor=self._fedprox_anchor(global_state_dict)
            if self.cfg.strategy.lower() == "fedprox" and self.cfg.fedprox_mu > 0.0
            else None,
        )

        local_test_metrics: dict = {}
        if len(self.local_test_loader.dataset) > 0:
            local_test_metrics = evaluate_model(
                self.model,
                self.local_test_loader,
                self.device,
                self.num_classes,
                amp=self.cfg.amp,
            )

        metrics_out = {
            "train_loss": train_loss,
            "n_local_train": self.n_train,
            "n_local_test": len(self.local_test_loader.dataset),
            "local_test": local_test_metrics,
        }

        logger.debug(
            "Client %d | Round %d | Loss=%.4f | LR=%.2e",
            self.partition_id, server_round, train_loss,
            self._cosine_lr(server_round),
        )

        return (
            {k: v.cpu() for k, v in self.model.state_dict().items()},
            self.n_train,
            metrics_out,
        )

    def _fedprox_anchor(
        self, global_state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Per-parameter copies of the broadcast global weights (trainable params only)."""
        anchor: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                anchor[name] = global_state_dict[name].to(self.device, non_blocking=True).clone()
        return anchor

    def _train_local(
        self,
        optimizer: optim.Optimizer,
        epochs: int,
        global_anchor: dict[str, torch.Tensor] | None = None,
    ) -> float:
        """Run `epochs` local gradient steps. Returns mean batch loss."""
        self.model.train()
        total_loss, n_batches = 0.0, 0
        mu = float(self.cfg.fedprox_mu)

        for _ in range(epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=bool(self.cfg.amp) and self.device.type == "cuda"):
                    logits = self.model(images)
                loss = self.loss_fn(logits.float(), labels)
                if global_anchor is not None:
                    prox = torch.zeros((), device=self.device, dtype=loss.dtype)
                    for name, p in self.model.named_parameters():
                        if p.requires_grad:
                            d = p - global_anchor[name]
                            prox = prox + d.pow(2).sum()
                    loss = loss + 0.5 * mu * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _cosine_lr(self, server_round: int) -> float:
        """Cosine decay: lr * 0.5 * (1 + cos(pi * t / T))."""
        t = server_round - 1
        T = max(self.cfg.num_rounds - 1, 1)
        return self.cfg.lr * 0.5 * (1.0 + math.cos(math.pi * t / T))
