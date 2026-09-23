"""
Loss functions for SkinFLNet++.

- FocalLoss (gamma=2): for ISIC2019 with severe class imbalance (NV dominates).
- LabelSmoothingCrossEntropy: for ISBI2016 to match paper's cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Handles class imbalance by down-weighting easy examples.
    Used for ISIC2019 where NV class dominates.

    Args:
        gamma:     Focusing parameter (≥0). Higher = more focus on hard examples.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw model outputs (before softmax).
            targets: (B,)   integer class labels.

        Returns:
            Scalar focal loss.
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Gather log-prob and prob at the target class
        log_pt = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'


class CrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss (wrapper around nn.CrossEntropyLoss).
    Used for ISBI2016 to match the paper's setup exactly.

    Args:
        class_weights: Optional (C,) tensor of per-class weights.
        label_smoothing: Small smoothing factor (default 0.0 → pure CE).
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._ce(logits, targets)


def get_loss_fn(dataset: str) -> nn.Module:
    """
    Return the appropriate loss function for the dataset.

    - ISIC2019 / ISIC2018 → FocalLoss(gamma=2)
    - ISBI2016 → CrossEntropyLoss
    """
    dataset = dataset.lower()
    if dataset in ("isic2019", "isic2018"):
        return FocalLoss(gamma=2.0)
    elif dataset == "isbi2016":
        return CrossEntropyLoss()
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose 'isic2019', 'isic2018', or 'isbi2016'."
        )
