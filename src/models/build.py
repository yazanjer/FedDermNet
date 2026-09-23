"""
Model builder for SkinFLNet++.

Loads a timm backbone (ImageNet-pretrained) and attaches the paper's
classification head:
    GlobalAvgPool → Dropout(0.2) → Linear(→128) → ReLU → Dropout(0.4) → Linear(→C)

For timm VGG variants (vgg11/13/16/19, with or without _bn), only the convolutional
trunk is kept (~7×7×512 at 224×224 input), then the head above with pool-flatten to 512.
Other timm models use their built-in pooling and feature dimension as before.

Older checkpoints trained with the full timm VGG (incl. pre_logits/ConvMlp) are not
load-compatible with the slim VGG trunk.

During training, ``freeze_backbone()`` is applied every FL round (and every centralized
epoch): only the input-near half of backbone stages is frozen; the deeper half trains
with the head.

Supported backbones (paper + modern):
  Paper:   vgg16, vgg16_bn, densenet169, inception_v3, xception, inception_resnet_v2
  Modern:  efficientnet_b3, convnext_tiny, vit_small_patch16_224
"""

from __future__ import annotations

import logging
from typing import Optional, cast

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Backbones that require a different input size
_INCEPTION_BACKBONES = {"inception_v3", "inception_resnet_v2"}

_VGG_PREFIXES = ("vgg11", "vgg13", "vgg16", "vgg19")

# Map timm backbone names to their feature dimension (auto-detected at build time)
_BACKBONE_ALIASES = {
    "vgg16": "vgg16",
    "vgg16_bn": "vgg16_bn",
    "densenet169": "densenet169",
    "inception_v3": "inception_v3",
    "xception": "xception",
    "inception_resnet_v2": "inception_resnet_v2",
    "efficientnet_b3": "efficientnet_b3",
    "convnext_tiny": "convnext_tiny",
    "vit_small_patch16_224": "vit_small_patch16_224",
}


def _is_timm_vgg(timm_name: str) -> bool:
    return any(timm_name.startswith(prefix) for prefix in _VGG_PREFIXES)


class VGGFeaturesBackbone(nn.Module):
    """ImageNet-pretrained VGG convolutional trunk only (no timm pre_logits / classifier)."""

    def __init__(self, timm_name: str, pretrained: bool) -> None:
        super().__init__()
        full = timm.create_model(timm_name, pretrained=pretrained, num_classes=1000)
        self.features = full.features
        del full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SkinFLNet(nn.Module):
    """
    timm backbone + paper's classification head.

    The backbone is accessed via self.backbone.
    The head is self.head.
    Training: freeze_backbone() each round — input-near half of the backbone stays frozen;
    unfreeze_backbone() is available for special cases (e.g. tests).
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()

        timm_name = _BACKBONE_ALIASES.get(backbone_name, backbone_name)

        if _is_timm_vgg(timm_name):
            self.backbone = VGGFeaturesBackbone(timm_name=timm_name, pretrained=pretrained)
            feat_dim = 512
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Dropout(p=0.2),
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(128, num_classes),
            )
            logger.debug(
                "Backbone '%s' → VGG conv trunk + GAP head (feat_dim=%d)", backbone_name, feat_dim
            )
        else:
            self.backbone = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feat_dim = self.backbone(dummy).shape[1]

            logger.debug("Backbone '%s' → feature dim %d", backbone_name, feat_dim)

            self.head = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(128, num_classes),
            )

        self.num_classes = num_classes
        self.backbone_name = backbone_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# ── Partial freeze (input-near stages) ─────────────────────────────────────────


def _stage_children(seq: nn.Module) -> list[nn.Module]:
    """Top-level modules inside a Sequential or ModuleList."""
    if isinstance(seq, (nn.Sequential, nn.ModuleList)):
        return list(seq.children())
    return []


def _module_has_params(mod: nn.Module) -> bool:
    return any(True for _ in mod.parameters())


def _is_vit_style(backbone: nn.Module) -> bool:
    return hasattr(backbone, "patch_embed") and hasattr(backbone, "blocks")


def _set_vit_root_params_requires_grad(backbone: nn.Module, requires_grad: bool) -> None:
    """cls_token / pos_embed / reg_token live on the ViT root, not inside patch_embed."""
    if not _is_vit_style(backbone):
        return
    for name in ("cls_token", "reg_token", "pos_embed"):
        p = getattr(backbone, name, None)
        if isinstance(p, nn.Parameter):
            p.requires_grad = requires_grad


def _ordered_backbone_stages(backbone: nn.Module) -> list[nn.Module]:
    """Ordered stage submodules from input-side toward the dense head (forward order)."""
    if isinstance(backbone, VGGFeaturesBackbone):
        stages = _stage_children(backbone.features)
        if stages:
            return stages

    if _is_vit_style(backbone):
        stages = [cast(nn.Module, backbone.patch_embed)]
        stages.extend(_stage_children(cast(nn.Module, backbone.blocks)))
        norm = getattr(backbone, "norm", None)
        if norm is not None and not isinstance(norm, nn.Identity):
            stages.append(cast(nn.Module, norm))
        if stages:
            return stages

    feat = getattr(backbone, "features", None)
    if feat is not None and isinstance(feat, nn.Sequential):
        stages = _stage_children(feat)
        if stages:
            return stages

    for attr in ("blocks", "stages"):
        b = getattr(backbone, attr, None)
        if b is not None and isinstance(b, (nn.Sequential, nn.ModuleList)):
            stages = _stage_children(cast(nn.Module, b))
            if stages:
                return stages

    stages = [c for c in backbone.children() if _module_has_params(c)]
    if stages:
        return stages

    return [backbone]


def freeze_backbone(model: SkinFLNet) -> None:
    """Freeze the input-near half of backbone stages; train the deeper half and the head.

    First ceil(N/2) stages (see `_ordered_backbone_stages`) have ``requires_grad=False``;
    remaining stages are trainable. The classification head is always fully trainable.
    """
    for param in model.head.parameters():
        param.requires_grad = True

    backbone = model.backbone
    stages = _ordered_backbone_stages(backbone)
    mid = (len(stages) + 1) // 2
    for i, stage in enumerate(stages):
        train = i >= mid
        for p in stage.parameters():
            p.requires_grad = train

    # ViT: tokens / positional embed are not inside patch_embed.
    if _is_vit_style(backbone):
        _set_vit_root_params_requires_grad(backbone, requires_grad=(mid < 1))

    trainable_bb = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total_bb = sum(p.numel() for p in backbone.parameters())
    logger.debug(
        "Partial backbone freeze: %d stages, first %d frozen | backbone params %s / %s trainable",
        len(stages), mid, f"{trainable_bb:,}", f"{total_bb:,}",
    )


def unfreeze_backbone(model: SkinFLNet) -> None:
    """Unfreeze all backbone parameters (and head) for full fine-tuning."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
    _set_vit_root_params_requires_grad(model.backbone, requires_grad=True)
    logger.debug("Backbone unfrozen.")


# ── Builder ───────────────────────────────────────────────────────────────────


def build_model(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> SkinFLNet:
    """
    Instantiate SkinFLNet on the target device.

    Args:
        backbone_name: One of the supported timm backbone names.
        num_classes:   Output classes (2 for ISBI2016, 8 for ISIC2019).
        pretrained:    Load ImageNet-pretrained weights.
        device:        Target device. Defaults to CUDA if available, else CPU.

    Returns:
        SkinFLNet model on the given device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import os

    if os.environ.get("SKINFL_NO_PRETRAINED", "").strip() in ("1", "true", "yes"):
        pretrained = False  # offline smoke tests only
    model = SkinFLNet(backbone_name=backbone_name, num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %s | Classes: %d | Params: %s total (%s trainable) | Device: %s",
        backbone_name, num_classes,
        f"{total:,}", f"{trainable:,}", device,
    )
    return model


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("No GPU found — running on CPU.")
    return device
