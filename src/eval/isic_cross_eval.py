"""
Cross-dataset evaluation: ISIC2019 global model on ISIC2018 test, and ISIC2018
global model on ISIC2019 test with SCC (label 7) removed.

Integer labels 0–6 are aligned across datasets for the shared seven classes
(see ``SHARED_SEVEN_CLASS_NAMES``). Index 3 is AK on ISIC2019 and AKIEC on
ISIC2018 — same slot in this codebase's encoding.

For 2019 → 2018, the eighth logit (SCC) is dropped at inference so metrics are
7-way and comparable to native ISIC2018 evaluation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.data.datasets import SkinDataset, dataloader_rng_seed, make_dataloader
from src.data.transforms import get_val_transforms
from src.eval.metrics import evaluate_model, predict_proba
from src.fl.config import dataset_folder_for
from src.models.build import build_model

logger = logging.getLogger(__name__)

# Indices 0–6: same ordering as ISIC2018 and the first seven of ISIC2019
# (ISIC2019 class 3 is "AK", ISIC2018 class 3 is "AKIEC").
SHARED_SEVEN_CLASS_NAMES: tuple[str, ...] = (
    "MEL",
    "NV",
    "BCC",
    "AK_or_AKIEC",
    "BKL",
    "DF",
    "VASC",
)

_ROUND_WEIGHTS_RE = re.compile(r"^model_round_(\d+)\.pth$")


class LogitSliceWrapper(nn.Module):
    """Wrap a classifier and return only the first ``n`` logits (inference-only)."""

    def __init__(self, inner: nn.Module, n: int) -> None:
        super().__init__()
        self.inner = inner
        self.n = int(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)[:, : self.n]


def load_global_state_dict(checkpoint_path: Path | str) -> dict[str, torch.Tensor]:
    """
    Load ``global_state_dict`` from a federated checkpoint file or run directory.

    Accepts:
    - ``checkpoint_latest.pt`` (payload with ``global_state_dict``)
    - ``model_round_NNN.pth`` (payload with ``global_state_dict``)
    - A raw state dict mapping (only if keys look like ``SkinFLNet``)
    """
    path = Path(checkpoint_path)
    if path.is_dir():
        path = resolve_checkpoint_file_in_run_dir(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint type at {path}: {type(payload).__name__}")

    if "global_state_dict" in payload:
        sd = payload["global_state_dict"]
    elif _looks_like_skinfl_state_dict(payload):
        sd = payload  # type: ignore[assignment]
    else:
        raise ValueError(
            f"Checkpoint at {path} has no 'global_state_dict' and does not look like a model state_dict."
        )

    if not isinstance(sd, dict) or not sd:
        raise ValueError(f"Empty or invalid global_state_dict in {path}")

    if not _looks_like_skinfl_state_dict(sd):
        raise ValueError(
            f"Loaded tensors at {path} do not look like SkinFLNet (expected backbone.* and head.* keys)."
        )

    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def resolve_checkpoint_file_in_run_dir(run_dir: Path) -> Path:
    """Prefer ``checkpoint_latest.pt``; else the highest ``model_round_NNN.pth``."""
    run_dir = Path(run_dir)
    latest = run_dir / "checkpoint_latest.pt"
    if latest.is_file():
        return latest

    best: tuple[int, Path] | None = None
    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        m = _ROUND_WEIGHTS_RE.match(p.name)
        if m:
            idx = int(m.group(1))
            if best is None or idx > best[0]:
                best = (idx, p)
    if best is None:
        raise FileNotFoundError(
            f"No checkpoint_latest.pt or model_round_*.pth under {run_dir}"
        )
    logger.info("Using round weights archive: %s", best[1])
    return best[1]


def _looks_like_skinfl_state_dict(d: dict[Any, Any]) -> bool:
    keys = list(d.keys())
    if not keys:
        return False
    has_bb = any(isinstance(k, str) and k.startswith("backbone.") for k in keys)
    has_head = any(isinstance(k, str) and k.startswith("head.") for k in keys)
    return has_bb and has_head


def infer_num_classes_from_state_dict(sd: dict[str, torch.Tensor]) -> int:
    """Read output dimension from the final Linear in ``SkinFLNet.head``."""
    # VGG-style head: Linear(128, C) is module index 6
    w = sd.get("head.6.weight")
    if w is not None and w.dim() == 2:
        return int(w.shape[0])
    # Non-VGG head: Linear(128, C) is module index 4
    w = sd.get("head.4.weight")
    if w is not None and w.dim() == 2:
        return int(w.shape[0])
    # Fallback: last head.*.weight with in_features 128
    candidates: list[tuple[str, torch.Tensor]] = []
    for k, v in sd.items():
        if not isinstance(k, str) or not k.startswith("head.") or not k.endswith(".weight"):
            continue
        if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[1] == 128:
            candidates.append((k, v))
    if not candidates:
        raise ValueError("Could not infer num_classes from state_dict (no head.* Linear with fan-in 128).")
    # Prefer largest module index in key
    def key_index(name: str) -> int:
        parts = name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return -1

    candidates.sort(key=lambda kv: key_index(kv[0]))
    return int(candidates[-1][1].shape[0])


def assert_backbone_style_matches_checkpoint(backbone: str, sd: dict[str, torch.Tensor]) -> None:
    """Lightweight sanity check that the checkpoint matches the declared backbone family."""
    bb = backbone.lower()
    has_vgg_features = any(k.startswith("backbone.features.") for k in sd)
    if bb.startswith("vgg"):
        if not has_vgg_features:
            raise ValueError(
                f"Backbone is {backbone!r} but checkpoint has no 'backbone.features.*' keys "
                "(expected VGG conv trunk checkpoint)."
            )
    elif has_vgg_features and not bb.startswith("vgg"):
        raise ValueError(
            f"Backbone is {backbone!r} but checkpoint looks like a VGG-style trunk (backbone.features.* present)."
        )


def build_model_and_load(
    backbone: str,
    num_classes: int,
    state_dict: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
) -> nn.Module:
    """Instantiate ``SkinFLNet`` (no ImageNet init for weights we overwrite) and load ``state_dict``."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert_backbone_style_matches_checkpoint(backbone, state_dict)
    inferred = infer_num_classes_from_state_dict(state_dict)
    if inferred != num_classes:
        raise ValueError(
            f"Checkpoint has num_classes={inferred} (from head weights) but caller passed num_classes={num_classes}."
        )
    model = build_model(backbone, num_classes=num_classes, pretrained=False, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected load_state_dict result: missing={missing}, unexpected={unexpected}")
    return model


def _exclude_seen(df_eval, other_manifest: Path, other_splits: tuple[str, ...]):
    """Drop evaluation rows whose image or lesion occurs in the other release's data.

    ISIC 2019 training data contain the HAM10000 (ISIC 2018 Task 3 training) images, so
    without this filter a model could be scored on images or lesions it was trained on.
    """
    import pandas as pd

    other = pd.read_csv(other_manifest)
    other = other[other["split"].isin(other_splits)]
    seen_img = set(other["image_id"].astype(str))
    seen_les = set(other["lesion_key"].astype(str)) if "lesion_key" in other.columns else set()
    seen_les = {k for k in seen_les if not k.startswith("img:")}
    m_img = df_eval["image_id"].astype(str).isin(seen_img)
    m_les = df_eval["lesion_key"].astype(str).isin(seen_les) if "lesion_key" in df_eval.columns else m_img & False
    drop = m_img | m_les
    return df_eval[~drop].reset_index(drop=True), {
        "n_dropped_seen_image": int(m_img.sum()),
        "n_dropped_seen_lesion_only": int((m_les & ~m_img).sum()),
    }


def eval_2019_model_on_2018_test(
    *,
    data_root: Path | str,
    checkpoint_path: Path | str,
    backbone: str = "vgg16_bn",
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load an ISIC2019 (8-class) federated global checkpoint and evaluate on ISIC2018 ``split=test``.

    Uses ``LogitSliceWrapper`` so only logits ``[:, :7]`` are scored (7-way metrics).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(data_root)
    sd = load_global_state_dict(checkpoint_path)
    assert_backbone_style_matches_checkpoint(backbone, sd)
    if infer_num_classes_from_state_dict(sd) != 8:
        raise ValueError(
            "eval_2019_model_on_2018_test expects an 8-class ISIC2019 checkpoint; "
            f"got {infer_num_classes_from_state_dict(sd)} classes."
        )

    inner = build_model_and_load(backbone, 8, sd, device=device)
    model = LogitSliceWrapper(inner, 7).to(device)

    manifest = data_root / dataset_folder_for("isic2018") / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"ISIC2018 manifest missing: {manifest}")

    df_full = SkinDataset.load_filtered_manifest_df(manifest, split="test", client_id=None)
    df, excl = _exclude_seen(
        df_full, data_root / dataset_folder_for("isic2019") / "manifest.csv", ("train", "val")
    )
    ds = SkinDataset.from_dataframe(manifest, df, transform=get_val_transforms(img_size))
    loader = make_dataloader(
        ds,
        batch_size=batch_size,
        weighted_sampling=False,
        shuffle=False,
        num_workers=num_workers,
        rng_seed=dataloader_rng_seed(seed, None, slot=9101),
    )

    metrics = evaluate_model(model, loader, device, num_classes=7, amp=True)
    meta = {
        "direction": "isic2019_on_isic2018_test",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "manifest": str(manifest.resolve()),
        "n_test": len(ds),
        "n_test_before_filter": len(df_full),
        **excl,
        "logit_slice": "[:, :7]",
        "shared_class_names": SHARED_SEVEN_CLASS_NAMES,
    }
    probs, labels = predict_proba(model, loader, device, amp=True)
    return {"metrics": metrics, "meta": meta, "probs": probs, "labels": labels,
            "image_id": ds.df["image_id"].astype(str).tolist()}


def eval_2018_model_on_2019_test_no_scc(
    *,
    data_root: Path | str,
    checkpoint_path: Path | str,
    backbone: str = "vgg16_bn",
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load an ISIC2018 (7-class) federated global checkpoint and evaluate on ISIC2019 ``split=test``
    with all rows ``label == 7`` (SCC) removed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(data_root)
    sd = load_global_state_dict(checkpoint_path)
    assert_backbone_style_matches_checkpoint(backbone, sd)
    if infer_num_classes_from_state_dict(sd) != 7:
        raise ValueError(
            "eval_2018_model_on_2019_test_no_scc expects a 7-class ISIC2018 checkpoint; "
            f"got {infer_num_classes_from_state_dict(sd)} classes."
        )

    model = build_model_and_load(backbone, 7, sd, device=device)

    manifest = data_root / dataset_folder_for("isic2019") / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"ISIC2019 manifest missing: {manifest}")

    df_full = SkinDataset.load_filtered_manifest_df(manifest, split="test", client_id=None)
    n_before = len(df_full)
    df = df_full[df_full["label"].astype(int) != 7].reset_index(drop=True)
    n_scc_dropped = n_before - len(df)
    df, excl = _exclude_seen(
        df, data_root / dataset_folder_for("isic2018") / "manifest.csv", ("train", "val", "test")
    )
    if len(df) == 0:
        raise RuntimeError("After removing SCC (label 7), no ISIC2019 test rows remain.")

    ds = SkinDataset.from_dataframe(manifest, df, transform=get_val_transforms(img_size))
    loader = make_dataloader(
        ds,
        batch_size=batch_size,
        weighted_sampling=False,
        shuffle=False,
        num_workers=num_workers,
        rng_seed=dataloader_rng_seed(seed, None, slot=9102),
    )

    metrics = evaluate_model(model, loader, device, num_classes=7, amp=True)
    meta = {
        "direction": "isic2018_on_isic2019_test_no_scc",
        **excl,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "manifest": str(manifest.resolve()),
        "n_test": len(ds),
        "n_test_before_filter": n_before,
        "n_scc_dropped": n_scc_dropped,
        "shared_class_names": SHARED_SEVEN_CLASS_NAMES,
    }
    probs, labels = predict_proba(model, loader, device, amp=True)
    return {"metrics": metrics, "meta": meta, "probs": probs, "labels": labels,
            "image_id": ds.df["image_id"].astype(str).tolist()}
