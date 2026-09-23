"""
PyTorch Dataset classes for ISIC2019, ISIC2018, and ISBI2016.

Both datasets read from a pre-built manifest CSV that contains:
  image_id, path, label (integer), split, client_id

Label encoding:
  ISBI2016:  benign=0, malignant=1
  ISIC2018:  MEL=0, NV=1, BCC=2, AKIEC=3, BKL=4, DF=5, VASC=6
  ISIC2019:  MEL=0, NV=1, BCC=2, AK=3, BKL=4, DF=5, VASC=6, SCC=7
"""

from __future__ import annotations

import functools
import logging
import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ── ISIC2019 ─────────────────────────────────────────────────────────────────

ISIC2019_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# ── ISIC2018 ─────────────────────────────────────────────────────────────────

ISIC2018_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# ── ISBI2016 ─────────────────────────────────────────────────────────────────

ISBI2016_LABEL_MAP = {"benign": 0, "malignant": 1}
ISBI2016_CLASSES = ["benign", "malignant"]

logger = logging.getLogger(__name__)


def dataloader_rng_seed(global_seed: int, partition_id: Optional[int], slot: int) -> int:
    """
    Stable integer seed for DataLoader / sampler RNG (does not depend on hash randomization).

    ``partition_id`` None denotes global loaders (server test, centralized pooled train).
    ``slot`` distinguishes loaders for the same client (train vs local test, etc.).
    """
    pid = 0 if partition_id is None else int(partition_id)
    return int(global_seed) + pid * 100_003 + int(slot) * 17_011


def _dataloader_worker_init_fn(worker_id: int, base_seed: int) -> None:
    wseed = int(base_seed) + int(worker_id)
    random.seed(wseed)
    np.random.seed(wseed % (2**32))
    torch.manual_seed(wseed)


# Case / format variants seen after unzip or cross-platform copies (Linux is case-sensitive).
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".BMP")


def _locate_image_file(expected: Path) -> Path | None:
    """Return a path that exists, trying alternate extensions next to ``expected``."""
    if expected.is_file():
        return expected
    parent = expected.parent
    stem = expected.stem
    for ext in _IMAGE_EXTS:
        cand = parent / f"{stem}{ext}"
        if cand.is_file():
            return cand
    matches = [p for p in parent.glob(f"{stem}.*") if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_image_path(expected: Path) -> Path:
    found = _locate_image_file(expected)
    if found is not None:
        return found
    raise FileNotFoundError(
        f"Image not found: {expected} (also tried other extensions under {expected.parent}). "
        "Upload the full ISIC/ISBI image folders to match the manifest, or set environment variable "
        "SKINFL_DROP_MISSING_IMAGES=1 to skip rows whose files are missing."
    )


class SkinDataset(Dataset):
    """
    Generic skin-lesion dataset backed by a manifest CSV.

    Args:
        manifest_path: Path to manifest CSV with columns:
                       image_id, path, label, split, client_id
        split:         One of 'train', 'val', 'test', or None (use all rows).
        client_id:     If given, only load images for this FL client.
        transform:     Optional callable applied to each PIL image.

    Paths in the CSV are overwritten at load time for ISIC2019/ISIC2018/ISBI2016 using
    ``manifest_path``'s parent folder, so manifests stay portable across machines
    (e.g. Windows paths in the CSV do not break Linux/Colab).
    """

    @staticmethod
    def _resolve_paths(df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
        """Rebuild ``path`` from ``image_id`` + dataset layout next to the manifest."""
        df = df.copy()
        root = manifest_path.resolve().parent
        if root.name == "partitions":  # per-run manifests live in <DS>/partitions/
            root = root.parent
        folder = root.name

        if folder == "ISIC2019":
            img_dir = root / "ISIC_2019_Training_Input"
            df["path"] = df["image_id"].astype(str).map(lambda iid: str(img_dir / f"{iid}.jpg"))
        elif folder == "ISIC2018":
            train_dir = root / "ISIC2018_Task3_Training_Input"
            val_dir = root / "ISIC2018_Task3_Validation_Input"
            test_dir = root / "ISIC2018_Task3_Test_Input"

            def row_path_2018(r) -> str:
                if r["split"] == "test":
                    base = test_dir
                elif r["split"] == "val":
                    base = val_dir
                else:
                    base = train_dir
                return str(base / f'{r["image_id"]}.jpg')

            df["path"] = df.apply(row_path_2018, axis=1)
        elif folder == "ISBI2016":
            train_dir = root / "ISBI2016_ISIC_Training_Data"
            test_dir = root / "ISBI2016_ISIC_Test_Data"

            def row_path(r) -> str:
                base = test_dir if r["split"] == "test" else train_dir
                return str(base / f'{r["image_id"]}.jpg')

            df["path"] = df.apply(row_path, axis=1)
        # else: keep ``path`` column from CSV

        return df

    @staticmethod
    def load_filtered_manifest_df(
        manifest_path: str | Path,
        split: Optional[str] = None,
        client_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load manifest CSV with path resolution, optional split/client filters,
        and optional SKINFL_DROP_MISSING_IMAGES row dropping.

        Shared by SkinDataset.__init__ and FL client local train/test splitting.
        """
        manifest_path = Path(manifest_path)
        df = pd.read_csv(manifest_path)
        df = SkinDataset._resolve_paths(df, manifest_path)

        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)

        if client_id is not None:
            df = df[df["client_id"] == client_id].reset_index(drop=True)

        drop_missing = os.environ.get("SKINFL_DROP_MISSING_IMAGES", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if drop_missing and len(df) > 0:
            before = len(df)
            mask = df["path"].map(lambda p: _locate_image_file(Path(p)) is not None)
            df = df[mask].reset_index(drop=True)
            dropped = before - len(df)
            if dropped:
                logger.warning(
                    "SKINFL_DROP_MISSING_IMAGES: dropped %d / %d rows with no file on disk",
                    dropped,
                    before,
                )
            if len(df) == 0:
                raise RuntimeError(
                    "After SKINFL_DROP_MISSING_IMAGES, no rows left — check data paths on Drive."
                )

        return df.reset_index(drop=True)

    @classmethod
    def from_dataframe(
        cls,
        manifest_path: str | Path,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
    ) -> SkinDataset:
        """Build a dataset from an already-filtered dataframe (columns include ``path``, ``label``)."""
        need = {"path", "label"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"from_dataframe missing required columns: {sorted(missing)}")
        obj = cls.__new__(cls)
        obj.manifest_path = Path(manifest_path)
        obj.transform = transform
        obj.df = df.reset_index(drop=True).copy()
        obj.labels = obj.df["label"].values.astype(int)
        obj._path_resolve_cache = {}
        return obj

    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        client_id: Optional[int] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        df = self.load_filtered_manifest_df(self.manifest_path, split=split, client_id=client_id)

        self.df = df
        self.labels = df["label"].values.astype(int)
        self._path_resolve_cache: dict[int, Path] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if idx not in self._path_resolve_cache:
            self._path_resolve_cache[idx] = _resolve_image_path(Path(row["path"]))
        img_path = self._path_resolve_cache[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        return image, label

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1

    def get_class_counts(self) -> list[int]:
        """Return per-class sample counts for WeightedRandomSampler."""
        counts = []
        for c in range(self.num_classes):
            counts.append(int((self.labels == c).sum()))
        return counts


def make_weighted_sampler(
    dataset: SkinDataset,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that up-weights minority classes.
    This replaces SMOTE (which is invalid on raw images).
    """
    class_counts = dataset.get_class_counts()
    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = [class_weights[label] for label in dataset.labels]
    kwargs: dict = dict(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
    if generator is not None:
        kwargs["generator"] = generator
    return WeightedRandomSampler(**kwargs)


def make_dataloader(
    dataset: SkinDataset,
    batch_size: int = 32,
    weighted_sampling: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    rng_seed: Optional[int] = None,
    persistent_workers: bool = False,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Create a DataLoader with optional WeightedRandomSampler.

    Pass ``rng_seed`` for reproducible shuffling / weighted sampling and worker RNG
    (recommended whenever ``cfg.seed`` is fixed).
    """
    generator: Optional[torch.Generator] = None
    worker_init_fn = None
    if rng_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(rng_seed))
        if num_workers > 0:
            worker_init_fn = functools.partial(
                _dataloader_worker_init_fn, base_seed=int(rng_seed)
            )

    sampler = None
    if weighted_sampling:
        sampler = make_weighted_sampler(dataset, generator=generator)
        shuffle = False
        dl_generator = None
    else:
        # Evaluation loaders must keep manifest order (predictions are archived
        # against image_id); callers pass shuffle=False for val/test.
        shuffle = (len(dataset) > 0) if shuffle is None else bool(shuffle)
        dl_generator = generator

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        generator=dl_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0 and persistent_workers),
        worker_init_fn=worker_init_fn,
    )
