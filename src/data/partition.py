"""
Client partitioning schemes for federated learning.

Three schemes:
  - iid:        Stratified random split across N clients.
  - dirichlet:  Dirichlet(alpha) per-class non-IID split (patient-level for ISIC2019).
  - patient:    Group by lesion_id (patient) then Dirichlet — for ISIC2019 source split.

Guarantees enforced (verified by tests/test_partition_disjoint.py):
  - No image appears in more than one client's training set.
  - All images with the same lesion_id (patient) go to the same client.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _assert_disjoint(df: pd.DataFrame, split: str = "train") -> None:
    """Raise AssertionError if any image_id appears in two or more clients."""
    train_df = df[df["split"] == split]
    duplicates = train_df.groupby("image_id")["client_id"].nunique()
    leaking = duplicates[duplicates > 1]
    assert len(leaking) == 0, (
        f"Partition is NOT disjoint! {len(leaking)} images appear in "
        f"multiple clients: {leaking.index.tolist()[:10]}"
    )


def _assert_lesion_disjoint(df: pd.DataFrame, patient_col: str) -> None:
    """Raise AssertionError if any lesion is split across two clients."""
    tr = df[(df["split"] == "train") & df[patient_col].notna()]
    n = tr.groupby(patient_col)["client_id"].nunique()
    bad = n[n > 1]
    assert len(bad) == 0, f"{len(bad)} lesions span several clients: {bad.index.tolist()[:10]}"


def _assign_clients_from_map(
    df: pd.DataFrame, image_to_client: dict[str, int]
) -> pd.DataFrame:
    df = df.copy()
    df["client_id"] = df["image_id"].map(image_to_client)
    # Images not in train split get client_id = -1
    df["client_id"] = df["client_id"].fillna(-1).astype(int)
    return df


# ── IID Partition ─────────────────────────────────────────────────────────────


def iid_partition(
    df: pd.DataFrame,
    n_clients: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified IID partition: each client gets an equal share of every class.
    Applied only to the 'train' split rows.

    Args:
        df:        Manifest DataFrame with columns [image_id, label, split, ...].
        n_clients: Number of FL clients.
        seed:      Random seed for reproducibility.

    Returns:
        df with 'client_id' column filled for train rows; -1 for val/test.
    """
    rng = np.random.RandomState(seed)
    train_df = df[df["split"] == "train"].copy()

    image_to_client: dict[str, int] = {}

    for cls in train_df["label"].unique():
        cls_images = list(train_df[train_df["label"] == cls]["image_id"].values)
        rng.shuffle(cls_images)
        assignments = np.array_split(cls_images, n_clients)
        for client_id, imgs in enumerate(assignments):
            for img in imgs:
                image_to_client[img] = client_id

    df = _assign_clients_from_map(df, image_to_client)
    _assert_disjoint(df)

    logger.info(
        "IID partition: %d clients, %d train images", n_clients, len(train_df)
    )
    return df


# ── Dirichlet Partition ───────────────────────────────────────────────────────


def dirichlet_partition(
    df: pd.DataFrame,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    patient_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Non-IID Dirichlet(alpha) partition.

    If patient_col is provided (e.g., 'lesion_id'), partitioning is done at the
    patient level — all images from the same patient go to the same client.
    This prevents intra-patient data leakage (a known silent inflator of metrics).

    Args:
        df:          Manifest DataFrame.
        n_clients:   Number of FL clients.
        alpha:       Dirichlet concentration parameter.
                     Smaller → more skewed (non-IID). Default 0.5.
        seed:        Random seed.
        patient_col: Column name for patient grouping (None = image-level).

    Returns:
        df with 'client_id' column.
    """
    rng = np.random.RandomState(seed)
    train_df = df[df["split"] == "train"].copy()

    image_to_client: dict[str, int] = {}

    if patient_col and patient_col in train_df.columns:
        # Lesion-level partitioning: every view of a lesion goes to one client.
        # Rows without an identifier form singleton groups (one image = one lesion).
        key = train_df[patient_col].astype("object")
        key = key.where(key.notna() & (key.astype(str) != ""), "img:" + train_df["image_id"].astype(str))
        train_df = train_df.assign(_gkey=key.astype(str).values)
        group_images = train_df.groupby("_gkey")["image_id"].apply(list).to_dict()
        group_label = train_df.groupby("_gkey")["label"].agg(lambda x: x.mode()[0])

        for cls in sorted(train_df["label"].unique()):
            cls_groups = sorted(group_label[group_label == cls].index.tolist())
            rng.shuffle(cls_groups)
            proportions = rng.dirichlet(alpha * np.ones(n_clients))
            splits = _proportional_split(proportions, len(cls_groups))
            idx = 0
            for client_id, count in enumerate(splits):
                for gk in cls_groups[idx: idx + count]:
                    for img in group_images[gk]:
                        image_to_client[img] = client_id
                idx += count
    else:
        # Image-level partitioning (used for ISBI2016)
        for cls in train_df["label"].unique():
            cls_images = list(train_df[train_df["label"] == cls]["image_id"].values)
            rng.shuffle(cls_images)
            proportions = rng.dirichlet(alpha * np.ones(n_clients))
            splits = _proportional_split(proportions, len(cls_images))

            idx = 0
            for client_id, count in enumerate(splits):
                for img in cls_images[idx: idx + count]:
                    image_to_client[img] = client_id
                idx += count

    df = _assign_clients_from_map(df, image_to_client)
    _assert_disjoint(df)
    if patient_col and patient_col in df.columns:
        _assert_lesion_disjoint(df, patient_col)

    client_counts = {
        cid: (df[(df["split"] == "train") & (df["client_id"] == cid)]).shape[0]
        for cid in range(n_clients)
    }
    logger.info(
        "Dirichlet(alpha=%.2f) partition: %d clients, sizes=%s",
        alpha, n_clients, client_counts,
    )
    return df


def _proportional_split(proportions: np.ndarray, total: int) -> np.ndarray:
    """Convert Dirichlet proportions to integer counts summing to total."""
    counts = (proportions * total).astype(int)
    # Distribute remainder to largest proportions
    remainder = total - counts.sum()
    if remainder > 0:
        top_k = np.argsort(proportions)[::-1][:remainder]
        counts[top_k] += 1
    return counts


# ── Patient / Source-Based Partition ─────────────────────────────────────────


def patient_partition(
    df: pd.DataFrame,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    patient_col: str = "lesion_id",
) -> pd.DataFrame:
    """
    Source-based partition using lesion_id as patient proxy.

    Each unique lesion_id is assigned to exactly one client. The assignment
    uses Dirichlet(alpha) to create realistic heterogeneity. Images without
    a lesion_id fall back to image-level Dirichlet.

    This replaces the 'source-based' scheme from the spec since the
    ISIC2019 metadata does not contain source institution information.
    See docs/02_rules.md §Design Decisions.
    """
    return dirichlet_partition(
        df=df,
        n_clients=n_clients,
        alpha=alpha,
        seed=seed,
        patient_col=patient_col,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def partition(
    df: pd.DataFrame,
    scheme: str,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    patient_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified entry point for all partitioning schemes.

    Args:
        df:          Manifest DataFrame with at minimum [image_id, label, split].
        scheme:      'iid' | 'dirichlet' | 'patient'
        n_clients:   Number of FL clients.
        alpha:       Dirichlet α (only used for 'dirichlet' and 'patient').
        seed:        Random seed.
        patient_col: Column name for patient-level grouping.

    Returns:
        df with 'client_id' column populated for train rows.
    """
    scheme = scheme.lower()
    if scheme == "iid":
        return iid_partition(df, n_clients=n_clients, seed=seed)
    elif scheme == "dirichlet":
        return dirichlet_partition(
            df, n_clients=n_clients, alpha=alpha, seed=seed, patient_col=patient_col
        )
    elif scheme in ("patient", "source"):
        return patient_partition(
            df, n_clients=n_clients, alpha=alpha, seed=seed,
            patient_col=patient_col or "lesion_id",
        )
    else:
        raise ValueError(
            f"Unknown partition scheme '{scheme}'. "
            "Choose from: 'iid', 'dirichlet', 'patient'."
        )
