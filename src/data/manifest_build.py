"""
Build manifest CSVs for SkinFLNet++ from raw dataset files.

Partitioning (client_id) is applied per-experiment by src/data/partition.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

ISIC2019_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
ISIC2018_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
ISBI2016_LABEL_MAP = {"benign": 0, "malignant": 1}

DatasetName = Literal["isic2019", "isic2018", "isbi2016"]


# ── Lesion-level helpers (revision R1: lesion-disjoint splits) ───────────────


def _load_api_lesion_map(folder: Path) -> dict[str, str]:
    """Optional ISIC-archive lesion IDs (``lesion_ids_isic_api.csv``: image_id,lesion_id).

    Written by ``scripts/fetch_isic_lesion_ids.py``. Used only to fill rows whose
    challenge metadata carries no lesion identifier.
    """
    path = folder / "lesion_ids_isic_api.csv"
    if not path.is_file():
        return {}
    m = pd.read_csv(path).dropna(subset=["lesion_id"])
    return dict(zip(m["image_id"].astype(str), m["lesion_id"].astype(str)))


def attach_lesion_key(df: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """Add ``lesion_key``: the lesion identifier, else the image itself (singleton group).

    ``lesion_source`` records where the key came from (challenge metadata, ISIC API,
    or singleton fallback) so the split audit can report coverage.
    """
    df = df.copy()
    if "lesion_id" not in df.columns:
        df["lesion_id"] = None
    meta = df["lesion_id"].astype("object")
    meta = meta.where(meta.notna() & (meta.astype(str).str.strip() != ""), None)
    api = _load_api_lesion_map(folder)
    api_ids = df["image_id"].astype(str).map(api) if api else pd.Series([None] * len(df), index=df.index)
    # Prefer the ISIC-archive identifier when available (one namespace across splits and
    # releases); fall back to the challenge metadata, then to a singleton group.
    lid = api_ids.where(api_ids.notna(), meta)
    source = pd.Series(np.where(api_ids.notna(), "isic_api", np.where(meta.notna(), "metadata", "")),
                       index=df.index, dtype="object")
    single = lid.isna()
    source[single] = "singleton"
    df["lesion_key"] = lid.where(~single, "img:" + df["image_id"].astype(str))
    df["lesion_source"] = source
    return df


def lesion_disjoint_split(
    df: pd.DataFrame, val_frac: float = 0.10, test_frac: float = 0.10, seed: int = 42
) -> pd.DataFrame:
    """Stratified train/val/test split at the LESION level.

    Unique lesions (``lesion_key``) are assigned to exactly one split, so no view of a
    lesion can appear in two splits. Stratification uses each lesion's majority label;
    ``StratifiedGroupKFold`` keeps image-level class proportions close to the targets.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_folds = int(round(1.0 / min(val_frac, test_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1)
    y = df["label"].values
    g = df["lesion_key"].astype(str).values
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(df)), y, g)):
        folds[idx] = k
    n_test = int(round(test_frac * n_folds))
    n_val = int(round(val_frac * n_folds))
    split = np.where(folds < n_test, "test", np.where(folds < n_test + n_val, "val", "train"))
    df = df.copy()
    df["split"] = split
    return df


def split_audit(df: pd.DataFrame, class_names: list[str]) -> dict:
    """Images, unique lesions and pairwise lesion/image intersections per split."""
    out: dict = {"n_images": int(len(df))}
    keys = {}
    for sp in ("train", "val", "test"):
        part = df[df["split"] == sp]
        keys[sp] = set(part["lesion_key"].astype(str))
        out[sp] = {
            "images": int(len(part)),
            "unique_lesions": int(part["lesion_key"].nunique()),
            "images_with_lesion_id": int((part.get("lesion_source", pd.Series(dtype=object)) != "singleton").sum()),
            "per_class_images": {class_names[int(c)]: int(n) for c, n in part["label"].value_counts().sort_index().items()},
            "per_class_lesions": {
                class_names[int(c)]: int(n)
                for c, n in part.groupby("label")["lesion_key"].nunique().sort_index().items()
            },
        }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        out[f"lesion_overlap_{a}_{b}"] = int(len(keys[a] & keys[b]))
        ia = set(df.loc[df["split"] == a, "image_id"]); ib = set(df.loc[df["split"] == b, "image_id"])
        out[f"image_overlap_{a}_{b}"] = int(len(ia & ib))
    if "lesion_source" in df.columns:
        out["lesion_key_source_counts"] = {str(k): int(v) for k, v in df["lesion_source"].value_counts().items()}
    return out


def write_split_audit(df: pd.DataFrame, class_names: list[str], out_path: Path) -> dict:
    import json

    audit = split_audit(df, class_names)
    out_path.write_text(json.dumps(audit, indent=2))
    logger.info(
        "Split audit -> %s | lesion overlaps train/val=%d train/test=%d val/test=%d",
        out_path, audit["lesion_overlap_train_val"], audit["lesion_overlap_train_test"],
        audit["lesion_overlap_val_test"],
    )
    return audit


def _require_inputs(name: str, paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"{name}: missing path(s) under data root (see README / RUN_EXPERIMENTS.md):\n{lines}"
        )


def isic2019_inputs(data_root: Path) -> list[Path]:
    return [
        data_root / "ISIC2019" / "ISIC_2019_Training_GroundTruth.csv",
        data_root / "ISIC2019" / "ISIC_2019_Training_Metadata.csv",
        data_root / "ISIC2019" / "ISIC_2019_Training_Input",
    ]


def isic2018_inputs(data_root: Path) -> list[Path]:
    base = data_root / "ISIC2018"
    return [
        base / "ISIC2018_Task3_Training_GroundTruth.csv",
        base / "ISIC2018_Task3_Validation_GroundTruth.csv",
        base / "ISIC2018_Task3_Test_GroundTruth.csv",
        base / "ISIC2018_Task3_Training_LesionGroupings.csv",
        base / "ISIC2018_Task3_Training_Input",
        base / "ISIC2018_Task3_Validation_Input",
        base / "ISIC2018_Task3_Test_Input",
    ]


def isbi2016_inputs(data_root: Path) -> list[Path]:
    return [
        data_root / "ISBI2016" / "ISBI2016_ISIC_Training_GroundTruth.csv",
        data_root / "ISBI2016" / "ISBI2016_ISIC_Test_GroundTruth.csv",
        data_root / "ISBI2016" / "ISBI2016_ISIC_Training_Data",
        data_root / "ISBI2016" / "ISBI2016_ISIC_Test_Data",
    ]


def build_isic2019_manifest(data_root: Path, seed: int = 42) -> pd.DataFrame:
    gt_path = data_root / "ISIC2019" / "ISIC_2019_Training_GroundTruth.csv"
    meta_path = data_root / "ISIC2019" / "ISIC_2019_Training_Metadata.csv"
    img_dir = data_root / "ISIC2019" / "ISIC_2019_Training_Input"

    logger.info("Loading ISIC2019 ground truth from %s", gt_path)
    gt = pd.read_csv(gt_path)

    class_cols = [c for c in ISIC2019_CLASSES if c in gt.columns]
    gt["label"] = gt[class_cols].values.argmax(axis=1)
    gt["label_name"] = gt["label"].apply(lambda i: ISIC2019_CLASSES[i])

    logger.info("Loading ISIC2019 metadata from %s", meta_path)
    meta = pd.read_csv(meta_path)

    df = gt[["image", "label", "label_name"]].merge(
        meta[["image", "lesion_id", "age_approx", "anatom_site_general", "sex"]],
        on="image",
        how="left",
    )
    df = df.rename(columns={"image": "image_id"})

    df["path"] = df["image_id"].apply(lambda iid: str(img_dir / f"{iid}.jpg"))

    missing = df[~df["path"].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        logger.warning(
            "%d images not found on disk (first 5: %s)",
            len(missing),
            missing["image_id"].head(5).tolist(),
        )

    df = attach_lesion_key(df, data_root / "ISIC2019")
    df = lesion_disjoint_split(df, val_frac=0.10, test_frac=0.10, seed=seed)
    df["client_id"] = -1

    logger.info(
        "ISIC2019 split: %d train | %d val | %d test",
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )

    for name, group in df[df["split"] == "train"].groupby("label_name"):
        logger.info("  Train class %s: %d samples", name, len(group))

    return df


def _load_isic2018_split(
    gt_path: Path,
    img_dir: Path,
    split: str,
    lesion_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    logger.info("Loading ISIC2018 %s ground truth from %s", split, gt_path)
    gt = pd.read_csv(gt_path)
    class_cols = [c for c in ISIC2018_CLASSES if c in gt.columns]
    if not class_cols:
        raise ValueError(f"No ISIC2018 class columns in {gt_path}")
    gt["label"] = gt[class_cols].values.argmax(axis=1)
    gt["label_name"] = gt["label"].apply(lambda i: ISIC2018_CLASSES[i])
    df = gt[["image", "label", "label_name"]].rename(columns={"image": "image_id"})
    df["split"] = split
    df["path"] = df["image_id"].apply(lambda iid: str(img_dir / f"{iid}.jpg"))
    if lesion_df is not None and split == "train":
        df = df.merge(
            lesion_df[["image", "lesion_id"]].rename(columns={"image": "image_id"}),
            on="image_id",
            how="left",
        )
    else:
        df["lesion_id"] = None
    return df


def build_isic2018_manifest(data_root: Path) -> pd.DataFrame:
    base = data_root / "ISIC2018"
    lesion_path = base / "ISIC2018_Task3_Training_LesionGroupings.csv"
    logger.info("Loading ISIC2018 lesion groupings from %s", lesion_path)
    lesion_df = pd.read_csv(lesion_path)

    train_df = _load_isic2018_split(
        base / "ISIC2018_Task3_Training_GroundTruth.csv",
        base / "ISIC2018_Task3_Training_Input",
        "train",
        lesion_df=lesion_df,
    )
    val_df = _load_isic2018_split(
        base / "ISIC2018_Task3_Validation_GroundTruth.csv",
        base / "ISIC2018_Task3_Validation_Input",
        "val",
    )
    test_df = _load_isic2018_split(
        base / "ISIC2018_Task3_Test_GroundTruth.csv",
        base / "ISIC2018_Task3_Test_Input",
        "test",
    )

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df["client_id"] = -1
    df = attach_lesion_key(df, base)
    # The official Task 3 folders are kept. Lesions whose views also occur in the
    # official validation or test folders are removed from TRAINING only, so the
    # training data are lesion-disjoint from the evaluation data.
    held = set(df.loc[df["split"].isin(["val", "test"]), "lesion_key"].astype(str))
    overlap = (df["split"] == "train") & df["lesion_key"].astype(str).isin(held)
    if overlap.any():
        logger.warning(
            "ISIC2018: removing %d training images (%d lesions) that share a lesion with val/test",
            int(overlap.sum()), int(df.loc[overlap, "lesion_key"].nunique()),
        )
    df = df[~overlap].reset_index(drop=True)
    df.attrs["n_train_removed_for_overlap"] = int(overlap.sum())

    for split_name in ("train", "val", "test"):
        part = df[df["split"] == split_name]
        missing = part[~part["path"].apply(lambda p: Path(p).exists())]
        if len(missing) > 0:
            logger.warning(
                "%s: %d images not found on disk (first 5: %s)",
                split_name,
                len(missing),
                missing["image_id"].head(5).tolist(),
            )

    logger.info(
        "ISIC2018 split: %d train | %d val | %d test",
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )
    for name, group in df[df["split"] == "train"].groupby("label_name"):
        logger.info("  Train class %s: %d samples", name, len(group))

    return df


def build_isbi2016_manifest(data_root: Path) -> pd.DataFrame:
    train_gt = data_root / "ISBI2016" / "ISBI2016_ISIC_Training_GroundTruth.csv"
    test_gt = data_root / "ISBI2016" / "ISBI2016_ISIC_Test_GroundTruth.csv"
    train_img_dir = data_root / "ISBI2016" / "ISBI2016_ISIC_Training_Data"
    test_img_dir = data_root / "ISBI2016" / "ISBI2016_ISIC_Test_Data"

    def _load(csv_path: Path, img_dir: Path, split: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, header=None, names=["image_id", "label_name"])
        df["label"] = df["label_name"].map(ISBI2016_LABEL_MAP)
        df["path"] = df["image_id"].apply(lambda iid: str(img_dir / f"{iid}.jpg"))
        df["split"] = split
        return df

    train_df = _load(train_gt, train_img_dir, "train")
    test_df = _load(test_gt, test_img_dir, "test")

    tr_ids, val_ids = train_test_split(
        train_df["image_id"].values,
        test_size=0.10,
        stratify=train_df["label"].values,
        random_state=42,
    )
    train_df.loc[train_df["image_id"].isin(val_ids), "split"] = "val"

    df = pd.concat([train_df, test_df], ignore_index=True)
    df["lesion_id"] = None
    df["client_id"] = -1

    logger.info(
        "ISBI2016 split: %d train | %d val | %d test",
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )
    return df


def ensure_manifest_for_dataset(data_root: Path | str, dataset: DatasetName) -> None:
    """Create manifest.csv for one dataset if missing; skip if file exists."""
    root = Path(data_root)
    if dataset == "isic2019":
        out = root / "ISIC2019" / "manifest.csv"
        if out.exists():
            logger.info("ISIC2019 manifest already exists at %s (skipping)", out)
            return
        _require_inputs("ISIC2019", isic2019_inputs(root))
        logger.info("Building ISIC2019 manifest...")
        df = build_isic2019_manifest(root, seed=42)
        df.to_csv(out, index=False)
        write_split_audit(df, ISIC2019_CLASSES, out.parent / "split_audit.json")
        logger.info("Saved: %s", out)
        return

    if dataset == "isic2018":
        out = root / "ISIC2018" / "manifest.csv"
        if out.exists():
            logger.info("ISIC2018 manifest already exists at %s (skipping)", out)
            return
        _require_inputs("ISIC2018", isic2018_inputs(root))
        logger.info("Building ISIC2018 manifest...")
        df = build_isic2018_manifest(root)
        df.to_csv(out, index=False)
        a = write_split_audit(df, ISIC2018_CLASSES, out.parent / "split_audit.json")
        a["n_train_removed_for_overlap"] = df.attrs.get("n_train_removed_for_overlap", 0)
        (out.parent / "split_audit.json").write_text(__import__("json").dumps(a, indent=2))
        logger.info("Saved: %s", out)
        return

    out = root / "ISBI2016" / "manifest.csv"
    if out.exists():
        logger.info("ISBI2016 manifest already exists at %s (skipping)", out)
        return
    _require_inputs("ISBI2016", isbi2016_inputs(root))
    logger.info("Building ISBI2016 manifest...")
    df = build_isbi2016_manifest(root)
    df.to_csv(out, index=False)
    logger.info("Saved: %s", out)


def build_all_manifests(
    data_root: Path | str,
    *,
    seed: int = 42,
    skip_isic2019: bool = False,
    skip_isic2018: bool = False,
    skip_isbi2016: bool = False,
) -> None:
    """Build manifest CSVs with CLI-parity skip flags."""
    if skip_isic2019 and skip_isic2018 and skip_isbi2016:
        raise ValueError("Nothing to do: all skip flags set.")

    root = Path(data_root)

    if not skip_isic2019:
        out = root / "ISIC2019" / "manifest.csv"
        if not out.exists():
            _require_inputs("ISIC2019", isic2019_inputs(root))
            logger.info("Building ISIC2019 manifest...")
            df19 = build_isic2019_manifest(root, seed=seed)
            df19.to_csv(out, index=False)
            write_split_audit(df19, ISIC2019_CLASSES, out.parent / "split_audit.json")
            logger.info("Saved: %s", out)
        else:
            logger.info("ISIC2019 manifest already exists at %s (skipping)", out)

    if not skip_isic2018:
        out = root / "ISIC2018" / "manifest.csv"
        if not out.exists():
            _require_inputs("ISIC2018", isic2018_inputs(root))
            logger.info("Building ISIC2018 manifest...")
            df18 = build_isic2018_manifest(root)
            df18.to_csv(out, index=False)
            a = write_split_audit(df18, ISIC2018_CLASSES, out.parent / "split_audit.json")
            a["n_train_removed_for_overlap"] = df18.attrs.get("n_train_removed_for_overlap", 0)
            (out.parent / "split_audit.json").write_text(__import__("json").dumps(a, indent=2))
            logger.info("Saved: %s", out)
        else:
            logger.info("ISIC2018 manifest already exists at %s (skipping)", out)

    if not skip_isbi2016:
        out = root / "ISBI2016" / "manifest.csv"
        if not out.exists():
            _require_inputs("ISBI2016", isbi2016_inputs(root))
            logger.info("Building ISBI2016 manifest...")
            df16 = build_isbi2016_manifest(root)
            df16.to_csv(out, index=False)
            logger.info("Saved: %s", out)
        else:
            logger.info("ISBI2016 manifest already exists at %s (skipping)", out)
