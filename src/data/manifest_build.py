"""
Build manifest CSVs for SkinFLNet++ from raw dataset files.

Partitioning (client_id) is applied per-experiment by src/data/partition.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

ISIC2019_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
ISIC2018_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
ISBI2016_LABEL_MAP = {"benign": 0, "malignant": 1}

DatasetName = Literal["isic2019", "isic2018", "isbi2016"]


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

    train_ids, test_val_ids = train_test_split(
        df["image_id"].values,
        test_size=0.20,
        stratify=df["label"].values,
        random_state=seed,
    )
    val_ids, test_ids = train_test_split(
        test_val_ids,
        test_size=0.50,
        stratify=df.set_index("image_id").loc[test_val_ids, "label"].values,
        random_state=seed,
    )

    split_map = {iid: "train" for iid in train_ids}
    split_map.update({iid: "val" for iid in val_ids})
    split_map.update({iid: "test" for iid in test_ids})
    df["split"] = df["image_id"].map(split_map)
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
