"""
Download ISIC 2018 Task 3 and ISIC 2019, pre-resize images, fetch lesion identifiers,
and build lesion-disjoint manifests with split audits.

  python scripts/prepare_data.py --data-root /workspace/data [--size 288] [--no-api]

Steps
  1. Download the official challenge archives (ISIC S3) and unzip them.
  2. Resize every image in place so that its SHORTER side is ``--size`` px (aspect kept).
     Training uses RandomResizedCrop(224) and evaluation Resize(255)+CenterCrop(224),
     so 288 px preserves every pixel the pipelines actually consume while cutting JPEG
     decode time several-fold.
  3. Optionally query the ISIC Archive API for each image's lesion identifier
     (``lesion_ids_isic_api.csv``). This yields one lesion namespace across the 2018
     and 2019 releases and covers images whose challenge metadata has no lesion_id.
  4. Build ``manifest.csv`` + ``split_audit.json`` for both releases.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prepare_data")

S3 = "https://isic-archive.s3.amazonaws.com/challenges"
FILES = {
    "ISIC2019": [
        f"{S3}/2019/ISIC_2019_Training_Input.zip",
        f"{S3}/2019/ISIC_2019_Training_GroundTruth.csv",
        f"{S3}/2019/ISIC_2019_Training_Metadata.csv",
    ],
    "ISIC2018": [
        f"{S3}/2018/ISIC2018_Task3_Training_Input.zip",
        f"{S3}/2018/ISIC2018_Task3_Training_GroundTruth.zip",
        f"{S3}/2018/ISIC2018_Task3_Training_LesionGroupings.csv",
        f"{S3}/2018/ISIC2018_Task3_Validation_Input.zip",
        f"{S3}/2018/ISIC2018_Task3_Validation_GroundTruth.zip",
        f"{S3}/2018/ISIC2018_Task3_Test_Input.zip",
        f"{S3}/2018/ISIC2018_Task3_Test_GroundTruth.zip",
    ],
}


def download(url: str, dest: Path, retries: int = 5) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "FedDermNet-R1/1.0"})
            with urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=4 << 20)
            tmp.rename(dest)
            log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return
        except Exception as e:  # noqa: BLE001
            log.warning("download %s failed (%s), retry %d", url, e, attempt + 1)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def unzip(zpath: Path, out_dir: Path) -> None:
    marker = out_dir / f".unzipped_{zpath.stem}"
    if marker.exists():
        return
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out_dir)
    # Ground-truth zips unpack into a folder of the same name; lift CSVs up one level.
    inner = out_dir / zpath.stem
    if inner.is_dir():
        for p in inner.glob("*.csv"):
            shutil.move(str(p), out_dir / p.name)
    marker.touch()
    log.info("unzipped %s", zpath.name)


def _resize_one(args):
    path, size = args
    from PIL import Image

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = min(w, h)
            if s <= size:
                return 0
            r = size / s
            im = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BICUBIC)
            im.save(path, "JPEG", quality=95)
        return 1
    except Exception as e:  # noqa: BLE001
        log.warning("resize failed %s: %s", path, e)
        return 0


def resize_tree(folder: Path, size: int, workers: int) -> None:
    marker = folder / f".resized_{size}"
    if marker.exists():
        return
    files = [p for p in folder.glob("*.jpg")]
    with Pool(workers) as pool:
        n = sum(pool.imap_unordered(_resize_one, [(p, size) for p in files], chunksize=64))
    marker.touch()
    log.info("resized %d / %d images in %s", n, len(files), folder.name)


def _find_lesion_id(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "lesion_id" and v:
                return str(v)
            r = _find_lesion_id(v)
            if r:
                return r
    return None


def fetch_api_lesion_ids(image_ids: list[str], out_csv: Path, workers: int = 24) -> None:
    """Query api.isic-archive.com for each image's lesion identifier (best effort)."""
    import pandas as pd

    done: dict[str, str | None] = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = dict(zip(prev["image_id"].astype(str), prev["lesion_id"]))
    todo = [i for i in image_ids if i not in done]
    log.info("ISIC API lesion lookup: %d cached, %d to fetch", len(done), len(todo))

    def one(iid: str):
        qid = iid.replace("_downsampled", "")  # some ISIC 2019 ids carry this suffix
        url = f"https://api.isic-archive.com/api/v2/images/{qid}/"
        for attempt in range(4):
            try:
                with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=30) as r:
                    return iid, _find_lesion_id(json.load(r))
            except Exception as e:  # noqa: BLE001
                if "404" in str(e):
                    return iid, None
                time.sleep(1.5 * (attempt + 1))
        return iid, None

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, i) for i in todo]
        for k, f in enumerate(as_completed(futs), 1):
            iid, lid = f.result()
            done[iid] = lid
            if k % 2000 == 0:
                log.info("  %d/%d (%.0fs)", k, len(todo), time.time() - t0)
                pd.DataFrame({"image_id": list(done), "lesion_id": list(done.values())}).to_csv(out_csv, index=False)
    pd.DataFrame({"image_id": list(done), "lesion_id": list(done.values())}).to_csv(out_csv, index=False)
    n_ok = sum(1 for v in done.values() if isinstance(v, str) and v)
    log.info("ISIC API lesion ids: %d / %d images resolved", n_ok, len(done))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--no-api", action="store_true", help="skip ISIC API lesion lookup")
    args = ap.parse_args()
    root = Path(args.data_root)

    for folder, urls in FILES.items():
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        for u in urls:
            dest = d / u.rsplit("/", 1)[1]
            if (d / (dest.name + ".done")).exists():
                continue
            download(u, dest)
            if dest.suffix == ".zip":
                unzip(dest, d)
                dest.unlink()  # free disk; marker file prevents re-download
                (d / (dest.name + ".done")).touch()

    for folder in [
        root / "ISIC2019" / "ISIC_2019_Training_Input",
        root / "ISIC2018" / "ISIC2018_Task3_Training_Input",
        root / "ISIC2018" / "ISIC2018_Task3_Validation_Input",
        root / "ISIC2018" / "ISIC2018_Task3_Test_Input",
    ]:
        resize_tree(folder, args.size, args.workers)

    import pandas as pd

    if not args.no_api:
        ids19 = pd.read_csv(root / "ISIC2019" / "ISIC_2019_Training_GroundTruth.csv")["image"].astype(str).tolist()
        ids18 = []
        for f in ("Training", "Validation", "Test"):
            ids18 += pd.read_csv(root / "ISIC2018" / f"ISIC2018_Task3_{f}_GroundTruth.csv")["image"].astype(str).tolist()
        allcsv = root / "lesion_ids_isic_api_all.csv"
        try:
            fetch_api_lesion_ids(sorted(set(ids19) | set(ids18)), allcsv)
            for folder in ("ISIC2019", "ISIC2018"):
                shutil.copy(allcsv, root / folder / "lesion_ids_isic_api.csv")
        except Exception as e:  # noqa: BLE001
            log.warning("ISIC API lookup failed (%s); using challenge metadata only", e)

    from src.data.manifest_build import ensure_manifest_for_dataset

    for ds in ("isic2019", "isic2018"):
        ensure_manifest_for_dataset(root, ds)  # type: ignore[arg-type]
    for folder in ("ISIC2019", "ISIC2018"):
        log.info("%s audit: %s", folder, (root / folder / "split_audit.json").read_text())


if __name__ == "__main__":
    main()
