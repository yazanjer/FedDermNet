"""
CLI for manifest construction. Core logic lives in src.data.manifest_build.

  python scripts/build_manifests.py
  python scripts/build_manifests.py --data-root /path/to/data --skip-isbi2016
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from src.data.manifest_build import build_all_manifests  # noqa: E402

DATA_ROOT = Path(__file__).parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest CSVs for SkinFLNet++")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-isic2019",
        action="store_true",
        help="Do not build ISIC2019/manifest.csv",
    )
    parser.add_argument(
        "--skip-isic2018",
        action="store_true",
        help="Do not build ISIC2018/manifest.csv",
    )
    parser.add_argument(
        "--skip-isbi2016",
        action="store_true",
        help="Do not build ISBI2016/manifest.csv (Colab if only ISIC2019)",
    )
    args = parser.parse_args()

    build_all_manifests(
        Path(args.data_root),
        seed=args.seed,
        skip_isic2019=args.skip_isic2019,
        skip_isic2018=args.skip_isic2018,
        skip_isbi2016=args.skip_isbi2016,
    )


if __name__ == "__main__":
    main()
