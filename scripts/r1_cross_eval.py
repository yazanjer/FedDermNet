"""
Cross-release transfer for every seed of the reference federated setting
(alpha = 0.5, K = 10, E = 2, FedAvg), using the best-validation global weights.

  python scripts/r1_cross_eval.py --data-root /workspace/data --results-dir results_r1

Images and lesions that the source model saw during training are removed from the
target test set (ISIC 2019 training data contain the HAM10000/ISIC 2018 images).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.isic_cross_eval import (  # noqa: E402
    eval_2018_model_on_2019_test_no_scc,
    eval_2019_model_on_2018_test,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--results-dir", default="results_r1")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--tag", default="fedavg_a0.5_K10_E2")
    args = ap.parse_args()
    res = Path(args.results_dir)
    out = res / "cross_eval"
    out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in args.seeds.split(",")]:
        target = out / f"cross_s{s}.json"
        if target.is_file():
            continue
        ck19 = res / f"isic2019_{args.tag}_s{s}" / "best_global.pt"
        ck18 = res / f"isic2018_{args.tag}_s{s}" / "best_global.pt"
        r1 = eval_2019_model_on_2018_test(data_root=args.data_root, checkpoint_path=ck19, seed=s, num_workers=8)
        r2 = eval_2018_model_on_2019_test_no_scc(data_root=args.data_root, checkpoint_path=ck18, seed=s, num_workers=8)
        payload = {}
        for key, r in (("isic2019_to_isic2018", r1), ("isic2018_to_isic2019", r2)):
            np.savez_compressed(out / f"{key}_s{s}.npz", probs=r["probs"].astype(np.float32),
                                labels=r["labels"].astype(np.int16), image_id=np.asarray(r["image_id"]))
            payload[key] = {"metrics": r["metrics"], "meta": r["meta"]}
        target.write_text(json.dumps(payload, indent=2, default=float))
        print(f"seed {s}: 2019->2018 F1={r1['metrics']['macro_f1']:.4f} | 2018->2019 F1={r2['metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
