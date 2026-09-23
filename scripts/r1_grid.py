"""
Experiment grid for the R1 revision (IEEE Access resubmission).

  python scripts/r1_grid.py tune      --out configs/r1        # strategy tuning (seed 42)
  python scripts/r1_grid.py main      --out configs/r1        # all ablations, 3 seeds
  python scripts/r1_grid.py strategy  --out configs/r1 --results results_r1
                                                             # tuned strategies, 3 seeds

Every command writes YAML configs and prints their paths (one per line) so they can be
piped into ``scripts/run_queue.py``.

Design
- One data split per release (seed 42, lesion-disjoint); the SEED varies the client
  partition, client sampling, initialisation of the head, augmentation and batching.
- Fixed budget: every run trains for T = 50 rounds (no early stopping); the reported
  checkpoint is the round with the highest validation macro-F1.
- Reference federated setting: alpha = 0.5, K = 10, E = 2, FedAvg, client lr 1e-4.
  Every ablation changes exactly one knob relative to this reference.
- Strategy comparison with an equal validation budget: three candidate settings per
  rule (FedAvg: client lr; FedProx: mu; FedAdam: server lr eta), chosen by validation
  macro-F1 on seed 42, then re-run on all seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DATASETS = ("isic2019", "isic2018")
SEEDS = (42, 43, 44)
REF = dict(alpha=0.5, num_clients=10, local_epochs=2, strategy="fedavg", lr=1e-4)

TUNE_GRID = {
    "fedavg": ("lr", [5e-5, 1e-4, 2e-4]),
    "fedprox": ("fedprox_mu", [1e-3, 1e-2, 1e-1]),
    "fedadam": ("fedadam_eta", [1e-5, 1e-4, 1e-3]),
}


def base(ds: str) -> dict:
    return dict(
        dataset=ds, backbone="vgg16_bn", partition="dirichlet", alpha=0.5, num_clients=10,
        num_rounds=50, local_epochs=2, fraction_fit=0.5, strategy="fedavg",
        fedprox_mu=0.01, fedadam_eta=1e-3, fedadam_tau=1e-3,
        lr=1e-4, weight_decay=1e-4, batch_size=32, early_stop_patience=0, seed=42,
        mode="federated", use_wandb=False, num_workers=5, resume=True,
        client_local_eval_fraction=0.0, amp=True, eval_test_every_round=True,
        centralized_best_on="val", show_round_progress=False, deterministic=False,
    )


def fmt(x: float) -> str:
    return f"{x:g}".replace("e-0", "e-")


def run_name(c: dict) -> str:
    if c["mode"] == "centralized":
        return f"{c['dataset']}_centralized_s{c['seed']}"
    tag = f"{c['strategy']}_a{fmt(c['alpha'])}_K{c['num_clients']}_E{c['local_epochs']}"
    if c["strategy"] == "fedprox":
        tag += f"_mu{fmt(c['fedprox_mu'])}"
    if c["strategy"] == "fedadam":
        tag += f"_eta{fmt(c['fedadam_eta'])}"
    if abs(c["lr"] - 1e-4) > 1e-12:
        tag += f"_lr{fmt(c['lr'])}"
    return f"{c['dataset']}_{tag}_s{c['seed']}"


def write(cfgs: list[dict], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in cfgs:
        c = dict(c)
        c["name"] = run_name(c)
        p = out / f"{c['name']}.yaml"
        p.write_text(yaml.safe_dump({"experiment": c}, sort_keys=False))
        paths.append(p)
    return paths


def tune_cfgs() -> list[dict]:
    cfgs = []
    for ds in DATASETS:
        for strat, (key, values) in TUNE_GRID.items():
            for v in values:
                c = base(ds) | REF | {"strategy": strat, key: v, "seed": 42}
                cfgs.append(c)
    return cfgs


def main_cfgs(seeds=SEEDS) -> list[dict]:
    cfgs = []
    for s in seeds:
        for ds in DATASETS:
            b = base(ds) | REF | {"seed": s}
            cfgs.append(b)                                            # reference (alpha 0.5)
            for a in (0.1, 1.0, 100.0):
                cfgs.append(b | {"alpha": a})                          # label-skew sweep
            for k in (5, 20):
                cfgs.append(b | {"num_clients": k})                    # federation size
            for e in (1, 5, 10):
                cfgs.append(b | {"local_epochs": e})                   # local work
            cfgs.append(b | {"mode": "centralized", "partition": "iid"})  # pooled reference
    return cfgs


def part_of(c: dict) -> str:
    """Assignment of main-grid runs to three pods (roughly equal GPU time).

    p1 holds every run the cross-release evaluation needs (reference setting) and the
    centralized references; p2 the local-epoch runs; p3 the label-skew and
    federation-size runs.
    """
    if c["mode"] == "centralized":
        return "p1"
    if c["alpha"] == 0.5 and c["num_clients"] == 10 and c["local_epochs"] == 2:
        return "p1"
    if c["local_epochs"] != 2:
        if c["dataset"] == "isic2019" or c["local_epochs"] == 10:
            return "p2"
        return "p3"
    return "p3"


def select_tuned(results: Path) -> dict:
    """Best candidate per (dataset, strategy) by validation macro-F1 at the selected round."""
    chosen: dict = {}
    for c in tune_cfgs():
        name = run_name(c)
        f = results / name / "summary.json"
        if not f.is_file():
            raise FileNotFoundError(f"tuning run missing: {f}")
        val = json.loads(f.read_text())["selected"]["val"]["macro_f1"]
        key = f"{c['dataset']}|{c['strategy']}"
        tkey, _ = TUNE_GRID[c["strategy"]]
        if key not in chosen or val > chosen[key]["val_macro_f1"]:
            chosen[key] = {"param": tkey, "value": c[tkey], "val_macro_f1": val, "run": name}
    return chosen


def strategy_cfgs(chosen: dict, seeds=SEEDS) -> list[dict]:
    cfgs = []
    for s in seeds:
        for ds in DATASETS:
            for strat in ("fedavg", "fedprox", "fedadam"):
                ch = chosen[f"{ds}|{strat}"]
                c = base(ds) | REF | {"strategy": strat, ch["param"]: ch["value"], "seed": s}
                cfgs.append(c)
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "main", "strategy"])
    ap.add_argument("--out", default="configs/r1")
    ap.add_argument("--results", default="results_r1")
    ap.add_argument("--part", default="all", choices=["all", "p1", "p2", "p3"])
    args = ap.parse_args()
    out = Path(args.out)
    if args.phase == "tune":
        paths = write(tune_cfgs(), out)
    elif args.phase == "main":
        cfgs = [c for c in main_cfgs() if args.part == "all" or part_of(c) == args.part]
        paths = write(cfgs, out)
    else:
        chosen = select_tuned(Path(args.results))
        (Path(args.results) / "tuning_selection.json").write_text(json.dumps(chosen, indent=2))
        paths = write(strategy_cfgs(chosen), out)
    seen = set()
    for p in paths:  # de-duplicate (reference runs recur across phases)
        if p not in seen:
            seen.add(p)
            print(p)


if __name__ == "__main__":
    main()
