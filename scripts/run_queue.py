"""
Run a list of experiment YAMLs with N concurrent workers on one GPU.

  python scripts/r1_grid.py main | python scripts/run_queue.py --jobs 4 \
      --data-root /workspace/data --results-dir results_r1 --figures-dir figures_r1 \
      --sync scripts/sync_results.sh

A run is skipped when ``<results-dir>/<name>/summary.json`` already exists, so the
queue can be restarted after an interruption (unfinished runs resume from their
``checkpoint_latest.pt``). After each finished run the optional ``--sync`` command
is executed (serialised by a lock) to publish lightweight result files.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
_sync_lock = threading.Lock()


def job_name(cfg_path: Path) -> str:
    return yaml.safe_load(cfg_path.read_text())["experiment"]["name"]


def run_one(cfg_path: Path, args) -> tuple[str, int, float]:
    name = job_name(cfg_path)
    done = Path(args.results_dir) / name / "summary.json"
    if done.is_file():
        return name, 0, 0.0
    log_dir = Path(args.results_dir) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_experiment.py"), str(cfg_path),
        "--data-root", args.data_root, "--results-dir", args.results_dir,
        "--figures-dir", args.figures_dir,
    ]
    rc = 1
    for attempt in range(args.retries + 1):
        with open(log_dir / f"{name}.log", "a") as log:
            log.write(f"\n=== attempt {attempt + 1} {time.ctime()} ===\n")
            log.flush()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if rc == 0 and done.is_file():
            break
    dt = time.time() - t0
    status = "OK" if rc == 0 and done.is_file() else f"FAIL rc={rc}"
    with open(Path(args.results_dir) / "_queue_status.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name}\t{status}\t{dt/60:.1f} min\n")
    if args.sync:
        with _sync_lock:
            subprocess.call(["bash", args.sync, name], cwd=ROOT)
    return name, rc, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", help="YAML paths (else read from stdin)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--results-dir", default="results_r1")
    ap.add_argument("--figures-dir", default="figures_r1")
    ap.add_argument("--sync", default="")
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()
    paths = [Path(p) for p in (args.configs or sys.stdin.read().split())]
    # Longest jobs first keeps the GPU busy at the tail of the queue.
    def cost(p: Path) -> float:
        e = yaml.safe_load(p.read_text())["experiment"]
        w = 2.0 if e["dataset"] == "isic2019" else 1.0
        if e.get("mode") == "centralized":
            return w * 1.5
        return w * e["local_epochs"] * e["fraction_fit"] * 2
    paths = sorted(dict.fromkeys(paths), key=cost, reverse=True)
    print(f"[queue] {len(paths)} jobs, {args.jobs} concurrent", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        for name, rc, dt in ex.map(lambda p: run_one(p, args), paths):
            print(f"[queue] {name}: rc={rc} {dt/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
