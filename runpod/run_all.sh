#!/usr/bin/env bash
# End-to-end R1 pipeline on a single RunPod GPU pod.
#   env: DEPLOY_KEY_B64 (base64 ssh private key with write access to the repo)
#        JOBS (concurrent runs, default 4), WORKERS (dataloader workers per loader)
set -euo pipefail
export PYTHONUNBUFFERED=1 SKINFL_FORCE_PLAIN_FL_LOG=1
JOBS="${JOBS:-4}"
W=/workspace
cd "$W"
REPO_URL_HTTPS=https://github.com/yazanjer/FedDermNet.git
REPO_URL_SSH=git@github.com:yazanjer/FedDermNet.git

# ── code ────────────────────────────────────────────────────────────────────
[ -d FedDermNet ] || git clone -q "$REPO_URL_HTTPS" FedDermNet
cd FedDermNet && git pull -q || true
(command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null))
pip install -q -r requirements.txt

# ── results branch (deploy key) ─────────────────────────────────────────────
if [ -n "${DEPLOY_KEY_B64:-}" ]; then
  mkdir -p ~/.ssh && echo "$DEPLOY_KEY_B64" | base64 -d > ~/.ssh/deploy_key && chmod 600 ~/.ssh/deploy_key
  ssh-keyscan -t ed25519,rsa github.com >> ~/.ssh/known_hosts 2>/dev/null
  export GIT_SSH_COMMAND="ssh -i ~/.ssh/deploy_key -o IdentitiesOnly=yes"
  if [ ! -d "$W/results_repo/.git" ]; then
    git clone -q "$REPO_URL_SSH" "$W/results_repo"
    cd "$W/results_repo"
    if git ls-remote --exit-code --heads origin results-r1 >/dev/null 2>&1; then
      git checkout -q -B results-r1 origin/results-r1
    else
      git checkout -q --orphan results-r1 && git rm -rqf . || true
      echo "# FedDermNet R1 result archive (generated on RunPod)" > README.md
      git add README.md && git -c user.name="FedDermNet RunPod" -c user.email="yazan.aljeroudi@gmail.com" commit -qm "init results-r1"
      git push -q origin results-r1
    fi
    cd "$W/FedDermNet"
  fi
fi
export RESULTS_REPO="$W/results_repo" RESULTS_DIR="$W/results_r1"
mkdir -p "$RESULTS_DIR"
python - <<'PY' > "$RESULTS_DIR/env.json"
import json, torch, platform, os
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "python": platform.python_version(), "cpus": os.cpu_count(),
  "pod": os.environ.get("RUNPOD_POD_ID")}, indent=2))
PY

# ── data ────────────────────────────────────────────────────────────────────
python scripts/prepare_data.py --data-root "$W/data"
cp "$W/data/ISIC2019/split_audit.json" "$RESULTS_DIR/split_audit_isic2019.json"
cp "$W/data/ISIC2018/split_audit.json" "$RESULTS_DIR/split_audit_isic2018.json"
mkdir -p "$RESULTS_DIR/manifests"
cp "$W/data/ISIC2019/manifest.csv" "$RESULTS_DIR/manifests/isic2019_manifest.csv"
cp "$W/data/ISIC2018/manifest.csv" "$RESULTS_DIR/manifests/isic2018_manifest.csv"
bash scripts/sync_results.sh data-ready

Q="python scripts/run_queue.py --jobs $JOBS --data-root $W/data --results-dir $RESULTS_DIR --figures-dir $W/figures_r1 --sync scripts/sync_results.sh"
# ── phase 1: strategy tuning (seed 42, validation only) ─────────────────────
python scripts/r1_grid.py tune --out configs/r1 | $Q
# ── phase 2: all ablations + tuned strategies, 3 seeds ──────────────────────
python scripts/r1_grid.py strategy --out configs/r1 --results "$RESULTS_DIR" > /tmp/strategy.txt
python scripts/r1_grid.py main --out configs/r1 > /tmp/main.txt
cat /tmp/main.txt /tmp/strategy.txt | $Q
# ── phase 3: cross-release transfer ─────────────────────────────────────────
python scripts/r1_cross_eval.py --data-root "$W/data" --results-dir "$RESULTS_DIR"
touch "$RESULTS_DIR/ALL_DONE"
bash scripts/sync_results.sh all-done
echo "ALL DONE $(date)"
# ── stop this pod (compute billing ends; /workspace volume keeps checkpoints) ─
if [ "${AUTO_STOP:-1}" = "1" ] && command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID" || true
fi
