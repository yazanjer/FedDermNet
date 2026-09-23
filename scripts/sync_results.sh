#!/usr/bin/env bash
# Publish lightweight result files to the `results-r1` branch (called by run_queue.py).
# Requires RESULTS_REPO (a clone of the repo checked out on results-r1) and a git remote
# that the pod can push to (deploy key configured by runpod/run_all.sh).
set -u
RES="${RESULTS_DIR:-/workspace/results_r1}"
REPO="${RESULTS_REPO:-/workspace/results_repo}"
[ -d "$REPO/.git" ] || exit 0
mkdir -p "$REPO/results_r1"
rsync -a --delete --prune-empty-dirs \
  --include='*/' \
  --include='summary.json' --include='round_[0-9][0-9][0-9].json' \
  --include='client_assignment.csv.gz' --include='test_predictions_best.npz' \
  --include='cross_s*.json' --include='*_s*.npz' --include='tuning_selection.json' \
  --include='_queue_status.tsv' --include='split_audit*.json' --include='env.json' --include='*_manifest.csv' \
  --exclude='*' "$RES/" "$REPO/results_r1/"
# tail of the logs for monitoring
mkdir -p "$REPO/results_r1/_logs_tail"
for f in "$RES"/_logs/*.log; do [ -f "$f" ] && tail -n 40 "$f" > "$REPO/results_r1/_logs_tail/$(basename "$f")"; done
[ -f /workspace/run.log ] && tail -n 200 /workspace/run.log > "$REPO/results_r1/_logs_tail/_run.log"
cd "$REPO" || exit 0
git add -A results_r1 >/dev/null 2>&1
git -c user.name="FedDermNet RunPod" -c user.email="yazan.aljeroudi@gmail.com" \
  commit -qm "results: ${1:-sync} ($(date -u +%FT%TZ))" >/dev/null 2>&1 || true
for i in 1 2 3; do git push -q origin HEAD:results-r1 && break; sleep 10; git pull -q --rebase origin results-r1 || true; done
