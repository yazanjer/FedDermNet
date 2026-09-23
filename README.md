# FedDermNet R1 result archive

Consolidated archives of every run of the revision (86 training runs on ISIC 2019 and
ISIC 2018 Task 3, seeds 42/43/44, fixed 50-round budget, validation-only selection),
produced on three RunPod A100 pods (`results-r1`, `results-r1-p2`, `results-r1-p3`
branches; merged here).

- `results_r1/<run>/summary.json` — per-round validation and test metrics, selected round
- `results_r1/<run>/test_predictions_best.npz` — test probabilities of the selected checkpoint
- `results_r1/<run>/client_assignment.csv.gz` — client of every training image
- `results_r1/cross_eval/` — cross-release transfer, 3 seeds
- `results_r1/manifests/`, `results_r1/split_audit_*.json` — lesion-disjoint splits
- `paper_r1/` — output of `scripts/r1_aggregate.py` (tables, statistics, figures)

Regenerate: `python scripts/r1_aggregate.py --results results_r1 --out paper_r1`
