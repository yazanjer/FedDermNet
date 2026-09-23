# FedDermNet

Federated dermoscopic skin-lesion classification under controlled label skew, evaluated on
ISIC 2019 (8 classes) and ISIC 2018 Task 3 (7 classes). This repository contains the code
used for the IEEE Access submission *"FedDermNet: A Federated Learning Framework for
Dermoscopic Skin Lesion Classification Under Heterogeneous Clients"* and its revision.

## What the pipeline does

| Stage | Module |
|---|---|
| Manifest construction, **lesion-disjoint** train/val/test split, split audit | `src/data/manifest_build.py` |
| Lesion-aware Dirichlet(α) client partition (all views of a lesion on one client) | `src/data/partition.py` |
| Federated simulation (FedAvg, FedProx, FedAdam), VGG-16-BN backbone | `src/fl/` |
| Pooled (centralized) reference with a matched sample budget | `src/train/centralized.py` |
| Metrics: accuracy, balanced accuracy, macro-F1, macro AUROC/AUPRC, per-class sensitivity, specificity, AUROC, AUPRC, confusion matrix | `src/eval/metrics.py` |
| Cross-release transfer with train-overlap filtering | `src/eval/isic_cross_eval.py` |

### Protocol (revision R1)

* **Lesion-disjoint splits.** ISIC 2019 has no public multi-class test labels, so its labelled
  training set is split 80/10/10 at the *lesion* level (`StratifiedGroupKFold` on
  `lesion_key`). ISIC 2018 keeps the official Task 3 folders; any training lesion that also
  occurs in the official validation or test folders is removed from training. Every build
  writes `split_audit.json` with image and unique-lesion counts per split and class and the
  pairwise lesion/image intersections (all zero by construction).
* **Model selection on validation only.** The server scores the global model on the
  validation split after every round; early stopping (patience 5) and the reported
  checkpoint use validation macro-F1. The test split is evaluated for reporting and curves
  only. The centralized reference uses the same rule.
* **Matched sample budget.** One centralized "round" is `fraction_fit × E` epochs over the
  pooled data (one epoch for the default recipe), the expected number of samples processed by
  one federated round.
* **Seeds.** One fixed data split per release (seed 42); seeds 42, 43, 44 vary the client
  partition, client sampling, head initialisation, augmentation and batching.
* **Strategy comparison with an equal tuning budget.** FedAvg (client learning rate),
  FedProx (µ) and FedAdam (server learning rate η) each receive three candidate settings,
  chosen by validation macro-F1 on seed 42 (`scripts/r1_grid.py`).
* FedAdam applies the adaptive server step to trainable parameters only; BatchNorm running
  statistics are averaged.

## Reproducing the experiments

```bash
pip install -r requirements.txt          # plus torch/torchvision (>= 2.2)
python scripts/prepare_data.py --data-root data     # download, resize, lesion IDs, manifests
python scripts/r1_grid.py tune | python scripts/run_queue.py --jobs 4 --data-root data
python scripts/r1_grid.py strategy > strat.txt && python scripts/r1_grid.py main > main.txt
cat main.txt strat.txt | python scripts/run_queue.py --jobs 4 --data-root data
python scripts/r1_cross_eval.py --data-root data
python scripts/r1_aggregate.py --results results_r1 --out paper_r1
```

`runpod/run_all.sh` runs the whole pipeline unattended on one GPU pod. Every run directory
contains `summary.json` (per-round validation and test metrics, the selected round and its
metrics), `client_assignment.csv.gz` (the exact client of every training image) and
`test_predictions_best.npz` (test-set probabilities of the selected checkpoint, used for
bootstrap confidence intervals). Result archives of the revision are on the `results-r1`
branch.

## Data

ISIC 2018 Task 3 and ISIC 2019 are downloaded from the official ISIC challenge archive and
are subject to the ISIC terms of use. Images are resized once so that the shorter side is
288 px (training uses `RandomResizedCrop(224)`, evaluation `Resize(255)+CenterCrop(224)`).

## Layout

```
configs/v1_original/   configurations of the first submission (kept for provenance)
scripts/               data preparation, experiment grid, queue runner, aggregation
src/                   library code
runpod/                unattended GPU pipeline
tests/                 synthetic-data smoke test
```

The first submission's archives used an image-level ISIC 2019 split and selected rounds on
the test split; those results are superseded by the revision protocol above.
