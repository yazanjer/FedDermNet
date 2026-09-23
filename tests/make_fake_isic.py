"""Tiny synthetic ISIC-like tree for offline smoke tests (random images, real file layout)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fakedata")
rng = np.random.RandomState(0)


def img(path, label):
    a = (rng.rand(48, 64, 3) * 60 + label * 25).clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


# ISIC 2019: 480 images, 8 classes, lesions with 1-3 views, some missing ids
d = root / "ISIC2019"; (d / "ISIC_2019_Training_Input").mkdir(parents=True, exist_ok=True)
cls = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
rows, meta = [], []
i = 0
lesion = 0
while i < 480:
    lab = rng.choice(8, p=[.18, .4, .12, .05, .1, .05, .05, .05])
    nviews = rng.choice([1, 2, 3])
    lid = f"HAM_{lesion:07d}" if rng.rand() > 0.15 else None
    lesion += 1
    for _ in range(nviews):
        iid = f"ISIC_{i:07d}" + ("_downsampled" if rng.rand() < 0.05 else "")
        img(d / "ISIC_2019_Training_Input" / f"{iid}.jpg", lab)
        r = {"image": iid, **{c: float(c == cls[lab]) for c in cls}, "UNK": 0.0}
        rows.append(r); meta.append({"image": iid, "lesion_id": lid, "age_approx": 50, "anatom_site_general": "x", "sex": "male"})
        i += 1
pd.DataFrame(rows).to_csv(d / "ISIC_2019_Training_GroundTruth.csv", index=False)
pd.DataFrame(meta).to_csv(d / "ISIC_2019_Training_Metadata.csv", index=False)

# ISIC 2018: reuse first 300 2019 images as training (as in reality), plus val/test
d8 = root / "ISIC2018"
c18 = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
for sp in ("Training", "Validation", "Test"):
    (d8 / f"ISIC2018_Task3_{sp}_Input").mkdir(parents=True, exist_ok=True)
gt = pd.DataFrame(rows[:300]); md = pd.DataFrame(meta[:300])
gt = gt[gt["SCC"] == 0]
tr = []
for _, r in gt.iterrows():
    lab = int(np.argmax([r[c] for c in cls[:7]]))
    img(d8 / "ISIC2018_Task3_Training_Input" / f"{r['image']}.jpg", lab)
    tr.append({"image": r["image"], **{c18[k]: float(k == lab) for k in range(7)}})
pd.DataFrame(tr).to_csv(d8 / "ISIC2018_Task3_Training_GroundTruth.csv", index=False)
md[md["image"].isin(gt["image"])][["image", "lesion_id"]].fillna("HAM_x").to_csv(
    d8 / "ISIC2018_Task3_Training_LesionGroupings.csv", index=False)
for sp, n, off in (("Validation", 60, 900000), ("Test", 90, 950000)):
    rr = []
    for k in range(n):
        lab = rng.choice(7); iid = f"ISIC_{off + k:07d}"
        img(d8 / f"ISIC2018_Task3_{sp}_Input" / f"{iid}.jpg", lab)
        rr.append({"image": iid, **{c18[j]: float(j == lab) for j in range(7)}})
    pd.DataFrame(rr).to_csv(d8 / f"ISIC2018_Task3_{sp}_GroundTruth.csv", index=False)
print("fake data at", root)
