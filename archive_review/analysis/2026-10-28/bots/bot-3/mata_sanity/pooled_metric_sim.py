"""Sanity: their competition_like_f1 pools patch-pixel polygons from all val patches.
Imports THEIR code read-only. GT = target object of each val patch only (ellipse at patch centre,
axes = obj_w/h_px * 256 / crop_size, see build_multiclass_dataset.py centring + dataset resize).
Predictors use NO image information. 'pooled' = their metric as is; 'per-patch' = same polygons
shifted by patch index*1000 px so matching can only happen inside one patch."""
import sys, numpy as np, pandas as pd
from shapely.geometry import Point, mapping
from shapely import affinity
REPO = "/workspace/koz3_shared/competitors/MataNerdy-Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab"
sys.path.insert(0, REPO)
from utils.polygon_metrics import competition_like_f1
v = pd.read_csv(f"{REPO}/splits/archaeology_5class_research_split_v1/val_split.csv")
v = v[v.modality.isin(["Li", "Ae", "SpOr"])].reset_index(drop=True)
rng = np.random.default_rng(0)
def ell(cx, cy, a, b):
    return affinity.scale(Point(cx, cy).buffer(1.0, 32), max(a, 1.5), max(b, 1.5))
def fc(polys):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(p), "properties": {"class_name": c, "confidence": 1.0}} for p, c in polys]}
def gt(shift):
    out = []
    for i, r in v.iterrows():
        s = 256.0 / r.crop_size
        out.append((ell(128 + shift * i, 128, r.obj_w_px * s / 2, r.obj_h_px * s / 2), r.class_name))
    return out
prior = v.class_name.value_counts(normalize=True)
blind_cls = rng.choice(prior.index.values, size=len(v), p=prior.values)
def blind(shift, cls):
    return [(ell(128 + shift * i, 128, 4, 4), c) for i, c in enumerate(cls)]
res = {}
for name, shift in (("pooled(their metric)", 0), ("per-patch", 1000)):
    G = fc(gt(shift))
    res[(name, "image-blind prior-sampled class, r=4px disc at centre")] = competition_like_f1(fc(blind(shift, blind_cls)), G)[0]
    res[(name, "image-blind all 'kurgany_povrezhdennye' disc")] = competition_like_f1(fc(blind(shift, ["kurgany_povrezhdennye"] * len(v))), G)[0]
for k, s in res.items():
    print(f"{k[0]:22s} | {k[1]:55s} | wF1={s:.4f}")
print("n_val", len(v), v.class_name.value_counts().to_dict())
