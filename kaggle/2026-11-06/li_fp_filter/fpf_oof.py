#!/usr/bin/env python3
"""OOF фильтра ложных курганов лидара (PREREG kaggle/2026-11-06/li_fp_filter/PREREG.md).

    python fpf_oof.py OUT_DIR [--relief /workspace/c12/relief12] [--oof /workspace/koz3/w2/oof/cnxin] [--final MODEL.txt]

- Детекции — OOF курганов c5-in ядер (`oof_cnx_c5-in_f*.geojson`), фолды — region_fold.json ядер.
- Признаки — expds.lidar_fpf.features по первым 5 каналам рельефа (= вход c5) и детекциям участка.
- Метка — центр детекции внутри полигона эталонного кургана лидара (gt_train_v2, markup li) того же участка.
- LightGBM (параметры PREREG) для фолда f учится на остальных фолдах; новая уверенность — вероятность.
- Выход: OUT/oof_cnx_c5rs-in_f{k}.geojson + region_fold.json, OUT/fpf_report.json (AUC уверенности и GBM),
  с --final — модель на всех фолдах (для сдачи).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
GT = Path("/workspace/koz3/w5data/gt_train_v2.geojson")
# PREREG: num_leaves 15, lr 0,05, 300 деревьев, min_data_in_leaf 30, feature_fraction 0,8, bagging 0,8 / 1, seed 0
PARAMS = dict(objective="binary", num_leaves=15, learning_rate=0.05, min_data_in_leaf=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, seed=0, verbose=-1, num_threads=4)
ROUNDS = 300


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--relief", type=Path, default=Path("/workspace/c12/relief12"))
    ap.add_argument("--oof", type=Path, default=Path("/workspace/koz3/w2/oof/cnxin"))
    ap.add_argument("--final", type=Path, default=None)
    a = ap.parse_args()
    import lightgbm as lgb
    from pyproj import Transformer
    from shapely import STRtree, points
    from shapely.geometry import shape

    from expds import lidar_fpf as F
    a.out.mkdir(parents=True, exist_ok=True)
    rf_p = sorted(glob.glob(str(a.oof / "**" / "region_fold.json"), recursive=True))[0]
    rf = json.loads(Path(rf_p).read_text(encoding="utf-8"))
    fold_of = {r: int(f) for r, f in rf["region_fold"].items()}
    dets: dict = {}
    for p in sorted(glob.glob(str(a.oof / "**" / "oof_cnx_c5-in_f*.geojson"), recursive=True)):
        for ft in json.loads(Path(p).read_text(encoding="utf-8"))["features"]:
            q = ft["properties"]
            if q["class_name"] in KUR:
                dets.setdefault(q["region_name"], []).append(ft)
    gt: dict = {}
    for ft in json.loads(GT.read_text(encoding="utf-8"))["features"]:
        q = ft["properties"]
        if q.get("markup_type") == "li" and q.get("class_name") in KUR:
            gt.setdefault(q["region_name"], []).append(shape(ft["geometry"]))
    Xs, ys, folds, refs = [], [], [], []
    for site, fts in sorted(dets.items()):
        geo = json.loads((a.relief / f"{site}.json").read_text(encoding="utf-8"))
        u8 = np.load(a.relief / f"{site}.npy", mmap_mode="r")
        tf = Transformer.from_crs("EPSG:3857", geo["crs"], always_xy=True).transform
        cen = [shape(ft["geometry"]).centroid for ft in fts]
        d = [(*tf(c.x, c.y), KUR.index(ft["properties"]["class_name"]), float(ft["properties"]["confidence"]))
             for c, ft in zip(cen, fts)]
        Xs.append(F.features(np.asarray(u8[:5]), float(geo["left"]), float(geo["top"]), float(geo["step"]), d))
        g = gt.get(site, [])
        if g:
            tree = STRtree(g)
            hit = np.zeros(len(cen), bool)
            pi, _ = tree.query(points([(c.x, c.y) for c in cen]), predicate="within")
            hit[np.unique(pi)] = True
        else:
            hit = np.zeros(len(cen), bool)
        ys.append(hit.astype(np.int8))
        folds += [fold_of[site]] * len(fts)
        refs += fts
        print(site, len(fts), int(hit.sum()), flush=True)
    X, y, fo = np.concatenate(Xs), np.concatenate(ys), np.array(folds)
    p_oof = np.zeros(len(y))
    for k in sorted(set(fo)):
        m = lgb.train(PARAMS, lgb.Dataset(X[fo != k], label=y[fo != k]), num_boost_round=ROUNDS)
        p_oof[fo == k] = m.predict(X[fo == k])
    rep = {"n": int(len(y)), "pos": int(y.sum()), "auc_conf": auc(y, X[:, 0]), "auc_gbm": auc(y, p_oof),
           "features": list(F.NAMES)}
    rep["stop_rule_gbm_worse"] = rep["auc_gbm"] <= rep["auc_conf"]
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    by_fold: dict = {}
    for ft, s, k in zip(refs, p_oof, fo):
        g = {"type": "Feature", "geometry": ft["geometry"], "properties": {**ft["properties"], "confidence": round(float(s), 5)}}
        by_fold.setdefault(int(k), []).append(g)
    for k, fs in by_fold.items():
        (a.out / f"oof_cnx_c5rs-in_f{k}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": fs},
                                                                        ensure_ascii=False), encoding="utf-8")
    (a.out / "region_fold.json").write_text(json.dumps(rf, ensure_ascii=False), encoding="utf-8")
    (a.out / "fpf_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.final is not None:
        m = lgb.train(PARAMS, lgb.Dataset(X, label=y), num_boost_round=ROUNDS)
        m.save_model(str(a.final))
        print("final model →", a.final, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
