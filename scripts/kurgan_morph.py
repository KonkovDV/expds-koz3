"""Классификатор подкласса курганов «повреждённый / целый» по морфометрии эталонных полигонов.

Честная проверка: leave-family-out, фолды = expds.metric.family_folds(..., site_family, k=10),
как в «честном балле» табло. Метрика — AUC (повреждённый = 1) по каждому фолду и по всем OOF.

Ограничение среды (28.10.2026): на машине ботов нет ЦМР/LAS реальных участков
(/workspace/data/train-expds-3 содержит только geojson/json). Поэтому высотные признаки
(h, h/D, плоская вершина, яма грабителя, склон, LRM) здесь НЕ считаются — только плановые
признаки формы полигона и контекст соседей. Скрипт готов принять растровые признаки
(--raster-features CSV с колонкой gt_id), когда ЦМР появятся.

Запуск (CPU, 2 потока):
  OMP_NUM_THREADS=2 nice -n 10 /workspace/rerank/venv/bin/python scripts/kurgan_morph.py \
     --gt /workspace/rerank/gt_train_v2.geojson --data /workspace/data/train-expds-3 \
     --out /workspace/koz3_shared/work/kurgan_morph.json
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expds.metric import family_folds          # noqa: E402
from expds.naming import site_family           # noqa: E402

DMG, INT = "kurgany_povrezhdennye", "kurgany_tselye"


def utm_transformers(data: Path, regions):
    from pyproj import Transformer
    out = {}
    for r in regions:
        p = data / r / "UTM.json"
        epsg = None
        if p.exists():
            crs = json.loads(p.read_text(encoding="utf-8")).get("crs", "")
            epsg = crs.rsplit(":", 1)[-1] if crs else None
        out[r] = Transformer.from_crs("EPSG:3857", f"EPSG:{epsg}", always_xy=True) if epsg else None
    return out


def shape_features(poly):
    import shapely
    a = poly.area
    per = poly.length
    hull = poly.convex_hull
    mrr = poly.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    e = [math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2)]
    L, S = max(e), max(min(e), 1e-6)
    i = 0 if e[0] >= e[1] else 1
    ang = math.degrees(math.atan2(ys[i + 1] - ys[i], xs[i + 1] - xs[i])) % 180.0
    c = poly.centroid
    pts = np.asarray(poly.exterior.coords)[:-1]
    rad = np.hypot(pts[:, 0] - c.x, pts[:, 1] - c.y) if len(pts) else np.array([0.0])
    mic = shapely.maximum_inscribed_circle(poly, tolerance=0.05) if hasattr(shapely, "maximum_inscribed_circle") else None
    r_in = mic.length if mic is not None else float("nan")
    return {
        "area": a, "log_area": math.log1p(a), "perimeter": per,
        "eq_diam": 2 * math.sqrt(a / math.pi),
        "compactness": 4 * math.pi * a / per ** 2 if per > 0 else 0.0,
        "convexity": a / hull.area if hull.area > 0 else 1.0,
        "elong": L / S, "mrr_fill": a / (L * S),
        "rad_cv": float(rad.std() / rad.mean()) if rad.mean() > 0 else 0.0,
        "rad_minmax": float(rad.min() / rad.max()) if rad.max() > 0 else 1.0,
        "inscr_ratio": (2 * r_in) / (2 * math.sqrt(a / math.pi)) if a > 0 else 0.0,
        "orient_sin2": math.sin(math.radians(2 * ang)), "orient_cos2": math.cos(math.radians(2 * ang)),
        "n_vert": len(pts),
    }, (c.x, c.y), ang


def load_kurgans(gt_path: Path, data: Path):
    import shapely
    from shapely.geometry import shape
    from shapely.ops import transform
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    feats = [f for f in gt["features"] if f["properties"]["class_name"] in (DMG, INT)
             and f["geometry"]["type"] == "Polygon"]
    regions = sorted({f["properties"]["region_name"] for f in feats})
    tr = utm_transformers(data, regions)
    rows, xy, ang = [], [], []
    for f in feats:
        p = f["properties"]
        g = shape(f["geometry"])
        t = tr.get(p["region_name"])
        if t is not None:
            g = transform(t.transform, g)
        else:   # запасной путь: масштаб Меркатора по широте
            lat = math.degrees(2 * math.atan(math.exp(g.centroid.y / 6378137.0)) - math.pi / 2)
            k = math.cos(math.radians(lat))
            g = shapely.affinity.scale(g, k, k, origin="centroid")
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty or g.area <= 0:
            continue
        if g.geom_type != "Polygon":
            g = max(g.geoms, key=lambda q: q.area)
        sf, c, a = shape_features(g)
        sf.update({"mk_li": p["markup_type"] == "li", "mk_ae": p["markup_type"] == "ae",
                   "mk_sp": p["markup_type"] == "sp_or"})
        rows.append({"region": p["region_name"], "markup": p["markup_type"], "y": int(p["class_name"] == DMG),
                     "gt_id": f"{p['region_name']}|{p['markup_type']}|{p['fid']}", **sf})
        xy.append(c); ang.append(a)
    return rows, np.asarray(xy), np.asarray(ang)


def context_features(rows, xy, ang):
    """Соседи без меток: плотность, расстояние до ближайшего, относительный размер, сонаправленность."""
    from scipy.spatial import cKDTree
    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r["region"], r["markup"]), []).append(i)
    for idx in by.values():
        idx = np.asarray(idx)
        P = xy[idx]
        tree = cKDTree(P)
        area = np.array([rows[i]["area"] for i in idx])
        med = float(np.median(area))
        d, _ = tree.query(P, k=min(2, len(idx)))
        nn = d[:, 1] if d.ndim == 2 and d.shape[1] > 1 else np.full(len(idx), 1e4)
        for j, i in enumerate(idx):
            r = rows[i]
            r["nn_dist"] = float(nn[j])
            r["nn_dist_rel"] = float(nn[j] / max(r["eq_diam"], 0.1))
            for rad in (25, 50, 100):
                r[f"n_{rad}"] = len(tree.query_ball_point(P[j], rad)) - 1
            nb = [k for k in tree.query_ball_point(P[j], 100) if k != j]
            r["area_rel_region"] = r["area"] / med if med > 0 else 1.0
            r["area_rel_nb"] = r["area"] / float(np.median(area[nb])) if nb else 1.0
            if nb:   # согласованность ориентации с соседями (распашка тянет в одну сторону)
                d2 = np.radians(2 * (ang[idx[nb]] - ang[i]))
                r["orient_agree"] = float(np.mean(np.cos(d2)))
            else:
                r["orient_agree"] = 0.0
            r["region_n"] = len(idx)


SHAPE = ["log_area", "eq_diam", "compactness", "convexity", "elong", "mrr_fill", "rad_cv", "rad_minmax",
         "inscr_ratio", "orient_sin2", "orient_cos2"]
MARKUP = ["mk_li", "mk_ae", "mk_sp"]
CONTEXT = ["nn_dist", "nn_dist_rel", "n_25", "n_50", "n_100", "area_rel_region", "area_rel_nb",
           "orient_agree", "region_n"]
STYLE = ["n_vert"]


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def lofo(rows, cols, folds, seed=0):
    import lightgbm as lgb
    X = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([r["y"] for r in rows])
    reg = np.array([r["region"] for r in rows])
    oof = np.full(len(rows), np.nan)
    per = []
    for k, fold in enumerate(folds):
        te = np.isin(reg, list(fold))
        if te.sum() == 0:
            continue
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                               num_threads=2, random_state=seed, verbose=-1)
        m.fit(X[~te], y[~te])
        oof[te] = m.predict_proba(X[te])[:, 1]
        per.append({"fold": k, "n": int(te.sum()), "n_dmg": int(y[te].sum()), "n_int": int((1 - y[te]).sum()),
                    "families": sorted({site_family(r) for r in fold}), "auc": auc(y[te], oof[te])})
    ok = ~np.isnan(oof)
    per_auc = [p["auc"] for p in per if not math.isnan(p["auc"])]
    # AUC внутри участка (снимает разницу априорных долей между участками)
    within = []
    for r in sorted(set(reg)):
        mk = (reg == r) & ok
        if len(set(y[mk])) == 2 and mk.sum() >= 10:
            within.append((auc(y[mk], oof[mk]), int(mk.sum())))
    w_auc = float(np.average([a for a, _ in within], weights=[n for _, n in within])) if within else float("nan")
    return {"cols": cols, "auc_pooled": auc(y[ok], oof[ok]), "auc_fold_mean": float(np.mean(per_auc)),
            "auc_fold_median": float(np.median(per_auc)), "auc_within_region_wmean": w_auc,
            "n_regions_within": len(within), "folds": per}, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--raster-features", type=Path, default=None)
    a = ap.parse_args()
    rows, xy, ang = load_kurgans(a.gt, a.data)
    context_features(rows, xy, ang)
    extra = []
    if a.raster_features:
        import csv
        rf = {r["gt_id"]: r for r in csv.DictReader(open(a.raster_features, encoding="utf-8"))}
        extra = [c for c in next(iter(rf.values())) if c != "gt_id"]
        for r in rows:
            for c in extra:
                r[c] = float(rf.get(r["gt_id"], {}).get(c, "nan"))
    regions = sorted({r["region"] for r in rows})
    folds = family_folds(regions, site_family, a.k)
    y = np.array([r["y"] for r in rows])
    reg = np.array([r["region"] for r in rows])
    # доля повреждённых по участкам: насколько подкласс «участковый»
    pure = 0
    for r in regions:
        m = reg == r
        if y[m].min() == y[m].max():
            pure += int(m.sum())
    res = {"n": len(rows), "n_dmg": int(y.sum()), "n_int": int((1 - y).sum()), "n_regions": len(regions),
           "n_families": len({site_family(r) for r in regions}), "k": a.k,
           "share_in_single_subclass_regions": pure / len(rows), "models": {}}
    sets = {"markup_only": MARKUP, "shape": SHAPE, "shape+markup": SHAPE + MARKUP,
            "shape+markup+context": SHAPE + MARKUP + CONTEXT,
            "all+style(n_vert)": SHAPE + MARKUP + CONTEXT + STYLE}
    if extra:
        sets["all+raster"] = SHAPE + MARKUP + CONTEXT + extra
    oofs = {}
    for name, cols in sets.items():
        res["models"][name], oofs[name] = lofo(rows, cols, folds)
        m = res["models"][name]
        print(f"{name:24s} pooled {m['auc_pooled']:.4f}  fold-mean {m['auc_fold_mean']:.4f}  "
              f"within-region {m['auc_within_region_wmean']:.4f}", flush=True)
    # одиночные признаки (знак не подбирается: AUC и 1−AUC, берём max — только для описания)
    res["single_feature_auc"] = {c: auc(y, np.array([float(r[c]) for r in rows]))
                                 for c in SHAPE + CONTEXT + STYLE}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(a.out.with_suffix(".oof.npz"), gt_id=np.array([r["gt_id"] for r in rows]), y=y,
                        **{k.replace("+", "_").replace("(", "").replace(")", ""): v for k, v in oofs.items()})
    print("written", a.out)


if __name__ == "__main__":
    main()
