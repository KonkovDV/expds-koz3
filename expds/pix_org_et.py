"""Площадные аномалии электроразведки по схеме организаторов, режим градиентного бустинга. По умолчанию выключено.

Растр EtGr (карта-срез или профиль) → рабочая сетка 512 px → шкала jet → z (медиана/MAD) → кандидаты-изолинии
→ 16 признаков (+5 у профилей) → LightGBM (вместо HistGradientBoosting: в среде инференса нет sklearn) →
NMS, на картах разрезание слившихся пятен на вложенных кандидатов. Модели «карта» и «профиль» раздельные,
порог каждой — внутренний LOSO по участкам обучения (meta). Флаг [geophys.pixgbm] org_etgr (по умолчанию выкл.).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

WORK_SIDE = 512
PALETTE_MAX_DIST = 110.0
SMOOTH = 2
BG_SIGMAS = [0, 20, 10]
LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5]
MIN_CAND_PX = 30
NMS_IOU = 0.2
LABEL_IOU = 0.1
SPLIT_RATIO = 0.5
COVERAGE = 0.5
MISSING = -9.0
THRS = np.round(np.arange(0.05, 0.91, 0.05), 2)
N_BASE, N_PROF = 16, 21
KINDS = ("map", "prof")
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=20, min_sum_hessian_in_leaf=1e-3,
           lambda_l2=0.0, max_bin=255, feature_fraction=1.0, bagging_fraction=1.0, verbose=-1, deterministic=True)
N_ITER = 300


def kind_of(path: Path) -> str:
    return "prof" if "профил" in Path(path).stem.lower() else "map"


def _stem(path: Path) -> str:
    s = Path(path).stem
    return s.split("_", 1)[1] if s[:2].isdigit() and "_" in s else s


def boundary_for(path: Path, boundaries) -> Path | None:
    want = f"{_stem(path)}_границы"
    for b in boundaries or []:
        if Path(b).stem == want:
            return Path(b)
    return None


# ---------------------------------------------------------------- растр

def load(path: Path, boundary: Path | None) -> dict | None:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from scipy import ndimage

    from .crs import guess_geojson_crs, to_crs
    from .pix_clean import jet_scalar
    with rasterio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / WORK_SIDE)
        H, W = int(round(src.height / scale)), int(round(src.width / scale))
        rgb = src.read([1, 2, 3], out_shape=(3, H, W), resampling=Resampling.nearest)
        tr = src.transform * src.transform.scale(src.width / W, src.height / H)
        crs = src.crs.to_string() if src.crs else None
    t, dist = jet_scalar(rgb)
    if boundary is not None:
        from shapely.geometry import shape
        d = json.loads(Path(boundary).read_text(encoding="utf-8-sig"))
        scrs = guess_geojson_crs(d, crs) or crs
        gs = [shape(f["geometry"]).buffer(0) for f in d.get("features", []) if f.get("geometry")]
        gs = [to_crs(g, scrs, crs) if scrs != crs else g for g in gs if not g.is_empty]
        bmask = rasterize([(g, 1) for g in gs], out_shape=(H, W), transform=tr, fill=0, dtype=np.uint8) > 0 \
            if gs else np.zeros((H, W), bool)
    else:
        bmask = np.ones((H, W), bool)
    mask = bmask & (dist < PALETTE_MAX_DIST)
    if mask.sum() < MIN_CAND_PX:
        return None
    if not mask.all():
        iy, ix = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        t = t[iy, ix]
    v = t[mask]
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-6
    return {"path": Path(path), "kind": kind_of(path), "tr": tr, "crs": crs, "z": ((t - med) / mad).astype(np.float32),
            "mask": mask}


def _shape(reg: np.ndarray):
    yy, xx = np.nonzero(reg)
    cy, cx = yy.mean(), xx.mean()
    myy = ((yy - cy) ** 2).mean() + 1 / 12; mxx = ((xx - cx) ** 2).mean() + 1 / 12; mxy = ((yy - cy) * (xx - cx)).mean()
    t = math.sqrt(((myy - mxx) / 2) ** 2 + mxy ** 2)
    l1, l2 = (myy + mxx) / 2 + t, max((myy + mxx) / 2 - t, 0.0)
    return math.sqrt(1 - l2 / l1) if l1 > 0 else 0.0, 4 * math.sqrt(l1)


def candidates(d: dict):
    """Кандидаты-изолинии и 16 признаков (как ячейка 17 ноутбука 02)."""
    from rasterio.features import shapes
    from scipy import ndimage
    from shapely.geometry import shape as shp
    m = d["mask"]
    edt = ndimage.distance_transform_edt(m)
    edt_n = edt / (edt.max() + 1e-6)
    valid = int(m.sum())
    zs = ndimage.gaussian_filter(d["z"], SMOOTH)
    z_std = float(np.std(zs[m])) + 1e-9
    px = abs(d["tr"].a)
    polys, feats = [], []
    st = np.ones((3, 3), bool)
    prof = float(d["kind"] == "prof")
    for bg in BG_SIGMAS:
        z = zs - ndimage.gaussian_filter(d["z"], bg) if bg else zs
        z = z / (float(np.std(z[m])) + 1e-9)
        for sign in (1, -1):
            for lv in LEVELS:
                lab, n = ndimage.label((sign * z > lv) & m, structure=st)
                if not n:
                    continue
                cnt = np.bincount(lab.ravel())
                for li, sl in enumerate(ndimage.find_objects(lab), start=1):
                    if sl is None or cnt[li] < MIN_CAND_PX or cnt[li] > 0.5 * valid:
                        continue
                    pad = 6
                    y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
                    y1, x1 = min(sl[0].stop + pad, z.shape[0]), min(sl[1].stop + pad, z.shape[1])
                    reg = lab[y0:y1, x0:x1] == li
                    ring = ndimage.binary_dilation(reg, iterations=5) & ~reg & m[y0:y1, x0:x1]
                    zin = sign * z[y0:y1, x0:x1][reg]
                    zring = sign * z[y0:y1, x0:x1][ring] if ring.any() else np.array([lv])
                    ctr = d["tr"] * d["tr"].translation(x0, y0)
                    best = None
                    for gj, _ in shapes(reg.astype(np.uint8), mask=reg, transform=ctr, connectivity=8):
                        g = shp(gj)
                        if best is None or g.area > best.area:
                            best = g
                    if best is None:
                        continue
                    poly = best.buffer(0)
                    if poly.is_empty:
                        continue
                    area = int(cnt[li])
                    ecc, major = _shape(reg)
                    hull = poly.convex_hull.area
                    per = poly.length / px
                    e = edt_n[y0:y1, x0:x1][reg]
                    polys.append(poly)
                    feats.append([sign, bg, lv, math.log(area), area / valid, zin.mean(), zin.max(),
                                  zin.mean() - zring.mean(), (sign * zs[y0:y1, x0:x1][reg]).mean() / z_std,
                                  min(poly.area / hull, 1.0) if hull > 0 else 1.0, ecc,
                                  4 * math.pi * area / max(per, 1) ** 2, e.mean(), e.min(), major / max(z.shape), prof])
                del lab
    return polys, np.array(feats, np.float32).reshape(-1, N_BASE)


def nested_children(polys, X):
    from shapely.strtree import STRtree
    kids = [[] for _ in polys]
    if not polys:
        return kids
    rp = [p.representative_point() for p in polys]
    tree = STRtree(rp)
    for i, p in enumerate(polys):
        for j in tree.query(p):
            j = int(j)
            if (j != i and X[j, 0] == X[i, 0] and X[j, 1] == X[i, 1] and abs(X[j, 2] - X[i, 2] - 0.5) < 1e-6
                    and p.contains(rp[j])):
                kids[i].append(j)
    return kids


def _mean_under(poly, e: dict, sign: float) -> float:
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds
    H, W = e["z"].shape
    win = from_bounds(*poly.bounds, transform=e["tr"]).round_offsets().round_lengths()
    r0, c0 = max(int(win.row_off), 0), max(int(win.col_off), 0)
    r1, c1 = min(int(win.row_off + win.height) + 1, H), min(int(win.col_off + win.width) + 1, W)
    if r1 <= r0 or c1 <= c0:
        return np.nan
    tr = e["tr"] * e["tr"].translation(c0, r0)
    m = rasterize([(poly, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, all_touched=True).astype(bool)
    m &= e["mask"][r0:r1, c0:c1]
    return float(sign * e["z"][r0:r1, c0:c1][m].mean()) if m.sum() >= 3 else np.nan


def profile_extra(d: dict, polys, X, maps: list[dict]) -> np.ndarray:
    import rasterio
    out = np.full((len(X), N_PROF - N_BASE), MISSING, np.float32)
    m = d["mask"]
    surf = None
    if m.shape[1] > m.shape[0]:
        cols = np.where(m.any(0))[0]
        top = np.array([np.argmax(m[:, c]) for c in cols])
        surf = rasterio.transform.xy(d["tr"], top, cols)
    for i, p in enumerate(polys):
        if surf is not None:
            ys = np.interp(p.centroid.x, np.asarray(surf[0]), np.asarray(surf[1]))
            out[i, :2] = [ys - p.bounds[3], ys - p.bounds[1]]
        v = np.array([_mean_under(p, e, X[i, 0]) for e in maps])
        v = v[~np.isnan(v)]
        if len(v):
            out[i, 2:] = [v.mean(), v.max(), (v > 0.5 * X[i, 2]).mean()]
    return out


def region_cands(rasters, boundaries) -> list[dict]:
    """Все растры EtGr участка: кандидаты, признаки, дети (карты), доп. признаки (профили)."""
    ds = []
    for p in rasters:
        d = load(p, boundary_for(p, boundaries))
        if d is not None:
            ds.append(d)
    maps = [d for d in ds if d["kind"] == "map"]
    out = []
    for d in ds:
        polys, X = candidates(d)
        c = {"path": d["path"], "kind": d["kind"], "crs": d["crs"], "polys": polys}
        if d["kind"] == "map":
            c["X"], c["kids"] = X, nested_children(polys, X)
        else:
            c["X"], c["kids"] = np.hstack([X, profile_extra(d, polys, X, maps)]), [[] for _ in polys]
        out.append(c)
    for d in ds:
        d.pop("z", None)
    return out


# ---------------------------------------------------------------- метки и метрика платформы

def _largest(g):
    return max(g.geoms, key=lambda p: p.area) if g.geom_type == "MultiPolygon" else g


def is_hit(p, g) -> tuple[bool, float]:
    m = min(p.area, g.area)
    if m <= 0:
        return False, 0.0
    cov = p.intersection(g).area / m
    pt = p.representative_point()
    return cov >= COVERAGE and (g.contains(pt) or g.boundary.distance(pt) < 1e-10), cov


def label(polys, gts) -> np.ndarray:
    """Положительный: засчитал бы объект по правилу платформы и IoU ≥ LABEL_IOU (с каким-нибудь объектом)."""
    from shapely.strtree import STRtree
    y = np.zeros(len(polys), bool)
    if not polys or not gts:
        return y
    gl = [_largest(g) for g in gts]
    tree = STRtree(gts)
    for i, p in enumerate(polys):
        hit, iou = False, 0.0
        pl = _largest(p)
        for j in tree.query(p):
            j = int(j)
            iou = max(iou, p.intersection(gts[j]).area / max(p.union(gts[j]).area, 1e-12))
            hit = hit or is_hit(pl, gl[j])[0]
        y[i] = hit and iou >= LABEL_IOU
    return y


def platform_counts(preds, confs, gts) -> tuple[int, int, int]:
    from shapely.strtree import STRtree
    preds = [_largest(p) for p in preds]
    gts = [_largest(g) for g in gts]
    if not preds or not gts:
        return 0, len(preds), len(gts)
    tree = STRtree(gts)
    used, tp = set(), 0
    for i in sorted(range(len(preds)), key=lambda i: -confs[i]):
        best, bc = -1, -np.inf
        for j in sorted(int(x) for x in tree.query(preds[i])):
            if j in used:
                continue
            h, cov = is_hit(preds[i], gts[j])
            if h and cov > bc + 1e-9:
                best, bc = j, cov
        if best >= 0:
            used.add(best); tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def f1_platform(tp, fp, fn) -> float:
    return 2 * tp / max(2 * tp + 0.5 * fp + fn, 1e-12)


# ---------------------------------------------------------------- модель

def fit(X, y, seed: int = 0, threads: int = 2):
    """LightGBM вместо HGB(max_iter 300, lr 0.05, balanced); ранняя остановка как у HGB early_stopping='auto'."""
    import lightgbm as lgb
    y = np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    w = np.where(y, len(y) / (2 * y.sum()), len(y) / (2 * (~y).sum())).astype(np.float32)
    p = dict(LGB, seed=seed, num_threads=threads)
    if len(y) > 10000:
        rng = np.random.default_rng(seed)
        va = np.zeros(len(y), bool)
        for cl in (True, False):
            idx = np.where(y == cl)[0]
            va[rng.choice(idx, max(1, int(round(0.1 * len(idx)))), replace=False)] = True
        dtr = lgb.Dataset(X[~va], label=y[~va].astype(np.float32), weight=w[~va])
        dva = lgb.Dataset(X[va], label=y[va].astype(np.float32), weight=w[va], reference=dtr)
        return lgb.train(dict(p, metric="binary_logloss"), dtr, num_boost_round=N_ITER, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(10, verbose=False)])
    return lgb.train(p, lgb.Dataset(X, label=y.astype(np.float32), weight=w), num_boost_round=N_ITER)


def predict(model, X) -> np.ndarray:
    if model is None or not len(X):
        return np.zeros(len(X))
    return model.predict(X, num_threads=1, num_iteration=model.best_iteration or None)


def nms(polys, scores, thr) -> list[int]:
    keep = []
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] < thr:
            break
        p = polys[i]
        ok = True
        for k in keep:
            inter = p.intersection(polys[k]).area
            if inter / max(p.union(polys[k]).area, 1e-12) >= NMS_IOU or inter >= 0.5 * min(p.area, polys[k].area):
                ok = False
                break
        if ok:
            keep.append(int(i))
    return keep


def predict_polys(c: dict, scores, thr) -> list[tuple[int, float]]:
    keep = nms(c["polys"], scores, thr)
    if c["kind"] == "map":
        out = []
        for i in keep:
            kids = [j for j in c["kids"][i] if scores[j] >= SPLIT_RATIO * thr]
            out += kids if len(kids) >= 2 else [i]
        keep = sorted(set(out))
    return [(i, float(scores[i])) for i in keep]


def pooled_f1(cs, scores, gts, thr) -> float:
    tot = np.zeros(3)
    for c, s, g in zip(cs, scores, gts):
        kp = predict_polys(c, s, thr)
        tot += platform_counts([c["polys"][i] for i, _ in kp], [v for _, v in kp], g)
    return f1_platform(*tot)


def pick_threshold(cs, scores, gts) -> float:
    """Порог по F1 платформы на пуле растров (как evaluate/THRS у организаторов)."""
    best = max(((pooled_f1(cs, scores, gts, t), -t) for t in THRS))
    return float(-best[1])


# ---------------------------------------------------------------- инференс

def to_feats(c: dict, scores, thr: float, reg_name: str) -> list[dict]:
    from .crs import to_crs
    out = []
    for i, s in predict_polys(c, scores, thr):
        g = c["polys"][i]
        out.append({"geom": to_crs(g, c["crs"], "EPSG:3857"), "class_name": "object_poly",
                    "confidence": float(np.clip(s, 0, 1)), "region_name": reg_name, "markup_type": "etgr",
                    "source": c["path"].name, "original_crs": c["crs"], "threshold_group": "etgr_org"})
    return out


def load_models(model_dir: Path):
    import lightgbm as lgb
    md = Path(model_dir)
    m = {k: (lgb.Booster(model_file=str(md / f"org_etgr_{k}.txt")) if (md / f"org_etgr_{k}.txt").exists() else None)
         for k in KINDS}
    meta = json.loads((md / "org_etgr_meta.json").read_text()) if (md / "org_etgr_meta.json").exists() else {}
    return m, meta


def region_feats(reg, cfg: dict, model_dir, status: dict | None = None) -> list[dict]:
    status = {} if status is None else status
    models, meta = load_models(model_dir)
    if not any(models.values()):
        return []
    rasters = list(reg.rasters.get("etgr", []) or [])
    if not rasters:
        return []
    feats = []
    for c in region_cands(rasters, reg.boundaries):
        mdl = models.get(c["kind"])
        if mdl is None:
            continue
        feats += to_feats(c, predict(mdl, c["X"]), float(meta.get("thr", {}).get(c["kind"], 0.5)), reg.name)
    done = status.setdefault("done", {})
    for f in feats:
        done.setdefault(("object_poly", "etgr"), set()).add(f["source"])
    return feats
