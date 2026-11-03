"""GEO-ORG1: кандидаты MagGr по бейзлайну организаторов (ноутбук 01_baseline_maggr_anomalies) внутри clean-room PIXGBM.

Источник идей и параметров — материалы организаторов (ноутбуки org_baselines, разрешено); EN-код не используется.
Отличия от ноутбука: без skimage/sklearn/cv2 (нет в окружении инференса) — экстремумы через maximum_filter, свойства
регионов через моменты, контур через rasterio.features.shapes, классификатор LightGBM (300 итераций, lr 0,05, баланс).
Флаг: [geophys.pixgbm] org_maggr = false (по умолчанию выкл.), docs/PREREG_w4_pix_clean.md (дополнение 29.10).

Точки: экстремумы обоих знаков на gauss(1,5 px) сетки 0,1 м, 16 признаков (диполь ±1,5 м, ранги, DoG, Гессиан), NMS 1,5 м.
Полигоны: уровни 0,5–2,5 std × фоны {0, 10 м, 5 м} × знак на сетке 0,2 м, 15 признаков региона, NMS (IoU 0,2 / 0,5 меньшего),
слитое пятно → ≥ 2 вложенных детей.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RES = 0.1
PT_SMOOTH, PT_MIN_DIST, PT_MIN_Z, PT_LABEL_M, PT_NMS_M = 1.5, 5, 1.0, 0.5, 1.5
POLY_K = 2                       # сетка полигонов = POLY_K·RES (у организаторов int(round(0,25/0,1)) = 2)
SMOOTH, BG_SIGMAS_M, LEVELS = 2, (0.0, 10.0, 5.0), (0.5, 1.0, 1.5, 2.0, 2.5)
MIN_CAND_M2, NMS_IOU, LABEL_IOU, SPLIT_RATIO = 4.0, 0.2, 0.1, 0.5
N_PT, N_POLY = 16, 15
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=20, is_unbalance=True,
           verbose=-1, deterministic=True, force_row_wise=True)
N_ITER = 300


# ---------------------------------------------------------------- растр

def pick_raster(reg) -> Path | None:
    files = sorted(reg.rasters.get("maggr", []) or [], key=lambda f: f.name)
    gray = [f for f in files if "сер" in f.stem.lower()]
    return (gray or files or [None])[0]


def _survey_mask(v: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    lab, _ = ndimage.label((v == 0) | (v == 255))
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, edge[edge > 0])
    return ndimage.binary_erosion(~bg, iterations=2)


def load(path: Path, boundary_3857=None) -> dict:
    """Серая карта на сетке 0,1 м: z = (v − med)/MAD, маска съёмки, вне маски — ближайшее валидное."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from scipy import ndimage

    with rasterio.open(path) as src:
        H = max(1, int(round(src.height * abs(src.res[1]) / RES)))
        W = max(1, int(round(src.width * abs(src.res[0]) / RES)))
        v = src.read(1, out_shape=(H, W), resampling=Resampling.nearest)
        tr = src.transform * src.transform.scale(src.width / W, src.height / H)
        crs = src.crs.to_string() if src.crs else None
    mask = _survey_mask(v)
    if boundary_3857 is not None and crs:
        from .crs import to_crs
        g = to_crs(boundary_3857, "EPSG:3857", crs)
        if not g.is_empty:
            mask &= rasterize([(g, 1)], out_shape=(H, W), transform=tr, fill=0, dtype=np.uint8) > 0
    if mask.sum() < 100:
        return {"z": None, "path": path}
    if mask.all():
        vf = v.astype(np.float32)
    else:                           # без массива расстояний: на 213 (≈ 1e8 пикс) иначе > 5 ГБ
        iy, ix = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        vf = v[iy, ix].astype(np.float32)
        del iy, ix
    del v
    med = float(np.median(vf[mask]))
    mad = 1.4826 * float(np.median(np.abs(vf[mask] - med))) + 1e-6
    vf -= med
    vf /= mad
    return {"z": vf, "mask": mask, "tr": tr, "crs": crs, "path": path}


# ---------------------------------------------------------------- точки

def _rank(v):
    from scipy.stats import rankdata
    return rankdata(v) / max(len(v), 1)


def point_candidates(d: dict):
    from scipy import ndimage
    from scipy.spatial import cKDTree
    z, m = d["z"], d["mask"]
    zs1 = ndimage.gaussian_filter(z, PT_SMOOTH)
    zs3 = ndimage.gaussian_filter(z, 3)
    rc, signs = [], []
    for sign in (-1, 1):
        a = np.where(m, sign * zs1, -np.inf)
        mx = ndimage.maximum_filter(a, size=2 * PT_MIN_DIST + 1, mode="constant", cval=-np.inf)
        r, c = np.nonzero((a == mx) & (a >= PT_MIN_Z))
        rc.append(np.c_[r, c]); signs.append(np.full(len(r), sign, np.float32))
        del a, mx
    rc = np.concatenate(rc).astype(np.int64); sg = np.concatenate(signs)
    if not len(rc):
        return np.zeros((0, 2)), np.zeros((0, N_PT), np.float32)
    r, c = rc[:, 0], rc[:, 1]
    dog1 = zs1[r, c] - ndimage.gaussian_filter(z, 10)[r, c]
    dog2 = zs3[r, c] - ndimage.gaussian_filter(z, 30)[r, c]
    mu = ndimage.uniform_filter(z, 31)
    noise = np.sqrt(np.maximum(ndimage.uniform_filter(z * z, 31) - mu ** 2, 0))[r, c]
    del mu
    # Гессиан: сглаживание σ=3, затем конечные разности (как use_gaussian_derivatives=False)
    g = ndimage.gaussian_filter(z, 3)
    gr, gc = np.gradient(g)
    del g
    hrr, hrc = np.gradient(gr)
    hcc = np.gradient(gc, axis=1)
    del gr, gc
    a_, b_, c_ = hrr[r, c], hrc[r, c], hcc[r, c]
    del hrr, hrc, hcc
    t = np.sqrt(((a_ - c_) / 2) ** 2 + b_ ** 2)
    e1, e2 = (a_ + c_) / 2 + t, (a_ + c_) / 2 - t
    edge = ndimage.distance_transform_edt(m)[r, c] * RES
    amp = sg * zs1[r, c]; amp3 = sg * zs3[r, c]
    opp, dy, dx = np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r))
    H, W = z.shape
    for i in range(len(r)):
        r0, r1, c0, c1 = max(r[i] - 15, 0), min(r[i] + 16, H), max(c[i] - 15, 0), min(c[i] + 16, W)
        w = -sg[i] * zs1[r0:r1, c0:c1]
        k = np.unravel_index(np.argmax(w), w.shape)
        opp[i] = w[k]; dy[i] = (k[0] + r0 - r[i]) * RES; dx[i] = (k[1] + c0 - c[i]) * RES
    xs = d["tr"].c + (c + 0.5) * d["tr"].a + (r + 0.5) * d["tr"].b
    ys = d["tr"].f + (c + 0.5) * d["tr"].d + (r + 0.5) * d["tr"].e
    xy = np.c_[xs, ys]
    dens = np.array([len(v) - 1 for v in cKDTree(xy).query_ball_point(xy, 3.0)])
    nz = noise + 1e-3
    X = np.c_[sg, dy, dx, np.hypot(dy, dx), edge, dens, _rank(amp), _rank(amp3), _rank(sg * dog1), _rank(sg * dog2),
              _rank(opp), amp / nz, sg * dog1 / nz, opp / nz, sg * e1 / nz, sg * e2 / nz]
    return xy, X.astype(np.float32)


def nms_points(xy, scores, r_m=PT_NMS_M):
    from scipy.spatial import cKDTree
    order = np.argsort(-scores, kind="stable")
    if not len(order):
        return []
    tree = cKDTree(xy)
    taken = np.zeros(len(xy), bool)
    keep = []
    for i in order:
        if taken[i]:
            continue
        keep.append(int(i))
        taken[tree.query_ball_point(xy[i], r_m)] = True
    return keep


# ---------------------------------------------------------------- полигоны

def _coarse(d):
    k = POLY_K
    z, m = d["z"], d["mask"]
    H, W = (z.shape[0] // k) * k, (z.shape[1] // k) * k
    zc = z[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3))
    mc = m[:H, :W].reshape(H // k, k, W // k, k).all((1, 3))
    return zc, mc, d["tr"] * d["tr"].scale(k, k)


def _region_shape(reg: np.ndarray, res: float):
    """area(px), eccentricity, major axis (м) по моментам — как regionprops."""
    yy, xx = np.nonzero(reg)
    n = len(yy)
    cy, cx = yy.mean(), xx.mean()
    myy = ((yy - cy) ** 2).mean() + 1 / 12; mxx = ((xx - cx) ** 2).mean() + 1 / 12; mxy = ((yy - cy) * (xx - cx)).mean()
    t = math.sqrt(((myy - mxx) / 2) ** 2 + mxy ** 2)
    l1, l2 = (myy + mxx) / 2 + t, max((myy + mxx) / 2 - t, 0.0)
    ecc = math.sqrt(1 - l2 / l1) if l1 > 0 else 0.0
    return n, ecc, 4 * math.sqrt(l1) * res


def poly_candidates(d: dict):
    from rasterio.features import shapes
    from scipy import ndimage
    from shapely.geometry import shape as shp
    z0, m, tr = _coarse(d)
    res = POLY_K * RES
    edge = ndimage.distance_transform_edt(m) * res
    valid = int(m.sum())
    zs = ndimage.gaussian_filter(z0, SMOOTH)
    z_std = float(np.std(zs[m])) + 1e-9
    min_px = MIN_CAND_M2 / res ** 2
    polys, feats = [], []
    st = np.ones((3, 3), bool)
    for bg_m in BG_SIGMAS_M:
        z = zs - ndimage.gaussian_filter(z0, bg_m / res) if bg_m else zs
        z = z / (float(np.std(z[m])) + 1e-9)
        for sign in (1, -1):
            for lv in LEVELS:
                lab, n = ndimage.label((sign * z > lv) & m, structure=st)
                if not n:
                    continue
                cnt = np.bincount(lab.ravel())
                for li, sl in enumerate(ndimage.find_objects(lab), start=1):
                    if sl is None or cnt[li] < min_px or cnt[li] > 0.5 * valid:
                        continue
                    pad = 6
                    y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
                    y1, x1 = min(sl[0].stop + pad, z.shape[0]), min(sl[1].stop + pad, z.shape[1])
                    reg = lab[y0:y1, x0:x1] == li
                    ring = ndimage.binary_dilation(reg, iterations=5) & ~reg & m[y0:y1, x0:x1]
                    zin = sign * z[y0:y1, x0:x1][reg]
                    zring = sign * z[y0:y1, x0:x1][ring] if ring.any() else np.array([lv])
                    ctr = tr * tr.translation(x0, y0)
                    best = None
                    for gj, val in shapes(reg.astype(np.uint8), mask=reg, transform=ctr, connectivity=8):
                        g = shp(gj)
                        if best is None or g.area > best.area:
                            best = g
                    if best is None:
                        continue
                    poly = best.buffer(0)
                    if poly.is_empty:
                        continue
                    poly = poly.simplify(res * 0.5)
                    area, ecc, major = _region_shape(reg, res)
                    hull = poly.convex_hull.area
                    e = edge[y0:y1, x0:x1][reg]
                    per = poly.length / res
                    polys.append(poly)
                    feats.append([sign, bg_m, lv, math.log(area), area / valid, zin.mean(),
                                  zin.max(), zin.mean() - zring.mean(), (sign * zs[y0:y1, x0:x1][reg]).mean() / z_std,
                                  min(poly.area / hull, 1.0) if hull > 0 else 1.0, ecc,
                                  4 * math.pi * area / max(per, 1) ** 2, e.mean(), e.min(), major])
                del lab
    return polys, np.array(feats, np.float32).reshape(-1, N_POLY)


def nested_children(polys, X):
    from shapely.strtree import STRtree
    kids = [[] for _ in polys]
    if not polys:
        return kids
    reps = [p.representative_point() for p in polys]
    tree = STRtree(reps)
    for i, p in enumerate(polys):
        for j in tree.query(p):
            j = int(j)
            if (j != i and X[j, 0] == X[i, 0] and X[j, 1] == X[i, 1] and abs(X[j, 2] - X[i, 2] - 0.5) < 1e-6
                    and p.contains(reps[j])):
                kids[i].append(j)
    return kids


def nms_polys(polys, scores, t_min=0.0):
    from shapely.strtree import STRtree
    keep = []
    if not polys:
        return keep
    tree = STRtree(polys)
    kept = np.zeros(len(polys), bool)
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] < t_min:
            break
        p = polys[i]
        ok = True
        for k in tree.query(p):
            if not kept[k]:
                continue
            q = polys[k]
            inter = p.intersection(q).area
            if inter / max(p.union(q).area, 1e-12) >= NMS_IOU or inter >= 0.5 * min(p.area, q.area):
                ok = False
                break
        if ok:
            kept[i] = True
            keep.append(int(i))
    return keep


def decode_polys(polys, X, scores, t_ref: float):
    """NMS без порога (порог — в post.finalize по группе maggr_org; NMS с порогом даёт то же над порогом), затем
    слитое пятно → вложенные дети (≥ 2 со скором ≥ SPLIT_RATIO·t_ref); детям даётся скор родителя."""
    keep = nms_polys(polys, scores)
    kids = nested_children(polys, X)
    out = []
    for i in keep:
        ks = [j for j in kids[i] if scores[j] >= SPLIT_RATIO * t_ref]
        if len(ks) >= 2 and scores[i] >= t_ref:
            out += [(j, float(scores[i])) for j in ks]
        else:
            out.append((i, float(scores[i])))
    seen, res = set(), []
    for j, s in out:
        if j not in seen:
            seen.add(j); res.append((j, s))
    return res


# ---------------------------------------------------------------- метки (обучение)

def label_points(xy, gxy):
    from scipy.spatial import cKDTree
    if not len(gxy) or not len(xy):
        return np.zeros(len(xy), bool)
    return cKDTree(gxy).query(xy)[0] <= PT_LABEL_M


def label_polys(polys, gts):
    from shapely.strtree import STRtree
    y = np.zeros(len(polys), bool)
    if not polys or not gts:
        return y
    tree = STRtree(gts)
    for i, p in enumerate(polys):
        for j in tree.query(p):
            g = gts[int(j)]
            inter = p.intersection(g).area
            iou = inter / max(p.union(g).area, 1e-12)
            if iou >= LABEL_IOU and inter >= 0.5 * min(p.area, g.area):
                y[i] = True
                break
    return y


def fit(X, y, seed: int = 0, threads: int = 2):
    import lightgbm as lgb
    p = dict(LGB, seed=seed, num_threads=threads)
    if y.sum() == 0:
        return None
    return lgb.train(p, lgb.Dataset(X, label=y.astype(np.float32)), num_boost_round=N_ITER)


def predict(model, X):
    if model is None or not len(X):
        return np.zeros(len(X))
    return model.predict(X, num_threads=1)


# ---------------------------------------------------------------- участок (инференс)

def candidates(reg) -> dict | None:
    """Кандидаты участка (без модели): для кэша обучения и для инференса."""
    from . import geophys
    path = pick_raster(reg)
    if path is None:
        return None
    import rasterio
    with rasterio.open(path) as src:
        rcrs = src.crs.to_string() if src.crs else None
    try:
        b = geophys.load_mask(reg.boundaries, "maggr", rcrs)
    except Exception:
        b = None
    d = load(path, b)
    if d["z"] is None:
        return None
    pxy, pX = point_candidates(d)
    polys, cX = poly_candidates(d)
    return {"path": path, "crs": d["crs"], "pxy": pxy, "pX": pX, "polys": polys, "cX": cX}


def to_feats(c: dict, pts_s, poly_s, reg_name: str, point_r_m: float, t_ref: float, classes) -> list[dict]:
    from .crs import to_crs
    from .raster import circle_3857
    out = []
    src, crs = c["path"].name, c["crs"]

    def f(geom, cls, conf, x, y):
        return {"geom": geom, "class_name": cls, "confidence": float(np.clip(conf, 0, 1)), "region_name": reg_name,
                "markup_type": "maggr", "source": src, "original_crs": crs, "threshold_group": "maggr_org",
                "_xy": (float(x), float(y), crs)}
    if "finds_points" in classes and len(c["pxy"]):
        for i in nms_points(c["pxy"], pts_s):
            x, y = c["pxy"][i]
            out.append(f(circle_3857(x, y, point_r_m, crs), "finds_points", pts_s[i], x, y))
    if "object_poly" in classes and c["polys"]:
        for j, s in decode_polys(c["polys"], c["cX"], poly_s, t_ref):
            g = c["polys"][j]
            pt = g.representative_point()
            out.append(f(to_crs(g, crs, "EPSG:3857"), "object_poly", s, pt.x, pt.y))
    return out


def load_models(model_dir: Path):
    import lightgbm as lgb
    md = Path(model_dir)
    m = {}
    for cls in ("finds_points", "object_poly"):
        f = md / f"org_maggr_{cls}.txt"
        m[cls] = lgb.Booster(model_file=str(f)) if f.exists() else None
    meta = json.loads((md / "org_maggr_meta.json").read_text()) if (md / "org_maggr_meta.json").exists() else {}
    return m, meta


def region_feats(reg, cfg: dict, model_dir, status: dict | None = None) -> list[dict]:
    pix = cfg["geophys"]["pixgbm"]
    geo_cfg = cfg.get("geophys", {})
    status = {} if status is None else status
    models, meta = load_models(model_dir)
    if not any(models.values()):
        return []
    c = candidates(reg)
    if c is None:
        return []
    classes = set(pix.get("org_classes", ["finds_points", "object_poly"]))
    feats = to_feats(c, predict(models["finds_points"], c["pX"]), predict(models["object_poly"], c["cX"]),
                     reg.name, float(geo_cfg.get("point_circle_m", 0.3)), float(pix.get("org_t_ref", 0.5)), classes)
    done = status.setdefault("done", {})
    for f in feats:
        done.setdefault((f["class_name"], "maggr"), set()).add(f["source"])
    return feats
