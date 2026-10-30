import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from shapely.geometry import shape, Point, Polygon
from shapely.strtree import STRtree
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, peak_local_max
from skimage.measure import label, regionprops
from sklearn.ensemble import HistGradientBoostingClassifier

from utils import (DATA_DIR, FIG_DIR, PRED_DIR, R_POINT, ROOT, SEED, extent, is_hit, largest_part,
    load_geoms, n_vertices, platform_counts, platform_prf, plot_polys, point_counts, to_platform_geojson)

pd.set_option("display.precision", 3)

RES = 0.1                      # рабочая сетка, м/px (все растры приводятся к ней)
HOLDOUT_REGION = "207"         # отложенный участок (целиком): на нём проверяем качество
# точки
PT_SMOOTH = 1.5                # сглаживание перед поиском экстремумов, px
PT_MIN_DIST = 5                # минимальное расстояние между экстремумами, px (0.5 м)
PT_MIN_Z = 1.0                 # минимальная амплитуда экстремума, в MAD
PT_LABEL_M = 0.5               # кандидат положительный, если точка разметки ближе, м
PT_NMS_M = 1.5                 # подавление дублей среди точек, м
# полигоны
POLY_RES = 0.25                # сетка для полигональных кандидатов, м/px
SMOOTH = 2                     # сглаживание, px этой сетки
BG_SIGMAS = [0, 40, 20]        # 0: само значение; 40, 20 px (10, 5 м): остаток после вычитания фона с таким σ
LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5]  # уровни нарезки, в std
MIN_CAND_M2 = 4.0              # минимальная площадь кандидата, м²
NMS_IOU = 0.2                  # подавление дублей среди полигонов
LABEL_IOU = 0.1                # метка полигонального кандидата: засчитан платформой и IoU >= LABEL_IOU
# ----CELL
# Серая карта участка вместе с точками, полигонами разметки и контуром съёмки.
# Имя растра и разметки на некоторых участках не совпадает (уч7 / 7), поэтому разметку ищем по типу.
UNITS = []
for tif in sorted(DATA_DIR.glob("*/07_*_MagGr_*/*_серый.tif")):
    region_dir = tif.parents[1]
    (pts,) = region_dir.glob("06_*/*_MagGr_FindsPoints.geojson")
    (obj,) = region_dir.glob("06_*/*_MagGr_ObjectPoly.geojson")
    (boundary,) = region_dir.glob("extra/*_MagGr_границы.geojson")
    UNITS.append({
        "region": region_dir.name.split("_", 1)[0],
        "site": region_dir.name.split("_", 1)[1],
        "raster": tif, "pts": pts, "obj": obj, "boundary": boundary,
    })

rows = []
for u in UNITS:
    with rasterio.open(u["raster"]) as src:
        rows.append({"участок": f"{u['region']} {u['site']}", "м/px": src.res[0], "px": f"{src.width}x{src.height}",
                     "CRS": str(src.crs)})
pd.DataFrame(rows)
# ----CELL
def survey_mask(v):
    # фон: чистые 0 или 255, связанные с краем растра
    lab, _ = ndimage.label((v == 0) | (v == 255))
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, edge[edge > 0])
    return ndimage.binary_erosion(~bg, iterations=2)


def load_unit(u):
    with rasterio.open(u["raster"]) as src:
        H, W = int(round(src.height * src.res[1] / RES)), int(round(src.width * src.res[0] / RES))
        v = src.read(1, out_shape=(H, W), resampling=Resampling.nearest)
        tr = src.transform * src.transform.scale(src.width / W, src.height / H)
        crs = src.crs
    bounds = load_geoms(u["boundary"], crs)
    mask = survey_mask(v) & (rasterize([(g, 1) for g in bounds], out_shape=(H, W), transform=tr, fill=0) > 0)
    # вне маски ставим ближайшее валидное значение, чтобы фильтры не «видели» фон
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    vf = v.astype(np.float32)[iy, ix]
    med = np.median(vf[mask])
    mad = 1.4826 * np.median(np.abs(vf[mask] - med)) + 1e-6
    pts = load_geoms(u["pts"], crs)
    polys = load_geoms(u["obj"], crs)
    return dict(u, z=(vf - med) / mad, mask=mask, tr=tr, crs=crs,
                pts=pts, pxy=np.array([(p.x, p.y) for p in pts]).reshape(-1, 2),
                polys=[g.buffer(0) for g in polys])


DS = [load_unit(u) for u in UNITS]
pd.DataFrame([{"участок": d["region"], "сетка, px": f"{d['z'].shape[1]}x{d['z'].shape[0]}",
               "в маске, %": round(100 * d["mask"].mean()), "точек": len(d["pts"]), "полигонов": len(d["polys"])}
              for d in DS])
# ----CELL
show = [d for d in DS if d["region"] in ("202", "205", "207", "210")]
fig, axes = plt.subplots(1, len(show), figsize=(6 * len(show), 9))
for ax, d in zip(axes, show):
    ext = extent(d["z"].shape, d["tr"])
    ax.imshow(np.where(d["mask"], d["z"], np.nan), cmap="gray", vmin=-3, vmax=3, extent=ext)
    ax.imshow(np.where(d["mask"], np.nan, 1.0), cmap="Greys", vmin=0, vmax=3, extent=ext)
    ax.plot(d["pxy"][:, 0], d["pxy"][:, 1], ".", color="red", ms=2)
    plot_polys(ax, d["polys"], color="cyan", lw=1)
    ax.set_title(f"{d['region']} {d['site']}", fontsize=10)
    ax.tick_params(labelsize=6)
plt.suptitle("MagGr: z, точки (красные) и полигоны (голубые) разметки", fontsize=13)
plt.tight_layout()
fig.savefig(FIG_DIR / "01_maggr_examples.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
R_WIN = 30
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
for ax, d in zip(axes.flat, DS):
    inv = ~d["tr"]
    acc = []
    for x, y in d["pxy"]:
        c, r = inv * (x, y)
        r, c = int(r), int(c)
        if R_WIN <= r < d["z"].shape[0] - R_WIN and R_WIN <= c < d["z"].shape[1] - R_WIN and d["mask"][r, c]:
            acc.append(d["z"][r - R_WIN:r + R_WIN + 1, c - R_WIN:c + R_WIN + 1])
    a = np.mean(acc, 0)
    ax.imshow(a, cmap="RdBu_r", vmin=-2, vmax=2, extent=(-3, 3, -3, 3))
    ax.plot(0, 0, "k+", ms=10)
    ax.set_title(f"{d['region']}: центр z={a[R_WIN, R_WIN]:.2f}", fontsize=9)
    ax.tick_params(labelsize=6)
plt.suptitle("Средний z в окне ±3 м вокруг точек разметки (крест в центре: сама точка)", fontsize=13)
plt.tight_layout()
fig.savefig(FIG_DIR / "01_maggr_point_dipole.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
TRAIN = [d for d in DS if d["region"] != HOLDOUT_REGION]
HOLDOUT = [d for d in DS if d["region"] == HOLDOUT_REGION]
print(f"train: {len(TRAIN)} участков, {sum(len(d['pts']) for d in TRAIN)} точек, {sum(len(d['polys']) for d in TRAIN)} полигонов")
print(f"отложенный участок {HOLDOUT_REGION}: {sum(len(d['pts']) for d in HOLDOUT)} точек, {sum(len(d['polys']) for d in HOLDOUT)} полигонов")
# ----CELL
def point_candidates(d):
    z, m = d["z"], d["mask"]
    zs = {s: ndimage.gaussian_filter(z, s) for s in (PT_SMOOTH, 3)}
    dog1 = zs[PT_SMOOTH] - ndimage.gaussian_filter(z, 10)
    dog2 = zs[3] - ndimage.gaussian_filter(z, 30)
    mu = ndimage.uniform_filter(z, 31)
    noise = np.sqrt(np.maximum(ndimage.uniform_filter(z * z, 31) - mu ** 2, 0))
    e1, e2 = hessian_matrix_eigvals(hessian_matrix(z, sigma=3, order="rc", use_gaussian_derivatives=False))
    edge = ndimage.distance_transform_edt(m) * RES
    rc, raw = [], []
    for sign in (-1, 1):
        a = sign * zs[PT_SMOOTH]
        for r, c in peak_local_max(np.where(m, a, -np.inf), min_distance=PT_MIN_DIST, threshold_abs=PT_MIN_Z,
                                   exclude_border=False):
            # пара диполя: экстремум противоположного знака в окне ±1.5 м
            r0, r1, c0, c1 = max(r - 15, 0), min(r + 16, z.shape[0]), max(c - 15, 0), min(c + 16, z.shape[1])
            w = -a[r0:r1, c0:c1]
            k = np.unravel_index(np.argmax(w), w.shape)
            dy, dx = (k[0] + r0 - r) * RES, (k[1] + c0 - c) * RES
            rc.append((r, c))
            raw.append([sign, a[r, c], sign * zs[3][r, c], sign * dog1[r, c], sign * dog2[r, c], w.max(),
                        noise[r, c], dy, dx, np.hypot(dy, dx), sign * e1[r, c], sign * e2[r, c], edge[r, c]])
    rc = np.array(rc).reshape(-1, 2)
    raw = np.array(raw, np.float32).reshape(-1, 13)
    xs, ys = rasterio.transform.xy(d["tr"], rc[:, 0], rc[:, 1])
    xy = np.c_[xs, ys]
    sign, amp, amp3, dg1, dg2, opp, nz, dy, dx, dd, h1, h2, edg = raw.T
    dens = np.array([len(v) - 1 for v in cKDTree(xy).query_ball_point(xy, 3.0)]) if len(xy) else np.zeros(0)
    rank = lambda v: rankdata(v) / len(v)
    X = np.c_[sign, dy, dx, dd, edg, dens, rank(amp), rank(amp3), rank(dg1), rank(dg2), rank(opp),
              amp / (nz + 1e-3), dg1 / (nz + 1e-3), opp / (nz + 1e-3), h1 / (nz + 1e-3), h2 / (nz + 1e-3)]
    return xy, X.astype(np.float32)


for d in DS:
    d["pt_xy"], d["pt_X"] = point_candidates(d)
    d["pt_dist"] = cKDTree(d["pxy"]).query(d["pt_xy"])[0] if len(d["pxy"]) else np.full(len(d["pt_xy"]), np.inf)
    d["pt_reached"] = cKDTree(d["pt_xy"]).query(d["pxy"])[0] <= R_POINT if len(d["pxy"]) else np.zeros(0, bool)
pd.DataFrame([{"участок": d["region"], "точек": len(d["pxy"]), "кандидатов": len(d["pt_xy"]),
               "положит.": (d["pt_dist"] <= PT_LABEL_M).sum(), "оракул, %": round(100 * d["pt_reached"].mean())}
              for d in DS])
# ----CELL
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def make_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                          random_state=SEED)


def fit_points(dd):
    X = np.concatenate([d["pt_X"] for d in dd])
    y = np.concatenate([d["pt_dist"] <= PT_LABEL_M for d in dd])
    return make_model().fit(X, y.astype(int))


def nms_points(xy, scores, thr):
    order = [i for i in np.argsort(-scores) if scores[i] >= thr]
    if not order:
        return []
    tree = cKDTree(xy[order])
    taken = np.zeros(len(order), bool)
    keep = []
    for k, i in enumerate(order):
        if taken[k]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(xy[i], PT_NMS_M):
            if j > k:
                taken[j] = True
    return keep


def eval_points(dd, scores, thr):
    tot = np.zeros(3, int)
    for d, s in zip(dd, scores):
        k = nms_points(d["pt_xy"], s, thr)
        tot += point_counts(d["pt_xy"][k], s[k], d["pxy"])
    return tuple(int(x) for x in tot), platform_prf(*tot)


pt_scores = [fit_points([e for e in TRAIN if e is not d]).predict_proba(d["pt_X"])[:, 1] for d in TRAIN]
PT_CURVE = [(t, *eval_points(TRAIN, pt_scores, t)) for t in THRS]
PT_THR, (tp, fp, fn), (p, r, f1) = max(PT_CURVE, key=lambda x: x[2][2])
PT_CLF = fit_points(TRAIN)
print(f"Порог по OOF {PT_THR}: P={p:.3f} R={r:.3f} F1={f1:.3f}  (TP={tp}, FP={fp}, FN={fn})")
pd.DataFrame([eval_points([d], [s], PT_THR)[1] for d, s in zip(TRAIN, pt_scores)],
             index=pd.Index([d["region"] for d in TRAIN], name="OOF"), columns=["P", "R", "F1"])
# ----CELL
POLY_FEATURES = ["знак", "фон σ", "уровень", "log площади", "доля площади", "z среднее", "z макс",
                 "контраст с кольцом", "z без фона", "solidity", "eccentricity", "компактность",
                 "удал. от края ср.", "удал. от края мин.", "длина, м"]


def coarse(d):
    k = int(round(POLY_RES / RES))
    H, W = (d["z"].shape[0] // k) * k, (d["z"].shape[1] // k) * k
    z = d["z"][:H, :W].reshape(H // k, k, W // k, k).mean((1, 3))
    m = d["mask"][:H, :W].reshape(H // k, k, W // k, k).all((1, 3))
    return z, m, d["tr"] * d["tr"].scale(k, k)


def poly_candidates(d):
    z0, m, tr = coarse(d)
    edge = ndimage.distance_transform_edt(m) * POLY_RES
    valid_area = m.sum()
    zs = ndimage.gaussian_filter(z0, SMOOTH)
    z_std = np.std(zs[m])
    min_px = MIN_CAND_M2 / POLY_RES ** 2
    polys, feats = [], []
    for bg in BG_SIGMAS:
        z = zs - ndimage.gaussian_filter(z0, bg) if bg else zs
        z = z / np.std(z[m])
        for sign in (1, -1):
            for lv in LEVELS:
                lab = label((sign * z > lv) & m, connectivity=2)
                for rp in regionprops(lab):
                    if rp.area < min_px or rp.area > 0.5 * valid_area:
                        continue
                    sl, pad = rp.slice, 6
                    y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
                    y1, x1 = min(sl[0].stop + pad, z.shape[0]), min(sl[1].stop + pad, z.shape[1])
                    reg = np.zeros((y1 - y0, x1 - x0), bool)
                    reg[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] = lab[sl] == rp.label
                    ring = ndimage.binary_dilation(reg, iterations=5) & ~reg & m[y0:y1, x0:x1]
                    zin = sign * z[y0:y1, x0:x1][reg]
                    zring = sign * z[y0:y1, x0:x1][ring] if ring.any() else np.array([lv])
                    cnts = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    cnt = max(cnts, key=cv2.contourArea)
                    if len(cnt) < 4:
                        continue
                    poly = Polygon([rasterio.transform.xy(tr, p[1] + y0, p[0] + x0)
                                    for p in cnt.squeeze(1)]).buffer(0)
                    if poly.is_empty:
                        continue
                    e = edge[y0:y1, x0:x1][reg]
                    polys.append(poly)
                    feats.append([sign, bg, lv, np.log(rp.area), rp.area / valid_area, zin.mean(), zin.max(),
                                  zin.mean() - zring.mean(), (sign * zs[y0:y1, x0:x1][reg]).mean() / z_std,
                                  rp.solidity, rp.eccentricity, 4 * np.pi * rp.area / max(rp.perimeter, 1) ** 2,
                                  e.mean(), e.min(), rp.axis_major_length * POLY_RES])
    return polys, np.array(feats, np.float32).reshape(-1, len(POLY_FEATURES))


def hit_flags(polys, gts):
    # какие кандидаты засчитали бы какой-нибудь объект, какие объекты кем-то засчитаны, и IoU кандидата
    cand, gt, iou = np.zeros(len(polys), bool), np.zeros(len(gts), bool), np.zeros(len(polys))
    if not polys or not gts:
        return cand, gt, iou
    gts_ = [largest_part(g) for g in gts]
    tree = STRtree(gts)
    for i, p in enumerate(polys):
        for j in tree.query(p):
            iou[i] = max(iou[i], p.intersection(gts[j]).area / max(p.union(gts[j]).area, 1e-12))
            if is_hit(largest_part(p), gts_[j])[0]:
                cand[i] = gt[j] = True
    return cand, gt, iou


for d in DS:
    d["cand_polys"], d["cand_X"] = poly_candidates(d)
    d["cand_hit"], d["gt_reached"], d["cand_iou"] = hit_flags(d["cand_polys"], d["polys"])
    d["cand_pos"] = d["cand_hit"] & (d["cand_iou"] >= LABEL_IOU)
pd.DataFrame([{"участок": d["region"], "полигонов": len(d["polys"]), "кандидатов": len(d["cand_polys"]),
               "положит.": d["cand_pos"].sum(), "оракул, %": round(100 * d["gt_reached"].mean())} for d in DS])
# ----CELL
def fit_polys(dd):
    X = np.concatenate([d["cand_X"] for d in dd])
    y = np.concatenate([d["cand_pos"] for d in dd])
    return make_model().fit(X, y.astype(int))


def nms_polys(polys, scores, thr):
    keep = []
    for i in np.argsort(-scores):
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
            keep.append(i)
    return keep


def pool_polys(dd, scores, thr):
    preds, confs, gts = [], [], []
    for d, s in zip(dd, scores):
        k = nms_polys(d["cand_polys"], s, thr)
        preds += [d["cand_polys"][i] for i in k]
        confs += [float(s[i]) for i in k]
        gts += d["polys"]
    return preds, confs, gts


def eval_polys(dd, scores, thr):
    tot = platform_counts(*pool_polys(dd, scores, thr))
    return tot, platform_prf(*tot)


poly_scores = [fit_polys([e for e in TRAIN if e is not d]).predict_proba(d["cand_X"])[:, 1] for d in TRAIN]
POLY_CURVE = [(t, *eval_polys(TRAIN, poly_scores, t)) for t in THRS]
POLY_THR, (tp, fp, fn), (p, r, f1) = max(POLY_CURVE, key=lambda x: x[2][2])
POLY_CLF = fit_polys(TRAIN)
print(f"Порог по OOF {POLY_THR}: P={p:.3f} R={r:.3f} F1={f1:.3f}  (TP={tp}, FP={fp}, FN={fn})")
pd.DataFrame([eval_polys([d], [s], POLY_THR)[1] for d, s in zip(TRAIN, poly_scores)],
             index=pd.Index([d["region"] for d in TRAIN], name="OOF"), columns=["P", "R", "F1"])
# ----CELL
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (name, c, thr) in zip(axes, [("точки", PT_CURVE, PT_THR), ("полигоны", POLY_CURVE, POLY_THR)]):
    ax.plot([x[0] for x in c], [x[2][0] for x in c], label="Precision")
    ax.plot([x[0] for x in c], [x[2][1] for x in c], label="Recall")
    ax.plot([x[0] for x in c], [x[2][2] for x in c], label="F1", lw=2)
    ax.axvline(thr, color="gray", ls="--")
    ax.set_xlabel("порог вероятности")
    ax.set_title(f"{name}: OOF на train-участках (метрика платформы)")
    ax.grid(True, alpha=0.3)
    ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "01_maggr_threshold.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
d = HOLDOUT[0]
pt_s = PT_CLF.predict_proba(d["pt_X"])[:, 1]
pt_k = nms_points(d["pt_xy"], pt_s, PT_THR)
HOLDOUT_PTS = (d["pt_xy"][pt_k], pt_s[pt_k])
poly_s = POLY_CLF.predict_proba(d["cand_X"])[:, 1]
HOLDOUT_POLYS = pool_polys([d], [poly_s], POLY_THR)[:2]

pt_counts = point_counts(*HOLDOUT_PTS, d["pxy"])
near = (cKDTree(HOLDOUT_PTS[0]).query(d["pxy"])[0] <= R_POINT).sum() if len(HOLDOUT_PTS[0]) else 0
poly_counts = platform_counts(*HOLDOUT_POLYS, d["polys"])
hit = sum(any(q.intersects(g) for q in HOLDOUT_POLYS[0]) for g in d["polys"])
pd.DataFrame([[*pt_counts, *platform_prf(*pt_counts), f"{near}/{len(d['pxy'])}"],
              [*poly_counts, *platform_prf(*poly_counts), f"{hit}/{len(d['polys'])}"]],
             index=["точки", "полигоны"], columns=["TP", "FP", "FN", "P", "R", "F1", "задето"])
# ----CELL
ext = extent(d["z"].shape, d["tr"])
fig, axes = plt.subplots(1, 2, figsize=(16, 16))
for k, (ax, title) in enumerate(zip(axes, ["разметка: точки (красные), полигоны (голубые)",
                                           "предсказание: точки (оранжевые), полигоны (зелёные)"])):
    ax.imshow(np.where(d["mask"], d["z"], np.nan), cmap="gray", vmin=-3, vmax=3, extent=ext)
    if k == 0:
        ax.plot(d["pxy"][:, 0], d["pxy"][:, 1], ".", color="red", ms=3)
        plot_polys(ax, d["polys"], color="cyan", lw=1.2)
    else:
        ax.plot(HOLDOUT_PTS[0][:, 0], HOLDOUT_PTS[0][:, 1], "o", mfc="none", color="orange", ms=4)
        plot_polys(ax, d["polys"], color="cyan", lw=0.8)
        plot_polys(ax, HOLDOUT_POLYS[0], color="lime", lw=1.5, ls="--")
    ax.set_title(f"{HOLDOUT_REGION}: {title}", fontsize=11)
    ax.tick_params(labelsize=6)
plt.tight_layout()
fig.savefig(FIG_DIR / "01_maggr_holdout_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
def loso(fit, key, evaluate):
    sc = [fit([e for e in DS if e is not d]).predict_proba(d[key])[:, 1] for d in DS]
    thr = max(THRS, key=lambda t: evaluate(DS, sc, t)[1][2])
    per = {d["region"]: evaluate([d], [s], thr)[1][2] for d, s in zip(DS, sc)}
    return thr, evaluate(DS, sc, thr)[1], per


rows = {}
for name, args in [("точки", (fit_points, "pt_X", eval_points)), ("полигоны", (fit_polys, "cand_X", eval_polys))]:
    thr, (p, r, f1), per = loso(*args)
    rows[name] = {"порог": thr, "P": p, "R": r, "F1": f1, **per}
pd.DataFrame.from_dict(rows, orient="index")
# ----CELL
out = PRED_DIR / f"01_maggr_holdout_{HOLDOUT_REGION}.geojson"
fc = to_platform_geojson([(d, [Point(*xy) for xy in HOLDOUT_PTS[0]], HOLDOUT_PTS[1]), (d, *HOLDOUT_POLYS)], "maggr", out)
print(f"{out.relative_to(ROOT)}: {len(HOLDOUT_PTS[0])} точек и {len(HOLDOUT_POLYS[0])} полигонов, "
      f"{out.stat().st_size / 1e6:.2f} МБ, макс. вершин {max(n_vertices(q) for q in HOLDOUT_POLYS[0])}")
print("Пример свойств:", fc["features"][0]["properties"])

# проверка: переводим файл обратно в UTM и считаем метрику
back = json.loads(out.read_text(encoding="utf-8"))["features"]
geo = [(shape(transform_geom("EPSG:3857", d["crs"], f["geometry"])), f["properties"]) for f in back]
f_pts = np.array([(g.x, g.y) for g, pr in geo if pr["class_name"] == "finds_points"]).reshape(-1, 2)
f_pt_conf = [pr["confidence"] for g, pr in geo if pr["class_name"] == "finds_points"]
f_polys = [g for g, pr in geo if pr["class_name"] == "object_poly"]
f_poly_conf = [pr["confidence"] for g, pr in geo if pr["class_name"] == "object_poly"]
for name, from_file, in_memory in [("точки", point_counts(f_pts, f_pt_conf, d["pxy"]), point_counts(*HOLDOUT_PTS, d["pxy"])),
                                   ("полигоны", platform_counts(f_polys, f_poly_conf, d["polys"]),
                                    platform_counts(*HOLDOUT_POLYS, d["polys"]))]:
    print(f"{name:>9}: из файла TP={from_file[0]} FP={from_file[1]} FN={from_file[2]} "
          f"F1={platform_prf(*from_file)[2]:.3f}; в памяти TP={in_memory[0]}; разница TP: {from_file[0] - in_memory[0]:+d}")