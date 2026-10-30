import json
import os
import re

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import cv2
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import timm
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import shape, Polygon
from shapely.strtree import STRtree
from skimage.measure import label, regionprops
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from utils import (DATA_DIR, FIG_DIR, PRED_DIR, ROOT, SEED, extent, is_hit, largest_part, load_polys,
    n_vertices, platform_counts, platform_prf, plot_polys, rasterize_polys, to_platform_geojson, touched)

pd.set_option("display.precision", 3)

WORK_SIDE = 512             # длинная сторона рабочей сетки, px (все растры приводятся к ней)
PALETTE_MAX_DIST = 110      # пиксели дальше от шкалы jet (рамка, подписи) в маску не входят
SMOOTH = 2                  # сглаживание перед нарезкой кандидатов, px
BG_SIGMAS = [0, 20, 10]     # 0: само значение; 20, 10: остаток после вычитания фона с таким σ
LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5]  # уровни нарезки, в std
MIN_CAND_PX = 30            # минимальная площадь кандидата, px
NMS_IOU = 0.2               # подавление дублей среди кандидатов

HOLDOUT_REGION = "216"      # отложенный участок: не участвует в обучении, на нём проверяем качество

LABEL_IOU = 0.1            # метка кандидата: засчитан платформой и IoU с разметкой >= LABEL_IOU (раздел 7)
SPLIT_RATIO = 0.5           # карты: пятно заменяется вложенными кандидатами, если их >= 2 со score >= SPLIT_RATIO * порог
CROP = 128                  # окно кандидата для CNN, px
CNN_EPOCHS = 8              # эпохи обучения CNN
CNN_SEEDS = [0, 1, 2]       # ансамбль из нескольких CNN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
# ----CELL
# Один растр EtGr вместе с его разметкой и контуром съёмки
UNITS = []
for tif in sorted(DATA_DIR.glob("*/08_*_EtGr_*/*.tif")):
    region_dir = tif.parents[1]
    stem = tif.stem.split("_", 1)[1]          # без порядкового номера файла
    (obj,) = region_dir.glob(f"06_*/{stem}_ObjectPoly.geojson")
    (boundary,) = region_dir.glob(f"extra/{stem}_границы.geojson")
    UNITS.append({
        "region": region_dir.name.split("_", 1)[0],
        "site": region_dir.name.split("_", 1)[1],
        "kind": "профиль" if "_профиль_" in stem else "карта",
        "stem": stem, "raster": tif, "obj": obj, "boundary": boundary,
    })

regions = sorted({u["region"] for u in UNITS})
print(f"Найдено {len(UNITS)} растров EtGr на {len(regions)} участках")
pd.crosstab([f"{u['region']} {u['site']}" for u in UNITS], [u["kind"] for u in UNITS], rownames=["участок"], colnames=["растров"])
# ----CELL
examples = []
for r in regions:
    for kind in ("карта", "профиль"):
        uu = [u for u in UNITS if u["region"] == r and u["kind"] == kind]
        if uu:
            examples.append(uu[0])

fig, axes = plt.subplots(3, 4, figsize=(20, 13))
for ax, u in zip(axes.flat, examples):
    with rasterio.open(u["raster"]) as src:
        scale = max(1.0, max(src.width, src.height) / 800)
        H, W = int(src.height / scale), int(src.width / scale)
        rgb = src.read([1, 2, 3], out_shape=(3, H, W), resampling=Resampling.nearest)
        tr = src.transform * src.transform.scale(src.width / W, src.height / H)
        crs = src.crs
    ax.imshow(rgb.transpose(1, 2, 0), extent=extent((H, W), tr))
    plot_polys(ax, load_polys(u["obj"], crs), color="black", lw=1)
    plot_polys(ax, load_polys(u["boundary"], crs), color="magenta", lw=1, ls="--")
    ax.set_title(f"{u['region']} {u['stem'][-24:]}", fontsize=9)
    ax.tick_params(labelsize=6)
for ax in axes.flat[len(examples):]:
    ax.axis("off")
plt.suptitle("EtGr: примеры карт и профилей (чёрным ObjectPoly, пунктиром контур съёмки)", fontsize=13)
plt.tight_layout()
fig.savefig(FIG_DIR / "02_etgr_examples.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
JET = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3] * 255
JET_TREE = cKDTree(JET)


def rgb_to_scalar(rgb):
    dist, idx = JET_TREE.query(rgb.reshape(3, -1).T.astype(np.float32))
    return (idx / 255.0).reshape(rgb.shape[1:]), dist.reshape(rgb.shape[1:])


def load_unit(u):
    with rasterio.open(u["raster"]) as src:
        scale = max(1.0, max(src.width, src.height) / WORK_SIDE)
        H, W = int(round(src.height / scale)), int(round(src.width / scale))
        rgb = src.read([1, 2, 3], out_shape=(3, H, W), resampling=Resampling.nearest)
        tr = src.transform * src.transform.scale(src.width / W, src.height / H)
        crs = src.crs
    t, dist = rgb_to_scalar(rgb)
    bmask = rasterize_polys(load_polys(u["boundary"], crs), (H, W), tr) > 0
    mask = bmask & (dist < PALETTE_MAX_DIST)
    # невалидные пиксели заполняем ближайшим валидным, чтобы фильтры не «видели» рамку
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    t = t[iy, ix]
    v = t[mask]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med)) + 1e-6
    polys = load_polys(u["obj"], crs)
    return dict(u, rgb=rgb, tr=tr, crs=crs, z=(t - med) / mad, mask=mask, dist=dist, bmask=bmask,
                polys=polys, gt=rasterize_polys(polys, (H, W), tr))


DS = [load_unit(u) for u in UNITS]
print(f"Загружено {len(DS)} растров")

d = [d for d in DS if d["region"] == "214" and d["kind"] == "карта"][2]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ext = extent(d["z"].shape, d["tr"])
axes[0].imshow(d["rgb"].transpose(1, 2, 0), extent=ext)
axes[0].set_title("исходный RGB")
im = axes[1].imshow(np.where(d["bmask"], d["dist"], np.nan), cmap="magma", vmin=0, vmax=200, extent=ext)
plt.colorbar(im, ax=axes[1], fraction=0.04)
axes[1].set_title("расстояние до шкалы jet (внутри контура)")
im = axes[2].imshow(np.where(d["mask"], d["z"], np.nan), cmap="RdBu_r", vmin=-3, vmax=3, extent=ext)
plt.colorbar(im, ax=axes[2], fraction=0.04)
plot_polys(axes[2], d["polys"], color="black", lw=1)
axes[2].set_title("z (нормированное значение) + ObjectPoly")
plt.suptitle(f"{d['region']} {d['stem']}: обращение цветовой шкалы", fontsize=13)
plt.tight_layout()
fig.savefig(FIG_DIR / "02_etgr_palette.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Доля пикселей внутри контура, отброшенных как «не шкала»: "
      f"{np.mean([1 - d['mask'].sum() / max(d['bmask'].sum(), 1) for d in DS]):.1%} (в среднем по растрам)")
# ----CELL
SITE_3D = "216"


def read_small(u, max_side):
    with rasterio.open(u["raster"]) as src:
        scale = max(1.0, max(src.width, src.height) / max_side)
        H, W = int(round(src.height / scale)), int(round(src.width / scale))
        rgb = src.read([1, 2, 3], out_shape=(3, H, W), resampling=Resampling.nearest)
        tr = src.transform * src.transform.scale(src.width / W, src.height / H)
        crs = src.crs
    _, dist = rgb_to_scalar(rgb)
    mask = (rasterize_polys(load_polys(u["boundary"], crs), (H, W), tr) > 0) & (dist < PALETTE_MAX_DIST)
    return rgb.transpose(1, 2, 0) / 255.0, mask, tr, crs


def profile_to_depth(u, max_side=600):
    # поверхность: верхний валидный пиксель каждого столбца, глубина отсчитывается от него вниз
    rgb, m, tr, crs = read_small(u, max_side)
    cols = np.where(m.any(0))[0]
    r0 = np.array([np.argmax(m[:, c]) for c in cols])
    x_s, y_s = rasterio.transform.xy(tr, r0, cols)
    dep = (np.arange(m.shape[0])[:, None] - r0[None, :]) * abs(tr.e)
    return dict(rgb=rgb[:, cols], mask=m[:, cols] & (dep >= 0), x=np.array(x_s), y_surf=np.array(y_s),
                dep=dep, polys=load_polys(u["obj"], crs), tr=tr, rgb_file=rgb, mask_file=m)


def map_depth(u):
    return float(re.search(r"_(\d+(?:\.\d+)?)$", u["stem"]).group(1))


maps_3d = [u for u in UNITS if u["region"] == SITE_3D and u["kind"] == "карта"]
profs_3d = [u for u in UNITS if u["region"] == SITE_3D and u["kind"] == "профиль"]

P = profile_to_depth(profs_3d[0])
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
H, W = P["rgb_file"].shape[:2]
axes[0].imshow(np.where(P["mask_file"][..., None], P["rgb_file"], 1.0), extent=extent((H, W), P["tr"]))
plot_polys(axes[0], P["polys"], color="black", lw=1)
axes[0].set_xlabel("восток, м (UTM)")
axes[0].set_ylabel("«север», м (UTM)")
axes[0].set_title(f"{profs_3d[0]['stem']} в GeoTIFF: верхний край: линия профиля, вниз отложена глубина")
dist_along = P["x"] - P["x"][0]
dmax = P["dep"][P["mask"]].max()
img = np.ones((int(dmax / abs(P["tr"].e)) + 1, len(dist_along), 3))
for j in range(len(dist_along)):
    col = P["rgb"][:, j][P["mask"][:, j]]
    img[:len(col), j] = col
axes[1].imshow(img, extent=(0, dist_along[-1], dmax, 0), aspect="auto")
for g in P["polys"]:
    for p in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
        x, y = np.array(p.exterior.xy)
        axes[1].plot(x - P["x"][0], np.interp(x, P["x"], P["y_surf"]) - y, color="black", lw=1)
axes[1].set_xlabel("расстояние вдоль профиля, м")
axes[1].set_ylabel("глубина, м")
axes[1].set_title("тот же профиль как вертикальный разрез (ObjectPoly чёрным)")
plt.tight_layout()
fig.savefig(FIG_DIR / "02_etgr_profile_depth.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
def add_map_slice(ax, u, alpha, max_side=120, with_polys=False):
    rgb, m, tr, crs = read_small(u, max_side)
    H, W = m.shape
    xs, ys = rasterio.transform.xy(tr, *np.mgrid[0:H, 0:W].reshape(2, -1))
    z = -map_depth(u)
    ax.plot_surface(np.array(xs).reshape(H, W), np.array(ys).reshape(H, W), np.full((H, W), z),
                    facecolors=np.dstack([rgb, np.where(m, alpha, 0.0)]), rstride=1, cstride=1,
                    shade=False, linewidth=0)
    if with_polys:
        for g in load_polys(u["obj"], crs):
            for p in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
                x, y = p.exterior.xy
                ax.plot(x, y, z, color="black", lw=0.8)


fig = plt.figure(figsize=(20, 9))
ax = fig.add_subplot(1, 2, 1, projection="3d")
for u in maps_3d:
    add_map_slice(ax, u, alpha=0.8, with_polys=True)
ax.set_box_aspect((37, 23, 20))
ax.set_zlabel("глубина, м")
ax.view_init(22, -65)
ax.set_title(f"{SITE_3D}: карты-срезы 0.25–1.75 м, сложенные по глубине (+ ObjectPoly)")

ax = fig.add_subplot(1, 2, 2, projection="3d")
for u in [p for p in profs_3d if p["stem"].split("_")[-1] in ("01", "04", "07", "23")]:
    P = profile_to_depth(u, 300)
    ax.plot_surface(np.broadcast_to(P["x"], P["dep"].shape), np.broadcast_to(P["y_surf"], P["dep"].shape),
                    -np.clip(P["dep"], 0, None), facecolors=np.dstack([P["rgb"], P["mask"] * 0.6]),
                    rstride=2, cstride=2, shade=False, linewidth=0)
    ax.text(P["x"][-1], P["y_surf"][-1], 0.5, "профиль " + u["stem"].split("_")[-1], fontsize=8)
for u in (maps_3d[0], maps_3d[-1]):
    add_map_slice(ax, u, alpha=0.45)
ax.set_box_aspect((37, 23, 20))
ax.set_zlabel("глубина, м")
ax.view_init(30, -30)
ax.set_title(f"{SITE_3D}: профили на своих линиях (fence diagram) + карты 0.25 и 1.75 м")
plt.tight_layout()
fig.savefig(FIG_DIR / "02_etgr_pseudo3d.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
def object_contrast(d, g):
    m = rasterize_polys([g], d["z"].shape, d["tr"]).astype(bool) & d["mask"]
    if m.sum() < 5:
        return None
    ring = ndimage.binary_dilation(m, iterations=5) & ~(d["gt"] > 0) & d["mask"]
    if ring.sum() < 5:
        return None
    return d["z"][m].mean() - d["z"][ring].mean()


rows = []
for r in regions:
    for kind in ("карта", "профиль"):
        dd = [d for d in DS if d["region"] == r and d["kind"] == kind]
        if not dd:
            continue
        cs = [c for d in dd for g in d["polys"] if (c := object_contrast(d, g)) is not None]
        n_gt = sum(len(d["polys"]) for d in dd)
        areas = [g.area for d in dd for g in d["polys"]]
        rows.append((r, kind, len(dd), n_gt, np.median(areas), np.mean(np.array(cs) > 0)))

display(pd.DataFrame(rows, columns=["участок", "вид", "растров", "полигонов", "медиана S, м²", "доля высокоомных"]))

fig, ax = plt.subplots(figsize=(10, 4))
labels = [f"{r}\n{k}" for r, k, *_ in rows]
hi = np.array([x[-1] for x in rows])
ax.bar(labels, hi, color="firebrick", label="высокоомные (z выше окружения)")
ax.bar(labels, 1 - hi, bottom=hi, color="steelblue", label="низкоомные (z ниже окружения)")
ax.set_ylabel("доля полигонов")
ax.set_title("Полярность размеченных аномалий по участкам")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
plt.tight_layout()
fig.savefig(FIG_DIR / "02_etgr_polarity.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
for kind in ("карта", "профиль"):
    tr_regs = sorted({d["region"] for d in DS if d["kind"] == kind and d["region"] != HOLDOUT_REGION})
    n_tr = sum(len(d["polys"]) for d in DS if d["kind"] == kind and d["region"] != HOLDOUT_REGION)
    n_va = sum(len(d["polys"]) for d in DS if d["kind"] == kind and d["region"] == HOLDOUT_REGION)
    print(f"{kind:>8}: train участки {tr_regs}, {n_tr} полигонов; отложенный {HOLDOUT_REGION}, {n_va} полигонов")
# ----CELL
FEATURE_NAMES = ["знак", "фон σ", "уровень", "log площади", "доля площади", "z среднее", "z макс",
                 "контраст с кольцом", "z без фона", "solidity", "eccentricity", "компактность",
                 "удал. от края ср.", "удал. от края мин.", "длина / размер растра", "профиль?"]


def candidates(d):
    m = d["mask"]
    edt = ndimage.distance_transform_edt(m)
    edt_n = edt / (edt.max() + 1e-6)
    valid_area = m.sum()
    zs = ndimage.gaussian_filter(d["z"], SMOOTH)
    z_std = np.std(zs[m])
    polys, feats = [], []
    for bg in BG_SIGMAS:
        z = zs - ndimage.gaussian_filter(d["z"], bg) if bg else zs
        z = z / np.std(z[m])
        for sign in (1, -1):
            for lv in LEVELS:
                lab = label((sign * z > lv) & m, connectivity=2)
                for rp in regionprops(lab):
                    if rp.area < MIN_CAND_PX or rp.area > 0.5 * valid_area:
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
                    poly = Polygon([rasterio.transform.xy(d["tr"], p[1] + y0, p[0] + x0)
                                    for p in cnt.squeeze(1)]).buffer(0)
                    if poly.is_empty:
                        continue
                    e = edt_n[y0:y1, x0:x1][reg]
                    polys.append(poly)
                    feats.append([sign, bg, lv, np.log(rp.area), rp.area / valid_area, zin.mean(), zin.max(),
                                  zin.mean() - zring.mean(), (sign * zs[y0:y1, x0:x1][reg]).mean() / z_std,
                                  rp.solidity, rp.eccentricity, 4 * np.pi * rp.area / max(rp.perimeter, 1) ** 2,
                                  e.mean(), e.min(), rp.axis_major_length / max(z.shape), d["kind"] == "профиль"])
    return polys, np.array(feats, np.float32).reshape(-1, len(FEATURE_NAMES))


def hit_flags(polys, gts):
    # по критерию платформы: какие кандидаты засчитали бы какой-нибудь объект и какие объекты кем-то засчитаны;
    # плюс максимальный IoU кандидата с полигонами разметки
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
    d["cand_polys"], d["cand_X"] = candidates(d)
    d["cand_hit"], d["gt_reached"], d["cand_iou"] = hit_flags(d["cand_polys"], d["polys"])
    d["cand_pos"] = d["cand_hit"] & (d["cand_iou"] >= LABEL_IOU)
print(f"Кандидатов: {sum(len(d['cand_polys']) for d in DS)}\n")

rows = []
for r in regions:
    for kind in ("карта", "профиль"):
        dd = [d for d in DS if d["region"] == r and d["kind"] == kind]
        if not dd:
            continue
        n_gt = sum(len(d["polys"]) for d in dd)
        hit = sum(d["gt_reached"].sum() for d in dd)
        rows.append({"участок": r, "вид": kind, "полигонов": n_gt, "кандидатов": sum(len(d["cand_polys"]) for d in dd),
                     "оракул, %": round(100 * hit / n_gt)})
pd.DataFrame(rows)
# ----CELL
d = [d for d in DS if d["region"] == "215" and d["kind"] == "карта"][1]
fig, axes = plt.subplots(1, 3, figsize=(15, 8))
ext = extent(d["z"].shape, d["tr"])
for ax, (title, sel) in zip(axes, [("все кандидаты", np.ones(len(d["cand_polys"]), bool)),
                                    (f"положительные: засчитаны платформой и IoU ≥ {LABEL_IOU}", d["cand_pos"]),
                                    ("разметка", None)]):
    ax.imshow(np.where(d["mask"], d["z"], np.nan), cmap="RdBu_r", vmin=-3, vmax=3, extent=ext)
    if sel is None:
        plot_polys(ax, d["polys"], color="black", lw=1.5)
    else:
        for p, s in zip(d["cand_polys"], sel):
            if s:
                plot_polys(ax, [p], color="lime", lw=0.6)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=6)
plt.suptitle(f"{d['region']} {d['stem']}: кандидаты", fontsize=13)
plt.tight_layout()
fig.savefig(FIG_DIR / "02_etgr_candidates.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
PROFILE_EXTRA = ["глубина верха, м", "глубина низа, м", "карты под полигоном: среднее", "карты под полигоном: макс.",
                 "доля карт с аномалией"]
PROFILE_FEATURE_NAMES = FEATURE_NAMES + PROFILE_EXTRA
MISSING = -9.0


def nested_children(d):
    # дети кандидата: кандидаты того же знака и фона на следующем уровне, лежащие внутри него
    P, X = d["cand_polys"], d["cand_X"]
    isg, ibg, ilv = (FEATURE_NAMES.index(k) for k in ("знак", "фон σ", "уровень"))
    kids = [[] for _ in P]
    if not P:
        return kids
    tree = STRtree([p.representative_point() for p in P])
    for i, p in enumerate(P):
        for j in tree.query(p):
            if (j != i and X[j, isg] == X[i, isg] and X[j, ibg] == X[i, ibg]
                    and abs(X[j, ilv] - X[i, ilv] - 0.5) < 1e-6 and p.contains(P[j].representative_point())):
                kids[i].append(j)
    return kids


def mean_under(poly, d, sign):
    # среднее sign * z карты d под полигоном (по координатам: сетки карт одного участка немного различаются)
    H, W = d["z"].shape
    win = from_bounds(*poly.bounds, transform=d["tr"]).round_offsets().round_lengths()
    r0, c0 = max(int(win.row_off), 0), max(int(win.col_off), 0)
    r1, c1 = min(int(win.row_off + win.height) + 1, H), min(int(win.col_off + win.width) + 1, W)
    if r1 <= r0 or c1 <= c0:
        return np.nan
    tr = d["tr"] * d["tr"].translation(c0, r0)
    m = rasterize([(poly, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, all_touched=True).astype(bool)
    m &= d["mask"][r0:r1, c0:c1]
    return float(sign * d["z"][r0:r1, c0:c1][m].mean()) if m.sum() >= 3 else np.nan


def profile_extra(d, maps):
    X = d["cand_X"]
    out = np.full((len(X), len(PROFILE_EXTRA)), MISSING, np.float32)
    m = d["mask"]
    surf = None
    if m.shape[1] > m.shape[0]:
        cols = np.where(m.any(0))[0]
        top = np.array([np.argmax(m[:, c]) for c in cols])
        surf = rasterio.transform.xy(d["tr"], top, cols)
    for i, p in enumerate(d["cand_polys"]):
        if surf is not None:
            y_surf = np.interp(p.centroid.x, surf[0], surf[1])
            out[i, :2] = [y_surf - p.bounds[3], y_surf - p.bounds[1]]
        v = np.array([mean_under(p, e, X[i, FEATURE_NAMES.index("знак")]) for e in maps])
        v = v[~np.isnan(v)]
        if len(v):
            out[i, 2:] = [v.mean(), v.max(), (v > 0.5 * X[i, FEATURE_NAMES.index("уровень")]).mean()]
    return out


for d in DS:
    if d["kind"] == "карта":
        d["kids"] = nested_children(d)
    else:
        maps = [e for e in DS if e["region"] == d["region"] and e["kind"] == "карта"]
        d["cand_X"] = np.hstack([d["cand_X"], profile_extra(d, maps)])
# ----CELL
def crop_box(img, r0, r1, c0, c1, pad=0.5, min_side=16):
    # квадратное окно вокруг bbox с полями pad (доля размера); вне растра нули
    H, W = img.shape[:2]
    side = max(r1 - r0, c1 - c0, min_side) * (1 + 2 * pad)
    cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
    a0, a1, b0, b1 = int(cy - side / 2), int(cy + side / 2), int(cx - side / 2), int(cx + side / 2)
    out = np.zeros((a1 - a0, b1 - b0, img.shape[2]), np.float32)
    y0, y1, x0, x1 = max(a0, 0), min(a1, H), max(b0, 0), min(b1, W)
    if y1 > y0 and x1 > x0:
        out[y0 - a0:y1 - a0, x0 - b0:x1 - b0] = img[y0:y1, x0:x1]
    return out


def candidate_crops(d):
    # N x 4 x CROP x CROP (uint8): RGB-окно и маска кандидата
    img = d["rgb"].transpose(1, 2, 0).astype(np.float32) / 255.0
    inv = ~d["tr"]
    out = np.zeros((len(d["cand_polys"]), 4, CROP, CROP), np.uint8)
    for i, p in enumerate(d["cand_polys"]):
        minx, miny, maxx, maxy = p.bounds
        c0, r0 = inv * (minx, maxy)
        c1, r1 = inv * (maxx, miny)
        box = (int(r0), int(r1) + 1, int(c0), int(c1) + 1)
        m = rasterize_polys([p], d["z"].shape, d["tr"]).astype(np.float32)[..., None]
        x = np.concatenate([crop_box(img, *box), crop_box(m, *box)], 2)
        x = F.interpolate(torch.from_numpy(x).permute(2, 0, 1)[None], size=(CROP, CROP), mode="bilinear",
                          align_corners=False)[0]
        out[i] = (x.numpy() * 255).round().astype(np.uint8)
    return out


NORM_MEAN = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(1, 4, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225, 0.5]).view(1, 4, 1, 1)


def augment(x):
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))


def train_cnn(X, y, seed, bs=128, lr=3e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = timm.create_model("resnet18", pretrained=True, num_classes=1, in_chans=4).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    steps = CNN_EPOCHS * max(1, len(neg) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    mean, std = NORM_MEAN.to(DEVICE), NORM_STD.to(DEVICE)
    net.train()
    for _ in range(steps):
        idx = np.r_[np.random.choice(pos, bs // 2), np.random.choice(neg, bs // 2)]  # сбалансированный батч
        xb = torch.from_numpy(X[idx]).to(DEVICE).float() / 255.0
        xb = torch.cat([augment(xb[i:i + 1]) for i in range(len(idx))])
        yb = torch.from_numpy(y[idx].astype(np.float32)).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            loss = F.binary_cross_entropy_with_logits(net((xb - mean) / std).squeeze(1).float(), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


@torch.no_grad()
def predict_cnn(net, X, bs=512):
    # среднее по отражениям и повороту
    mean, std = NORM_MEAN.to(DEVICE), NORM_STD.to(DEVICE)
    out = []
    for i in range(0, len(X), bs):
        xb = (torch.from_numpy(X[i:i + bs]).to(DEVICE).float() / 255.0 - mean) / std
        views = [xb, xb.flip(3), xb.flip(2), torch.rot90(xb, 1, (2, 3))]
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            out.append(torch.stack([torch.sigmoid(net(v).squeeze(1).float()) for v in views]).mean(0).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit_scorer(dd, kind):
    # возвращает функцию, которая оценивает кандидатов растра, а для профилей ещё и сам классификатор
    y = np.concatenate([d["cand_pos"] for d in dd])
    if kind == "карта":
        X = np.concatenate([d["crops"] for d in dd])
        nets = [train_cnn(X, y, seed) for seed in CNN_SEEDS]
        return (lambda d: np.mean([predict_cnn(n, d["crops"]) for n in nets], 0)), None
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                         random_state=SEED).fit(np.concatenate([d["cand_X"] for d in dd]), y.astype(int))
    return (lambda d: clf.predict_proba(d["cand_X"])[:, 1] if len(d["cand_X"]) else np.zeros(0)), clf


def nms(polys, scores, thr):
    # индексы принятых кандидатов
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


def predict_polys(d, scores, thr):
    # NMS, затем на картах разрезание слившихся пятен на вложенных кандидатов
    keep = nms(d["cand_polys"], scores, thr)
    if d["kind"] == "карта":
        out = []
        for i in keep:
            kids = [j for j in d["kids"][i] if scores[j] >= SPLIT_RATIO * thr]
            out += kids if len(kids) >= 2 else [i]
        keep = sorted(set(out))
    return [d["cand_polys"][i] for i in keep], [float(scores[i]) for i in keep]


def pool(dd, scores, thr):
    # предсказания и разметка нескольких растров одним пулом, как на платформе
    preds, confs, gts = [], [], []
    for d, s in zip(dd, scores):
        p, c = predict_polys(d, s, thr)
        preds += p
        confs += c
        gts += d["polys"]
    return preds, confs, gts


def evaluate(dd, scores, thr):
    tot = platform_counts(*pool(dd, scores, thr))
    return tot, platform_prf(*tot)


for d in DS:
    if d["kind"] == "карта":
        d["crops"] = candidate_crops(d)

THRS = np.round(np.arange(0.05, 0.91, 0.05), 2)
MODELS = {}

for kind in ("карта", "профиль"):
    train = [d for d in DS if d["kind"] == kind and d["region"] != HOLDOUT_REGION]
    tr_regs = sorted({d["region"] for d in train})
    oof = {}
    for r in tr_regs:
        scorer, _ = fit_scorer([d for d in train if d["region"] != r], kind)
        for d in train:
            if d["region"] == r:
                oof[id(d)] = scorer(d)
    oof_scores = [oof[id(d)] for d in train]
    curve = [(t, *evaluate(train, oof_scores, t)) for t in THRS]
    thr, (tp, fp, fn), (p, r_, f1) = max(curve, key=lambda x: x[2][2])
    scorer, clf = fit_scorer(train, kind)
    MODELS[kind] = dict(scorer=scorer, clf=clf, thr=thr, train=train, oof=oof_scores, curve=curve)

    print(f"{kind}: порог по OOF {thr}: P={p:.3f} R={r_:.3f} F1={f1:.3f}  (TP={tp}, FP={fp}, FN={fn})")
    per = [evaluate([d for d in train if d["region"] == r],
                    [s for d, s in zip(train, oof_scores) if d["region"] == r], thr)[1] for r in tr_regs]
    display(pd.DataFrame(per, index=pd.Index(tr_regs, name="OOF"), columns=["P", "R", "F1"]))
# ----CELL
fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
for ax, kind in zip(axes[:2], ("карта", "профиль")):
    c = MODELS[kind]["curve"]
    ax.plot([x[0] for x in c], [x[2][0] for x in c], label="Precision")
    ax.plot([x[0] for x in c], [x[2][1] for x in c], label="Recall")
    ax.plot([x[0] for x in c], [x[2][2] for x in c], label="F1", lw=2)
    ax.axvline(MODELS[kind]["thr"], color="gray", ls="--")
    ax.set_xlabel("порог вероятности")
    ax.set_title(f"{kind}: OOF на train-участках (метрика платформы)")
    ax.grid(True, alpha=0.3)
    ax.legend()
# permutation importance градиентного бустинга профилей: насколько падает average precision, если перемешать признак
tr = MODELS["профиль"]["train"]
X = np.concatenate([d["cand_X"] for d in tr])
y = np.concatenate([d["cand_pos"] for d in tr]).astype(int)
imp = permutation_importance(MODELS["профиль"]["clf"], X, y, scoring="average_precision",
                             n_repeats=5, random_state=SEED, n_jobs=-1).importances_mean
axes[2].bar(np.arange(len(PROFILE_FEATURE_NAMES)), imp)
axes[2].set_xticks(np.arange(len(PROFILE_FEATURE_NAMES)))
axes[2].set_xticklabels(PROFILE_FEATURE_NAMES, rotation=60, ha="right", fontsize=8)
axes[2].set_title("Профили: permutation importance (train, average precision)")
plt.tight_layout()
fig.savefig(FIG_DIR / "02_etgr_threshold_importance.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
HOLDOUT_PRED = {}
for kind in ("карта", "профиль"):
    mdl = MODELS[kind]
    hold = [d for d in DS if d["kind"] == kind and d["region"] == HOLDOUT_REGION]
    HOLDOUT_PRED[kind] = (hold, [mdl["scorer"](d) for d in hold])


def report(preds, confs, gts):
    tp, fp, fn = platform_counts(preds, confs, gts)
    return [tp, fp, fn, *platform_prf(tp, fp, fn), f"{touched(preds, gts)}/{len(gts)}"]


rows = {}
all_p, all_c, all_g = [], [], []
for kind in ("карта", "профиль"):
    hold, scores = HOLDOUT_PRED[kind]
    ml = pool(hold, scores, MODELS[kind]["thr"])
    rows[f"{kind}: кандидаты + ML"] = report(*ml)
    all_p += ml[0]
    all_c += ml[1]
    all_g += ml[2]
rows[f"{HOLDOUT_REGION} целиком: кандидаты + ML"] = report(all_p, all_c, all_g)
display(pd.DataFrame.from_dict(rows, orient="index", columns=["TP", "FP", "FN", "P", "R", "F1", "задето"]))

per_raster = []
for kind in ("карта", "профиль"):
    hold, scores = HOLDOUT_PRED[kind]
    for d, s in zip(hold, scores):
        tp, fp, fn = platform_counts(*pool([d], [s], MODELS[kind]["thr"]))
        per_raster.append({"растр (кандидаты + ML)": d["stem"], "GT": len(d["polys"]), "TP": tp, "FP": fp, "FN": fn,
                           "F1": platform_prf(tp, fp, fn)[2]})
pd.DataFrame(per_raster)
# ----CELL
show = HOLDOUT_PRED["карта"][0][:3] + HOLDOUT_PRED["профиль"][0][:3]
scores = HOLDOUT_PRED["карта"][1][:3] + HOLDOUT_PRED["профиль"][1][:3]
fig, axes = plt.subplots(len(show), 2, figsize=(18, 4 * len(show)))
for row, d, s in zip(axes, show, scores):
    ext = extent(d["z"].shape, d["tr"])
    pred, conf = predict_polys(d, s, MODELS[d["kind"]]["thr"])
    row[0].imshow(d["rgb"].transpose(1, 2, 0), extent=ext)
    plot_polys(row[0], d["polys"], color="black", lw=1.2)
    row[0].set_title(f"{d['stem']}: разметка", fontsize=9)
    row[1].imshow(np.where(d["mask"], d["z"], np.nan), cmap="RdBu_r", vmin=-3, vmax=3, extent=ext)
    plot_polys(row[1], d["polys"], color="black", lw=1.2)
    plot_polys(row[1], pred, color="lime", lw=1.5, ls="--")
    tp, fp, fn = platform_counts(pred, conf, d["polys"])
    row[1].set_title(f"предсказание (зелёный пунктир): TP={tp} FP={fp} FN={fn}", fontsize=9)
    for ax in row:
        ax.tick_params(labelsize=6)
plt.suptitle(f"Отложенный участок {HOLDOUT_REGION}: разметка (чёрный) и предсказание (зелёный)", fontsize=13)
plt.tight_layout()
fig.savefig(FIG_DIR / "02_etgr_holdout_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
# ----CELL
def loso(kind):
    dd = [d for d in DS if d["kind"] == kind]
    regs = sorted({d["region"] for d in dd})
    sc = {}
    for r in regs:
        scorer, _ = fit_scorer([d for d in dd if d["region"] != r], kind)
        for d in dd:
            if d["region"] == r:
                sc[id(d)] = scorer(d)
    scores = [sc[id(d)] for d in dd]
    thr = max(THRS, key=lambda t: evaluate(dd, scores, t)[1][2])
    per = {r: evaluate([d for d in dd if d["region"] == r], [sc[id(d)] for d in dd if d["region"] == r], thr)[1]
           for r in regs}
    return thr, evaluate(dd, scores, thr)[1], per


rows = {}
for kind in ("карта", "профиль"):
    thr, (p, rr, f1), per = loso(kind)
    rows[kind] = {"порог": thr, "P": p, "R": rr, "F1": f1, **{r: v[2] for r, v in per.items()}}
pd.DataFrame.from_dict(rows, orient="index")
# ----CELL
items = [(d, *predict_polys(d, s, MODELS[kind]["thr"]))
         for kind in ("карта", "профиль") for d, s in zip(*HOLDOUT_PRED[kind])]
out = PRED_DIR / f"02_etgr_holdout_{HOLDOUT_REGION}.geojson"
fc = to_platform_geojson(items, "etgr", out)
print(f"{out.relative_to(ROOT)}: {len(fc['features'])} полигонов, {out.stat().st_size / 1e6:.2f} МБ, "
      f"макс. вершин {max(n_vertices(q) for _, pp, _ in items for q in pp)}")
print("Пример свойств:", fc["features"][0]["properties"])

# проверка: переводим файл обратно в UTM и считаем метрику
crs_of = {f"{d['region']}_{d['site']}": d["crs"] for d in DS}
back = json.loads(out.read_text(encoding="utf-8"))["features"]
polys = [shape(transform_geom("EPSG:3857", crs_of[f["properties"]["region_name"]], f["geometry"])) for f in back]
confs = [f["properties"]["confidence"] for f in back]
gts = [g for d, _, _ in items for g in d["polys"]]
from_file = platform_counts(polys, confs, gts)
print(f"Из файла: TP={from_file[0]} FP={from_file[1]} FN={from_file[2]}  F1={platform_prf(*from_file)[2]:.3f}")
in_memory = platform_counts([q for _, pp, _ in items for q in pp], [c for _, _, cc in items for c in cc], gts)
print(f"В памяти: TP={in_memory[0]} FP={in_memory[1]} FN={in_memory[2]}; разница TP: {from_file[0] - in_memory[0]:+d}")