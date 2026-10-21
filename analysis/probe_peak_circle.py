"""Один отложенный могильник: полигон маски против круга в пике вероятности.

Модель фолда 0 не видела семью «Курганы». Круг засчитывается, если его центр
внутри эталона: покрытие меньшей фигуры равно 1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import Point, mapping, shape

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.crs import WEB_MERCATOR, to_crs  # noqa: E402
from expds.infer_nn import SPLIT_RADIUS_M, Segmenter, split_by_peaks  # noqa: E402
from expds.metric import sweep  # noqa: E402
from expds.pipeline import _raster_with_model  # noqa: E402
from expds.tiles import MAX_UPSAMPLE, band_stats, fix_channels, to_uint8, working_res  # noqa: E402
from expds import accel  # noqa: E402

METRIC = r"C:\Sloi\files\materials\compute_metrics.py"
REGION = "163_Курганы_34"
TIF = Path(r"F:\expds3\train-expds-3\163_Курганы_34\04_Курганы_34_SpOr_спутник\01_Курганы_34_SpOr_спутник.tif")
MODEL = Path(r"F:\expds3\work\spor_r1_oof\fold0\sp_or.pt")
GT_PATH = Path(r"F:\expds3\work\gt_train.geojson")
RADII = (6.0, 10.0, 15.0, 22.0)


def gt_region():
    fc = json.loads(GT_PATH.read_text(encoding="utf-8"))
    feats = [f for f in fc["features"]
             if (f.get("properties") or {}).get("region_name") == REGION
             and (f.get("properties") or {}).get("markup_type") == "sp_or"
             and (f.get("properties") or {}).get("class_name") in ("kurgany_povrezhdennye", "kurgany_tselye")]
    return {"type": "FeatureCollection", "features": feats}


def as_fc(feats):
    out = []
    for f in feats:
        g = f["geom"]
        pr = {k: v for k, v in f.items() if k != "geom"}
        out.append({"type": "Feature", "geometry": mapping(g), "properties": pr})
    return {"type": "FeatureCollection", "features": out}


def circles(seg, path: Path, radius_m: float) -> list[dict]:
    stats = band_stats(str(path), "sp_or")
    feats = []
    with rasterio.open(path) as ds:
        res = working_res(abs(ds.res[0]), seg.res, seg.meta.get("max_upsample") or MAX_UPSAMPLE)
        scale = res / abs(ds.res[0])
        w, h = max(1, int(ds.width / scale)), max(1, int(ds.height / scale))
        n = min(ds.count, 3)
        crs = ds.crs.to_string() if ds.crs else None
        block, overlap = 4096, 256
        half = overlap // 2
        step = block - overlap
        xs = list(range(0, max(w - overlap, 1), step))
        ys = list(range(0, max(h - overlap, 1), step))
        for iy, y0 in enumerate(ys):
            for ix, x0 in enumerate(xs):
                bw, bh = min(block, w - x0), min(block, h - y0)
                win = Window(x0 * scale, y0 * scale, bw * scale, bh * scale)
                a = ds.read(list(range(1, n + 1)), window=win, out_shape=(n, bh, bw),
                            resampling=Resampling.average if scale > 1 else Resampling.bilinear)
                if not a.any():
                    continue
                tr = ds.window_transform(win) @ rasterio.Affine.scale(win.width / bw, win.height / bh)
                img = fix_channels(to_uint8(a, "sp_or", stats), int(seg.meta["in_channels"]))
                probs = seg.predict(img)
                c0 = (half if ix > 0 else 0, half if iy > 0 else 0)
                c1 = (bw - (half if ix < len(xs) - 1 else 0), bh - (half if iy < len(ys) - 1 else 0))
                for k, cls in enumerate(seg.classes):
                    p = probs[k]
                    mask = p >= 0.05
                    if not mask.any():
                        continue
                    rad = int(round(SPLIT_RADIUS_M.get(cls, 0.0) / res)) if SPLIT_RADIUS_M.get(cls, 0.0) else 0
                    inst = split_by_peaks(p, mask, rad, 0.1)
                    nlab = int(inst.max())
                    if nlab == 0:
                        continue
                    peak = accel.label_max(inst, p, nlab)
                    flat = inst.ravel()
                    score = np.where(inst > 0, p, -np.inf).ravel()
                    order = np.argsort(score, kind="stable")
                    best_i = np.full(nlab + 1, -1)
                    best_i[flat[order]] = order
                    for L in range(1, nlab + 1):
                        i = best_i[L]
                        if i < 0 or peak[L] < 0.3:
                            continue
                        rr, cc = divmod(int(i), inst.shape[1])
                        if not (c0[1] <= rr < c1[1] and c0[0] <= cc < c1[0]):
                            continue
                        x, y = tr * (cc + 0.5, rr + 0.5)
                        geom = Point(x, y).buffer(radius_m, quad_segs=4)
                        if crs and crs != WEB_MERCATOR:
                            geom = to_crs(geom, crs, WEB_MERCATOR)
                        feats.append({"geom": geom, "class_name": cls, "confidence": float(peak[L]),
                                      "region_name": REGION, "markup_type": "sp_or", "source": path.name,
                                      "original_crs": crs})
    return feats


def report(name, feats, gt):
    res = sweep(as_fc(feats), gt, METRIC)
    bits = []
    for cls, v in res["per_class"].items():
        if v["n_gt"]:
            bits.append(f"{cls.split('_')[-1][:4]} F1 {v['f1']:.3f} порог {v['threshold']:.3f} GT {v['n_gt']}")
    print(f"{name}: объектов {len(feats)}  балл {res['estimated_score']:.3f}  " + " | ".join(bits), flush=True)


def main():
    print("модель", MODEL.exists(), "снимок", TIF.exists(), flush=True)
    gt = gt_region()
    print("эталон", len(gt["features"]), flush=True)
    seg = Segmenter(MODEL)
    polys = _raster_with_model(seg, TIF, REGION, "sp_or")
    report("полигон", polys, gt)
    for r in RADII:
        report(f"круг {r:.0f} м", circles(seg, TIF, r), gt)


if __name__ == "__main__":
    main()
