"""Детектор окон спутника (YOLO26, TorchScript) — план docs/plans/OPTICS_DETECTOR_PLAN_20261105.md.

Без ultralytics на площадке: веса экспортируются в TorchScript (`model.export(format="torchscript")`), выход головы —
тензор [1, 4 + nc, N]: cx, cy, w, h в пикселях входа imgsz и уверенности классов (сигмоида уже применена).
Разбор и NMS по классам (torchvision.ops.batched_nms) — здесь. OOF для порогов считается этим же кодом
(kaggle/2026-11-06/opt_yolo), поэтому шкала уверенности у прода и OOF одна по построению.

Цепочка как у выгрузки окон (kaggle/2026-11-06/opt_yolo/export_spor.py):
1. GeoTIFF → UTM участка с шагом gsd (вниз — average, вверх — bilinear), alpha/нули → вне снимка.
2. Окна tile×tile с перекрытием; окно с долей снимка < 0,4 пропускается.
3. JPEG q95 туда и обратно — как окна обучения.
4. Ресайз до imgsz (cv2 INTER_LINEAR, как ultralytics; без cv2 — PIL BILINEAR).
5. Центр рамки → мир → дубли участка по радиусу класса → круг 0,5 м (курганы) / диск 0,05 м (прочие).

``replace_sp`` снимает уже найденный спутник участка. Проба 06.11 в этом режиме дала 0,38441.
Чужой плюс 0,440→0,489 дал дополнительный детектор поверх своих, не замена ветки.
По умолчанию ``replace_sp`` выключен: новые объекты только далеко от уже принятых.
"""
from __future__ import annotations

import io
import logging

import numpy as np

log = logging.getLogger(__name__)

CLASSES = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
RADIUS_M = {"kurgany_povrezhdennye": 6.0, "kurgany_tselye": 6.0, "arkhitektury": 6.0, "fortifikatsii": 10.0,
            "gorodishcha": 40.0}
CIRCLE_M = {"kurgany_povrezhdennye": 0.5, "kurgany_tselye": 0.5}
DEFAULTS = {"enabled": False, "weights": "", "gsd": 0.5, "tile": 1024, "overlap": 0.25, "imgsz": 640,
            "conf_min": 0.01, "iou": 0.6, "max_det": 300, "batch": 8, "threshold_group": "sp_yolo",
            "classes": list(CLASSES), "jpeg_q": 95, "replace_sp": False, "dedupe_m": 8.0}


def cfg_of(cfg: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update((cfg or {}).get("sp_yolo") or {})
    return o


def warp(src_path, crs: str, gsd: float):
    """GeoTIFF → (rgb uint8 [3,H,W], valid bool [H,W], affine) в crs с шагом gsd — как export_spor.warp."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    with rasterio.open(src_path) as src:
        tr, W, H = calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds, resolution=gsd)
        src_res = abs(src.transform.a) if src.crs and src.crs.is_projected else None
        rs = Resampling.average if (src_res is None or src_res < gsd) else Resampling.bilinear
        nb = min(src.count, 4)
        out = np.zeros((nb, H, W), np.uint8)
        for b in range(nb):
            band = src.read(b + 1)
            if band.dtype != np.uint8:
                hi = np.percentile(band[band > 0], 99.5) if (band > 0).any() else 1.0
                band = np.clip(band.astype(np.float32) / max(hi, 1e-6) * 255, 0, 255).astype(np.uint8)
            reproject(band, out[b], src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=crs,
                      resampling=rs, src_nodata=None, dst_nodata=0)
    rgb = out[:3] if nb >= 3 else np.repeat(out[:1], 3, axis=0)
    valid = (out[3] > 0) if nb >= 4 else (rgb.max(axis=0) > 0)
    return rgb, valid, tr


def windows(H: int, W: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    step = max(1, int(tile * (1 - overlap)))
    ys = sorted(set(list(range(0, max(1, H - tile), step)) + [max(0, H - tile)]))
    xs = sorted(set(list(range(0, max(1, W - tile), step)) + [max(0, W - tile)]))
    return [(y, x) for y in ys for x in xs]


def tile_rgb(rgb: np.ndarray, yy: int, xx: int, tile: int) -> np.ndarray:
    t = np.zeros((tile, tile, 3), np.uint8)
    crop = rgb[:, yy:yy + tile, xx:xx + tile].transpose(1, 2, 0)
    t[:crop.shape[0], :crop.shape[1]] = crop
    return t


def jpeg_roundtrip(t: np.ndarray, q: int) -> np.ndarray:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(t).save(buf, format="JPEG", quality=int(q))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def resize(t: np.ndarray, imgsz: int) -> np.ndarray:
    if t.shape[0] == imgsz and t.shape[1] == imgsz:
        return t
    try:
        import cv2
        return cv2.resize(t, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(t).resize((imgsz, imgsz), Image.BILINEAR))


class Detector:
    """TorchScript YOLO: predict(tiles RGB uint8 [n,S,S,3]) → [(xyxy в пикселях окна S, conf, cls), …] на окно."""

    def __init__(self, path, device: str | None = None, imgsz: int = 640, conf_min: float = 0.01, iou: float = 0.6,
                 max_det: int = 300):
        import torch
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = torch.jit.load(str(path), map_location=self.device).eval()
        self.imgsz, self.conf_min, self.iou, self.max_det = int(imgsz), float(conf_min), float(iou), int(max_det)

    def predict(self, tiles: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        import torchvision
        torch = self.torch
        if not tiles:
            return []
        S = tiles[0].shape[0]
        x = np.stack([resize(t, self.imgsz) for t in tiles]).astype(np.float32) / 255.0
        xb = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(self.device)
        with torch.inference_mode():
            y = self.net(xb)
        y = (y[0] if isinstance(y, (list, tuple)) else y).float()
        out = []
        k = S / self.imgsz
        for p in y:                                   # [4 + nc, N]
            box, sc = p[:4].T, p[4:].T                # [N, 4], [N, nc]
            conf, cls = sc.max(1)
            m = conf >= self.conf_min
            box, conf, cls = box[m], conf[m], cls[m]
            if not len(conf):
                out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)))
                continue
            xyxy = torch.stack([box[:, 0] - box[:, 2] / 2, box[:, 1] - box[:, 3] / 2,
                                box[:, 0] + box[:, 2] / 2, box[:, 1] + box[:, 3] / 2], 1)
            keep = torchvision.ops.batched_nms(xyxy, conf, cls, self.iou)[:self.max_det]
            out.append(((xyxy[keep] * k).cpu().numpy(), conf[keep].cpu().numpy(), cls[keep].cpu().numpy()))
        return out


def dedup_features(points: dict, region: str, group: str, classes=CLASSES) -> list[dict]:
    """points[(class_idx)] = [(conf, X, Y, crs)] в CRS участка → объекты сдачи в EPSG:3857 (жадно по радиусу класса)."""
    from pyproj import Transformer
    from shapely.geometry import Point, mapping
    feats, tfs = [], {}
    for k, lst in sorted(points.items()):
        name = classes[k]
        lst = sorted(lst, key=lambda t: -t[0])
        rad2 = RADIUS_M.get(name, 6.0) ** 2
        keep: list = []
        for c, X, Y, crs in lst:
            if any((X - kx) ** 2 + (Y - ky) ** 2 < rad2 for _, kx, ky, _ in keep):
                continue
            keep.append((c, X, Y, crs))
        for c, X, Y, crs in keep:
            if crs not in tfs:
                tfs[crs] = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform
            gx, gy = tfs[crs](X, Y)
            r_m = CIRCLE_M.get(name, 0.05)
            g = Point(gx, gy).buffer(r_m / max(np.cos(np.arctan(np.sinh(gy / 6378137.0))), 1e-6), quad_segs=4)
            feats.append({"type": "Feature", "geometry": mapping(g), "geom": g, "class_name": name,
                          "confidence": round(float(c), 5), "region_name": region, "markup_type": "sp_or",
                          "threshold_group": group})
    return feats


def site_feats(det: Detector, rasters, crs: str, opts: dict, region: str, deadline=None) -> list[dict]:
    """Все снимки SpOr участка → объекты группы opts['threshold_group'] (до порогов и полов)."""
    points: dict = {}
    for path in rasters:
        if deadline is not None and deadline():
            log.warning("%s: sp_yolo — время на исходе, остальные снимки без YOLO", region)
            break
        rgb, valid, tr = warp(path, crs, float(opts["gsd"]))
        S = int(opts["tile"])
        win = [(yy, xx) for yy, xx in windows(*valid.shape, S, float(opts["overlap"]))
               if valid[yy:yy + S, xx:xx + S].size and valid[yy:yy + S, xx:xx + S].mean() >= 0.4]
        for i in range(0, len(win), int(opts["batch"])):
            part = win[i:i + int(opts["batch"])]
            tiles = [jpeg_roundtrip(tile_rgb(rgb, yy, xx, S), opts["jpeg_q"]) for yy, xx in part]
            for (yy, xx), (xyxy, conf, cls) in zip(part, det.predict(tiles)):
                for (bx0, by0, bx1, by1), c, k in zip(xyxy, conf, cls):
                    X, Y = tr * (xx + (bx0 + bx1) / 2, yy + (by0 + by1) / 2)
                    points.setdefault(int(k), []).append((float(c), X, Y, crs))
        del rgb, valid
    keep = set(opts.get("classes") or CLASSES)
    feats = dedup_features(points, region, str(opts["threshold_group"]))
    return [f for f in feats if f["class_name"] in keep]


def dedupe_against(fresh: list[dict], pool: list[dict], passes, dist_m: float = 8.0) -> list[dict]:
    """Вариант K: объект YOLO снимается, если ближе dist_m есть объект того же класса участка, проходящий порог
    своей группы (passes(f) -> bool). Расстояние — между центрами в EPSG:3857 с поправкой масштаба по широте."""
    from shapely.geometry import shape
    if not fresh:
        return []
    anchors: dict = {}
    for f in pool:
        if f.get("class_name") in CLASSES and passes(f):
            g = f.get("geom") or shape(f["geometry"])
            c = g.centroid
            anchors.setdefault(f["class_name"], []).append((c.x, c.y))
    out = []
    for f in fresh:
        c = f["geom"].centroid
        k = max(np.cos(np.arctan(np.sinh(c.y / 6378137.0))), 1e-6)       # метры EPSG:3857 → метры на местности
        lim2 = (dist_m / k) ** 2
        if any((c.x - ax) ** 2 + (c.y - ay) ** 2 < lim2 for ax, ay in anchors.get(f["class_name"], [])):
            continue
        out.append(f)
    return out
