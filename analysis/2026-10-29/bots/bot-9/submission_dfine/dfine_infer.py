"""D-FINE SpOr (вариант R) — самодостаточный модуль инференса для пакета КОЗ №3 v3 (бот 9 → бот 7).

Повторяет инференс обучающего ядра bot-9/kaggle/dfine_spor/src/dfine_main.py (Scene :262-295, infer_region :459-505):
  WarpedVRT → UTM участка, 0,5 м/пкс (average при уменьшении, иначе bilinear), nodata 0, RGB uint8 (не-uint8 — перцентили 1–99),
  окна 640 с перекрытием 160, пропуск окон с >95 % nodata, flip-TTA, fp16 autocast на CUDA, top-100 по (query×класс),
  min_score 0,05, batched NMS 0,5 по классу на весь tif, эллипс в рамке → EPSG:3857.
Затем — круги варианта R/C, как metric/prep5.py fam C (= bot-10 coco_preds_to_stack): кластер по участку и классу
(центр ≤ 4 м на земле или покрытие рамок ≥ 0,5), круг r = 4 м в центре лучшего, confidence лучшего.

Отличия от ядра (на результат при одном фолде не влияют):
  * чтение окон в потоках с предвыборкой (у каждого потока свой rasterio-датасет); порядок окон и батчей тот же;
  * несколько фолдов → детекции всех моделей конкатенируются и проходят тот же batched NMS (как flip-TTA). НЕ ПРОВЕРЕНО OOF;
  * дедлайн: новые батчи не начинаются после deadline (time.time()), уже найденное возвращается.

Использование (в пакете):
    from dfine_infer import DFineSpOr
    det = DFineSpOr([".../dfine_f0"], device="cuda", vendor_dir=".../vendor")   # загрузка ~ секунды
    feats = det.run_regions([(region.name, region.rasters.get("sp_or", []), region.utm_json) for region in regions],
                            deadline=t_end)                                    # list[GeoJSON Feature dict], EPSG:3857
Сеть не нужна: веса — локальные каталоги save_pretrained (config.json + model.safetensors), HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import concurrent.futures as cf
import collections
import json
import logging
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("dfine_spor")
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
TRANSFORMERS_PIN = "5.21.0"
CFG = dict(res=0.5, tile=640, overlap=160, bs=8, min_score=0.05, nms_iou=0.5, max_det=100, nodata_skip=0.95,
           circle_r_m=4.0, merge_m=4.0, cover=0.5, cell=100.0)
GROUP, MARKUP = "sp_dfine", "sp_or"


def _prepare_imports(vendor_dir: str | os.PathLike | None) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1"); os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    if vendor_dir and str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))


# ── модель: загрузка под замком + починка meta (dfine_main.py:121-178) ─────────────────────────────────────────────
MODEL_LOCK = threading.Lock()
_PTM_ORIG: dict = {}


def _meta_names(m):
    return [n for n, p in m.named_parameters(remove_duplicate=False) if p.is_meta] + \
           [n for n, b in m.named_buffers(remove_duplicate=False) if b.is_meta]


def _repair_meta(m, meta):
    log.warning("D-FINE: на meta %d тензоров (%s…) — tie_weights()", len(meta), ", ".join(meta[:3]))
    _PTM_ORIG.get("tie_weights", type(m).tie_weights)(m)
    meta = _meta_names(m)
    if meta:
        log.warning("D-FINE: после tie_weights на meta ещё %d — to_empty + _init_weights", len(meta))
        for mn in sorted({n.rsplit(".", 1)[0] if "." in n else "" for n in meta}):
            mod = m.get_submodule(mn); mod.to_empty(device="cpu", recurse=False); m._init_weights(mod)
        _PTM_ORIG.get("tie_weights", type(m).tie_weights)(m)
        meta = _meta_names(m)
    return meta


def check_model(m) -> dict:
    meta = _meta_names(m)
    if meta:
        raise RuntimeError(f"D-FINE: {len(meta)} тензоров на meta ({', '.join(meta[:4])}…)")
    K, d = len(CLASSES), m.config.d_model; dec = m.model.decoder
    assert m.config.num_labels == K, f"num_labels {m.config.num_labels} != {K}"
    for i, l in enumerate(dec.class_embed):
        assert tuple(l.weight.shape) == (K, d), f"decoder.class_embed[{i}] {tuple(l.weight.shape)}"
    assert tuple(m.model.enc_score_head.weight.shape) == (K, d)
    for a, b in [(m.class_embed, dec.class_embed), (m.bbox_embed, dec.bbox_embed)]:
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert pa.data_ptr() == pb.data_ptr(), "class_embed/bbox_embed не привязаны к model.decoder.*"
    return {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "num_labels": K}


def load_model(ckpt_dir: str | os.PathLike, device="cuda"):
    import torch
    import transformers
    from transformers import DFineForObjectDetection, PreTrainedModel
    if transformers.__version__ != TRANSFORMERS_PIN:
        raise RuntimeError(f"transformers {transformers.__version__} != {TRANSFORMERS_PIN} (пин D-FINE)")
    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / "config.json").is_file() or not (ckpt_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"нет config.json/model.safetensors в {ckpt_dir}")
    with MODEL_LOCK:
        if not _PTM_ORIG:
            _PTM_ORIG.update({n: PreTrainedModel.__dict__[n] for n in ("tie_weights", "init_weights") if n in PreTrainedModel.__dict__})
        for n, f in _PTM_ORIG.items():
            if PreTrainedModel.__dict__.get(n) is not f:
                setattr(PreTrainedModel, n, f)
        # dtype=float32: веса fp16 (export_fp16.py) поднимаются до fp32, вычисления — как в ядре (autocast fp16 на CUDA)
        m = DFineForObjectDetection.from_pretrained(str(ckpt_dir), dtype=torch.float32, device_map=None, local_files_only=True)
        meta = _meta_names(m)
        if meta:
            meta = _repair_meta(m, meta)
        info = check_model(m)
    m.eval().to(device)
    return m, info


# ── растр (dfine_main.py:238-295) ───────────────────────────────────────────────────────────────────────────────
def target_crs(utm_json, ds) -> str:
    if utm_json and Path(utm_json).is_file():
        try:
            d = json.loads(Path(utm_json).read_text(encoding="utf-8"))
            m = re.search(r"EPSG::?(\d+)", str(d.get("crs", "")))
            if m:
                return f"EPSG:{m.group(1)}"
        except Exception as e:  # битый UTM.json → зона по центру, как в ядре
            log.warning("UTM.json не прочитан (%s): %s", utm_json, e)
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(*ds.xy(ds.height // 2, ds.width // 2))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"


class SceneGrid:
    """Параметры сетки UTM 0,5 м для одного tif (считаются один раз в главном потоке)."""
    def __init__(self, path, utm_json):
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform
        self.path = str(path)
        with rasterio.open(self.path) as src:
            self.crs = target_crs(utm_json, src)
            tr, w, h = calculate_default_transform(src.crs, self.crs, src.width, src.height, *src.bounds, resolution=CFG["res"])
            try:
                src_m = abs(calculate_default_transform(src.crs, self.crs, src.width, src.height, *src.bounds)[0].a)
            except Exception:
                src_m = CFG["res"]
            self.rs = Resampling.average if src_m < CFG["res"] else Resampling.bilinear
            self.W, self.H, self.T = w, h, tr
            self.bands = list(range(1, min(3, src.count) + 1)); self.dtype = src.dtypes[0]
            self.stretch = None
            if self.dtype != "uint8":
                ov = src.read(1, out_shape=(min(1024, src.height), min(1024, src.width)))
                v = ov[ov > 0]; self.stretch = tuple(np.percentile(v, [1, 99])) if v.size else (0, 1)


class _ThreadReader:
    """WarpedVRT на поток (rasterio-датасеты не потокобезопасны; GDAL отпускает GIL при чтении)."""
    def __init__(self, g: SceneGrid):
        self.g = g; self.tl = threading.local(); self.all = []; self.lk = threading.Lock()

    def _vrt(self):
        v = getattr(self.tl, "vrt", None)
        if v is None:
            import rasterio
            from rasterio.vrt import WarpedVRT
            src = rasterio.open(self.g.path)
            v = WarpedVRT(src, crs=self.g.crs, transform=self.g.T, width=self.g.W, height=self.g.H, resampling=self.g.rs, nodata=0)
            self.tl.vrt = v
            with self.lk:
                self.all.append((v, src))
        return v

    def read(self, x0, y0, n):
        from rasterio.windows import Window
        g = self.g; a = np.zeros((len(g.bands), n, n), dtype=g.dtype)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(g.W, x0 + n), min(g.H, y0 + n)
        if cx1 > cx0 and cy1 > cy0:
            a[:, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = self._vrt().read(g.bands, window=Window(cx0, cy0, cx1 - cx0, cy1 - cy0))
        if a.shape[0] < 3:
            a = np.repeat(a[:1], 3, 0)
        if g.stretch:
            lo, hi = g.stretch; a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return np.ascontiguousarray(a.astype(np.uint8).transpose(1, 2, 0))

    def batch(self, wins):
        chunk, imgs = [], []
        for x, y in wins:
            im = self.read(x, y, CFG["tile"])
            if (im.max(2) == 0).mean() > CFG["nodata_skip"]:
                continue
            chunk.append((x, y)); imgs.append(im)
        return chunk, (np.stack(imgs) if imgs else None)

    def close(self):
        for v, s in self.all:
            v.close(); s.close()


def windows(g: SceneGrid):
    S = CFG["tile"] - CFG["overlap"]
    xs = list(range(0, max(1, g.W - CFG["overlap"]), S)) or [0]
    ys = list(range(0, max(1, g.H - CFG["overlap"]), S)) or [0]
    return [(x, y) for y in ys for x in xs]


# ── круги варианта R/C (metric/prep5.py:112-146) ────────────────────────────────────────────────────────────────
def mercator_scale(y: float) -> float:
    lat = np.arctan(np.sinh(float(y) / 6378137.0))
    return float(1.0 / max(np.cos(lat), 1e-6))


def circles(cands: list[tuple], region: str, cls: str) -> list[dict]:
    """cands: (cx, cy, conf, (x0, y0, x1, y1)) в EPSG:3857 → круги r = 4 м (одна точка на кластер)."""
    from shapely.geometry import Point
    CELL, MERGE, R = CFG["cell"], CFG["merge_m"], CFG["circle_r_m"]

    def cover(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
        if w <= 0 or h <= 0:
            return 0.0
        sm = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])); return w * h / sm if sm > 0 else 0.0

    def cells_of(bx, pad):
        return {(i, j) for i in range(int(math.floor((bx[0] - pad) / CELL)), int(math.floor((bx[2] + pad) / CELL)) + 1)
                for j in range(int(math.floor((bx[1] - pad) / CELL)), int(math.floor((bx[3] + pad) / CELL)) + 1)}

    items = sorted(cands, key=lambda c: -c[2]); grid = collections.defaultdict(list); clusters = []
    for c in items:
        x, y, sc, bx = c; hit = None; seen = set()
        for cell in cells_of(bx, MERGE * mercator_scale(y)):
            for k in grid.get(cell, ()):
                if k in seen:
                    continue
                seen.add(k)
                for m in clusters[k]:
                    if math.hypot(x - m[0], y - m[1]) / mercator_scale((y + m[1]) / 2) <= MERGE or cover(bx, m[3]) >= CFG["cover"]:
                        hit = k if hit is None else min(hit, k); break
        if hit is None:
            clusters.append([c]); hit = len(clusters) - 1
        else:
            clusters[hit].append(c)
        for cell in cells_of(bx, 0):
            if hit not in grid[cell]:
                grid[cell].append(hit)
    out = []
    for cl in clusters:
        b = cl[0]
        geo = Point(b[0], b[1]).buffer(R * mercator_scale(b[1]), quad_segs=4)
        out.append({"type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[round(u, 2), round(v, 2)] for u, v in geo.exterior.coords]]},
                    "properties": {"class_name": cls, "confidence": round(b[2], 6), "region_name": region, "crs": "EPSG:3857",
                                   "markup_type": MARKUP, "threshold_group": GROUP, "views": len(cl)}})
    return out


class DFineSpOr:
    def __init__(self, ckpt_dirs, device=None, vendor_dir=None, flip=True, read_threads=8, prefetch=3, amp=True):
        _prepare_imports(vendor_dir)
        import torch
        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.flip, self.read_threads, self.prefetch = flip, max(1, int(read_threads)), max(1, int(prefetch))
        self.amp = bool(amp) and self.dev.type == "cuda"   # как в ядре: autocast fp16 только на CUDA
        t = time.time(); self.models, self.info = [], []
        for d in ckpt_dirs:
            m, info = load_model(d, self.dev); self.models.append(m); self.info.append(dict(info, ckpt=str(d)))
        self.stats = collections.Counter(); self.stats["load_s"] = round(time.time() - t, 2)
        log.info("D-FINE: %d модел(ей) на %s за %.1f с", len(self.models), self.dev, time.time() - t)

    def _forward(self, px, chunk):
        torch = self.torch; T = CFG["tile"]; B, L, C = [], [], []
        for m in self.models:
            for flip in ((False, True) if self.flip else (False,)):
                inp = px.flip(3) if flip else px
                with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.amp):
                    o = m(pixel_values=inp)
                prob = o.logits.float().sigmoid(); bx = o.pred_boxes.float()
                if flip:
                    bx = bx.clone(); bx[..., 0] = 1 - bx[..., 0]
                Q, K = prob.shape[1:]
                sc_, idx = prob.flatten(1).topk(min(CFG["max_det"], Q * K), 1)
                for i, (x, y) in enumerate(chunk):
                    keep = sc_[i] >= CFG["min_score"]; q = idx[i][keep] // K; c = idx[i][keep] % K; bb = bx[i][q] * T
                    xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2 + x, bb[:, 1] - bb[:, 3] / 2 + y,
                                        bb[:, 0] + bb[:, 2] / 2 + x, bb[:, 1] + bb[:, 3] / 2 + y], 1)
                    B.append(xyxy.cpu()); L.append(sc_[i][keep].cpu()); C.append(c.cpu())
        return B, L, C

    def detect_tif(self, path, utm_json=None, deadline=None, max_windows=0):
        """Детекции одного tif после NMS: (crs, transform, boxes_px[N,4], scores[N], cls[N]); boxes в пикселях сетки 0,5 м."""
        torch = self.torch
        from torchvision.ops import batched_nms
        g = SceneGrid(path, utm_json); wins = windows(g)
        if max_windows:
            wins = wins[:max_windows]
        rd = _ThreadReader(g); B, L, C = [], [], []; t0 = time.time(); stopped = False
        batches = [wins[b:b + CFG["bs"]] for b in range(0, len(wins), CFG["bs"])]
        with torch.no_grad(), cf.ThreadPoolExecutor(self.read_threads) as ex:
            q = collections.deque(); it = iter(batches)
            for _ in range(self.read_threads * self.prefetch):
                nb = next(it, None)
                if nb is None: break
                q.append(ex.submit(rd.batch, nb))
            while q:
                if deadline is not None and time.time() > deadline:
                    stopped = True; [f.cancel() for f in q]; break
                chunk, arr = q.popleft().result()
                nb = next(it, None)
                if nb is not None:
                    q.append(ex.submit(rd.batch, nb))
                if arr is None:
                    continue
                px = torch.from_numpy(arr.transpose(0, 3, 1, 2).astype(np.float32) / 255).to(self.dev)
                b, l, c = self._forward(px, chunk); B += b; L += l; C += c
        rd.close()
        self.stats["windows"] += len(wins); self.stats["tifs"] += 1; self.stats["deadline_stops"] += int(stopped)
        if not B or not sum(len(x) for x in B):
            return g, np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
        B, L, C = torch.cat(B), torch.cat(L), torch.cat(C)
        k = batched_nms(B, L, C, CFG["nms_iou"]); B, L, C = B[k], L[k], C[k]
        log.info("D-FINE %s: %d окон, %d детекций, %.1f с%s", Path(path).name, len(wins), len(k), time.time() - t0, " (дедлайн)" if stopped else "")
        return g, B.numpy(), L.numpy(), C.numpy()

    @staticmethod
    def ellipse_bounds_3857(g, boxes):
        """Эллипс в рамке (32 точки, как в ядре) → EPSG:3857; рамки эллипсов (x0, y0, x1, y1) и кольца."""
        from pyproj import Transformer
        x0, y0, x1, y1 = boxes.T
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = np.maximum((x1 - x0) / 2, .5), np.maximum((y1 - y0) / 2, .5)
        t = np.linspace(0, 2 * np.pi, 33)[:-1]
        px = cx[:, None] + rx[:, None] * np.cos(t)[None]; py = cy[:, None] + ry[:, None] * np.sin(t)[None]
        a, b, c, d, e, f = g.T.a, g.T.b, g.T.c, g.T.d, g.T.e, g.T.f
        ux, uy = a * px + b * py + c, d * px + e * py + f
        X, Y = Transformer.from_crs(g.crs, "EPSG:3857", always_xy=True).transform(ux.ravel(), uy.ravel())
        X, Y = np.asarray(X).reshape(ux.shape), np.asarray(Y).reshape(uy.shape)
        return np.stack([X.min(1), Y.min(1), X.max(1), Y.max(1)], 1), X, Y

    def run_regions(self, regions, deadline=None, output="circles", max_windows=0):
        """regions: iterable (region_name, [пути SpOr tif], путь UTM.json | None).
        output="circles" — вариант R/C (круги r=4 м, threshold_group=sp_dfine); "ellipses" — сырые эллипсы (вариант E)."""
        feats = []
        for name, tifs, utm in regions:
            cand = collections.defaultdict(list)
            for p in sorted(map(str, tifs)):
                if deadline is not None and time.time() > deadline:
                    log.warning("D-FINE: дедлайн, участок %s пропущен с %s", name, Path(p).name); self.stats["tifs_skipped"] += 1; continue
                try:
                    g, bx, sc, cl = self.detect_tif(p, utm, deadline, max_windows)
                except Exception as e:  # один битый растр не роняет весь шаг
                    log.exception("D-FINE: %s не обработан: %s", p, e); self.stats["tif_errors"] += 1; continue
                if not len(bx):
                    continue
                sc = np.round(sc.astype(np.float64), 5)     # ядро писало confidence с 5 знаками; круги брали их
                bounds, X, Y = self.ellipse_bounds_3857(g, bx)
                if output == "ellipses":
                    from shapely.geometry import Polygon, mapping
                    for i in range(len(bx)):
                        poly = Polygon(np.c_[X[i], Y[i]])
                        if not poly.is_valid:
                            poly = poly.buffer(0); self.stats["buffer0"] += 1
                        if poly.is_empty or poly.geom_type != "Polygon":
                            self.stats["dropped_geom"] += 1; continue
                        feats.append({"type": "Feature", "geometry": mapping(poly),
                                      "properties": {"class_name": CLASSES[int(cl[i])], "confidence": float(sc[i]), "region_name": name,
                                                     "crs": "EPSG:3857", "markup_type": MARKUP, "threshold_group": GROUP}})
                else:
                    for i in range(len(bx)):
                        b = bounds[i]; cand[CLASSES[int(cl[i])]].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, float(sc[i]), tuple(b)))
            for cls in sorted(cand):
                feats += circles(cand[cls], name, cls)
        self.stats["features"] = len(feats)
        return feats


def run_package_regions(catalog_regions, ckpt_dirs, deadline=None, vendor_dir=None, **kw):
    """Адаптер к expds.catalog.Region: rasters['sp_or'] и utm_json."""
    det = DFineSpOr(ckpt_dirs, vendor_dir=vendor_dir, **kw)
    regs = [(r.name, list(r.rasters.get("sp_or", [])), r.utm_json) for r in catalog_regions if r.rasters.get("sp_or")]
    return det.run_regions(regs, deadline=deadline), dict(det.stats)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="D-FINE SpOr: участок/tif → GeoJSON EPSG:3857")
    ap.add_argument("--ckpt", action="append", required=True); ap.add_argument("--tif", action="append", required=True)
    ap.add_argument("--region", default="region"); ap.add_argument("--utm", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None); ap.add_argument("--vendor", default=None); ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--max-windows", type=int, default=0); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--output", default="circles", choices=["circles", "ellipses"])
    a = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    det = DFineSpOr(a.ckpt, device=a.device, vendor_dir=a.vendor, flip=not a.no_flip, read_threads=a.threads)
    t = time.time(); fs = det.run_regions([(a.region, a.tif, a.utm)], output=a.output, max_windows=a.max_windows)
    json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": fs},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps({"features": len(fs), "sec": round(time.time() - t, 2), "stats": dict(det.stats), "models": det.info}, ensure_ascii=False))
