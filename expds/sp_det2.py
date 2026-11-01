"""sat2 / sp_det2: CenterNet (centerhead) on SpOr 3-ch, REPLACE of sp_* kurgans. Default OFF.

Weights: fold ensemble sp_det2_cnxt_f{0..4}.pt (tu-convnext_tiny, ImageNet norm, tile 512 / overlap 0.25).
Features: markup_type=sp_or, threshold_group=sp_det2. Requires [thresholds_by_markup.sp_det2] for kurgans.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("expds.sp_det2")

GROUP = "sp_det2"
KURGAN = ("kurgany_povrezhdennye", "kurgany_tselye")
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None] * 255
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None] * 255

DEFAULTS = {
    "enabled": False,
    "weights": [
        "models/sp_det2/sp_det2_cnxt_f0.pt",
        "models/sp_det2/sp_det2_cnxt_f1.pt",
        "models/sp_det2/sp_det2_cnxt_f2.pt",
        "models/sp_det2/sp_det2_cnxt_f3.pt",
        "models/sp_det2/sp_det2_cnxt_f4.pt",
    ],
    "encoder": "tu-convnext_tiny",
    "tile": 512,
    "overlap": 0.25,
    "batch": 8,
    "min_score": 0.03,
    "circle_r_m": 4.0,
    "res_m": 0.5,
    "drop_prefixes": ["sp_"],  # markup_type prefixes to drop for kurgan classes on completed regions
    "soft_need_s": 90.0,
    "strict": True,
}


def settings(cfg: dict) -> dict:
    own = dict(cfg.get("sp_det2") or {})
    out = dict(DEFAULTS)
    out.update(own)
    return out


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("sp_det2") or {}).get("enabled", False))


class SpDet2Unavailable(RuntimeError):
    pass


def preflight(cfg: dict, root: Path) -> dict:
    st = settings(cfg)
    problems = []
    by = (cfg.get("thresholds_by_markup") or {}).get(GROUP) or {}
    missing = [c for c in KURGAN if c not in by]
    if missing:
        problems.append("нет порогов [thresholds_by_markup.sp_det2] для " + ", ".join(missing))
    paths = []
    for w in st["weights"]:
        p = root / w if not Path(w).is_absolute() else Path(w)
        if not p.is_file():
            problems.append(f"нет весов {w}")
        else:
            paths.append(p)
    try:
        import torch  # noqa: F401
        import segmentation_models_pytorch  # noqa: F401
        import timm  # noqa: F401
    except Exception as e:
        problems.append(f"deps: {type(e).__name__}: {e}")
    try:
        from centerhead.head import build_unet_centernet  # noqa: F401
        from centerhead.decode import decode  # noqa: F401
    except Exception as e:
        problems.append(f"centerhead: {type(e).__name__}: {e}")
    if problems:
        msg = "sp_det2 включён, но не готов: " + "; ".join(problems)
        if st.get("strict", True):
            raise SpDet2Unavailable(msg)
        log.error(msg)
    return {"weights": len(paths), "encoder": st["encoder"]}


def drop_kurgan_sp(feats: list[dict], prefixes, regions) -> list[dict]:
    """REPLACE: drop kurgan feats whose markup_type starts with any prefix, on named regions only."""
    regs = set(regions)
    prefs = tuple(prefixes or ())

    def drop(f: dict) -> bool:
        if f.get("region_name") not in regs:
            return False
        if f.get("class_name") not in KURGAN:
            return False
        mk = str(f.get("markup_type") or "")
        return any(mk.startswith(p) for p in prefs)

    return [f for f in feats if not drop(f)]


class _Scene:
    """SpOr raster → UTM @ res_m, RGB uint8 windows (same contract as sp_dfine.Scene)."""

    def __init__(self, path: Path, crs: str, res_m: float = 0.5):
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import calculate_default_transform

        self.src = rasterio.open(path)
        self.crs = crs
        tr, w, h = calculate_default_transform(
            self.src.crs, crs, self.src.width, self.src.height, *self.src.bounds, resolution=res_m
        )
        try:
            src_m = abs(
                calculate_default_transform(
                    self.src.crs, crs, self.src.width, self.src.height, *self.src.bounds
                )[0].a
            )
        except Exception:
            src_m = res_m
        rs = Resampling.average if src_m < res_m else Resampling.bilinear
        self.vrt = WarpedVRT(self.src, crs=crs, transform=tr, width=w, height=h, resampling=rs, nodata=0)
        self.W, self.H, self.T = w, h, tr
        self.bands = list(range(1, min(3, self.src.count) + 1))
        self.stretch = None
        if self.src.dtypes[0] != "uint8":
            ov = self.src.read(1, out_shape=(min(1024, self.src.height), min(1024, self.src.width)))
            v = ov[ov > 0]
            self.stretch = tuple(np.percentile(v, [1, 99])) if v.size else (0, 1)

    def read_chw(self, x0: int, y0: int, n: int) -> np.ndarray:
        from rasterio.windows import Window

        a = np.zeros((len(self.bands), n, n), dtype=self.src.dtypes[0])
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(self.W, x0 + n), min(self.H, y0 + n)
        if cx1 > cx0 and cy1 > cy0:
            a[:, cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = self.vrt.read(
                self.bands, window=Window(cx0, cy0, cx1 - cx0, cy1 - cy0)
            )
        if a.shape[0] < 3:
            a = np.repeat(a[:1], 3, 0)
        if self.stretch:
            lo, hi = self.stretch
            a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return np.ascontiguousarray(a.astype(np.uint8))

    def close(self) -> None:
        self.vrt.close()
        self.src.close()


def _windows(W: int, H: int, tile: int, overlap_frac: float) -> list[tuple[int, int]]:
    step = max(1, int(round(tile * (1.0 - overlap_frac))))
    # align last tile like centerhead.tile_origins (multiple of stride 4 handled in decode grid)
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]
    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))
    # unique sorted
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    return [(x, y) for y in ys for x in xs]


def _load_models(st: dict, root: Path, device):
    import torch
    from centerhead.config import cfg_with
    from centerhead.head import build_unet_centernet

    cfg = cfg_with(markup_type="sp_or", circle_r_m=float(st["circle_r_m"]), min_score=float(st["min_score"]))
    models = []
    for w in st["weights"]:
        p = root / w if not Path(w).is_absolute() else Path(w)
        blob = torch.load(p, map_location="cpu", weights_only=False)
        enc = blob.get("enc") or st["encoder"]
        m = build_unet_centernet(3, enc, None, cfg)
        m.load_state_dict(blob["state_dict"])
        m.eval().to(device)
        models.append(m)
    return models, cfg


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def detect_scene(models, scene: _Scene, st: dict, ch_cfg, device, keep_going=lambda: True):
    """Run fold-ensemble CenterNet; own-cell dedupe across overlapping windows; return feature dicts."""
    import torch
    from centerhead.decode import decode, concat
    from centerhead.export import candidates_to_feats
    from scipy.spatial import cKDTree

    T = int(st["tile"])
    wins = _windows(scene.W, scene.H, T, float(st["overlap"]))
    if not wins:
        return [], 0, 0
    # world coords of each window origin (pixel → CRS via affine)
    tr = scene.T
    step_m = float(st["res_m"])
    # build rows like sat2 OOF
    rows = []
    for x0, y0 in wins:
        left = tr.c + x0 * tr.a
        top = tr.f + y0 * tr.e  # e is negative usually
        # centerhead uses left/top with positive step; for north-up e<0, top is northing of row0
        rows.append({"x0": x0, "y0": y0, "left": float(left), "top": float(top), "step": step_m})

    mean = torch.tensor(IMNET_MEAN, device=device)
    std = torch.tensor(IMNET_STD, device=device)
    IB = max(1, int(st["batch"]))
    st_stride = 4
    cands = []
    done = 0
    with torch.no_grad():
        for b0 in range(0, len(rows), IB):
            if not keep_going():
                break
            ids = list(range(b0, min(len(rows), b0 + IB)))
            imgs = np.stack([scene.read_chw(rows[i]["x0"], rows[i]["y0"], T) for i in ids])  # B,3,S,S
            x = (torch.from_numpy(imgs).to(device).float() - mean) / std
            # average hm/wh/off across folds
            hm_acc = wh_acc = off_acc = None
            for m in models:
                with torch.autocast(device if device != "cpu" else "cpu", dtype=torch.float16, enabled=device == "cuda"):
                    o = m(x)
                hm = torch.sigmoid(o["hm"].float())
                wh = o["wh"].float()
                off = o["off"].float()
                hm_acc = hm if hm_acc is None else hm_acc + hm
                wh_acc = wh if wh_acc is None else wh_acc + wh
                off_acc = off if off_acc is None else off_acc + off
            n = float(len(models))
            hm_np = (hm_acc / n).cpu().numpy()
            wh_np = (wh_acc / n).cpu().numpy()
            off_np = (off_acc / n).cpu().numpy()
            for j, i in enumerate(ids):
                r = rows[i]
                t = decode(
                    hm_np[j],
                    r["left"],
                    r["top"],
                    r["step"] * st_stride,
                    wh_np[j],
                    off_np[j],
                    cfg=ch_cfg,
                )
                if len(t["X"]):
                    t["win"] = np.full(len(t["X"]), i, np.int32)
                    cands.append(t)
            done += len(ids)

    if not cands:
        return [], done, len(wins)

    allc = concat(cands)
    # own-cell dedupe: keep peak from the window whose center is nearest (L∞) — same as sat2 kernel
    # Approximate: for each detection, find which window owns it by nearest window center in pixel space
    win_cx = np.array([rows[i]["x0"] + T / 2 for i in range(len(rows))])
    win_cy = np.array([rows[i]["y0"] + T / 2 for i in range(len(rows))])
    # map world X,Y → pixel
    # x_pix = (X - tr.c) / tr.a ; y_pix = (Y - tr.f) / tr.e
    Xs, Ys = allc["X"], allc["Y"]
    x_pix = (Xs - tr.c) / tr.a
    y_pix = (Ys - tr.f) / tr.e
    tree = cKDTree(np.stack([win_cx, win_cy], 1))
    _, nn = tree.query(np.stack([x_pix, y_pix], 1), p=np.inf)
    keep = allc["win"] == nn
    allc = {k: np.asarray(v)[keep] for k, v in allc.items() if k != "win"}
    if not len(allc.get("X", [])):
        return [], done, len(wins)

    allc["crs"] = np.array([scene.crs] * len(allc["X"]))
    allc["region"] = np.array(["_"] * len(allc["X"]))  # filled by caller
    return allc, done, len(wins)


def region_feats(reg, cfg: dict, guard, cache: dict, root: Path, crs_for=None, soft_ok=None) -> tuple[list[dict], bool]:
    st = settings(cfg)
    from centerhead.export import candidates_to_feats, to_fc

    paths = list(reg.rasters.get("sp_or") or [])
    if not paths:
        return [], False
    if "models" not in cache:
        t0 = time.monotonic()
        try:
            cache["models"], cache["ch_cfg"] = _load_models(st, root, _device())
            cache["device"] = _device()
            log.info("sp_det2: моделей %d на %s за %.1f с", len(cache["models"]), cache["device"], time.monotonic() - t0)
        except Exception as e:
            if st.get("strict", True):
                raise SpDet2Unavailable(f"sp_det2: модели не загрузились ({type(e).__name__}: {e})") from e
            log.exception("sp_det2: модели не загрузились")
            cache["models"] = []
            return [], False
    if not cache.get("models"):
        return [], False

    crs = None
    if crs_for is not None:
        try:
            crs = crs_for(reg)
        except Exception:
            crs = None
    feats_all = []
    complete = True
    for path in paths:
        path = Path(path)
        if soft_ok is not None and not soft_ok(f"sp_det2 {path.name}", float(st["soft_need_s"])):
            complete = False
            break
        if guard is not None and hasattr(guard, "ok") and not guard.ok(30):
            complete = False
            break
        try:
            if crs is None:
                import rasterio

                with rasterio.open(path) as ds:
                    crs = str(ds.crs) if ds.crs else "EPSG:3857"
            sc = _Scene(path, crs, float(st["res_m"]))
            try:
                keep = (lambda: guard.ok(5)) if guard is not None and hasattr(guard, "ok") else (lambda: True)
                blob, done, total = detect_scene(
                    cache["models"], sc, st, cache["ch_cfg"], cache["device"], keep_going=keep
                )
                if done < total:
                    complete = False
                if blob and len(blob.get("X", [])):
                    blob["region"] = np.array([reg.name] * len(blob["X"]))
                    blob["crs"] = np.array([crs] * len(blob["X"]))
                    feats = candidates_to_feats(blob, reg.name, crs, cfg=cache["ch_cfg"])
                    for f in feats:
                        f["threshold_group"] = GROUP
                        f.setdefault("markup_type", "sp_or")
                    feats_all.extend(feats)
                log.info(
                    "sp_det2: %s %dx%d px, окон %d/%d, боксов %d",
                    path.name,
                    sc.W,
                    sc.H,
                    done,
                    total,
                    len(feats_all),
                )
            finally:
                sc.close()
        except Exception:
            log.exception("sp_det2: %s пропущен", path.name)
            complete = False
    return feats_all, complete
