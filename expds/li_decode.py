"""Point decoding of lidar-net probability maps (off by default: ``[li] decode = "peaks"``).

Idea (recipe studied in EhimenNathan-EXpedition, re-implemented here, see REPORT2.md §7):
the metric accepts a predicted polygon when its representative point lies inside the reference
object and the overlap / min(area) >= 0.5, so a small circle placed at a reliable interior point
scores like a perfect polygon. Hence objects are decoded as points, not as mask outlines:

* kurgans: local maxima of the kurgan score map S = max(p_povr, p_tsel), suppressed within
  ``nms_m`` (Euclidean, strongest first); the class is the one with the larger mean probability in
  a small window; the other class is emitted too only when its share >= ``dual``.
  confidence = S(peak) * share(class).
* mask classes listed in ``radius_m`` (gor/fort by default): connected components of p >= ``comp_thr``
  with true area >= ``min_area_m2[cls]``; one circle at the component pixel farthest from the
  component edge (always inside the component); confidence = mean p over the component.
* every other class falls back to ``infer_nn.probs_to_features`` (unchanged behaviour).

Circles are built in the map's own metric CRS (UTM), so ``radius_m`` is in true metres, and are then
projected to EPSG:3857 like every other candidate. Thresholds on confidence are NOT set here: they
are fit per (group, class) downstream (expds.metric sweeps, nested within folds in evaluation).

Optional dump for offline evaluation: env ``EXPDS_LI_DUMP=<dir>`` makes ``dump_probs`` write one
compressed .npz per (region, source) with float16 maps, transform, CRS and the relief candidates.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")

DEFAULTS: dict = {
    "peak_min": 0.10,       # minimum kurgan score at a peak
    "nms_m": 2.5,           # centre-NMS distance between kurgan peaks
    "share_win_px": 2,      # half-size of the class-share window ((2w+1)^2 pixels)
    "dual": 0.30,           # emit the minority kurgan class when its share >= dual
    "comp_thr": 0.50,       # mask threshold for component classes
    "radius_m": {"kurgany_povrezhdennye": 1.0, "kurgany_tselye": 1.0,
                 "gorodishcha": 1.0, "fortifikatsii": 1.0},
    "min_area_m2": {"gorodishcha": 150.0, "fortifikatsii": 7.5},
}


def params(li_cfg: dict | None) -> dict:
    """DEFAULTS overridden by ``[li.peaks]`` (nested dicts merged one level deep)."""
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in ((li_cfg or {}).get("peaks") or {}).items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    return p


def enabled(li_cfg: dict | None) -> bool:
    return str((li_cfg or {}).get("decode", "legacy")).lower() == "peaks"


def find_peaks(score: np.ndarray, min_height: float, min_dist_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local maxima (3x3 plateau-safe) >= min_height, then greedy Euclidean NMS, strongest first.

    Returns rows, cols, heights sorted by height (descending).
    """
    from scipy.ndimage import maximum_filter
    from scipy.spatial import cKDTree

    s = np.nan_to_num(np.asarray(score, np.float32), nan=0.0)
    cand = (s >= min_height) & (s == maximum_filter(s, size=3, mode="constant", cval=-1.0))
    rows, cols = np.nonzero(cand)
    if rows.size == 0:
        return rows, cols, np.zeros(0, np.float32)
    h = s[rows, cols]
    order = np.lexsort((cols, rows, -h))          # deterministic: height desc, then row, col
    rows, cols, h = rows[order], cols[order], h[order]
    if min_dist_px <= 1.0 or rows.size == 1:
        return rows, cols, h
    tree = cKDTree(np.column_stack([rows, cols]).astype(np.float64))
    alive = np.ones(rows.size, bool)
    for i in range(rows.size):
        if not alive[i]:
            continue
        for j in tree.query_ball_point((rows[i], cols[i]), r=min_dist_px - 1e-9):
            if j > i:
                alive[j] = False
    return rows[alive], cols[alive], h[alive]


def _window_mean(p: np.ndarray, rows: np.ndarray, cols: np.ndarray, w: int) -> np.ndarray:
    H, W = p.shape
    out = np.empty(rows.size, np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        out[i] = float(np.nanmean(p[max(0, r - w):min(H, r + w + 1), max(0, c - w):min(W, c + w + 1)]))
    return out


def decode_kurgans(kp: np.ndarray, classes: list[str], res: float, prm: dict):
    """kp: 2xHxW (povr, tsel order of ``classes``) → list of (row, col, class, confidence)."""
    rows, cols, h = find_peaks(kp.max(axis=0), float(prm["peak_min"]), float(prm["nms_m"]) / res)
    if rows.size == 0:
        return []
    m = np.stack([_window_mean(kp[i], rows, cols, int(prm["share_win_px"])) for i in range(len(classes))], 1)
    share = m / np.maximum(m.sum(1, keepdims=True), 1e-12)
    best = share.argmax(1)
    out = []
    for i in range(rows.size):
        for k, cls in enumerate(classes):
            if k == best[i] or share[i, k] >= float(prm["dual"]):
                out.append((int(rows[i]), int(cols[i]), cls, float(h[i] * share[i, k])))
    return out


def decode_components(p: np.ndarray, res: float, thr: float, min_area_m2: float):
    """Components of p >= thr with area >= min_area_m2 → list of (row, col, mean_prob)."""
    from scipy.ndimage import distance_transform_edt, label

    lab, n = label(np.nan_to_num(p, nan=0.0) >= thr, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    min_px = max(1, int(np.ceil(min_area_m2 / (res * res))))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sums = np.bincount(lab.ravel(), weights=np.nan_to_num(p, nan=0.0).ravel(), minlength=n + 1)
    dist = distance_transform_edt(np.pad(lab > 0, 1))[1:-1, 1:-1]
    out = []
    flat_lab, flat_d = lab.ravel(), dist.ravel()
    order = np.lexsort((-flat_d, flat_lab))       # within each label, deepest pixel first
    first = np.r_[True, flat_lab[order][1:] != flat_lab[order][:-1]]
    for idx in order[first]:
        k = flat_lab[idx]
        if k == 0 or sizes[k] < min_px:
            continue
        r, c = divmod(int(idx), p.shape[1])
        out.append((r, c, float(sums[k] / sizes[k])))
    return out


def _circles(pts, transform, crs, radius_of) -> list:
    from shapely import buffer, points

    from .crs import WEB_MERCATOR, to_crs

    if not pts:
        return []
    rows = np.array([q[0] for q in pts], np.float64)
    cols = np.array([q[1] for q in pts], np.float64)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    rad = np.array([radius_of(q) for q in pts], np.float64)
    geoms = buffer(points(xs, ys), rad, quad_segs=8)
    if crs and crs != WEB_MERCATOR:
        geoms = to_crs(geoms, crs, WEB_MERCATOR)
    return list(geoms)


def decode_li(probs: np.ndarray, classes: list[str], transform, crs: str | None, res: float, region: str,
              markup: str, source: str, li_cfg: dict | None = None,
              min_conf: dict[str, float] | None = None) -> list[dict]:
    """KxHxW probabilities of one lidar net → pipeline features with point-style decoding."""
    prm = params(li_cfg)
    rad = prm["radius_m"]
    feats: list[dict] = []
    base = {"region_name": region, "markup_type": markup, "source": source, "original_crs": crs}
    kur = [c for c in KURGANS if c in classes and c in rad]
    if kur:
        kp = np.stack([probs[classes.index(c)] for c in kur]).astype(np.float32)
        pts = decode_kurgans(kp, kur, res, prm)
        for g, (_, _, cls, conf) in zip(_circles(pts, transform, crs, lambda q: rad[q[2]]), pts):
            feats.append({"geom": g, "class_name": cls, "confidence": conf, **base})
    comp = [c for c in classes if c in rad and c not in KURGANS]
    for cls in comp:
        pts = decode_components(probs[classes.index(cls)], res, float(prm["comp_thr"]),
                                float(prm["min_area_m2"].get(cls, 0.0)))
        for g, (_, _, conf) in zip(_circles(pts, transform, crs, lambda q, c=cls: rad[c]), pts):
            feats.append({"geom": g, "class_name": cls, "confidence": conf, **base})
    rest = [c for c in classes if c not in kur and c not in comp]
    if rest:
        from .infer_nn import probs_to_features

        feats += probs_to_features(probs[[classes.index(c) for c in rest]], rest, transform, crs, res,
                                   region, markup, source, min_conf=min_conf)
    if min_conf:
        feats = [f for f in feats if f["confidence"] >= min_conf.get(f["class_name"], -1.0)]
    return feats


def dump_probs(probs: np.ndarray, classes: list[str], transform, crs: str | None, region: str, source: str,
               cand: dict | None = None, model_tag: str = "") -> Path | None:
    """Write maps for offline decoding when EXPDS_LI_DUMP is set; returns the path or None."""
    root = os.environ.get("EXPDS_LI_DUMP")
    if not root:
        return None
    d = Path(root)
    d.mkdir(parents=True, exist_ok=True)
    org = f"{int(round(transform.c))}_{int(round(transform.f))}"   # mosaic pieces of one LAS get distinct files
    safe = re.sub(r"[^\w.-]+", "_", f"{region}__{Path(source).stem}__{model_tag}__{org}")
    dst = d / f"{safe}.npz"
    extra = {}
    if cand is not None and len(cand.get("x", [])):
        extra = {"cand_x": np.asarray(cand["x"], np.float64), "cand_y": np.asarray(cand["y"], np.float64)}
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, probs=np.asarray(probs, np.float16), classes=np.array(classes),
                        transform=np.array(tuple(transform)[:6], np.float64), crs=np.array(crs or ""),
                        region=np.array(region), source=np.array(source), **extra)
    os.replace(tmp, dst)
    return dst
