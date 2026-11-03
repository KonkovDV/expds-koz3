"""B1/B3 (PROMPT_LIDAR_AUG_SEEDS_20261101): physical DTM augmentation banks + c7 channels. Own implementation.

B1: relief channels are recomputed from a TRANSFORMED DTM (never by rescaling clipped u8 relief):
  height h acts on the residual r = z - lowpass(z, 50 m):  z' = lowpass + h * r
  zoom s resamples the DTM in XY (bicubic) keeping the 0.5 m pixel -> features become s x larger in metres
  relief = same metric windows / clips as inference (XL.relief_channels, RX.extra_channels with stats of bank (1,1)).
  Bank (1,1) is computed through the very same path as inference relief -> parity asserted in the kernel.
B3: c7 = lrm_s, lrm_l, slope, svf_r, opp_r (positive openness), opn_r (negative openness), msrm
  msrm: multi-scale |TPI| at 5/10/20/40 m, each divided by a fixed metric clip; the scale with max |TPI/clip|
  wins (signed), clipped to [-1, 1] -> u8 1..255 (0 outside survey). Fixed metric clips (no site stats) keep the
  height augmentation visible.
"""
from __future__ import annotations
import numpy as np

TAGS3 = ("lrm_s", "lrm_l", "slope")
TAGS5 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r")
TAGS7 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r", "opn_r", "msrm")
BANK9 = tuple((z, h) for z in (0.75, 1.0, 1.3) for h in (0.6, 1.0, 1.7))
BANK7 = ((0.75, 0.6), (0.75, 1.7), (1.0, 0.6), (1.0, 1.0), (1.0, 1.7), (1.3, 0.6), (1.3, 1.7))
LOWPASS_M = 50.0
MSRM_R_M = (5.0, 10.0, 20.0, 40.0)
MSRM_CLIP_M = (0.25, 0.5, 1.0, 2.0)


def tags_for(nch: int):
    return {3: TAGS3, 5: TAGS5, 7: TAGS7}[nch]


def box_mean(z, r):
    """NaN-aware (2r+1)^2 box mean via integral image (float64)."""
    fin = np.isfinite(z)
    v = np.where(fin, z, 0).astype(np.float64)
    s = np.pad(v.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n = np.pad(fin.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = z.shape
    r0, r1 = np.clip(np.arange(h) - r, 0, h), np.clip(np.arange(h) + r + 1, 0, h)
    c0, c1 = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
    tot = s[r1][:, c1] - s[r0][:, c1] - s[r1][:, c0] + s[r0][:, c0]
    cnt = n[r1][:, c1] - n[r0][:, c1] - n[r1][:, c0] + n[r0][:, c0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def transform_dtm(z, step, zoom=1.0, h=1.0):
    """z (float32, NaN outside survey) on the 0.5 m work grid -> transformed DTM. (1,1) returns z unchanged."""
    z = np.asarray(z, np.float32)
    if zoom == 1.0 and h == 1.0:
        return z
    valid = np.isfinite(z)
    lp = None
    if h != 1.0:
        lp = box_mean(z, max(1, int(round(LOWPASS_M / step))))
        z = np.where(valid, lp + h * (z.astype(np.float64) - lp), np.nan).astype(np.float32)
    if zoom != 1.0:
        from scipy import ndimage as ndi
        if lp is None:
            lp = box_mean(z, max(1, int(round(LOWPASS_M / step))))
        mean = np.nanmean(z) if valid.any() else 0.0
        fill = np.where(valid, z, np.where(np.isfinite(lp), lp, mean)).astype(np.float64)
        H, W = z.shape
        oh, ow = max(1, int(round(H * zoom))), max(1, int(round(W * zoom)))
        zr = ndi.zoom(fill, (oh / H, ow / W), order=3, mode="nearest", grid_mode=True)
        vr = ndi.zoom(valid.astype(np.uint8), (oh / H, ow / W), order=0, mode="nearest", grid_mode=True) > 0
        z = np.where(vr, zr, np.nan).astype(np.float32)
    return z


def msrm_u8(z, step, valid):
    zc = np.where(valid, z - np.nanmean(z), np.nan) if valid.any() else z
    best = np.zeros(z.shape, np.float64)
    for r_m, c in zip(MSRM_R_M, MSRM_CLIP_M):
        t = (zc - box_mean(zc, max(1, int(round(r_m / step))))) / c
        t = np.nan_to_num(t, nan=0.0)
        best = np.where(np.abs(t) > np.abs(best), t, best)
    u = np.clip(best, -1, 1) * 0.5 + 0.5
    return np.where(valid, np.rint(1 + u * 254), 0).astype(np.uint8)


def channels(z, step, nch, XL, RX, accel=None, stats=None):
    """C x H x W u8 relief from a work-grid DTM z (step = 0.5 m) via the inference functions.
    Returns (arr, stats) where stats are the RX site stats (pass bank (1,1) stats to the other banks)."""
    d = XL.Dem(dem=np.ascontiguousarray(z, np.float32), left=0.0, top=0.0, res=float(step))
    base, _ = XL.relief_channels(d, float(step), TAGS3)
    if nch == 3:
        return base, None
    zz = np.where(base[0] > 0, z, np.nan).astype(np.float32)
    ext_tags = tags_for(nch)[3:6] if nch == 7 else tags_for(nch)[3:]
    ext, st = RX.extra_channels(zz, step, 0.0, 0.0, accel=accel, tags=ext_tags,
                                stats=None if stats is None else {k: stats[k] for k in ext_tags})
    parts = [base, ext]
    if nch == 7:
        parts.append(msrm_u8(zz, step, base[0] > 0)[None])
    return np.concatenate(parts).astype(np.uint8), st


def resize_mask(mask, zoom, shape):
    if zoom == 1.0 and mask.shape == tuple(shape):
        return mask
    H, W = mask.shape
    yi = np.minimum(((np.arange(shape[0]) + 0.5) * H / shape[0]).astype(np.int64), H - 1)
    xi = np.minimum(((np.arange(shape[1]) + 0.5) * W / shape[1]).astype(np.int64), W - 1)
    return mask[yi][:, xi]


def site_probs(n_obj, alpha):
    """B2: p(site) proportional to n_obj(site)^alpha over sites with objects; alpha=0 -> uniform."""
    n = np.asarray(n_obj, np.float64)
    w = np.where(n > 0, n ** float(alpha), 0.0)
    return w / max(w.sum(), 1e-12)
