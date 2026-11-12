"""Фильтр ложных курганов лидара поверх c5-in (PREREG kaggle/2026-11-06/li_fp_filter/PREREG.md).

Гейт 06.11 фильтр не принял: вариант R дал Δ+0,0238 при P 1,0, но nolidar −0,00108,
ниже пола −0,001. Сдача этот модуль не вызывает, в configs/default.toml флага нет.

Признаки — только из 5 каналов входа c5 (u8: lrm_s, lrm_l, slope, svf_r, opp_r при шаге step) и самих детекций,
одна функция на прод (expds/lidar_cnx_head.py) и офлайн (kaggle/2026-11-06/li_fp_filter/fpf_oof.py):
  1. уверенность и класс;
  2. по каналу: среднее в круге 2 м, кольцах 4–6 м и 8–12 м, разности «круг − кольцо» (5 × 5);
  3. анизотропия структурного тензора lrm_s и slope в окне радиуса 8 м: (λ1 − λ2)/(λ1 + λ2) и λ2/λ1;
  4. доля съёмки в окне;
  5. соседи: число детекций в 15/30/60 м и максимальная уверенность соседа в 15 м.
"""
from __future__ import annotations

import numpy as np

NAMES = (["conf", "cls"] + [f"c{c}_{k}" for c in range(5) for k in ("d2", "r46", "r812", "d_r46", "d_r812")]
         + ["aniso_lrm", "ratio_lrm", "aniso_slope", "ratio_slope", "valid", "nb15", "nb30", "nb60", "nbmax15"])


def _masks(step: float):
    R = int(np.ceil(12.0 / step))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.hypot(yy, xx) * step
    return R, d <= 2.0, (d >= 4.0) & (d <= 6.0), (d >= 8.0) & (d <= 12.0), d <= 8.0


def _patch(a: np.ndarray, r: int, c: int, R: int) -> np.ndarray:
    H, W = a.shape[-2:]
    out = np.zeros(a.shape[:-2] + (2 * R + 1, 2 * R + 1), a.dtype)
    y0, y1, x0, x1 = max(0, r - R), min(H, r + R + 1), max(0, c - R), min(W, c + R + 1)
    if y0 < y1 and x0 < x1:
        out[..., y0 - (r - R):y1 - (r - R), x0 - (c - R):x1 - (c - R)] = a[..., y0:y1, x0:x1]
    return out


def _aniso(p: np.ndarray, m: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    gy, gx = np.gradient(p.astype(np.float32))
    w = (m & v).astype(np.float32)
    if w.sum() < 9:
        return np.nan, np.nan
    jxx, jyy, jxy = float((gx * gx * w).sum()), float((gy * gy * w).sum()), float((gx * gy * w).sum())
    tr, det = jxx + jyy, jxx * jyy - jxy * jxy
    disc = max(tr * tr / 4 - det, 0.0) ** 0.5
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 0.0)
    return (l1 - l2) / (l1 + l2 + 1e-6), l2 / (l1 + 1e-6)


def features(u8: np.ndarray, left: float, top: float, step: float, dets) -> np.ndarray:
    """u8 [≥5, H, W] (0 = вне съёмки), dets — [(X, Y, cls, conf), …] в CRS сетки → признаки [n, len(NAMES)]."""
    R, m2, m46, m812, m8 = _masks(step)
    u = u8[:5]
    XY = np.array([(d[0], d[1]) for d in dets], np.float64).reshape(-1, 2)
    conf = np.array([float(d[3]) for d in dets], np.float64)
    out = np.full((len(dets), len(NAMES)), np.nan, np.float32)
    for i, (X, Y, k, s) in enumerate(dets):
        c, r = int(np.floor((X - left) / step)), int(np.floor((top - Y) / step))
        p = _patch(u, r, c, R)
        v = p[0] > 0
        row = [float(s), float(k)]
        for ch in range(5):
            pc = p[ch].astype(np.float32)
            mm = [float(pc[m & v].mean()) if (m & v).any() else np.nan for m in (m2, m46, m812)]
            row += mm + [mm[0] - mm[1], mm[0] - mm[2]]
        a1 = _aniso(p[0], m8, v)
        a2 = _aniso(p[2], m8, v)
        dd = np.hypot(XY[:, 0] - X, XY[:, 1] - Y)
        dd[i] = np.inf
        near15 = dd <= 15.0
        row += [a1[0], a1[1], a2[0], a2[1], float(v[m8].mean()), float(near15.sum()), float((dd <= 30.0).sum()),
                float((dd <= 60.0).sum()), float(conf[near15].max()) if near15.any() else 0.0]
        out[i] = row
    return out


def u8_of(x: np.ndarray) -> np.ndarray:
    """Вход модели x = (u8 − 128)/127 → u8 (точно, как cnx_prep.u8_to_input наоборот)."""
    return np.clip(np.rint(x[:5] * 127.0 + 128.0), 0, 255).astype(np.uint8)
