"""Каналы рельефа *_r (site_robust) для li-модели — заготовка для ядра бота 9 (RELIEF.md).

Одна функция `extra_channels` строит каналы и для кэша тайлов, и для инференса: иначе вход модели
на тесте не совпадёт с обучением. База — expds/lidar.py (relief_channels, _smooth_valid, _curvature,
intensity_grid) и expds/accel.py (openness_svf, box_mean, local_relief); параметры — по yakirill99
configs/dataset_006.yaml:62-71, нормировка — yakirill99 src/expds/tiles/dataset.py:38,77-100.
"""
from __future__ import annotations

import numpy as np

#: Порядок = контракт модели (пишется в meta.json → li_tags). Базовые три — как у li_r18_hard2.
BASE_TAGS = ("lrm_s", "lrm_l", "slope")
EXTRA_TAGS = ("svf_r", "opp_r", "opn_r", "tpi5_r", "tri3_r", "curv_r", "slrm8_r", "dens_r", "int_r")
#: Параметры в МЕТРАХ; пиксели = round(м / шаг). Шаг сети 0,5 м.
P = {"hor_r_m": 10.0, "hor_dirs": 16,       # SVF, открытость ±: yakirill openness_radius 10, n_dir 16
     "tpi_r_m": 5.0,                         # yakirill tpi_radii [5, …] (пикс. при 1 м)
     "tri_r_m": 3.0,                         # yakirill tri_radii [3, …]
     "curv_sigma_m": 1.5,                    # наш RELIEF["curv_sigma_m"]; yakirill σ 2 — абляция
     "slrm_sigma_m": 8.0,                    # yakirill slrm_sigma_8.0
     "int_fill_m": 2.0,                      # наш INTENSITY_FILL_M
     "pct": (2.0, 50.0, 98.0), "eps": 1e-6}  # site_robust


def smooth_valid(z, valid, sigma_px):
    """Гаусс по пикселям съёмки (нормированная свёртка), как expds.lidar._smooth_valid."""
    from scipy.ndimage import gaussian_filter
    w = gaussian_filter(valid.astype(np.float64), sigma_px, mode="constant")
    s = gaussian_filter(np.where(valid, z, 0.0).astype(np.float64), sigma_px, mode="constant")
    return np.where(w > 1e-6, s / np.maximum(w, 1e-6), np.nan)


def box_mean(z, r):
    """Среднее окна (2r+1)² без NaN (NumPy-ветка expds.accel.box_mean)."""
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
        return np.where(cnt > 0, tot / cnt, np.nan)


def horizon(z, step, r_px, n_dir, accel=None):
    """(положительная открытость °, SVF). Если есть expds.accel — его openness_svf (Rust/NumPy),
    иначе rvt.vis.sky_view_factor (pip install rvt-py --no-deps), иначе ошибка (без np.roll: он
    заворачивает край, как у yakirill relief.py:183-231)."""
    zf = np.ascontiguousarray(z, np.float32)
    if accel is not None:
        return accel.openness_svf(zf, float(step), int(r_px), int(n_dir))
    import rvt.vis
    out = rvt.vis.sky_view_factor(zf.copy(), resolution=float(step), compute_svf=True, compute_opns=True,
                                  svf_n_dir=int(n_dir), svf_r_max=int(r_px), svf_noise=0, no_data=None)
    return out["opns"], out["svf"]


def raw_channels(z, step, dens=None, inten=None, accel=None, tags=EXTRA_TAGS):
    """Сырые значения (float32, NaN вне съёмки). z — DEM на шаге сети (to_work_grid), NaN вне контура.
    dens, inten — сетки того же размера из одного прохода по LAS (cloud_grids)."""
    valid = np.isfinite(z)
    zc = np.where(valid, z - np.nanmean(z), np.nan)       # сдвиг до квадратов: float32 не теряет дм
    px = lambda m: max(1, int(round(m / step)))
    out, hor_p, hor_n = {}, None, None
    for t in tags:
        if t in ("svf_r", "opp_r"):
            if hor_p is None:
                hor_p = horizon(zc, step, px(P["hor_r_m"]), P["hor_dirs"], accel)
            v = hor_p[1] if t == "svf_r" else hor_p[0]
        elif t == "opn_r":
            if hor_n is None:
                hor_n = horizon(-zc, step, px(P["hor_r_m"]), P["hor_dirs"], accel)
            v = hor_n[0]                                   # отрицательная открытость = открытость −z
        elif t == "tpi5_r":
            v = zc - box_mean(zc, px(P["tpi_r_m"]))
        elif t == "tri3_r":
            r = px(P["tri_r_m"]); m = box_mean(zc, r)
            v = np.sqrt(np.clip(box_mean(zc * zc, r) - m * m, 0, None))   # yakirill relief.py:368-392
        elif t == "curv_r":
            sm = smooth_valid(zc, valid, P["curv_sigma_m"] / step)
            p = np.pad(np.where(np.isfinite(sm), sm, 0.0), 1, mode="edge")
            v = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]) / step ** 2
        elif t == "slrm8_r":
            v = zc - smooth_valid(zc, valid, P["slrm_sigma_m"] / step)  # yakirill relief.py:128-149
        elif t == "dens_r":
            v = np.log1p(dens) if dens is not None else np.full(z.shape, np.nan)
        elif t == "int_r":
            if inten is None:
                v = np.full(z.shape, np.nan)
            else:
                near = box_mean(inten, px(P["int_fill_m"]))
                v = np.where(np.isfinite(inten), inten, near)
        else:
            raise ValueError(t)
        out[t] = np.where(valid, v, np.nan).astype(np.float32)
    return out


def site_stats(raw):
    """{tag: (p2, median, p98)} по валидным пикселям ВСЕГО участка (без эталона)."""
    st = {}
    for t, v in raw.items():
        x = v[np.isfinite(v)]
        st[t] = tuple(float(q) for q in np.percentile(x, P["pct"])) if x.size else (0.0, 0.0, 1.0)
    return st


def encode(raw, stats, valid):
    """site_robust → uint8: y = (x − med)/(p98 − p2), y∈[−1,1] → 1…255 (0 — вне съёмки; y=0 → 128)."""
    chans = []
    for t, v in raw.items():
        lo, med, hi = stats[t]
        y = (v - med) / max(hi - lo, P["eps"])
        u = np.clip((np.nan_to_num(y, nan=0.0) + 1.0) / 2.0, 0.0, 1.0)
        chans.append(np.where(valid, np.rint(1 + u * 254), 0).astype(np.uint8))
    return np.stack(chans)


def cloud_grids(las_path, left, top, step, shape, ground_class=2, chunk=2_000_000):
    """Один проход laspy по LAS: плотность ВСЕХ точек, 1/м² (как yakirill build_site.py:375) и
    средняя интенсивность точек земли (как expds.lidar.intensity_grid:447-473; без класса 2 — все)."""
    import laspy
    h, w = shape
    cnt_all = np.zeros(h * w); s_int = np.zeros(h * w); c_int = np.zeros(h * w)
    with laspy.open(str(las_path)) as f:
        has_g = None
        for pts in f.chunk_iterator(chunk):
            x, y = np.asarray(pts.x), np.asarray(pts.y)
            c = np.floor((x - left) / step); r = np.floor((top - y) / step)
            ok = (c >= 0) & (r >= 0) & (c < w) & (r < h)
            flat = r[ok].astype(np.int64) * w + c[ok].astype(np.int64)
            cnt_all += np.bincount(flat, minlength=h * w)
            cls = np.asarray(pts.classification)[ok]
            if has_g is None:
                has_g = bool((cls == ground_class).any())   # приближение: по первому куску
            g = (cls == ground_class) if has_g else np.ones(len(flat), bool)
            s_int += np.bincount(flat[g], weights=np.asarray(pts.intensity, np.float64)[ok][g], minlength=h * w)
            c_int += np.bincount(flat[g], minlength=h * w)
    dens = (cnt_all / (step * step)).reshape(h, w)
    inten = np.where(c_int > 0, s_int / np.maximum(c_int, 1), np.nan).reshape(h, w)
    return dens.astype(np.float32), inten.astype(np.float32)


def extra_channels(z, step, left, top, las_path=None, accel=None, stats=None, tags=EXTRA_TAGS):
    """C×H×W uint8 дополнительных каналов + stats (сохранить в site_stats.json).
    stats=None — считать по этому z (весь участок); для окон мозаики передать stats участка."""
    valid = np.isfinite(z)
    dens = inten = None
    if las_path is not None and ({"dens_r", "int_r"} & set(tags)):
        dens, inten = cloud_grids(las_path, left, top, step, z.shape)
    raw = raw_channels(z, step, dens, inten, accel, tags)
    stats = stats or site_stats(raw)
    return encode(raw, stats, valid), stats
