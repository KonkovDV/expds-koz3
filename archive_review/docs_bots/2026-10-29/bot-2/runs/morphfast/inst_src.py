"""Задача 12: быстрый экстрактор ТОЛЬКО 45 признаков варианта M (bot-4/PREREG_MORPH.md).

30 сырых morph_* (определения 1:1 как в morph_li.py, e918a3b) + 15 morph_rz_* (медиана/MAD региона).
Исключены лучи и дальние окна: tpi15/30, opp*/opn*/svf* (и их rz) — лучи не собираются вовсе.
Ускорения против morph_li.features: окно только r≤20 м + решётка 2 м (без 16×30 лучей), сбор по плоским
смещениям без проверок границ (медленный путь только у кандидатов у края), одна reduceat по кольцам 0,5 м
для фона/профиля/центра, секторы через перестановку + reduceat, float32, nanmedian-резерв только где нужен.

Опции:
  --conf-min T     признаки только для кандидатов с conf ≥ T (остальные NaN); rz-статистика региона по
                   этим кандидатам + случайной подвыборке остальных (--rz-sample, seed 0), см. REPORT2 §12.
  --dem-from-li    ЦМР не строится заново: берётся сетка li-пайплайна 0,1 м (expds.accel.las_grid, до
                   заполнения) → блоковое среднее 5×5 со сдвигом на сетку 0,5 м → пирамидальное заполнение.
                   В пайплайне это массив, который build_dem уже прочитал; здесь эмулируется его построением.
Метки, классы и фолды не используются; conf — только для отсева.
"""
import argparse, json, os, resource, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from expds import accel  # noqa: E402
from expds.lidar import fill_interior_holes  # noqa: E402
import morph_li as ML  # noqa: E402  (геометрия окна и ядра — те же объекты)

RAW = ["morph_valid", "morph_bg_slope", "morph_bg_rough", "morph_h0", "morph_hmax", "morph_r_peak", "morph_ring",
       "morph_ring_rel", "morph_r_half", "morph_r_edge", "morph_slope_edge", "morph_vol", "morph_shape",
       "morph_top_ratio", "morph_asym4", "morph_secmin4", "morph_secmin8", "morph_elong", "morph_cent_off",
       "morph_peak_off", "morph_tpi2", "morph_tpi5", "morph_tpi8", "morph_std3", "morph_std8", "morph_slope2",
       "morph_slope5", "morph_slope_max5", "morph_curv1_5", "morph_curv3"]
RZ = ["morph_bg_rough", "morph_h0", "morph_hmax", "morph_ring", "morph_r_half", "morph_r_edge", "morph_slope_edge",
      "morph_vol", "morph_tpi2", "morph_tpi5", "morph_tpi8", "morph_std3", "morph_std8", "morph_curv1_5", "morph_curv3"]
NAMES45 = RAW + ["morph_rz_" + n[6:] for n in RZ]
assert len(NAMES45) == 45
J = {n: i for i, n in enumerate(RAW)}

RES, HW, NB, RBIN = ML.RES, ML.HW, ML.NB, ML.RBIN
n20, DS, XS, YS = ML.n20, ML.DS, ML.XS, ML.YS
OFFY, OFFX = ML.OFFY[:n20], ML.OFFX[:n20]
LY, LX, L_ANN, A_lat = ML.LY, ML.LX, ML.L_ANN, ML.A_lat
cut, cutlt = ML._cut, ML._cutlt
# кольца 0,5 м до 20 м (0..39); фон 12–20 м = кольца 24..39, центр r<1 = кольца 0..1, профиль = 0..31
E40 = np.array([cutlt(k * RBIN) for k in range(40)] + [n20])
assert E40[-1] == cut(20.0) and cutlt(12.0) == E40[24] and cutlt(1.0) == E40[2]
assert np.array_equal(E40[:NB + 1], ML.RB_EDGES)
n1, n3, n5, n8, n16 = cutlt(1.0), cut(3.0), cut(5.0), cut(8.0), cut(16.0)
nTPI = [cut(r) for r in (0.75, 2, 5, 8)]
# секторы: перестановка пикселей кольца по номеру сектора
SECP = []
for r0, r1 in ((1, 4), (3, 8)):
    i0, i1 = cutlt(r0), cut(r1)
    ss = ML.SEC[i0:i1]
    perm = np.argsort(ss, kind="stable") + i0
    starts = np.searchsorted(ss[perm - i0], np.arange(8))
    SECP.append((perm, starts))
X8, Y8 = XS[:n8].astype(np.float32), YS[:n8].astype(np.float32)
XS20, mYS20 = XS[:n20].astype(np.float32), (-YS[:n20]).astype(np.float32)
NBN, NBS, NBE, NBW, NB9, _n5p = ML.NBN, ML.NBS, ML.NBE, ML.NBW, ML.NB9, ML._n5p
LOG15S, nL15, LOG3S, nL3 = ML.LOG15S, ML.nL15, ML.LOG3S, ML.nL3
AAT = (A_lat[:, :, None] * A_lat[:, None, :]).reshape(-1, 9)
RC = ((np.arange(NB) + 0.5) * RBIN).astype(np.float32)
REF_N = cut(1.0)


def _rsum(v, edges):
    return np.add.reduceat(v[:, edges[0]:edges[-1]], edges[:-1] - edges[0], axis=1)


_T=[]
def features(z, zl):
    _T.append((-1, __import__('time').perf_counter()))
    """z — B×n20 (r≤20 м, отн. опорной высоты), zl — B×nL решётка 2 м → B×30 (порядок RAW)."""
    _T.append((0, __import__('time').perf_counter()))
    B = z.shape[0]
    _T.append((1, __import__('time').perf_counter()))
    ar = np.arange(B)
    _T.append((2, __import__('time').perf_counter()))
    out = np.full((B, 30), np.nan, np.float32)
    _T.append((3, __import__('time').perf_counter()))
    fin = np.isfinite(z)
    _T.append((4, __import__('time').perf_counter()))
    finf = fin.astype(np.float32)
    _T.append((5, __import__('time').perf_counter()))
    finl = np.isfinite(zl)
    _T.append((6, __import__('time').perf_counter()))
    out[:, 0] = finl.mean(1)
    _T.append((7, __import__('time').perf_counter()))
    Z = zl[:, L_ANN].astype(np.float64); m = finl[:, L_ANN]; w = m.astype(np.float64); Zf = np.where(m, Z, 0)
    _T.append((8, __import__('time').perf_counter()))
    AtA = (w @ AAT).reshape(B, 3, 3)
    _T.append((9, __import__('time').perf_counter()))
    Atz = Zf @ A_lat
    _T.append((10, __import__('time').perf_counter()))
    ok = w.sum(1) > 20
    _T.append((11, __import__('time').perf_counter()))
    coef = np.full((B, 3), np.nan)
    _T.append((12, __import__('time').perf_counter()))
    if ok.any():
        coef[ok] = np.linalg.solve(AtA[ok] + 1e-9 * np.eye(3), Atz[ok][..., None])[..., 0]
        _T.append((13, __import__('time').perf_counter()))
    out[:, 1] = np.degrees(np.arctan(np.hypot(coef[:, 0], coef[:, 1])))
    _T.append((14, __import__('time').perf_counter()))
    resid = np.where(m, Z - (coef[:, 0:1] * A_lat[:, 0] + coef[:, 1:2] * A_lat[:, 1] + coef[:, 2:3]), np.nan)
    _T.append((15, __import__('time').perf_counter()))
    with np.errstate(all="ignore"):
        out[:, 2] = np.nanstd(resid, 1)
        _T.append((16, __import__('time').perf_counter()))
        c32 = coef.astype(np.float32)
        _T.append((17, __import__('time').perf_counter()))
        zp = z - (c32[:, 0:1] * XS20 + c32[:, 1:2] * mYS20 + c32[:, 2:3])       # NaN там, где z NaN
        _T.append((18, __import__('time').perf_counter()))
        zp0 = np.where(fin, zp, np.float32(0))
        _T.append((19, __import__('time').perf_counter()))
        S = _rsum(zp0, E40).astype(np.float64); N = _rsum(finf, E40).astype(np.float64)
        _T.append((20, __import__('time').perf_counter()))
        bg = S[:, 24:].sum(1) / N[:, 24:].sum(1)
        _T.append((21, __import__('time').perf_counter()))
        c1 = S[:, :2].sum(1) / N[:, :2].sum(1)
        _T.append((22, __import__('time').perf_counter()))
        prof = np.where(N[:, :NB] > 0, S[:, :NB] / np.maximum(N[:, :NB], 1), np.nan)
        _T.append((23, __import__('time').perf_counter()))
        out[:, 3] = c1 - bg
        _T.append((24, __import__('time').perf_counter()))
        p10 = prof[:, :20]
        _T.append((25, __import__('time').perf_counter()))
        allnan = ~np.isfinite(p10).any(1)
        _T.append((26, __import__('time').perf_counter()))
        kpk = np.argmax(np.where(np.isfinite(p10), p10, -np.inf), 1)
        _T.append((27, __import__('time').perf_counter()))
        pmax = p10[ar, kpk]
        _T.append((28, __import__('time').perf_counter()))
        hmax = pmax - bg
        _T.append((29, __import__('time').perf_counter()))
        out[:, 4] = hmax
        _T.append((30, __import__('time').perf_counter()))
        out[:, 5] = np.where(allnan, np.nan, RC[kpk])
        _T.append((31, __import__('time').perf_counter()))
        out[:, 6] = pmax - c1
        _T.append((32, __import__('time').perf_counter()))
        out[:, 7] = (pmax - c1) / np.maximum(hmax, 0.05)
        _T.append((33, __import__('time').perf_counter()))
        rel = prof - bg[:, None]
        _T.append((34, __import__('time').perf_counter()))
        beyond = np.arange(NB)[None, :] > kpk[:, None]
        _T.append((35, __import__('time').perf_counter()))
        below = beyond & (rel < (hmax / 2)[:, None])
        _T.append((36, __import__('time').perf_counter()))
        has = below.any(1)
        _T.append((37, __import__('time').perf_counter()))
        kh = np.argmax(below, 1)
        _T.append((38, __import__('time').perf_counter()))
        r_half = np.where(has, RC[kh], np.nan)
        _T.append((39, __import__('time').perf_counter()))
        out[:, 8] = r_half
        _T.append((40, __import__('time').perf_counter()))
        dp = -(np.diff(prof, axis=1)) / RBIN
        _T.append((41, __import__('time').perf_counter()))
        dpm = np.where(np.arange(NB - 1)[None] >= kpk[:, None], dp, np.nan)
        _T.append((42, __import__('time').perf_counter()))
        dpm = np.where(np.isfinite(dpm), dpm, -np.inf)
        _T.append((43, __import__('time').perf_counter()))
        ke = np.argmax(dpm, 1); se = dpm[ar, ke]
        _T.append((44, __import__('time').perf_counter()))
        out[:, 9] = np.where(np.isfinite(se), (ke + 1) * RBIN, np.nan)
        _T.append((45, __import__('time').perf_counter()))
        out[:, 10] = np.where(np.isfinite(se), se, np.nan)
        _T.append((46, __import__('time').perf_counter()))
        rv = np.where(np.isfinite(r_half), np.minimum(2 * r_half, 16.0), 16.0).astype(np.float32)
        _T.append((47, __import__('time').perf_counter()))
        bg32 = bg.astype(np.float32)[:, None]
        _T.append((48, __import__('time').perf_counter()))
        pos = np.maximum(zp0[:, :n16] - bg32, 0) * finf[:, :n16]
        _T.append((49, __import__('time').perf_counter()))
        out[:, 11] = np.where(DS[None, :n16] <= rv[:, None], pos, 0).sum(1) * RES * RES
        _T.append((50, __import__('time').perf_counter()))
        kq = np.where(np.isfinite(r_half), np.clip(np.nan_to_num(r_half / 2 / RBIN, nan=0).astype(int), 0, NB - 1), 0)
        _T.append((51, __import__('time').perf_counter()))
        out[:, 12] = np.where(np.isfinite(r_half), rel[ar, kq] / np.maximum(hmax, 0.05), np.nan)
        _T.append((52, __import__('time').perf_counter()))
        out[:, 13] = (c1 - bg) / np.maximum(hmax, 0.05)
        _T.append((53, __import__('time').perf_counter()))
        for (perm, st), ia, im in zip(SECP, (14, None), (15, 16)):
            sv = zp0[:, perm]; sf = finf[:, perm]
            s_ = np.add.reduceat(sv, st, axis=1); n_ = np.add.reduceat(sf, st, axis=1)
            sm = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan) - bg[:, None]
            if ia is not None:
                out[:, ia] = np.nanstd(sm, 1) / (np.abs(np.nanmean(sm, 1)) + 0.05)
            out[:, im] = np.nanmin(sm, 1)
        pw = pos[:, :n8]
        _T.append((54, __import__('time').perf_counter()))
        M0 = pw.sum(1, dtype=np.float64) + 1e-12
        _T.append((55, __import__('time').perf_counter()))
        mx = pw @ X8 / M0; my = pw @ Y8 / M0
        _T.append((56, __import__('time').perf_counter()))
        sxx = pw @ (X8 * X8) / M0 - mx ** 2; syy = pw @ (Y8 * Y8) / M0 - my ** 2; sxy = pw @ (X8 * Y8) / M0 - mx * my
        _T.append((57, __import__('time').perf_counter()))
        tr = sxx + syy; det = sxx * syy - sxy ** 2
        _T.append((58, __import__('time').perf_counter()))
        disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0)); l1, l2 = tr / 2 + disc, tr / 2 - disc
        _T.append((59, __import__('time').perf_counter()))
        good = M0 > 1e-3
        _T.append((60, __import__('time').perf_counter()))
        out[:, 17] = np.where(good, np.sqrt(l1 / np.maximum(l2, 1e-6)), np.nan)
        _T.append((61, __import__('time').perf_counter()))
        out[:, 18] = np.where(good, np.hypot(mx, my), np.nan)
        _T.append((62, __import__('time').perf_counter()))
        z0f = np.where(fin, z, np.float32(0))
        _T.append((63, __import__('time').perf_counter()))
        s9 = z0f[:, NB9].sum(1); c9 = fin[:, NB9].sum(1)
        _T.append((64, __import__('time').perf_counter()))
        sm9 = np.where(c9 > 0, s9 / np.maximum(c9, 1), -np.inf)
        _T.append((65, __import__('time').perf_counter()))
        ki = np.argmax(sm9, 1)
        _T.append((66, __import__('time').perf_counter()))
        out[:, 19] = np.where(np.isfinite(sm9[ar, ki]), DS[ki], np.nan)
        _T.append((67, __import__('time').perf_counter()))
        e = np.array([0] + nTPI)
        _T.append((68, __import__('time').perf_counter()))
        ss_ = _rsum(z0f, e).cumsum(1); nn_ = _rsum(finf, e).cumsum(1)
        _T.append((69, __import__('time').perf_counter()))
        dm = np.where(nn_ > 0, ss_ / np.maximum(nn_, 1), np.nan)
        _T.append((70, __import__('time').perf_counter()))
        z0 = dm[:, 0]
        _T.append((71, __import__('time').perf_counter()))
        out[:, 20:23] = z0[:, None] - dm[:, 1:4]
        _T.append((72, __import__('time').perf_counter()))
        for j, n in ((23, n3), (24, n8)):
            out[:, j] = np.nanstd(zp[:, :n], 1)
        gy = (z[:, NBS] - z[:, NBN]) / (2 * RES); gx = (z[:, NBE] - z[:, NBW]) / (2 * RES)
        _T.append((73, __import__('time').perf_counter()))
        sl = np.degrees(np.arctan(np.hypot(gx, gy)))
        _T.append((74, __import__('time').perf_counter()))
        out[:, 25] = np.nanmean(sl[:, :cut(2.0)], 1)
        _T.append((75, __import__('time').perf_counter()))
        out[:, 26] = np.nanmean(sl, 1)
        _T.append((76, __import__('time').perf_counter()))
        out[:, 27] = np.nanmax(sl, 1)
        _T.append((77, __import__('time').perf_counter()))
        for j, K, n in ((28, LOG15S, nL15), (29, LOG3S, nL3)):
            okk = fin[:, :n].all(1)
            out[:, j] = np.where(okk, zp0[:, :n] @ K, np.nan)
    out[out[:, 0] < 0.05, 1:] = np.nan
    _T.append((78, __import__('time').perf_counter()))
    out[~np.isfinite(z0), 3:] = np.nan
    _T.append((79, __import__('time').perf_counter()))
    return out


H20 = int(np.ceil(20.0 / RES))                 # 40 пикс.
IDX20 = (OFFY + H20) * (2 * H20 + 1) + (OFFX + H20)
HL = HW // 4                                    # решётка 2 м: шаг 4 пикс.
assert np.all(LY % 4 == 0) and np.all(LX % 4 == 0)
IDXL = (LY // 4 + HL) * (2 * HL + 1) + (LX // 4 + HL)


def windows(dem, left, top, cx, cy):
    """окна r≤20 м и решётка 2 м; высоты относительно медианы r≤1 м (как morph_li.windows)."""
    H, Wd = dem.shape
    col = np.floor((cx - left) / RES).astype(np.int64)
    row = np.floor((top - cy) / RES).astype(np.int64)
    flat = dem.ravel()
    inner = (row >= HW) & (row < H - HW) & (col >= HW) & (col < Wd - HW)
    z = np.empty((cx.size, n20), np.float32); zl = np.empty((cx.size, LY.size), np.float32)
    if inner.any():
        from numpy.lib.stride_tricks import as_strided
        s0, s1 = dem.strides
        r_i, c_i = row[inner], col[inner]
        v20 = as_strided(dem, (H, Wd, 2 * H20 + 1, 2 * H20 + 1), (s0, s1, s0, s1))
        z[inner] = v20[r_i - H20, c_i - H20].reshape(-1, (2 * H20 + 1) ** 2)[:, IDX20]
        vl = as_strided(dem, (H, Wd, 2 * HL + 1, 2 * HL + 1), (s0, s1, 4 * s0, 4 * s1))
        zl[inner] = vl[r_i - HW, c_i - HW].reshape(-1, (2 * HL + 1) ** 2)[:, IDXL]
    if (~inner).any():
        for oy, ox, dst in ((OFFY, OFFX, z), (LY, LX, zl)):
            rr = row[~inner][:, None] + oy; cc = col[~inner][:, None] + ox
            okk = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < Wd)
            dst[~inner] = np.where(okk, flat[np.clip(rr, 0, H - 1) * Wd + np.clip(cc, 0, Wd - 1)], np.nan)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        ref = np.nanmedian(z[:, :REF_N], 1)
        bad = ~np.isfinite(ref)
        if bad.any():
            r2 = np.nanmedian(z[bad], 1)
            ref[bad] = np.where(np.isfinite(r2), r2, 0.0)
    ref = ref.astype(np.float32)[:, None]
    z -= ref; zl -= ref
    return z, zl


def dem_from_li(laz):
    """эмуляция переиспользования: сетка li-пайплайна 0,1 м → 0,5 м (сдвиг на выравнивание, среднее 5×5 по
    непустым ячейкам) → пирамида. Возвращает dem, left, top и время только пересчёта (без чтения LAS)."""
    import laspy
    from expds.accel import grid_spec
    d01, g01 = accel.las_grid(str(laz), 0.1, 2, 60_000_000)
    if abs(g01["res"] - 0.1) > 1e-9:
        raise RuntimeError(f"li-сетка огрублена до {g01['res']} — в пайплайне это мозаика 0,5 м")
    t0 = time.perf_counter()
    with laspy.open(str(laz)) as f:
        h = f.header
        s05 = grid_spec(h.mins[0], h.mins[1], h.maxs[0], h.maxs[1], 0.5, 60_000_000)
    dc = int(round((g01["left"] - s05["left"]) / 0.1)); dr = int(round((s05["top"] - g01["top"]) / 0.1))
    H5, W5 = s05["height"], s05["width"]
    big = np.full((H5 * 5, W5 * 5), np.nan, np.float32)
    hh = min(d01.shape[0], H5 * 5 - dr); ww = min(d01.shape[1], W5 * 5 - dc)
    big[dr:dr + hh, dc:dc + ww] = d01[:hh, :ww]
    del d01
    b = big.reshape(H5, 5, W5, 5)
    fin = np.isfinite(b)
    s = np.where(fin, b, 0).sum((1, 3), dtype=np.float64); n = fin.sum((1, 3))
    del big, b, fin
    dem = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
    t1 = time.perf_counter()
    dem = fill_interior_holes(dem, method="pyramid")
    return dem, s05["left"], s05["top"], time.perf_counter() - t0, (dr, dc)


def rz_stats(F, sel_stat):
    out = np.full((F.shape[0], len(RZ)), np.nan, np.float32)
    for k, n in enumerate(RZ):
        x = F[:, J[n]].astype(np.float64)
        xs = x[sel_stat]
        if np.isfinite(xs).sum() >= 10:
            med = np.nanmedian(xs); mad = 1.4826 * np.nanmedian(np.abs(xs - med))
            out[:, k] = (x - med) / max(mad, 1e-6)
    return out


def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--root", default="/workspace/data/train-expds-3")
    ap.add_argument("--out", required=True, help="папка: <регион>.npz (stack_idx, feats45, names) + .json")
    ap.add_argument("--only", default="")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--conf-min", type=float, default=0.0)
    ap.add_argument("--rz-sample", type=int, default=2000)
    ap.add_argument("--dem-from-li", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    C = np.load(a.cands, allow_pickle=False)
    sidx, reg, cx, cy = C["stack_idx"], C["region"], C["cx"], C["cy"]
    conf = C["conf"] if a.conf_min > 0 else None
    regions = sorted(set(reg.tolist()))
    if a.only:
        regions = [r for r in regions if r in a.only.split(",")]
    for region in regions:
        m = np.flatnonzero(reg == region)
        info = {"region": region, "n": int(m.size), "conf_min": a.conf_min, "dem_from_li": a.dem_from_li}
        F = np.full((m.size, 30), np.nan, np.float32)
        laz = ML.find_laz(a.root, region)
        if conf is not None:
            hi = conf[m] >= a.conf_min
            rng = np.random.default_rng(0)
            lo = np.flatnonzero(~hi)
            samp = rng.choice(lo, min(a.rz_sample, lo.size), replace=False) if lo.size else lo
            todo = np.zeros(m.size, bool); todo[hi] = True; todo[samp] = True
            stat = todo.copy()
            info.update(n_hi=int(hi.sum()), n_rz_sample=int(samp.size))
        else:
            todo = np.ones(m.size, bool); stat = todo
        info["n_feat"] = int(todo.sum())
        if laz is None:
            info["status"] = "no_laz"
        else:
            ccx, ccy, note = ML.align_coords(laz, cx[m], cy[m])
            info["coords"] = note
            t0 = time.perf_counter()
            if a.dem_from_li:
                dem, left, top, t_conv, shift = dem_from_li(laz)
                t2 = time.perf_counter()
                info.update(t_li_grid_s=round(t2 - t0 - t_conv, 2), t_reuse_s=round(t_conv, 2), shift_rc=list(shift))
            else:
                dem, gi = accel.las_grid(str(laz), RES, 2, 60_000_000)
                t1 = time.perf_counter()
                dem = fill_interior_holes(dem, method="pyramid")
                t2 = time.perf_counter()
                left, top = gi["left"], gi["top"]
                info.update(t_grid_s=round(t1 - t0, 2), t_fill_s=round(t2 - t1, 2))
                if abs(float(gi["res"]) - RES) > 1e-9:
                    info["status"] = f"coarsened_res_{gi['res']}"
            info["rss_after_dem_mb"] = round(maxrss_mb(), 1)
            if "status" not in info:
                ii = np.flatnonzero(todo)
                for s in range(0, ii.size, a.batch):
                    j = ii[s:s + a.batch]
                    F[j] = features(*windows(dem, left, top, ccx[j], ccy[j]))
                info["status"] = "ok"
            t3 = time.perf_counter()
            info["t_feat_s"] = round(t3 - t2, 3)
            info["ms_per_cand_feat"] = round(1000 * (t3 - t2) / max(int(todo.sum()), 1), 4)
            info["t_total_s"] = round(t3 - t0, 2)
            del dem
        rz = rz_stats(F, stat)
        np.savez(out / f"{region}.npz", stack_idx=sidx[m].astype(np.int32), feats45=np.hstack([F, rz]),
                 computed=todo, names=np.array(NAMES45))
        info["maxrss_mb"] = round(maxrss_mb(), 1)
        (out / f"{region}.json").write_text(json.dumps(info, ensure_ascii=False))
        print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
