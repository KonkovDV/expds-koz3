"""Задача 10 (OSINT п.3): безучебные морфометрические признаки рельефа для лидарных кандидатов курганов.

Вход: li_kurgan_candidates.npz (bot-4; используются только stack_idx, region, cx, cy) и LAZ участка.
Метки, классы, conf, фолды и эталон в расчёте НЕ используются.
DEM: средняя высота точек земли (класс 2; если его нет — всех точек) на сетке 0,5 м
(expds.accel.las_grid), пустоты внутри выпуклой оболочки — пирамида (fill_interior_holes, pyramid).
Окно ±32 м вокруг пикселя центроида: все пиксели 0,5 м в r≤20 м + решётка 2 м в r≤32 м + 16 лучей до 30 м;
признаки — векторно по пачкам окон.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expds import accel  # noqa: E402
from expds.lidar import fill_interior_holes  # noqa: E402

RES = 0.5
HW = 64                       # полуокно, пикс. (32 м)
W = 2 * HW + 1
yy, xx = np.mgrid[-HW:HW + 1, -HW:HW + 1]
D = np.hypot(yy, xx) * RES    # расстояние до центра, м
ANG = np.arctan2(-yy, xx)     # угол (север вверх)
RBIN = 0.5                    # шаг радиального профиля, м
NB = 32                       # профиль до 16 м
rb = np.floor(D / RBIN).astype(int)

FEATS = [  # имя, единицы, описание
    ("morph_valid", "доля", "доля валидных узлов решётки 2 м в круге r≤32 м"),
    ("morph_bg_slope", "град", "уклон плоскости, подогнанной по узлам решётки 2 м в кольце 16–32 м"),
    ("morph_bg_rough", "м", "СКО остатков плоскости (решётка 2 м, кольцо 16–32 м)"),
    ("morph_h0", "м", "z' в r<1 м минус фон (среднее z' в кольце 12–20 м); z' = z − плоскость"),
    ("morph_hmax", "м", "максимум радиального профиля z' при r≤10 м минус фон"),
    ("morph_r_peak", "м", "радиус максимума профиля (0 — купол, >0 — кольцо)"),
    ("morph_ring", "м", "глубина центральной ямы: max профиля r≤10 м − z'(r<1 м)"),
    ("morph_ring_rel", "1", "morph_ring / max(morph_hmax, 0,05)"),
    ("morph_r_half", "м", "радиус, где профиль падает ниже фон+hmax/2 (за пиком; NaN если нет до 16 м)"),
    ("morph_r_edge", "м", "радиус наибольшего спада профиля за пиком"),
    ("morph_slope_edge", "м/м", "наибольший спад профиля −dp/dr за пиком"),
    ("morph_vol", "м³", "объём (z'−фон)+ в круге r≤min(2·r_half,16) м"),
    ("morph_shape", "1", "(p(r_half/2)−фон)/hmax: купол ≈0,8–0,9, конус ≈0,5"),
    ("morph_top_ratio", "1", "h0/hmax"),
    ("morph_asym4", "1", "СКО средних по 8 секторам в кольце 1–4 м / (|среднее|+0,05)"),
    ("morph_secmin4", "м", "минимум средних по 8 секторам в кольце 1–4 м минус фон"),
    ("morph_secmin8", "м", "то же для кольца 3–8 м"),
    ("morph_elong", "1", "отношение собственных чисел вторых моментов (z'−фон)+ в r≤8 м"),
    ("morph_cent_off", "м", "смещение центра масс (z'−фон)+ в r≤8 м от центроида"),
    ("morph_peak_off", "м", "расстояние до максимума сглаженного z' в r≤5 м"),
    ("morph_tpi2", "м", "z(r<0,75 м) − среднее z в круге r≤2 м"),
    ("morph_tpi5", "м", "то же, r≤5 м"),
    ("morph_tpi8", "м", "то же, r≤8 м (≈ lrm_s без клипа)"),
    ("morph_tpi15", "м", "то же, r≤15 м (среднее по решётке 2 м)"),
    ("morph_tpi30", "м", "то же, r≤30 м (среднее по решётке 2 м; ≈ lrm_l без клипа)"),
    ("morph_std3", "м", "СКО z' в r≤3 м"),
    ("morph_std8", "м", "СКО z' в r≤8 м"),
    ("morph_slope2", "град", "средний уклон в r≤2 м"),
    ("morph_slope5", "град", "средний уклон в r≤5 м"),
    ("morph_slope_max5", "град", "максимальный уклон в r≤5 м"),
    ("morph_curv1_5", "1/м", "LoG σ=1,5 м в центре (минус — купол)"),
    ("morph_curv3", "1/м", "LoG σ=3 м в центре"),
    ("morph_opp10", "град", "положительная открытость, 16 направлений, луч 10 м"),
    ("morph_opn10", "град", "отрицательная открытость, луч 10 м"),
    ("morph_svf10", "1", "SVF = 1 − ср. max(0, sin γmax), луч 10 м"),
    ("morph_opp30", "град", "положительная открытость, луч 30 м (шаг 1 м)"),
    ("morph_svf30", "1", "SVF, луч 30 м (шаг 1 м)"),
]
NAMES = [f[0] for f in FEATS]


def _log_kernel(sig_m):
    s = sig_m / RES
    r2 = (xx ** 2 + yy ** 2).astype(np.float64)
    g = np.exp(-r2 / (2 * s * s))
    k = (r2 - 2 * s * s) / (s ** 4) * g
    k[r2 > (4 * s) ** 2] = 0
    k -= k[r2 <= (4 * s) ** 2].mean() * (r2 <= (4 * s) ** 2)  # нулевая сумма
    k /= g[r2 <= (4 * s) ** 2].sum()
    return (k / RES ** 2).astype(np.float32)


LOG15, LOG3 = _log_kernel(1.5), _log_kernel(3.0)
NDIR = 16
_dirs = np.arange(NDIR) * 2 * np.pi / NDIR


def _rays(rmax):
    d = np.arange(RBIN, rmax + 1e-9, RBIN)
    ry = np.rint(-np.sin(_dirs)[:, None] * d / RES).astype(int) + HW
    rx = np.rint(np.cos(_dirs)[:, None] * d / RES).astype(int) + HW
    return ry, rx, d


RAY10 = _rays(10.0)


# ── пиксели окна, отсортированные по расстоянию до центра ──
_ord = np.argsort(D.ravel(), kind="stable")
_ord = _ord[D.ravel()[_ord] <= 32.0]
DS = D.ravel()[_ord]                      # расстояния (возр.)
YS = (yy.ravel()[_ord] * RES).astype(np.float32)   # вниз — +
XS = (xx.ravel()[_ord] * RES).astype(np.float32)
AS = ANG.ravel()[_ord]
NP = DS.size
OFFY = yy.ravel()[_ord]; OFFX = xx.ravel()[_ord]
n20 = int(np.searchsorted(DS, 20.0, "right"))
def _cut(r):           # число пикселей с D<=r
    return int(np.searchsorted(DS, r, "right"))
def _cutlt(r):
    return int(np.searchsorted(DS, r, "left"))
# кольца профиля: границы в отсортированном порядке
RB_EDGES = np.array([_cutlt(k * RBIN) for k in range(NB + 1)])
BG0, BG1 = _cutlt(12.0), _cut(20.0)
SEC = (np.floor((AS + np.pi) / (2 * np.pi) * 8).astype(int) % 8)
n8 = _cut(8.0); n5 = _cut(5.0)
# ядра LoG в отсортированном порядке (до 4σ)
def _sorted_kernel(K):
    k = K.ravel()[_ord]
    n = int(np.max(np.flatnonzero(k != 0))) + 1
    return k[:n].astype(np.float32), n
LOG15S, nL15 = _sorted_kernel(LOG15)
LOG3S, nL3 = _sorted_kernel(LOG3)
# лучи: индексы в отсортированном порядке
_pos = {(int(a_), int(b_)): i for i, (a_, b_) in enumerate(zip(OFFY, OFFX))}
def _ray_idx(rr):
    ry, rx, d = rr
    return np.vectorize(lambda a_, b_: _pos[(a_ - HW, b_ - HW)])(ry, rx), d.astype(np.float32)
RI10, RD10 = _ray_idx(RAY10)
# решётка 2 м (каждый 4-й пиксель) в круге r<=32 м: плоскость, фон-шероховатость, TPI15/30
_lat = (yy % 4 == 0) & (xx % 4 == 0) & (D <= 32.0)
_lo = np.argsort(D[_lat], kind="stable")
LY = yy[_lat][_lo]; LX = xx[_lat][_lo]; LD = D[_lat][_lo]
L_ANN = LD >= 16.0
A_lat = np.stack([LX[L_ANN] * RES, -LY[L_ANN] * RES, np.ones(int(L_ANN.sum()))], 1).astype(np.float64)
# лучи до 30 м с шагом 1 м (16 направлений) — отдельная выборка
_d30 = np.arange(1.0, 30.0 + 1e-9, 1.0)
R30Y = np.rint(-np.sin(_dirs)[:, None] * _d30 / RES).astype(int)
R30X = np.rint(np.cos(_dirs)[:, None] * _d30 / RES).astype(int)
RD30 = _d30.astype(np.float32)
# соседи для уклона (центральные разности) внутри r<=5 м
_n5p = _cut(5.0)
def _nb(dy, dx):
    return np.array([_pos[(int(a_) + dy, int(b_) + dx)] for a_, b_ in zip(OFFY[:_n5p], OFFX[:_n5p])])
NBN, NBS, NBE, NBW = _nb(-1, 0), _nb(1, 0), _nb(0, 1), _nb(0, -1)
# 3×3 среднее для поиска пика в r<=5 м
NB9 = np.stack([_nb(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)


def _segmean(v, f, edges):
    """средние по сегментам [edges[i], edges[i+1]) с NaN-маской f; v,f: B×N."""
    e = np.asarray(edges)
    vv = np.where(f, v, 0).astype(np.float64)[:, e[0]:e[-1]]
    ff = f[:, e[0]:e[-1]].astype(np.float64)
    st = e[:-1] - e[0]
    s_ = np.add.reduceat(vv, st, axis=1); n = np.add.reduceat(ff, st, axis=1)
    empty = (e[1:] - e[:-1]) <= 0
    s_[:, empty] = 0; n[:, empty] = 0
    with np.errstate(all="ignore"):
        return np.where(n > 0, s_ / n, np.nan)


def features(zz):
    """zz = (z, zl, zr): z — B×n20 пиксели r<=20 м по возрастанию расстояния, zl — B×nL решётка 2 м,
    zr — B×16×30 лучи до 30 м; высоты относительно опорной точки; NaN вне съёмки → B×len(NAMES)."""
    z, zl, zr30 = zz
    import warnings
    warnings.simplefilter("ignore")
    B = z.shape[0]
    ar = np.arange(B)
    out = np.full((B, len(NAMES)), np.nan, np.float32)
    fin = np.isfinite(z)
    finl = np.isfinite(zl)
    out[:, 0] = finl.mean(1)
    # плоскость по решётке 2 м в кольце 16–32 м
    A_ann = A_lat
    Z = zl[:, L_ANN].astype(np.float64); m = np.isfinite(Z); w = m.astype(np.float64); Zf = np.where(m, Z, 0)
    AtA = (w @ (A_ann[:, :, None] * A_ann[:, None, :]).reshape(-1, 9)).reshape(B, 3, 3)
    Atz = Zf @ A_ann
    ok = w.sum(1) > 20
    coef = np.full((B, 3), np.nan)
    if ok.any():
        coef[ok] = np.linalg.solve(AtA[ok] + 1e-9 * np.eye(3), Atz[ok][..., None])[..., 0]
    out[:, 1] = np.degrees(np.arctan(np.hypot(coef[:, 0], coef[:, 1])))
    resid = Z - (coef[:, 0:1] * A_ann[:, 0] + coef[:, 1:2] * A_ann[:, 1] + coef[:, 2:3])
    with np.errstate(all="ignore"):
        out[:, 2] = np.nanstd(np.where(m, resid, np.nan), 1)
        zp = (z[:, :n20] - (coef[:, 0:1] * XS[:n20] + coef[:, 1:2] * (-YS[:n20]) + coef[:, 2:3])).astype(np.float32)
        fp = fin[:, :n20]
        bg = _segmean(zp, fp, [BG0, BG1])[:, 0]
        prof = _segmean(zp, fp, RB_EDGES)                     # B×NB
        rc = (np.arange(NB) + 0.5) * RBIN
        c1 = _segmean(zp, fp, [0, _cutlt(1.0)])[:, 0]
        out[:, 3] = c1 - bg
        p10 = prof[:, :20]
        allnan = ~np.isfinite(p10).any(1)
        kpk = np.argmax(np.where(np.isfinite(p10), p10, -np.inf), 1)
        pmax = p10[ar, kpk]
        hmax = pmax - bg
        out[:, 4] = hmax
        out[:, 5] = np.where(allnan, np.nan, rc[kpk])
        out[:, 6] = pmax - c1
        out[:, 7] = (pmax - c1) / np.maximum(hmax, 0.05)
        rel = prof - bg[:, None]
        beyond = np.arange(NB)[None, :] > kpk[:, None]
        below = beyond & (rel < (hmax / 2)[:, None])
        has = below.any(1)
        kh = np.argmax(below, 1)
        r_half = np.where(has, rc[kh], np.nan)
        out[:, 8] = r_half
        dp = -(np.diff(prof, axis=1)) / RBIN
        dpm = np.where(np.arange(NB - 1)[None] >= kpk[:, None], dp, np.nan)
        dpm = np.where(np.isfinite(dpm), dpm, -np.inf)
        ke = np.argmax(dpm, 1); se = dpm[ar, ke]
        out[:, 9] = np.where(np.isfinite(se), (ke + 1) * RBIN, np.nan)
        out[:, 10] = np.where(np.isfinite(se), se, np.nan)
        rv = np.where(np.isfinite(r_half), np.minimum(2 * r_half, 16.0), 16.0)
        pos = np.where(fp, np.clip(zp - bg[:, None], 0, None), 0)
        inside = DS[None, :n20] <= rv[:, None]
        out[:, 11] = np.where(inside, pos, 0).sum(1) * RES * RES
        kq = np.where(np.isfinite(r_half), np.clip(np.nan_to_num(r_half / 2 / RBIN, nan=0).astype(int), 0, NB - 1), 0)
        out[:, 12] = np.where(np.isfinite(r_half), rel[ar, kq] / np.maximum(hmax, 0.05), np.nan)
        out[:, 13] = (c1 - bg) / np.maximum(hmax, 0.05)
        for r0, r1, ia, im in ((1, 4, 14, 15), (3, 8, None, 16)):
            i0, i1 = _cutlt(r0), _cut(r1)
            sv = zp[:, i0:i1]; sf = fp[:, i0:i1]; ss = SEC[i0:i1]
            sm = np.stack([np.nanmean(np.where(sf[:, ss == q], sv[:, ss == q], np.nan), 1) for q in range(8)], 1) - bg[:, None]
            if ia is not None:
                out[:, ia] = np.nanstd(sm, 1) / (np.abs(np.nanmean(sm, 1)) + 0.05)
            out[:, im] = np.nanmin(sm, 1)
        pw = pos[:, :n8].astype(np.float64); X8 = XS[:n8]; Y8 = YS[:n8]
        M0 = pw.sum(1) + 1e-12
        mx = pw @ X8 / M0; my = pw @ Y8 / M0
        sxx = pw @ (X8 * X8) / M0 - mx ** 2; syy = pw @ (Y8 * Y8) / M0 - my ** 2; sxy = pw @ (X8 * Y8) / M0 - mx * my
        tr = sxx + syy; det = sxx * syy - sxy ** 2
        disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0)); l1, l2 = tr / 2 + disc, tr / 2 - disc
        good = M0 > 1e-3
        out[:, 17] = np.where(good, np.sqrt(l1 / np.maximum(l2, 1e-6)), np.nan)
        out[:, 18] = np.where(good, np.hypot(mx, my), np.nan)
        s9 = np.zeros((B, _n5p), np.float32); c9 = np.zeros((B, _n5p), np.float32)
        for k in range(9):
            idx = NB9[k]; s9 += np.where(fin[:, idx], z[:, idx], 0); c9 += fin[:, idx]
        sm9 = np.where(c9 > 0, s9 / np.maximum(c9, 1), -np.inf)
        ki = np.argmax(sm9, 1)
        out[:, 19] = np.where(np.isfinite(sm9[ar, ki]), DS[ki], np.nan)
        # TPI по сырому z (диски — префиксы отсортированного порядка)
        RD_ = (0.75, 2, 5, 8)
        e = [0] + [_cut(r) for r in RD_]
        ss_ = np.add.reduceat(np.where(fin, z, 0)[:, :e[-1]], e[:-1], axis=1).cumsum(1)
        nn_ = np.add.reduceat(fin[:, :e[-1]].astype(np.float32), e[:-1], axis=1).cumsum(1)
        dm = np.where(nn_ > 0, ss_ / np.maximum(nn_, 1), np.nan)
        z0 = dm[:, 0]
        for j, k in zip(range(20, 23), range(1, 4)):
            out[:, j] = z0 - dm[:, k]
        for j, r in ((23, 15.0), (24, 30.0)):
            sel = LD <= r
            out[:, j] = z0 - np.nanmean(zl[:, sel], 1)
        n3 = _cut(3.0)
        out[:, 25] = np.nanstd(np.where(fp[:, :n3], zp[:, :n3], np.nan), 1)
        out[:, 26] = np.nanstd(np.where(fp[:, :n8], zp[:, :n8], np.nan), 1)
        gy = (z[:, NBS] - z[:, NBN]) / (2 * RES); gx = (z[:, NBE] - z[:, NBW]) / (2 * RES)
        sl = np.degrees(np.arctan(np.hypot(gx, gy)))
        n2 = _cut(2.0)
        out[:, 27] = np.nanmean(sl[:, :n2], 1)
        out[:, 28] = np.nanmean(sl, 1)
        out[:, 29] = np.nanmax(sl, 1)
        for j, K, n in ((30, LOG15S, nL15), (31, LOG3S, nL3)):
            okk = fin[:, :n].all(1)
            out[:, j] = np.where(okk, np.where(fp[:, :n], zp[:, :n], 0) @ K, np.nan)
        for zr, RD, jo, jn, js in ((z[:, RI10], RD10, 32, 33, 34), (zr30, RD30, 35, None, 36)):
            g = np.arctan((zr - z0[:, None, None]) / RD[None, None, :])
            gmax = np.nanmax(g, 2); gmin = np.nanmin(g, 2)
            out[:, jo] = np.nanmean(90 - np.degrees(gmax), 1)
            if jn is not None:
                out[:, jn] = np.nanmean(90 + np.degrees(gmin), 1)
            out[:, js] = 1 - np.nanmean(np.maximum(0, np.sin(gmax)), 1)
    out[out[:, 0] < 0.05, 1:] = np.nan
    out[~np.isfinite(z0), 3:] = np.nan
    return out


def windows(dem, left, top, cx, cy):
    """выборки окна для каждого кандидата (высоты относительно медианы ближних r<=1 м)."""
    col = np.floor((cx - left) / RES).astype(np.int64)
    row = np.floor((top - cy) / RES).astype(np.int64)
    H, Wd = dem.shape
    flat = dem.ravel()

    def gather(oy, ox):
        rr = row.reshape((-1,) + (1,) * oy.ndim) + oy[None]
        cc = col.reshape((-1,) + (1,) * ox.ndim) + ox[None]
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < Wd)
        v = flat[np.clip(rr, 0, H - 1) * Wd + np.clip(cc, 0, Wd - 1)]
        return np.where(ok, v, np.nan)

    z = gather(OFFY[:n20], OFFX[:n20])
    ref = np.nanmedian(z[:, :_cut(1.0)], 1)
    ref = np.where(np.isfinite(ref), ref, np.nanmedian(z, 1))
    ref = np.where(np.isfinite(ref), ref, 0.0)[:, None]
    zl = gather(LY, LX) - ref
    zr = gather(R30Y, R30X) - ref[:, :, None]
    return (z - ref).astype(np.float32), zl.astype(np.float32), zr.astype(np.float32)


def find_laz(root, region):
    d = Path(root) / region
    c = sorted(d.glob("01_*La*/*.la[sz]"))
    return c[0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--root", default="/workspace/data/train-expds-3")
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    C = np.load(a.cands, allow_pickle=False)
    sidx, reg, cx, cy = C["stack_idx"], C["region"], C["cx"], C["cy"]   # только эти поля
    regions = sorted(set(reg.tolist()))
    if a.only:
        regions = [r for r in regions if r in a.only.split(",")]
    for region in regions:
        outf = tmp / f"{region}.npz"
        if outf.exists():
            continue
        m = np.flatnonzero(reg == region)
        info = {"region": region, "n": int(m.size)}
        laz = find_laz(a.root, region)
        feats = np.full((m.size, len(NAMES)), np.nan, np.float32)
        if laz is None:
            info["status"] = "no_laz"
        else:
            t0 = time.perf_counter()
            dem, gi = accel.las_grid(str(laz), RES, 2, 60_000_000)
            t1 = time.perf_counter()
            dem = fill_interior_holes(dem, method="pyramid")
            t2 = time.perf_counter()
            info.update(t_grid_s=round(t1 - t0, 2), t_fill_s=round(t2 - t1, 2), shape=list(dem.shape),
                        res=float(gi["res"]), used_all=bool(gi.get("used_all", False)),
                        points_used=int(gi.get("points_used", -1)))
            if abs(float(gi["res"]) - RES) > 1e-9:
                info["status"] = f"coarsened_res_{gi['res']}"
            else:
                for s in range(0, m.size, a.batch):
                    j = m[s:s + a.batch]
                    feats[s:s + j.size] = features(windows(dem, gi["left"], gi["top"], cx[j], cy[j]))
                info["status"] = "ok"
            t3 = time.perf_counter()
            info["t_feat_s"] = round(t3 - t2, 2)
            info["s_per_cand_feat"] = (t3 - t2) / max(m.size, 1)
            del dem
        np.savez(outf, stack_idx=sidx[m].astype(np.int32), feats=feats)
        (tmp / f"{region}.json").write_text(json.dumps(info, ensure_ascii=False))
        print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
