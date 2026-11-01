"""W1 → W4: RR1s-cf inference plug-in for expds.w5_cloudfeat (flag ``w5_cloudfeat``), offline, no network / S3.

Contract (expds/w5_cloudfeat.py, bots/w4-v3on 18ba8b7):
* ``region_features(reg, li_feats, cfg) -> None`` — lidar pass, while the site LAS is at hand: puts the 10 cloud
  features (expds.cloudfeat.NAMES, from the test LAS + UTM.json, same code as training) into ``f["w5cf_feat"]``
  for li objects that can enter the ranker pool; records that the site has LAS (``has_las``).
* ``score(feats, cfg) -> list[float | None]`` — finish(), on ALL run objects: builds the 50 ranker features
  (X4 order, ``meta.json["names"]``) from the run's own objects, scores pool rows with the per-family LightGBM
  (trained on all train regions), None for rows outside the pool / unknown class / no model.

Features: 35 stack-context features (feats.py/feats2.py of W5, recomputed over the run's objects; ``gq`` uses the
train (grp, cls) conf distribution, ``gq_ref.npz``), 5 ``cnx_*`` (W2 ConvNeXt seeds — NOT produced by the run →
always NaN, as for train regions without cnx coverage; use the ``nocnx`` artifacts trained the same way), 10 ``cf_*``.
Site without LAS / CRS / lidar pass skipped / time left < estimate / error: cf_* = NaN and has_las = 0 (explicit
fallback, the model saw such rows in training: optical-only regions). Geometry is projected exactly as the official
metric does (EPSG:3857 → UTM zone by centroid, Point → circle r = 2 m, 32 points; MultiPolygon → largest part).

Artifacts: ``[w5cf] infer_dir`` (default models/w5cf_infer, relative to the archive root): meta.json, fam<k>.txt,
gq_ref.npz — built by tools/rr1s_cf/train_final.py. Time knobs: ``[w5cf] cf_deadline_s`` (default 1300, seconds since
process start: no cloud features for a site whose LAS would not be read before it), ``cf_mb_per_s`` (default 60).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

KEY = "w5cf_feat"
_ROOT = Path(__file__).resolve().parents[1]
_STATE: dict = {"las_regions": set(), "art": {}}
R_TOL, CIRCLE_PTS = 2.0, 32


# ───────────────────────────── artifacts ─────────────────────────────

def _spec(cfg: dict) -> dict:
    s = (cfg or {}).get("w5cf") or {}
    d = Path(s.get("infer_dir", "models/w5cf_infer"))
    return {"dir": d if d.is_absolute() else _ROOT / d, "deadline": float(s.get("cf_deadline_s", 1300)),
            "mbps": float(s.get("cf_mb_per_s", 60.0)), "margin": float(s.get("cf_pool_margin", 0.0))}


def load_artifacts(cfg: dict | None = None) -> dict:
    d = _spec(cfg or {})["dir"]
    if str(d) in _STATE["art"]:
        return _STATE["art"][str(d)]
    import lightgbm as lgb

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    models = {int(f): lgb.Booster(model_file=str(d / v["file"])) for f, v in meta["models"].items()}
    z = np.load(d / "gq_ref.npz", allow_pickle=False)
    off, val, cnt = z["off"], z["val"], z["cnt"]
    gq = {}
    keys = meta["gq_keys"] if "gq_keys" in meta else z["keys"].tolist()   # числовой npz: ключи — в meta.json
    for i, k in enumerate(keys):
        v, c = val[off[i]:off[i + 1]], cnt[off[i]:off[i + 1]]
        gq[k] = (v, np.concatenate([[0], np.cumsum(c)]))
    art = dict(meta=meta, models=models, gq=gq, names=list(meta["names"]), CL=list(meta["CL"]),
               cid={c: i for i, c in enumerate(meta["CL"])}, grp_codes=list(meta["grp_codes"]), mk_codes=list(meta["mk_codes"]),
               pool_thr={k: (-np.inf if v is None else float(v)) for k, v in meta["pool_thr"].items()})
    _STATE["art"][str(d)] = art
    return art


def reset() -> None:
    """Forget per-run state (sites with LAS). Call between runs in one process (tests)."""
    _STATE["las_regions"] = set()


# ───────────────────────────── geometry (as the official metric) ─────────────────────────────

def _shape(f: dict):
    import shapely.geometry
    from shapely.validation import make_valid

    g = f.get("geom")
    if g is None and f.get("geometry"):
        g = shapely.geometry.shape(f["geometry"])
    if g is None or g.is_empty:
        return None
    t = g.geom_type
    if t == "Point":
        return g
    if t == "MultiPolygon":
        parts = []
        for p in g.geoms:
            p = p if p.is_valid else make_valid(p)
            if p.is_valid and not p.is_empty:
                parts.append(p)
        if not parts:
            return None
        g = max(parts, key=lambda p: p.area)
    elif t != "Polygon":
        return None
    if not g.is_valid:
        g = make_valid(g)
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)
    return g if (g.geom_type in ("Polygon", "Point") and g.is_valid and not g.is_empty) else None


def _to_utm(geoms: np.ndarray) -> np.ndarray:
    import shapely
    from pyproj import Transformer

    out = geoms.copy()
    ok = np.array([g is not None for g in geoms])
    if not ok.any():
        return out
    gg = geoms[ok]
    b = shapely.bounds(gg)
    big = np.maximum(np.abs(b[:, 0]), np.abs(b[:, 2])) > 1_000_000
    idx = np.flatnonzero(ok)
    if big.any():
        sub = gg[big]
        cen = shapely.centroid(sub)
        lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(shapely.get_x(cen), shapely.get_y(cen))
        zones = np.clip(((np.asarray(lon) + 180.0) // 6).astype(np.int32) + 1, 1, 60)
        epsg = np.where(np.asarray(lat) >= 0, 32600, 32700) + zones
        proj = np.empty(len(sub), dtype=object)
        for e in np.unique(epsg):
            m = np.flatnonzero(epsg == e)
            tr = Transformer.from_crs(3857, int(e), always_xy=True)
            proj[m] = shapely.transform(sub[m], lambda c, _t=tr: np.column_stack(_t.transform(c[:, 0], c[:, 1])))
        gg = gg.copy(); gg[big] = proj
    pts = np.flatnonzero(shapely.get_type_id(gg) == 0)
    for i in pts:
        p = gg[i]
        ang = 2 * math.pi * np.arange(CIRCLE_PTS) / CIRCLE_PTS
        ring = [(p.x + R_TOL * math.cos(a), p.y + R_TOL * math.sin(a)) for a in ang]
        gg[i] = shapely.Polygon(ring + [ring[0]])
    out[idx] = gg
    return out


# ───────────────────────────── table from run objects ─────────────────────────────

def _conf(f: dict) -> float:
    try:
        return float(f.get("confidence", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _grp(f: dict) -> str:
    return str(f.get("threshold_group") or f.get("markup_type") or "")


def table(feats: list[dict], art: dict) -> dict:
    """Arrays over run objects with a known class and a usable geometry (rows), as prep.py builds the train table."""
    import shapely

    cid = art["cid"]
    rows = [i for i, f in enumerate(feats) if f.get("class_name") in cid]
    geoms = np.array([_shape(feats[i]) for i in rows], dtype=object)
    keep = np.array([g is not None for g in geoms], bool)
    rows = np.asarray(rows, int)[keep] if len(rows) else np.zeros(0, int)
    g = _to_utm(geoms[keep]) if len(rows) else np.zeros(0, dtype=object)
    F = [feats[i] for i in rows]
    cen = shapely.centroid(g) if len(g) else np.zeros(0)
    b = shapely.bounds(g) if len(g) else np.zeros((0, 4))
    return dict(rows=rows, cls=np.array([cid[f["class_name"]] for f in F], int), conf=np.array([_conf(f) for f in F], float),
                reg=np.array([str(f.get("region_name", "default_source")) for f in F], dtype=object),
                grp=np.array([_grp(f) for f in F], dtype=object), mk=np.array([str(f.get("markup_type") or "") for f in F], dtype=object),
                area=shapely.area(g) if len(g) else np.zeros(0), perim=shapely.length(g) if len(g) else np.zeros(0),
                cx=shapely.get_x(cen) if len(g) else np.zeros(0), cy=shapely.get_y(cen) if len(g) else np.zeros(0),
                nv=shapely.get_num_coordinates(g) if len(g) else np.zeros(0), bw=b[:, 2] - b[:, 0], bh=b[:, 3] - b[:, 1],
                cf=np.array([np.asarray(f[KEY], np.float32) if f.get(KEY) is not None else np.full(10, np.nan, np.float32)
                             for f in F], np.float32).reshape(-1, 10),
                has_las=np.array([str(f.get("region_name", "default_source")) in _STATE["las_regions"] for f in F], float))


def _codes(vals, codes):
    """searchsorted code of a train category; unseen → NaN (LightGBM: missing)."""
    pos = {c: float(i) for i, c in enumerate(codes)}
    return np.array([pos.get(v, np.nan) for v in vals], float)


def gq_of(grp, cls, conf, art) -> np.ndarray:
    """Global (grp, cls) quantile of conf in the train table (train: stable-rank / (n − 1); ties → mid position)."""
    out = np.zeros(len(conf))
    for k in set(zip(grp.tolist(), cls.tolist())):
        m = (grp == k[0]) & (cls == k[1])
        ref = art["gq"].get(f"{k[0]}|{k[1]}")
        if ref is None:
            continue                                   # unseen group: 0, as a lone row in train
        v, cum = ref
        n = int(cum[-1])
        lo = cum[np.searchsorted(v, conf[m], side="left")]
        hi = cum[np.searchsorted(v, conf[m], side="right")]
        pos = np.where(hi > lo, (lo + hi - 1) / 2.0, lo - 0.5 * (lo > 0))   # unseen value: between neighbours
        out[m] = np.clip(pos, 0, n - 1) / max(n - 1, 1)
    return out


def in_pool(grp, cls, conf, art, margin: float = 0.0) -> np.ndarray:
    thr = np.array([art["pool_thr"].get(f"{g}|{c}", np.inf) for g, c in zip(grp.tolist(), cls.tolist())], float)
    return conf >= thr - margin


def features(t: dict, art: dict) -> np.ndarray:
    """(n, 50) float32 in meta['names'] order — feats.py + feats2.py of W5 over the run's objects."""
    from scipy.spatial import cKDTree

    cls, conf, grp, reg, mk = t["cls"], t["conf"], t["grp"], t["reg"], t["mk"]
    N = len(cls)
    F: dict[str, np.ndarray] = {}
    cc = np.clip(conf, 1e-6, 1 - 1e-6)
    F["conf"] = conf; F["logit"] = np.log(cc / (1 - cc)); F["cls"] = cls.astype(float)
    F["grp"] = _codes(grp, art["grp_codes"]); F["mk"] = _codes(mk, art["mk_codes"])
    A, Pm = t["area"], t["perim"]
    F["area"] = np.log1p(A); F["compact"] = 4 * np.pi * A / np.maximum(Pm, 1e-6) ** 2
    F["bw"] = np.log1p(np.maximum(t["bw"], t["bh"]))
    F["aspect"] = np.minimum(t["bw"], t["bh"]) / np.maximum(np.maximum(t["bw"], t["bh"]), 1e-6)
    F["nv"] = np.log1p(t["nv"])
    XY = np.c_[t["cx"], t["cy"]]
    # kurgan twins (P/T of the same (reg, grp) within 0.5 m), as an1.py
    twin = np.full(N, -1)
    kp, kt = art["cid"].get("kurgany_povrezhdennye", 0), art["cid"].get("kurgany_tselye", 1)
    kur = np.flatnonzero((cls == kp) | (cls == kt))
    if len(kur):
        kk = np.array([f"{r}|{g}" for r, g in zip(reg[kur], grp[kur])])
        for k in np.unique(kk):
            ii = kur[kk == k]; a = ii[cls[ii] == kp]; b = ii[cls[ii] == kt]
            if not len(a) or not len(b):
                continue
            d, j = cKDTree(XY[b]).query(XY[a]); ok = d < 0.5
            twin[a[ok]] = b[j[ok]]; twin[b[j[ok]]] = a[ok]
    ht = twin >= 0
    F["has_twin"] = ht.astype(float); F["twin_conf"] = np.where(ht, conf[np.maximum(twin, 0)], -1)
    F["twin_diff"] = np.where(ht, conf - conf[np.maximum(twin, 0)], 0)
    F["has_las"] = t["has_las"].astype(float)
    # (reg, grp, cls) stats
    key = np.array([f"{r}|{g}|{c}" for r, g, c in zip(reg, grp, cls)], dtype=object)
    _, inv = np.unique(key, return_inverse=True) if N else (None, np.zeros(0, int))
    inv = inv.ravel(); cnt = np.bincount(inv) if N else np.zeros(0, int)
    F["rg_n"] = np.log1p(cnt[inv]) if N else np.zeros(0)
    order = np.lexsort((-conf, inv)); starts = np.r_[0, np.cumsum(cnt)[:-1]] if N else np.zeros(0, int)
    rank = np.empty(N); rank[order] = np.arange(N) - starts[inv[order]]
    F["rg_rank"] = np.log1p(rank); F["rg_pct"] = rank / np.maximum(cnt[inv] - 1, 1) if N else np.zeros(0)
    mx = np.full(len(cnt), -1.0); np.maximum.at(mx, inv, conf)
    F["rg_max"] = mx[inv]; F["conf_rel"] = conf / np.maximum(mx[inv], 1e-6)
    F["rg_mean"] = np.bincount(inv, conf, minlength=len(cnt))[inv] / np.maximum(cnt[inv], 1) if N else np.zeros(0)
    q = gq_of(grp, cls, conf, art); F["gq"] = q
    fam = np.where(cls <= 1, 0, cls); strong = q >= 0.98
    for nm in ("nb_same10", "nb_same30", "nb_same100", "nb_strong30", "nb_strong100", "og_max5", "og_n5", "nn_dist", "nb_otherfam50"):
        F[nm] = np.zeros(N)
    E = {k: np.zeros(N) for k in ("d_nhigher_g", "d_maxq_other_g", "d_localmax", "d_maxq_othersub", "d_n8")}
    for r in np.unique(reg) if N else []:
        ir = np.flatnonzero(reg == r)
        for f_ in np.unique(fam[ir]):
            ii = ir[fam[ir] == f_]; tr = cKDTree(XY[ii])
            for rad, nm in ((10, "nb_same10"), (30, "nb_same30"), (100, "nb_same100")):
                F[nm][ii] = np.log1p(tr.query_ball_point(XY[ii], rad, return_length=True))
            if len(ii) > 1:
                d, _ = tr.query(XY[ii], k=min(3, len(ii)))
                dd = np.where(d > 0.5, d, np.inf).min(1)
            else:
                dd = np.full(len(ii), np.inf)
            F["nn_dist"][ii] = np.log1p(np.minimum(dd, 1e4))
            js = ii[strong[ii]]
            if len(js):
                ts = cKDTree(XY[js])
                for rad, nm in ((30, "nb_strong30"), (100, "nb_strong100")):
                    F[nm][ii] = np.log1p(ts.query_ball_point(XY[ii], rad, return_length=True))
            # other-group agreement within 5 m: max gq and number of distinct other groups
            pr = tr.query_pairs(5.0, output_type="ndarray")
            if len(pr):
                a_, b_ = np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]]
                gi = grp[ii]; m = gi[a_] != gi[b_]; a_, b_ = a_[m], b_[m]
                om = np.zeros(len(ii)); np.maximum.at(om, a_, q[ii[b_]])
                gcode = np.unique(gi, return_inverse=True)[1].ravel()
                u = np.unique(a_.astype(np.int64) * (gcode.max() + 1) + gcode[b_])
                on = np.bincount(u // (gcode.max() + 1), minlength=len(ii))
                F["og_max5"][ii] = om; F["og_n5"][ii] = on
        for f_ in np.unique(fam[ir]):
            ii = ir[fam[ir] == f_]; js = ir[(fam[ir] != f_) & strong[ir]]
            if len(js):
                F["nb_otherfam50"][ii] = np.log1p(cKDTree(XY[js]).query_ball_point(XY[ii], 50, return_length=True))
        # duplicate / NMS context within 8 m, per class (feats2.py)
        for c in np.unique(cls[ir]):
            ii = ir[cls[ir] == c]; n = len(ii)
            pr = cKDTree(XY[ii]).query_pairs(8.0, output_type="ndarray")
            a_, b_ = (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]]) if len(pr) else (np.zeros(0, int), np.zeros(0, int))
            qi, gi = q[ii], grp[ii]
            nn = np.bincount(a_, minlength=n); E["d_n8"][ii] = nn
            same = gi[a_] == gi[b_]
            E["d_nhigher_g"][ii] = np.bincount(a_[same & (qi[b_] > qi[a_])], minlength=n)
            mo = np.zeros(n); np.maximum.at(mo, a_[~same], qi[b_[~same]]); E["d_maxq_other_g"][ii] = mo
            ma = np.full(n, -np.inf); np.maximum.at(ma, a_, qi[b_])
            E["d_localmax"][ii] = np.where(nn == 0, 1.0, (qi >= ma).astype(float))
            if c in (0, 1):
                jj = ir[cls[ir] == 1 - c]
                if len(jj):
                    lst = cKDTree(XY[jj]).query_ball_point(XY[ii], 8.0)
                    E["d_maxq_othersub"][ii] = [q[jj[np.asarray(l)]].max() if l else 0 for l in lst]
    F.update(E)
    for k in ("cnx_s0", "cnx_s1", "cnx_mean", "cnx_agree", "cnx_samecls"):
        F[k] = np.full(N, np.nan)                      # W2 ConvNeXt seeds are not part of the run
    from .cloudfeat import NAMES
    for j, k in enumerate(NAMES):
        F[k] = t["cf"][:, j].astype(float)
    return np.column_stack([F[n] for n in art["names"]]).astype(np.float32) if N else np.zeros((0, len(art["names"])), np.float32)


# ───────────────────────────── plug-in API ─────────────────────────────

def _proc_elapsed() -> float:
    """Seconds since this process started (the pipeline watchdog counts from the same point)."""
    try:
        st = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        start = int(st[19]) / os.sysconf("SC_CLK_TCK")
        up = float(Path("/proc/uptime").read_text().split()[0])
        return max(0.0, up - start)
    except Exception:
        return time.monotonic() - _T0


_T0 = time.monotonic()


def region_features(reg, li_feats: list[dict], cfg: dict) -> None:
    """Cloud features of the site's li objects that can enter the pool (train conf thresholds − margin)."""
    from .cloudfeat import cand_boxes, compute

    las = [Path(p) for p in (getattr(reg, "las", None) or [])]
    name = str(getattr(reg, "name", ""))
    if not las:
        return                                          # no LAS: cf_* NaN, has_las = 0
    _STATE["las_regions"].add(name)
    if not li_feats:
        return
    sp = _spec(cfg); art = load_artifacts(cfg)
    mb = sum(p.stat().st_size for p in las if p.exists()) / 2 ** 20
    need = mb / max(sp["mbps"], 1e-6)
    if _proc_elapsed() + need > sp["deadline"]:
        log.warning("w5cf_infer: %s: no time for cloud features (%.0f MB, ~%.0f s), cf_* = NaN", name, mb, need)
        return
    from .catalog import read_utm_json

    crs = read_utm_json(getattr(reg, "utm_json", None))
    if crs is None:
        log.warning("w5cf_infer: %s: no UTM.json CRS, cf_* = NaN", name)
        return
    cid = art["cid"]
    cand = [f for f in li_feats if f.get("class_name") in cid]
    if not cand:
        return
    grp = np.array([_grp(f) for f in cand], dtype=object); cls = np.array([cid[f["class_name"]] for f in cand])
    conf = np.array([_conf(f) for f in cand])
    sel = np.flatnonzero(in_pool(grp, cls, conf, art, sp["margin"]))
    if not len(sel):
        return
    geoms = [_shape(cand[i]) for i in sel]
    ok = [k for k, g in enumerate(geoms) if g is not None]
    t = time.monotonic(); st: dict = {}
    Fv = compute(las, crs, cand_boxes([geoms[k] for k in ok]), stats=st)
    for k, row in zip(ok, Fv):
        cand[sel[k]][KEY] = row
    log.info("w5cf_infer: %s: cloud features %d/%d li objects, %.1f s, %s", name, len(ok), len(cand), time.monotonic() - t, st)


def score(feats: list[dict], cfg: dict) -> list:
    """Ranker score for every run object (None outside the pool / unknown class / family without a model)."""
    art = load_artifacts(cfg)
    out: list = [None] * len(feats)
    t0 = time.monotonic()
    tb = table(feats, art)
    if not len(tb["rows"]):
        return out
    X = features(tb, art)
    pool = in_pool(tb["grp"], tb["cls"], tb["conf"], art)
    fam = np.where(tb["cls"] <= 1, 0, tb["cls"])
    n = 0
    for f_, m in art["models"].items():
        ii = np.flatnonzero(pool & (fam == f_))
        if len(ii):
            s = m.predict(X[ii], num_threads=int(((cfg or {}).get("w5cf") or {}).get("threads", 4)))
            for i, v in zip(ii, s):
                out[int(tb["rows"][i])] = float(v)
            n += len(ii)
    log.info("w5cf_infer: %d objects, pool scored %d, cf rows %d, %.1f s", len(feats), n,
             int(np.isfinite(tb["cf"][:, 0]).sum()), time.monotonic() - t0)
    return out
