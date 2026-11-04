"""W4 E1_G1 clean-room MagGr/EtGr geo chain for inference (PLAN_NEXT step 7 / PREP_PIXGBM_CLEANROOM P1).

Packs the frozen offline recipe FREEZE_E1_G1 (30.10 23:04 MSK) into one module, built ONLY from W4 clean code:
expds.pix_clean (features/predict), expds.pix_org_et (ORG3), and the W4 offline scripts
w4geo/work/{night/mkens.py, sweep/finds_sweep.py decode, sweep/raster_groups.py, gnz/{loops,emit_loops,add_loops}.py,
etfp/mk_E.py, night/mkM_any.py + magp/feats.py nms}. DOC-1: no code of the EN line is imported or reused.

Chain (per region, all GT-free test-time ops):
  1. MagGr finds: mean prob map of the ens6 members (n40r8, n40r6, n40r10, aug, gw1d, gw3d; same order as mkens,
     float16 quantisation as in the offline npz) -> decode sigma 3 px / NMS 0.5 m / p_min 0.02, conf = smoothed peak
     ("raw") -> masks -> cross-map dedupe 1.0 m.  (replaces classic maggr finds, pipeline.pix_merge replace=True)
  2. EtGr object_poly: ORG3 (pix_org_et.region_feats) replaces classic etgr object_poly (group etgr_org).
  3. after merge_within_markup + finalize (as in the offline compose):
     a. threshold_group per MagGr raster: maggr_pix_{cv|gr}{lo|hi}, lo = < cut (300) candidates on that raster;
     b. closed-isoline loops on MagGr "цвет_2" maps (dark 20, area 0.05..3 m2) union (r 0.75 m) / boost;
     c. EtGr top-K 25 per raster by conf and UTM area >= 0.2 m2;
     d. per-raster norm c' = 0.9 c^A (c/q_Q)^(1-A) (Q .9, A .25) + soft-NMS (D 1.75 m, x0.6) on MagGr finds.
Classic TerFl / classic MagGr object_poly stay as they are.
"""
from __future__ import annotations

import collections
import logging
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

MEMBERS = ("n40r8", "n40r6", "n40r10", "aug", "gw1d", "gw3d")   # order == night/mkens.py argv for probs_ens6m
DEFAULTS = {
    "sigma_px": 3.0, "nms_r_m": 0.5, "p_min": 0.02, "max_per_raster": 5000, "cross_map_r_m": 1.0,
    "backend": "lightgbm", "lleaves_target": "x86-64-v2",   # P3: "lleaves" = compiled members, fallback lightgbm
    "f16": True,                       # offline maps went through float16 npz (member dump and mean) — keep for parity
    "group_cut": 300,
    "loops": True, "loops_name_re": "цвет_2", "loops_dark": 20.0, "loops_amin_m2": 0.05, "loops_amax_m2": 3.0,
    "loops_r_m": 0.75,
    "etgr_top_k": 25, "etgr_min_area_m2": 0.2,
    "norm_q": 0.9, "norm_a": 0.25, "nms_d_m": 1.75, "nms_soft": 0.6,
}


def params(pix: dict) -> dict:
    p = dict(DEFAULTS)
    p.update((pix or {}).get("e1g1") or {})
    return p


# ------------------------------------------------------------------ 1. ens6 MagGr finds

def member_dirs(model_dir: Path, members=MEMBERS) -> list[Path]:
    return [Path(model_dir) / m for m in members]


class _LLeaves:
    """lleaves-compiled member (P3 budget): same finds.txt trees, bit-identical probs on 200/202/203 (P3 timing.json).
    Object cache <member>/finds.lleaves.<target>.o (+ .sha256) built offline in the official image (p3/compile_generic.py, x86-64-v2)."""

    def __init__(self, txt: Path, obj: Path, target: str):
        import lleaves
        self.m = lleaves.Model(model_file=str(txt))
        self.m.compile(cache=str(obj), target_cpu=target, target_cpu_features="")

    def predict(self, X, num_threads: int = 1):
        return self.m.predict(np.ascontiguousarray(X, dtype=np.float64), n_jobs=max(1, int(num_threads)))


_LL_CACHE: dict = {}


def _vendor_path() -> None:
    import sys
    v = Path(__file__).resolve().parents[1] / "vendor"
    if v.is_dir() and str(v) not in sys.path:
        sys.path.append(str(v))


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_sums(d: Path, obj: Path) -> None:
    """Fail closed: <obj>.sha256 = '<sha obj>  <sha finds.txt>' must match, else this member runs on LightGBM."""
    want = obj.with_name(obj.name + ".sha256").read_text().split()
    if len(want) < 2 or want[0] != _sha(obj) or want[1] != _sha(d / "finds.txt"):
        raise ValueError("sha256 mismatch of lleaves object / finds.txt")


def load_members(dirs: list[Path], backend: str = "lightgbm", target: str = "x86-64-v2") -> list:
    """backend "lleaves": member = compiled object cache next to finds.txt; any failure -> LightGBM for that member."""
    from . import pix_clean as pc
    out = []
    for d in dirs:
        if backend == "lleaves":
            obj = Path(d) / f"finds.lleaves.{target}.o"
            key = str(obj.resolve())
            if key not in _LL_CACHE and obj.exists():
                try:
                    _check_sums(Path(d), obj)
                    _vendor_path()
                    _LL_CACHE[key] = _LLeaves(Path(d) / "finds.txt", obj, target)
                except Exception as exc:   # no lleaves/llvmlite or bad object -> lightgbm
                    log.warning("pix_e1g1: lleaves %s: %s: %s — lightgbm", obj, type(exc).__name__, exc)
                    _LL_CACHE[key] = None
            if _LL_CACHE.get(key) is not None:
                out.append(_LL_CACHE[key])
                continue
        models, _ = pc._models_cached(Path(d), ("finds_points",))
        if "finds_points" not in models:
            raise FileNotFoundError(f"pix_e1g1: no finds model in {d}")
        out.append(models["finds_points"])
    return out


def mean_map(p, boosters: list, threads: int, f16: bool = True, deadline: float | None = None) -> np.ndarray:
    """Mean of member prob maps; one feature pass per tile, every member predicts on it."""
    from . import pix_clean as pc
    named = {f"m{i}": b for i, b in enumerate(boosters)}
    maps = pc.predict_maps(p, named, threads, deadline=deadline)
    acc = None
    for i in range(len(boosters)):
        a = maps.pop(f"m{i}")
        if f16:
            a = a.astype(np.float16).astype(np.float32)
        acc = a if acc is None else acc + a
    m = acc / len(boosters)
    return m.astype(np.float16).astype(np.float32) if f16 else m


def decode(prob: np.ndarray, valid: np.ndarray, step: float, sigma: float, nms_r_m: float, p_min: float,
           max_n: int = 5000) -> list[tuple]:
    """Local maxima of the sigma-smoothed map; (row, col, conf=smoothed peak). == finds_sweep.decode."""
    sm = ndi.gaussian_filter(prob, sigma, mode="nearest") if sigma > 0 else prob
    rad = max(1, int(round(nms_r_m / step)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    foot = (yy * yy + xx * xx) <= rad * rad
    mx = ndi.maximum_filter(sm, footprint=foot, mode="constant", cval=0.0)
    r, c = np.nonzero((sm >= mx) & (sm >= p_min) & valid)
    if not len(r):
        return []
    order = np.lexsort((c, r, -sm[r, c]))[: max_n * 4]
    r, c = r[order], c[order]
    pts = np.stack([r, c], 1).astype(np.float64)
    tree = cKDTree(pts)
    keep, taken = [], np.zeros(len(r), bool)
    for i in range(len(r)):
        if taken[i]:
            continue
        keep.append(i)
        taken[tree.query_ball_point(pts[i], rad)] = True
        if len(keep) >= max_n:
            break
    H, Wd = prob.shape
    out = []
    for i in keep:
        r0, c0 = int(r[i]), int(c[i])
        ra, rb, ca, cb = max(0, r0 - 1), min(H, r0 + 2), max(0, c0 - 1), min(Wd, c0 + 2)
        w = sm[ra:rb, ca:cb].astype(np.float64)
        w = np.maximum(w - w.min(), 0) + 1e-9
        gy, gx = np.mgrid[ra:rb, ca:cb]
        out.append(((w * gy).sum() / w.sum(), (w * gx).sum() / w.sum(), float(sm[r0, c0])))
    return out


def raster_finds(prob, valid, transform, crs, step, reg_name, src, point_r, ep) -> list[dict]:
    from . import pix_clean as pc
    from .raster import circle_3857, pixel_to_xy
    peaks = decode(prob, valid, step, float(ep["sigma_px"]), float(ep["nms_r_m"]), float(ep["p_min"]),
                   int(ep["max_per_raster"]))
    if not peaks:
        return []
    rr = np.array([q[0] for q in peaks]); cc = np.array([q[1] for q in peaks])
    xs, ys = pixel_to_xy(transform, rr, cc)
    return [pc._feat(circle_3857(x, y, point_r, crs), "finds_points", "maggr", reg_name, src, crs, q[2])
            for x, y, q in zip(xs, ys, peaks)]


def region_finds(reg, cfg: dict, model_dirs, guard=None, status: dict | None = None,
                 prob_cache: Path | None = None) -> list[dict]:
    """ens6 MagGr finds of one region. model_dirs: list of member dirs (6) or a callable reg->list.
    status as pix_clean.region_feats (done/fell) so pipeline.pix_merge keeps classic on failed rasters."""
    import pickle
    from . import geophys, pix_clean as pc
    from .pipeline import region_crs
    status = {} if status is None else status
    done = status.setdefault("done", {}); fell = status.setdefault("fell", {})
    geo = cfg.get("geophys") or {}; pix = geo.get("pixgbm") or {}; ep = params(pix)
    files = pc.select_files(reg, "maggr", pix)
    if not files:
        return []
    dirs = model_dirs(reg) if callable(model_dirs) else model_dirs
    boosters = load_members([Path(d) for d in dirs], str(ep.get("backend", "lightgbm")), str(ep.get("lleaves_target", "x86-64-v2")))
    threads = pc._threads(pix)
    point_r = float(geo.get("point_circle_m", 0.3))
    crs_region = region_crs(reg, cfg)
    boundary = geophys.survey_boundary(reg.boundaries, "maggr", crs_region, geo)
    noise = geophys.load_mask(reg.noise, "maggr", crs_region)
    mine: list[dict] = []
    for path in files:
        t = time.monotonic()
        try:
            left = pc.time_left(pix, guard)
            if left < 5:
                raise TimeoutError("pix_e1g1: phase budget exhausted")
            cf = (Path(prob_cache) / reg.name / (path.stem + ".npz")) if prob_cache else None
            if cf is not None and cf.exists():
                z = np.load(cf); prob = z["prob"].astype(np.float32); valid = z["valid"]
                pp = pickle.loads(cf.with_suffix(".prep.pkl").read_bytes())
                tr, crs, step = pp["transform"], pp["crs"], pp["step"]
            else:
                p = pc.prepare(path, "maggr", crs_region)
                prob = mean_map(p, boosters, threads, bool(ep["f16"]), deadline=time.monotonic() + max(1.0, left))
                valid, tr, crs, step = p.valid, p.transform, p.crs, p.step
                del p
            feats = raster_finds(prob, valid, tr, crs, step, reg.name, path.name, point_r, ep)
            b_map, n_map = geophys.sane_masks(path, boundary, noise, geo, crs_region)
            feats = geophys.apply_masks(feats, b_map, n_map)
        except Exception as exc:
            log.warning("pix_e1g1: %s: %s: %s — classic on this raster", path.name, type(exc).__name__, exc)
            fell[("finds_points", "maggr")] = True
            continue
        done.setdefault(("finds_points", "maggr"), set()).add(path.name)
        mine += feats
        log.info("pix_e1g1 %s %s: %d finds %.1fs", reg.name, path.name, len(feats), time.monotonic() - t)
    mine = pc.dedupe_cross_maps(mine, float(ep["cross_map_r_m"])) if mine else []
    for f in mine:
        f.pop("_xy", None)
    return mine


# ------------------------------------------------------------------ 3a. raster groups (Gtype_n<cut>)

def _is_finds(f):
    return f.get("markup_type") == "maggr" and f.get("class_name") == "finds_points"


def assign_groups(feats: list[dict], cut: int = 300) -> dict:
    """threshold_group of pix MagGr finds by raster type x candidate count; returns {source: group}."""
    n = collections.Counter(f["source"] for f in feats if _is_finds(f))
    grp = {s: ("cv" if "цвет" in s else "gr") + ("lo" if k < cut else "hi") for s, k in n.items()}
    for f in feats:
        if _is_finds(f):
            f["threshold_group"] = "maggr_pix_" + grp[f["source"]]
    return grp


# ------------------------------------------------------------------ 3b. isoline loops (gnz)

def loops_raster(path, dark=20.0, amax_m2=3.0, amin_m2=0.003, win_m=0.15):
    """== gnz/loops.py loops(): small closed isoline loops on a contour-colour map.

    Memory-lean but value-identical rewrite: bands are read one at a time as uint8, luma is accumulated
    in float32 in the same order, the 60th percentile is taken before the median filter, component
    areas use bincount (== ndi.sum over ones) and centroids are computed per kept label inside its
    bounding box (== ndi.center_of_mass over ones up to float64 ulp)."""
    import rasterio
    with rasterio.open(path) as r:
        tr = r.transform; crs = r.crs; res = abs(r.res[0])
        luma = None; hi = None; lo = None
        for k, wgt in ((1, 0.299), (2, 0.587), (3, 0.114)):
            b = r.read(k)
            term = wgt * b.astype(np.float32)
            luma = term if luma is None else (luma + term)
            del term
            h = b >= 253; l = b <= 2
            hi = h if hi is None else (hi & h); lo = l if lo is None else (lo & l)
            del b, h, l
    bgmask = hi | lo; del hi, lo
    w = max(3, int(round(win_m / res)) | 1)
    vals = luma[~bgmask]; p60 = np.percentile(vals, 60, overwrite_input=True); del vals
    med = ndi.median_filter(luma, size=w) if w <= 9 else ndi.uniform_filter(luma, w)
    line = (luma < med - dark * 0.5) & (luma < p60)
    del med, luma
    line = ndi.binary_closing(line, iterations=1)
    lab, n = ndi.label(~line & ~bgmask); del line
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64) * res * res
    touch = np.zeros(n + 1, bool)
    edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]); touch[np.unique(edge)] = True
    bgd = ndi.binary_dilation(bgmask, iterations=2); touch[np.unique(lab[bgd])] = True; del bgd, bgmask
    idx = np.arange(1, n + 1); keep = idx[(area >= amin_m2) & (area <= amax_m2) & ~touch[1:]]
    if not len(keep):
        return np.array([]), np.array([]), np.array([]), crs, res
    sl = ndi.find_objects(lab)
    cy = np.empty(len(keep)); cx = np.empty(len(keep))
    for t, k in enumerate(keep):
        s0, s1 = sl[k - 1]
        rr, cc = np.nonzero(lab[s0, s1] == k)
        cy[t] = rr.astype(np.float64).mean() + s0.start; cx[t] = cc.astype(np.float64).mean() + s1.start
    import rasterio.transform as rt
    xs, ys = rt.xy(tr, cy, cx)
    return np.array(xs), np.array(ys), area[keep - 1], crs, res


def _loop_conf(a: float) -> float:
    return 0.55 if a < 0.3 else (0.75 if a < 1.0 else 0.65)


def region_loops(reg, cfg: dict, ep: dict) -> list[dict]:
    import re
    from . import geophys, pix_clean as pc
    from .pipeline import region_crs
    from .raster import circle_3857
    geo = cfg.get("geophys") or {}; pr = float(geo.get("point_circle_m", 0.3))
    files = [f for f in pc.select_files(reg, "maggr", {}) if re.search(str(ep["loops_name_re"]), f.name)]
    if not files:
        return []
    crs_r = region_crs(reg, cfg)
    b = geophys.survey_boundary(reg.boundaries, "maggr", crs_r, geo); n = geophys.load_mask(reg.noise, "maggr", crs_r)
    out = []
    for path in files:
        xs, ys, ar, crs, _ = loops_raster(str(path), float(ep["loops_dark"]))
        m = (ar >= float(ep["loops_amin_m2"])) & (ar <= float(ep["loops_amax_m2"])) if len(ar) else np.zeros(0, bool)
        fs = [pc._feat(circle_3857(x, y, pr, crs.to_string()), "finds_points", "maggr", reg.name, path.name,
                       crs.to_string(), _loop_conf(a)) for x, y, a in zip(xs[m], ys[m], ar[m])]
        bm, nm = geophys.sane_masks(path, b, n, geo, crs_r)
        out += geophys.apply_masks(fs, bm, nm)
    for f in out:
        f.pop("_xy", None)
    return out


def _k3857(y: float) -> float:
    lat = math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)
    return 1 / math.cos(math.radians(lat))


def add_loops(feats: list[dict], loops: list[dict], r_m: float = 0.75) -> tuple[list[dict], int, int]:
    """== gnz/add_loops.py union: loop near a pix find (<= r_m on the ground) boosts its conf, else is added."""
    pix = [f for f in feats if _is_finds(f)]
    grp = {}
    for f in pix:
        grp.setdefault(f["source"], f.get("threshold_group"))
    byreg = collections.defaultdict(list)
    for i, f in enumerate(pix):
        byreg[f["region_name"]].append(i)
    xy = lambda f: (f["geom"].centroid.x, f["geom"].centroid.y)
    trees = {r: (cKDTree(np.array([xy(pix[i]) for i in ix])), ix) for r, ix in byreg.items()}
    add, boosted = [], 0
    for lf in loops:
        x, y = xy(lf); r = r_m * _k3857(y)
        t = trees.get(lf["region_name"])
        hit = [t[1][j] for j in t[0].query_ball_point((x, y), r)] if t else []
        if hit:
            for i in hit:
                if lf["confidence"] > pix[i]["confidence"]:
                    pix[i]["confidence"] = lf["confidence"]; boosted += 1
            continue
        g = grp.get(lf["source"]) or ("maggr_pix_cvlo" if "цвет" in lf["source"] else "maggr_pix_grlo")
        add.append(dict(lf, threshold_group=g))
    return feats + add, len(add), boosted


# ------------------------------------------------------------------ 3c. EtGr top-K / area (etfp/mk_E.py)



_TR: dict = {}


def _to_utm(geom3857):
    """As the official metric _to_utm(geom, 'auto') on the written geometry (writer rounds coords to 0.01 m):
    zone by centroid (3857 -> WGS84), whole geometry transformed to that UTM zone."""
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.ops import transform as sh_tr
    if geom3857.geom_type == "Polygon":
        geom3857 = Polygon([(round(x, 2), round(y, 2)) for x, y, *_ in geom3857.exterior.coords],
                           [[(round(x, 2), round(y, 2)) for x, y, *_ in r.coords] for r in geom3857.interiors])
    if "wgs" not in _TR:
        _TR["wgs"] = Transformer.from_crs(3857, 4326, always_xy=True)
    c = geom3857.centroid
    lon, lat = _TR["wgs"].transform(c.x, c.y)
    epsg = (32600 if lat >= 0 else 32700) + min(max(int((lon + 180) // 6) + 1, 1), 60)
    if epsg not in _TR:
        _TR[epsg] = Transformer.from_crs(3857, epsg, always_xy=True)
    return sh_tr(_TR[epsg].transform, geom3857)


def utm_area_of(geom3857) -> float:
    return _to_utm(geom3857).area


def utm_xy_of(geom3857) -> tuple[float, float]:
    return _to_utm(geom3857).centroid.coords[0]


def etgr_topk(feats: list[dict], k: int = 25, min_area_m2: float = 0.2, area_fn=utm_area_of) -> tuple[list[dict], int]:
    by = collections.defaultdict(list)
    for i, f in enumerate(feats):
        if f.get("markup_type") == "etgr":
            by[f.get("source")].append(i)
    drop = set()
    for ix in by.values():
        for r, i in enumerate(sorted(ix, key=lambda i: -float(feats[i].get("confidence", 0.0)))):
            if r >= k or area_fn(feats[i]["geom"]) + 1e-3 < min_area_m2:
                drop.add(i)
    return [f for i, f in enumerate(feats) if i not in drop], len(drop)


# ------------------------------------------------------------------ 3d. per-raster norm + soft-NMS (mkM_any)

def norm_softnms(feats: list[dict], q=0.9, a=0.25, d_m=1.75, soft=0.6, xy_fn=utm_xy_of) -> int:
    idx = [i for i, f in enumerate(feats) if _is_finds(f)]
    if not idx:
        return 0
    F = [feats[i] for i in idx]
    c = np.array([float(f["confidence"]) for f in F])
    src = np.array([f["source"] for f in F]); reg = np.array([f["region_name"] for f in F])
    if a < 1:
        for s in np.unique(src):
            m = src == s; v = c[m]
            c[m] = np.clip(0.9 * v ** a * (v / np.quantile(v, q)) ** (1 - a), 0, 1)
    if soft < 1:
        xy = np.array([xy_fn(f["geom"]) for f in F])
        o = np.argsort(-c, kind="stable"); nb = cKDTree(xy).query_ball_point(xy, d_m); newc = c.copy()
        for i in o:
            for j in nb[i]:
                if j == i or reg[j] != reg[i] or c[j] > c[i] or (c[j] == c[i] and j < i):
                    continue
                newc[j] *= soft
        c = newc
    for f, cc in zip(F, c):
        f["confidence"] = round(float(cc), 6)
    return len(idx)


# ------------------------------------------------------------------ 3. whole post chain on one region's geo feats

def post_region(feats: list[dict], reg, cfg: dict, stats: dict | None = None) -> list[dict]:
    """Steps 3a–3d on the region's geo features (already pix_merge'd, merge_within_markup'd, finalized)."""
    ep = params(((cfg.get("geophys") or {}).get("pixgbm")) or {})
    st = stats if stats is not None else {}
    assign_groups(feats, int(ep["group_cut"]))
    if ep["loops"]:
        lp = region_loops(reg, cfg, ep)
        feats, st["loops_added"], st["loops_boosted"] = add_loops(feats, lp, float(ep["loops_r_m"]))
    feats, st["etgr_dropped"] = etgr_topk(feats, int(ep["etgr_top_k"]), float(ep["etgr_min_area_m2"]))
    st["maggr_finds"] = norm_softnms(feats, float(ep["norm_q"]), float(ep["norm_a"]), float(ep["nms_d_m"]),
                                     float(ep["nms_soft"]))
    return feats
