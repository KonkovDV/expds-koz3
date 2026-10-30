#!/usr/bin/env python3
"""exec A: MagGr finds decode sweep (DOC-1 clean: only expds.pix_clean / own decode, no EN code).

dump   : fold model → finds prob maps per maggr raster → probs/<reg>/<raster>.npz (+light prep pkl)
decode : variants (sigma, nms, p_min, calib) → cache/pix/<tag>/<reg>.pkl = base tag (d10) with maggr finds replaced
"""
from __future__ import annotations
import argparse, pickle, sys, time, types, itertools, json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

WP = Path("/workspace/koz3/wt/w4-pix"); sys.path.insert(0, str(WP)); sys.path.insert(0, str(WP / "scripts"))
from geo_predict import apply_override  # noqa
from expds import pix_clean as pc, geophys  # noqa
from expds.catalog import scan_root  # noqa
from expds.pipeline import load_config, region_crs  # noqa

FOLDS = {"F1": ["200"], "F2": [str(i) for i in range(201, 208)], "F3": ["208", "209", "210", "215"],
         "F4": ["211", "212", "216"], "F5": ["213", "214", "217"]}
W = Path("/workspace/koz3/w4geo/work"); T = Path("/workspace/koz3/w4geo/train")


def cfg_():
    cfg = load_config(str(WP / "configs/v3_on.toml"))
    for s in ['geophys.pixgbm.impl="clean"', 'geophys.pixgbm.maps=["maggr","etgr","terfl"]']:
        apply_override(cfg, s)
    return cfg


def fold_of(name):
    return next(k for k, v in FOLDS.items() if name.split("_")[0] in v)


def cmd_dump(a):
    cfg = cfg_(); geo = cfg["geophys"]; pix = geo["pixgbm"]
    for reg in scan_root(T):
        files = pc.select_files(reg, "maggr", pix)
        if not files:
            continue
        f = fold_of(reg.name)
        models, _ = pc.load_models(W / a.models / f"{f}_s0", ("finds_points",))
        crs_region = region_crs(reg, cfg)
        d = W / a.probs / reg.name; d.mkdir(parents=True, exist_ok=True)
        for path in files:
            o = d / (path.stem + ".npz")
            if o.exists():
                continue
            t = time.monotonic()
            p = pc.prepare(path, "maggr", crs_region)
            prob = pc.predict_maps(p, models, a.threads)["finds_points"]
            np.savez_compressed(o.with_suffix(".tmp.npz"), prob=prob.astype(np.float16), valid=p.valid)
            (d / (path.stem + ".prep.pkl")).write_bytes(pickle.dumps(
                {"transform": p.transform, "crs": p.crs, "step": p.step, "path": str(path)}))
            o.with_suffix(".tmp.npz").replace(o)
            print(f"dump {reg.name} {path.name} {prob.shape} {time.monotonic()-t:.0f}s", flush=True)


def decode(prob, valid, step, sigma, nms_r_m, p_min, max_n=5000):
    sm = ndi.gaussian_filter(prob, sigma, mode="nearest") if sigma > 0 else prob
    rad = max(1, int(round(nms_r_m / step)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    foot = (yy * yy + xx * xx) <= rad * rad
    mx = ndi.maximum_filter(sm, footprint=foot, mode="constant", cval=0.0)
    r, c = np.nonzero((sm >= mx) & (sm >= p_min) & valid)
    if not len(r):
        return sm, []
    order = np.lexsort((c, r, -sm[r, c]))[: max_n * 4]
    r, c = r[order], c[order]
    pts = np.stack([r, c], 1).astype(np.float64); tree = cKDTree(pts)
    keep, taken = [], np.zeros(len(r), bool)
    for i in range(len(r)):
        if taken[i]:
            continue
        keep.append(i)
        taken[tree.query_ball_point(pts[i], rad)] = True
        if len(keep) >= max_n:
            break
    H, Wd = prob.shape; out = []
    for i in keep:
        r0, c0 = int(r[i]), int(c[i])
        ra, rb, ca, cb = max(0, r0 - 1), min(H, r0 + 2), max(0, c0 - 1), min(Wd, c0 + 2)
        w = sm[ra:rb, ca:cb].astype(np.float64); w = np.maximum(w - w.min(), 0) + 1e-9
        gy, gx = np.mgrid[ra:rb, ca:cb]
        out.append(((w * gy).sum() / w.sum(), (w * gx).sum() / w.sum(), float(sm[r0, c0])))
    return sm, out


def variants(grid):
    if grid == "small":
        for n in (0.75, 1.0):
            yield dict(sigma=1.0, nms=n, pmin=0.01)
    if grid == "main":
        for s, n in itertools.product((1.0, 2.0, 3.0), (0.5, 0.75, 1.0, 1.5, 2.0)):
            yield dict(sigma=s, nms=n, pmin=0.02)
        yield dict(sigma=1.0, nms=1.0, pmin=0.005)
        yield dict(sigma=2.0, nms=1.0, pmin=0.005)


def tag_of(v, cal, xr):
    return f"fs{v['sigma']:g}_n{v['nms']:g}_p{v['pmin']:g}_{cal}_x{xr:g}"


def cmd_decode(a):
    from expds.raster import circle_3857, pixel_to_xy
    cfg = cfg_(); geo = cfg["geophys"]; pix = geo["pixgbm"]
    point_r = float(geo.get("point_circle_m", 0.3))
    cals = a.calib; xrs = a.xr
    vs = list(variants(a.grid))
    tags_done = []
    for reg in scan_root(T):
        files = pc.select_files(reg, "maggr", pix)
        base = pickle.loads((W / "cache/pix" / a.base / f"{reg.name}.pkl").read_bytes())
        if not files:
            per = {}
        else:
            crs_region = region_crs(reg, cfg)
            boundary = geophys.survey_boundary(reg.boundaries, "maggr", crs_region, geo)
            noise = geophys.load_mask(reg.noise, "maggr", crs_region)
            per = {}  # tag -> feats
            t = time.monotonic()
            for path in files:
                d = W / a.probs / reg.name
                z = np.load(d / (path.stem + ".npz")); prob = z["prob"].astype(np.float32); valid = z["valid"]
                pp = pickle.loads((d / (path.stem + ".prep.pkl")).read_bytes())
                b_map, n_map = geophys.sane_masks(path, boundary, noise, geo, crs_region)
                for v in vs:
                    sm, peaks = decode(prob, valid, pp["step"], v["sigma"], v["nms"], v["pmin"])
                    if not peaks:
                        continue
                    vv = sm[valid]
                    stats = {"q999": float(np.quantile(vv, 0.999)) if vv.size else 1.0,
                             "max": float(vv.max()) if vv.size else 1.0,
                             "q99": float(np.quantile(vv, 0.99)) if vv.size else 1.0}
                    rr = np.array([q[0] for q in peaks]); cc = np.array([q[1] for q in peaks])
                    xs, ys = pixel_to_xy(pp["transform"], rr, cc)
                    raw = np.array([q[2] for q in peaks])
                    for cal in cals:
                        if cal == "raw":
                            conf = raw
                        elif cal in ("q999", "q99", "max"):
                            conf = np.clip(raw / max(stats[cal], 1e-6), 0, 1) * 0.999
                        elif cal == "rank":  # within-raster rank of peak → (0,1]
                            o = np.argsort(np.argsort(-raw)); conf = 1.0 / (1.0 + o / 10.0)
                        feats = [pc._feat(circle_3857(x, y, point_r, pp["crs"]), "finds_points", "maggr", reg.name,
                                          path.name, pp["crs"], cf) for x, y, cf in zip(xs, ys, conf)]
                        feats = geophys.apply_masks(feats, b_map, n_map)
                        for xr in xrs:
                            per.setdefault(a.prefix + tag_of(v, cal, xr), []).extend(feats)
            print(f"decode {reg.name}: {len(files)} rasters {time.monotonic()-t:.0f}s", flush=True)
        old = [x for x in base["new"] if not (x["class_name"] == "finds_points" and x["markup_type"] == "maggr")]
        for v in vs:
            for cal in cals:
                for xr in xrs:
                    tg = a.prefix + tag_of(v, cal, xr)
                    new = pc.dedupe_cross_maps(per.get(tg, []), xr) if per.get(tg) else []
                    for f in new:
                        f.pop("_xy", None)
                    od = W / "cache/pix" / tg; od.mkdir(parents=True, exist_ok=True)
                    (od / f"{reg.name}.pkl").write_bytes(pickle.dumps({"new": old + new, "pst": base["pst"],
                                                                        "model_dir": base.get("model_dir")}))
                    if tg not in tags_done:
                        tags_done.append(tg)
    (W / "sweep" / f"tags_{a.prefix}{a.grid}.txt").write_text("\n".join(tags_done))
    print("tags", len(tags_done))


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", choices=["dump", "decode"])
  ap.add_argument("--threads", type=int, default=3)
  ap.add_argument("--base", default="d10")
  ap.add_argument("--grid", default="main")
  ap.add_argument("--calib", nargs="*", default=["raw", "q999", "max"])
  ap.add_argument("--models", default="models")
  ap.add_argument("--probs", default="probs")
  ap.add_argument("--prefix", default="")
  ap.add_argument("--xr", nargs="*", type=float, default=[1.0])
  a = ap.parse_args()
  {"dump": cmd_dump, "decode": cmd_decode}[a.cmd](a)
