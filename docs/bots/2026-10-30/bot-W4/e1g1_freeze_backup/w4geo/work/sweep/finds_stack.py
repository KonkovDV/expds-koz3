#!/usr/bin/env python3
"""exec A: MagGr finds stacker (DOC-1 clean). Candidates = pix_clean OOF peaks ∪ classic finds;
features from own prob maps + classic conf; LightGBM by geo5 fold (train on other folds' OOF cands);
greedy NMS by score → cache/pix/<tag>/<reg>.pkl (d10 base, maggr finds replaced)."""
from __future__ import annotations
import argparse, pickle, sys, json, math
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
import finds_sweep as FS  # noqa  (defines paths, cfg_, decode)
from finds_sweep import W, T, FOLDS, pc, geophys, scan_root, region_crs, fold_of

def gxy(geom):
    c = geom.centroid; k = math.cos(math.atan(math.sinh(c.y / 6378137.0))); return c.x * k, c.y * k

def build(a):
    from expds.raster import circle_3857, pixel_to_xy
    from pyproj import Transformer
    cfg = FS.cfg_(); geo = cfg["geophys"]; pix = geo["pixgbm"]; pr_ = float(geo.get("point_circle_m", 0.3))
    gt = json.loads(Path("/workspace/koz3/w5data/gt_train_v2.geojson").read_text())
    from shapely.geometry import shape
    G = {}
    for f in gt["features"]:
        p = f["properties"]
        if p.get("class_name") == "finds_points":
            G.setdefault(p["region_name"], []).append(gxy(shape(f["geometry"])))
    rows = []
    for reg in scan_root(T):
        files = pc.select_files(reg, "maggr", pix)
        if not files:
            continue
        cl = [f for f in pickle.loads((W / "cache/classic" / f"{reg.name}.pkl").read_bytes())
              if f["class_name"] == "finds_points" and f["markup_type"] == "maggr"]
        crs_region = region_crs(reg, cfg)
        boundary = geophys.survey_boundary(reg.boundaries, "maggr", crs_region, geo)
        noise = geophys.load_mask(reg.noise, "maggr", crs_region)
        cands = []; rstats = []
        cl_samp = np.zeros((len(cl), 4)) - 1  # max over rasters: p, pct, p/q999, p/max
        for ri, path in enumerate(files):
            d = W / "probs" / reg.name
            z = np.load(d / (path.stem + ".npz")); prob = z["prob"].astype(np.float32); valid = z["valid"]
            pp = pickle.loads((d / (path.stem + ".prep.pkl")).read_bytes())
            sm, peaks = FS.decode(prob, valid, pp["step"], 1.0, a.nms, 0.01)
            vv = sm[valid]; rs = np.random.default_rng(0)
            samp = np.sort(rs.choice(vv, min(200000, vv.size), replace=False)) if vv.size else np.array([0.0])
            q99, q999, mx, mean = np.quantile(vv, .99), np.quantile(vv, .999), vv.max(), vv.mean()
            area = valid.sum() * pp["step"] ** 2
            st = [q99, q999, mx, mean, (vv > .5).mean(), (vv > .2).mean(), area]
            b_map, n_map = geophys.sane_masks(path, boundary, noise, geo, crs_region)
            if peaks:
                rr = np.array([q[0] for q in peaks]); cc = np.array([q[1] for q in peaks]); pv = np.array([q[2] for q in peaks])
                xs, ys = pixel_to_xy(pp["transform"], rr, cc)
                fe = [pc._feat(circle_3857(x, y, pr_, pp["crs"]), "finds_points", "maggr", reg.name, path.name, pp["crs"], v)
                      for x, y, v in zip(xs, ys, pv)]
                for f, v in zip(fe, pv):
                    f["_p"] = float(v)
                fe = geophys.apply_masks(fe, b_map, n_map)
                dens = sum(f["_p"] > .3 for f in fe) / max(area, 1) * 100
                for f in fe:
                    v = f["_p"]
                    cands.append(dict(f=f, src=0, ri=ri, color=int("цвет" in path.name), p=v,
                                      pct=np.searchsorted(samp, v) / len(samp), pq=v / max(q999, 1e-6), pm=v / max(mx, 1e-6),
                                      st=st + [dens]))
            # sample prob at classic points
            if cl:
                tr = Transformer.from_crs("EPSG:3857", pp["crs"], always_xy=True)
                inv = ~pp["transform"]
                for i, f in enumerate(cl):
                    c = f["geom"].centroid; x, y = tr.transform(c.x, c.y); col, row = inv * (x, y)
                    r0, c0 = int(round(row)), int(round(col))
                    if 0 <= r0 < sm.shape[0] and 0 <= c0 < sm.shape[1] and valid[r0, c0]:
                        ra, rb, ca, cb = max(0, r0 - 5), r0 + 6, max(0, c0 - 5), c0 + 6
                        v = float(sm[ra:rb, ca:cb].max())
                        if v > cl_samp[i, 0]:
                            cl_samp[i] = [v, np.searchsorted(samp, v) / len(samp), v / max(q999, 1e-6), v / max(mx, 1e-6)]
            rstats.append(st)
        rmean = list(np.mean(rstats, 0)) + [0.0]
        for i, f in enumerate(cl):
            f = dict(f)
            cands.append(dict(f=f, src=1, ri=-1, color=-1, p=cl_samp[i, 0], pct=cl_samp[i, 1], pq=cl_samp[i, 2],
                              pm=cl_samp[i, 3], st=rmean, cconf=float(f["confidence"])))
        if not cands:
            continue
        P = np.array([gxy(c["f"]["geom"]) for c in cands]); tree = cKDTree(P)
        isc = np.array([c["src"] == 1 for c in cands]); pvals = np.array([c["p"] for c in cands])
        Pc = P[isc]; cc_conf = np.array([c.get("cconf", 0) for c in cands])[isc]
        tc = cKDTree(Pc) if len(Pc) else None
        g = np.array(G.get(reg.name, [])).reshape(-1, 2); tg = cKDTree(g) if len(g) else None
        for k, c in enumerate(cands):
            nb2 = tree.query_ball_point(P[k], 2.0); nb5 = tree.query_ball_point(P[k], 5.0)
            pix2 = [j for j in nb2 if not isc[j] and j != k]; pix5 = [j for j in nb5 if not isc[j] and j != k]
            if tc is not None:
                dd, jj = tc.query(P[k], k=2)
                dd = np.atleast_1d(dd); jj = np.atleast_1d(jj)
                if c["src"] == 1:  # skip self
                    dd, jj = dd[1:], jj[1:]
                dcl = float(dd[0]) if len(dd) and np.isfinite(dd[0]) else 99.0
                ccn = float(cc_conf[jj[0]]) if len(jj) and np.isfinite(dd[0]) and dcl <= 2.0 else -1.0
            else:
                dcl, ccn = 99.0, -1.0
            x = [c["src"], c["color"], c["p"], c["pct"], c["pq"], c["pm"], *c["st"], len(pix2), len(pix5),
                 max([pvals[j] for j in pix2], default=0), min(dcl, 20), ccn, c.get("cconf", -1)]
            y = int(tg is not None and len(tg.query_ball_point(P[k], 2.0)) > 0)
            rows.append(dict(reg=reg.name, fold=fold_of(reg.name), x=x, y=y, f=c["f"], xy=P[k]))
        print(f"build {reg.name}: {len(cands)} cands, pos {sum(r['y'] for r in rows if r['reg']==reg.name)}, GT {len(g)}", flush=True)
    (W / "sweep" / f"stack_n{a.nms:g}.pkl").write_bytes(pickle.dumps(rows))

def fit(a):
    import lightgbm as lgb
    rows = pickle.loads((W / "sweep" / f"stack_n{a.nms:g}.pkl").read_bytes())
    X = np.array([r["x"] for r in rows], float); y = np.array([r["y"] for r in rows]); fo = np.array([r["fold"] for r in rows])
    S = np.zeros(len(rows))
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3, seed=0)
    for k in sorted(set(fo)):
        m = fo == k
        b = lgb.train(prm, lgb.Dataset(X[~m], y[~m]), num_boost_round=a.rounds)
        S[m] = b.predict(X[m])
    if a.feat_mode == "pixonly":
        pass
    from scipy.stats import rankdata
    rk = rankdata(S); npos = y.sum(); auc = (rk[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))
    print("OOF AUC", round(float(auc), 4), "pos", int(npos), "n", len(y))
    for k in sorted(set(fo)):
        m = fo == k; yy, ss = y[m], S[m]; r2 = rankdata(ss); p2 = yy.sum()
        if 0 < p2 < len(yy): print(" ", k, round(float((r2[yy == 1].sum() - p2 * (p2 + 1) / 2) / (p2 * (len(yy) - p2))), 4))
    base_dir = W / "cache/pix" / "d10"
    regs = sorted({r["reg"] for r in rows})
    for nmsr in a.final_nms:
        tag = f"stk_n{a.nms:g}_f{nmsr:g}"
        od = W / "cache/pix" / tag; od.mkdir(parents=True, exist_ok=True)
        for reg in [p.stem for p in base_dir.glob("*.pkl")]:
            base = pickle.loads((base_dir / f"{reg}.pkl").read_bytes())
            old = [x for x in base["new"] if not (x["class_name"] == "finds_points" and x["markup_type"] == "maggr")]
            idx = [i for i, r in enumerate(rows) if r["reg"] == reg]
            new = []
            if idx:
                idx = sorted(idx, key=lambda i: -S[i]); P = np.array([rows[i]["xy"] for i in idx]); tr = cKDTree(P)
                dead = np.zeros(len(idx), bool)
                for j, i in enumerate(idx):
                    if dead[j]:
                        continue
                    f = dict(rows[i]["f"]); f.pop("_xy", None); f.pop("_p", None)
                    f["confidence"] = float(S[i]); f["threshold_group"] = "maggr_stk"; new.append(f)
                    for q in tr.query_ball_point(P[j], nmsr):
                        dead[q] = True
            (od / f"{reg}.pkl").write_bytes(pickle.dumps({"new": old + new, "pst": base["pst"], "model_dir": "stack"}))
        print("tag", tag, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["build", "fit"])
ap.add_argument("--nms", type=float, default=0.75)
ap.add_argument("--rounds", type=int, default=300)
ap.add_argument("--final-nms", nargs="*", type=float, default=[1.0, 1.5, 2.0])
ap.add_argument("--feat-mode", default="all")
a = ap.parse_args()
{"build": build, "fit": fit}[a.cmd](a)
