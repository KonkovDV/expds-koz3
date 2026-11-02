#!/usr/bin/env python3
"""W4 Q0: new34 loss breakdown by scene / class / scene size (PREREG docs/bots/2026-11-02/bot-W4/q0_new34/PREREG.md)."""
import json, math, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "/workspace/koz3/wt/w1"); sys.path.insert(0, "/workspace/koz3/wt/w1/scripts")
from expds import metric as M
from shapely.geometry import shape
METRIC = "/workspace/koz3/wt/w1/materials/compute_metrics.py"
T = json.load(open("/workspace/koz3/orch/bot-W1/work/v3prod/t_taus.json"))
NEW = set(T["new"])
FB = json.load(open("/workspace/koz3/repo/docs/bots/2026-10-29/bot-10/work11/frame_boxes.json"))["boxes"]
GT = json.loads(Path("/workspace/koz3/w5data/gt_train_v2.geojson").read_bytes())
GTR = {cl: {r: v[2] for r, v in d.items()} for cl, d in T["counts_B0"].items()}
W = M.CLASS_WEIGHTS

from pyproj import Transformer
import shapely
_ZONE, _TR = {}, {}
def _zone(reg):
    if reg not in _ZONE:
        xs = sorted(shape(f["geometry"]).centroid.x for f in GT["features"] if f["properties"]["region_name"] == reg)
        lon = math.degrees(xs[len(xs)//2] / 6378137.0) if xs else 0.0
        _ZONE[reg] = 32600 + int((lon + 180) // 6) + 1
    return _ZONE[reg]
N_UTM = [0]
def to3857(g, reg):
    g = shape(g)
    if abs(g.centroid.x) < 1.0e6 and 1.0e6 < g.centroid.y < 1.0e7:   # UTM-looking coords (metric handles them; scene lookup needs 3857)
        ep = _zone(reg); tr = _TR.setdefault(ep, Transformer.from_crs(ep, 3857, always_xy=True)); N_UTM[0] += 1
        g = shapely.transform(g, lambda xy: __import__("numpy").c_[tr.transform(xy[:, 0], xy[:, 1])])
    return g
def scene_of(geom, reg):
    b = FB.get(reg, {}).get("sp_or")
    if not b: return "nosat"
    pt = to3857(geom, reg).representative_point(); best = None
    for k, (x0, y0, x1, y1) in enumerate(b):
        hw, hh = (x1-x0)/2, (y1-y0)/2
        if hw > 0 and hh > 0 and x0 <= pt.x <= x1 and y0 <= pt.y <= y1:
            d = max(abs(pt.x-x0-hw)/hw, abs(pt.y-y0-hh)/hh)
            if best is None or d < best[0]: best = (d, k)
    return f"s{best[1]}" if best else "nosat"

def mpix(reg, k):
    x0, y0, x1, y1 = FB[reg]["sp_or"][k]
    lat = math.atan(math.sinh(((y0+y1)/2)/6378137.0)); c = math.cos(lat)
    return (x1-x0)*c/0.5 * (y1-y0)*c/0.5 / 1e6, (x1-x0)*c, (y1-y0)*c

def relabel(fc):
    out = []
    for f in fc["features"]:
        p = dict(f["properties"]); r = p.get("region_name")
        if r in NEW: p["region_name"] = r + "#" + scene_of(f["geometry"], r)
        out.append({"type": "Feature", "geometry": f["geometry"], "properties": p})
    return {"type": "FeatureCollection", "features": out}

GTL = relabel(GT)
def load(p): return json.loads(Path(p).read_bytes())
def counts(path):
    fc = path if isinstance(path, dict) else load(path)
    cur = M.build_curves(relabel(fc), GTL, METRIC, "global")
    out = {}
    for cl, c in cur.items():
        per = {}
        for r, n in c.gt_regions.items():
            if r.split("#")[0] in NEW: per.setdefault(r, [0, 0, 0])[2] += int(n)
        for r, h in zip(c.pred_region, c.hit):
            if r.split("#")[0] in NEW: per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
        out[cl] = per
    return out

def sc(cnt): return M.score_counts(cnt)[0]
def swap(dst, src, units):
    return {cl: {u: (src.get(cl, {}).get(u, [0, 0, 0]) if u in units else v) for u, v in
                 {**d, **{u: [0, 0, 0] for u in units if u in src.get(cl, {})}}.items()} for cl, d in
            {cl: dict(dst.get(cl, {})) for cl in set(dst) | set(src)}.items()}

def per_class(cnt):
    return {cl: x for cl, x in M.score_counts(cnt)[1].items()}

RUNS = {"SATREC": ("/workspace/koz3/w1/satrec/out", ["R1", "C1"]), "SATREC2": ("/workspace/koz3/w1/satrec2/out", ["A1"]),
        "SATDET2": ("/workspace/koz3/w1/satdet/out", ["P1", "P2"])}
def size_bin(u):
    r, s = u.split("#")
    if s == "nosat": return "nosat"
    m = mpix(r, int(s[1:]))[0]
    return "<1" if m < 1 else "1-2" if m < 2 else "2-4" if m < 4 else "4-8" if m < 8 else ">=8"

RES = {"scenes": {}, "runs": {}}
for fam, (od, vs) in RUNS.items():
    B = counts(f"{od}/out_base.geojson")
    for v in vs:
        V = counts(f"{od}/out_{v}.geojson")
        units = sorted({u for d in (B, V) for cd in d.values() for u in cd})
        sB, sV = sc(B), sc(V); dT = sV - sB
        pcB, pcV = per_class(B), per_class(V)
        den = sum(W.get(c, 0) for c, x in pcB.items() if x["n_gt"] > 0)
        cls = {c: {"dF1": pcV[c]["f1"] - pcB[c]["f1"], "contrib": W.get(c, 0) * (pcV[c]["f1"] - pcB[c]["f1"]) / den,
                   "dTP": pcV[c]["tp"] - pcB[c]["tp"], "dFP": pcV[c]["fp"] - pcB[c]["fp"], "n_gt": pcB[c]["n_gt"],
                   "base": {k: pcB[c][k] for k in ("tp", "fp", "fn")}} for c in pcB if pcB[c]["n_gt"] > 0}
        U = {}
        for u in units:
            a = sc(swap(B, V, {u})) - sB; l = sV - sc(swap(V, B, {u}))
            dd = {}
            for c in set(B) | set(V):
                b_ = B.get(c, {}).get(u, [0, 0, 0]); v_ = V.get(c, {}).get(u, [0, 0, 0])
                if b_ != v_: dd[c] = {"dTP": v_[0]-b_[0], "dFP": v_[1]-b_[1], "n_gt": b_[2], "base_tp_fp": b_[:2]}
            if dd or abs(a) > 1e-12: U[u] = {"swap_in": a, "leave_out": l, "contrib": (a+l)/2, "bin": size_bin(u), "changes": dd}
        bins = defaultdict(set)
        for u in units: bins[size_bin(u)].add(u)
        BIN = {}
        for bn, us in bins.items():
            a = sc(swap(B, V, us)) - sB; l = sV - sc(swap(V, B, us))
            BIN[bn] = {"n_scenes": len(us), "n_changed": sum(u in U for u in us), "swap_in": a, "leave_out": l, "contrib": (a+l)/2}
        fb, fv = load(f"{od}/out_base.geojson"), load(f"{od}/out_{v}.geojson")
        issp = lambda f: f["properties"].get("markup_type") == "sp_or"
        mix = {"type": "FeatureCollection", "features": [f for f in fb["features"] if not issp(f)] + [f for f in fv["features"] if issp(f)]}
        sMix = sc(counts(mix)); del fb, fv, mix
        nonsp = {"d_new34_sp_only(nonsat_frozen)": sMix - sB, "d_new34_nonsat_side_effect": sV - sMix}
        RES["runs"][f"{fam}/{v}"] = {"decomp_sat_vs_nonsat": nonsp, "base_new34": sB, "var_new34": sV, "d_new34": dT, "class": cls, "bins": BIN,
                                    "scenes": dict(sorted(U.items(), key=lambda kv: kv[1]["contrib"]))}
        print(fam, v, nonsp, f"base {sB:.6f} var {sV:.6f} d {dT:+.6f}", {k: round(x["contrib"], 5) for k, x in BIN.items()}, flush=True)
for r in sorted(NEW):
    for k, b in enumerate(FB.get(r, {}).get("sp_or", [])):
        m, w, h = mpix(r, k); RES["scenes"][f"{r}#s{k}"] = {"mpix": m, "w_m": w, "h_m": h}
RES["n_utm_like_relocated"] = N_UTM[0]
json.dump(RES, open(sys.argv[1], "w"), ensure_ascii=False, indent=1, default=float)
