#!/usr/bin/env python3
"""W3 Q3/Q4/Q5 gates vs V3P (prereg/Q3_GORSWAP, Q4_LISPTWIN, Q5_SATHEDGE). Clean-room. Usage: q345.py BOOT Q3|Q4|Q5"""
import importlib.util, json, sys, gc, pickle
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D, M, log, site_family = G.D, G.M, G.log, G.site_family
BOOT, Q = int(sys.argv[1]), sys.argv[2]
OUT = Path(f"/workspace/koz3/orch/bot-W3/runs/gate_{Q}.json")
PO, TS, GOR = "kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha"; KC = (PO, TS); LIG = {"li", "li_dev", "li_mosaic"}
PMIN = 1 - 0.1 / 3
res = {"q": Q, "boot": BOOT, "sanity": {}}
def save(): OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
TP = Path("/workspace/koz3/w5data/work/table.pkl")
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(TP, stk)
cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10); del cA; gc.collect()
kis = {site_family(r) for r in regs if "Кисловод" in r}
slices = {"all": None, "new34": stk.new, "nolidar": stk.optical, "noKislovodsk": {r for r in regs if site_family(r) not in kis}}
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
dfx, dreg = G.dfine_feats(); SP = G.SP
drop = lambda g, r: g == "etgr" or (g in SP and r in dreg)
b3 = G.drop_engine(tab, drop)
cP, _ = b3.curves(c19 + dfx)
nB, _, tB = D.honest(cP, folds)
sB = M.score_counts(nB)[0]; res["sanity"]["V3P"] = sB; log(f"V3P honest {sB:.6f}"); save()
CC = {"Q3": (GOR,), "Q4": KC, "Q5": KC}[Q]
ex = b3._extra(c19 + dfx)
pr = M._collect(b3.mod, b3.calc, D.S.fc(c19 + dfx), False); M._UTM_GEOM_CACHE.clear()
R = {}
for c in CC:
    b, e = b3.base[c], ex.get(c); nb = len(b["conf"])
    conf = np.concatenate([b["conf"], e["conf"]]); reg = np.concatenate([b["reg"], e["reg"]]); grp = np.concatenate([b["grp"], e["grp"]])
    pairs = dict(b["pairs"]); pairs.update({nb + i: v for i, v in e["pairs"].items()})
    xy = np.full((len(conf), 2), np.nan); ar = np.full(len(conf), np.nan)
    lst = pr.get(c, []); assert len(lst) == len(e["conf"]) and np.allclose([x[1] for x in lst], e["conf"])
    for i, x in enumerate(lst): ct = x[0].centroid; xy[nb + i] = (ct.x, ct.y); ar[nb + i] = x[0].area
    R[c] = dict(conf=conf.astype(np.float64), reg=reg.astype(str), grp=grp.astype(str), pairs=pairs, nb=nb, xy=xy, ar=ar)
del pr, ex; gc.collect()
with open(TP, "rb") as fh: T = pickle.load(fh)
for k, c in enumerate(T["CL"]):
    if c not in CC: continue
    idx = np.flatnonzero(T["cls"] == k)
    keep = np.array([not drop(g, r) for g, r in zip(tab.base[c]["grp"], tab.base[c]["reg"])], bool)
    gi = idx[keep]; assert len(gi) == R[c]["nb"]
    R[c]["xy"][:R[c]["nb"]] = np.c_[T["cx"][gi], T["cy"][gi]]; R[c]["gi"] = gi
del T, tab; gc.collect()
def mk(c, cf, rg, gp, prs):
    order = np.argsort(-cf, kind="stable"); rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    hit = np.zeros(len(order), bool); used = set()
    for i in sorted(prs, key=lambda i: rank[i]):
        best, bj = float("-inf"), -1
        for j, s_ in sorted(prs[i]):
            if j not in used and s_ > best + 1e-9: best, bj = s_, j
        if bj >= 0: used.add(bj); hit[rank[i]] = True
    return M.ClassCurve(conf=cf[order], hit=hit, pred_region=rg[order].tolist(), n_gt=cP[c].n_gt, gt_regions=cP[c].gt_regions, pred_group=gp[order].tolist())
def build(c, conf=None, keep=None, add=None):
    A = R[c]; cf = A["conf"] if conf is None else conf; rg, gp, prs = A["reg"], A["grp"], A["pairs"]
    if keep is not None:
        ix = np.flatnonzero(keep); rm = {int(o): n for n, o in enumerate(ix)}
        cf, rg, gp = cf[ix], rg[ix], gp[ix]; prs = {rm[i]: v for i, v in prs.items() if i in rm}
    if add is not None and len(add["conf"]):
        n = len(cf); cf = np.concatenate([cf, add["conf"]]); rg = np.concatenate([rg, add["reg"]]); gp = np.concatenate([gp, add["grp"]])
        prs = dict(prs); prs.update({n + i: v for i, v in add["pairs"].items()})
    return mk(c, cf, rg, gp, prs)
for c in CC:
    cv = build(c); assert np.array_equal(cv.conf, cP[c].conf) and np.array_equal(cv.hit, cP[c].hit) and cv.pred_group == cP[c].pred_group
res["sanity"]["rebuild_equal"] = True; log("rebuild == V3P"); save()
ocrs = {f["properties"]["region_name"]: f["properties"]["original_crs"] for f in dfx}
def near(c, o, rad):   # for each row of o: list of rows of c within rad in the same region (sp_dfine rows only)
    out = defaultdict(list); A, B = R[c], R[o]
    for r in set(B["reg"][B["grp"] == "sp_dfine"]):
        ia = np.flatnonzero((A["reg"] == r) & (A["grp"] == "sp_dfine")); ib = np.flatnonzero((B["reg"] == r) & (B["grp"] == "sp_dfine"))
        if not len(ia): continue
        tr = cKDTree(A["xy"][ia])
        for p, js in zip(ib, tr.query_ball_point(B["xy"][ib], rad)): out[int(p)] = [int(ia[j]) for j in js]
    return out
# ---------------- variants ----------------
GRID = []
if Q == "Q5":
    NB = {o: near(KC[1 - KC.index(o)], o, 4.0) for o in KC}
    res["sanity"]["q5_rows_with_partner"] = {o[:8]: sum(1 for v in NB[o].values() if v) for o in KC}
    def variant(h):
        out = {}
        for o in KC:
            c = KC[1 - KC.index(o)]; cf = R[o]["conf"].copy(); n = 0
            for p, js in NB[o].items():
                if js and cf[p] >= h:
                    v = 0.99 * float(R[c]["conf"][js].max())
                    if v > cf[p]: cf[p] = v; n += 1
            out[o] = build(o, conf=cf)
        return out
    GRID = [0.2, 0.3, 0.4, 0.5]
elif Q == "Q4":
    want = {}
    for c in KC:
        A = R[c]; o = KC[1 - KC.index(c)]
        li = np.flatnonzero(np.isin(A["grp"], list(LIG)) & np.isin(A["reg"], list(dreg)) & (np.arange(len(A["conf"])) < A["nb"]))
        keep = []
        for r in set(A["reg"][li]):
            ir = li[A["reg"][li] == r]
            S = np.concatenate([R[k]["xy"][(R[k]["reg"] == r) & (R[k]["grp"] == "sp_dfine")] for k in KC])
            if len(S): d, _ = cKDTree(S).query(A["xy"][ir]); ir = ir[d > 8.0]
            keep += ir.tolist()
        for p in keep: want[int(A["gi"][p])] = (c, p)
    log(f"Q4 twins wanted {len(want)} {Counter(v[0][:8] for v in want.values())}")
    feats, src = [], []
    for i, (f, _) in enumerate(D.iter_features(Path("/workspace/koz3/w5data/stack_arch.geojson"))):
        if i in want:
            c, p = want[i]; q = dict(f["properties"]); assert q["class_name"] == c and q["region_name"] == R[c]["reg"][p]
            q.update(markup_type="sp_or", threshold_group="li_sp_twin", confidence=float(R[c]["conf"][p]))
            feats.append({"type": "Feature", "geometry": f["geometry"], "properties": q}); src.append((c, p))
    assert len(feats) == len(want)
    TW = b3._extra(feats); res["sanity"]["q4_twins"] = {c[:8]: int(len(TW[c]["conf"])) if c in TW else 0 for c in KC}
    for c in KC:
        if c in TW: assert set(TW[c]["grp"]) <= {"li_sp_twin"}, set(TW[c]["grp"])
    def variant(a):
        out = {}
        for c in KC:
            if c not in TW: out[c] = build(c); continue
            t = TW[c]; out[c] = build(c, add=dict(conf=a * t["conf"], reg=t["reg"], grp=t["grp"], pairs=t["pairs"]))
        return out
    GRID = [0.1, 0.3, 0.5, 1.0]
elif Q == "Q3":
    from pyproj import Transformer
    from shapely.geometry import Point, mapping
    from shapely.ops import transform as stf
    GP = Path("/workspace/koz3/repo/docs/bots/2026-10-29/bot-5/analysis/gor_patch")
    meta = json.loads((GP / "meta.json").read_text()); sc = json.loads((GP / "scores_cand7.json").read_text())
    pp = {}
    for x in sc: pp[int(x["patch"])] = float(x["p"])
    files = defaultdict(set); byreg = defaultdict(list)
    for i, m in enumerate(meta): files[m["reg"]].add(m["file"]); byreg[m["reg"]].append(i)
    from shapely.geometry import shape as shp
    dpt = defaultdict(list)
    for f in dfx: c_ = shp(f["geometry"]).centroid; dpt[f["properties"]["region_name"]].append((c_.x, c_.y))
    A = R[GOR]; sp = A["grp"] == "sp_dfine"
    gf = [f for f in dfx if f["properties"]["class_name"] == GOR]; isp = np.flatnonzero(sp); assert len(gf) == len(isp) and np.allclose([f["properties"]["confidence"] for f in gf], A["conf"][isp])
    for i_, f in zip(isp, gf):   # positions in the region's original UTM (same frame as bot-5 meta)
        c_ = shp(f["geometry"]).centroid; A["xy"][i_] = Transformer.from_crs("EPSG:3857", ocrs[f["properties"]["region_name"]], always_xy=True).transform(c_.x, c_.y)
    rr = float(np.sqrt(np.nanmedian(A["ar"][sp]) / np.pi))
    SC = {}; insd = []
    for r, ix in byreg.items():
        if r not in dreg or len(files[r]) != 1: continue
        ps = [pp[i] for i in ix if i in pp]
        if not ps: continue
        q_ = np.array(dpt.get(r, [(np.nan, np.nan)])); tx, ty = Transformer.from_crs("EPSG:3857", ocrs[r], always_xy=True).transform(q_[:, 0], q_[:, 1])
        cr = np.r_[np.array([c_ for i in ix for c_ in meta[i]["corners"]]), np.c_[tx, ty][np.isfinite(tx)]]; u = np.array([(meta[i]["ux"], meta[i]["uy"]) for i in ix])
        w, h = np.ptp(cr[:, 0]), np.ptp(cr[:, 1]); gsd = float(np.median([meta[i]["gsd"] for i in ix]))
        cen = ((cr[:, 0].min() + cr[:, 0].max()) / 2, (cr[:, 1].min() + cr[:, 1].max()) / 2)
        ir = np.flatnonzero((A["reg"] == r) & sp)
        if len(ir): insd.append(float(np.mean((A["xy"][ir, 0] >= cr[:, 0].min()) & (A["xy"][ir, 0] <= cr[:, 0].max()) & (A["xy"][ir, 1] >= cr[:, 1].min()) & (A["xy"][ir, 1] <= cr[:, 1].max()))))
        SC[r] = dict(p=max(ps), mpx=w * h / gsd ** 2 / 1e6, half=min(w, h) / 2, cen=cen, rows=ir, d=np.hypot(*(A["xy"][ir] - np.array(cen)).T) if len(ir) else np.zeros(0))
    res["sanity"]["q3"] = {"scenes": len(SC), "mpx_lt": {a: sum(v["mpx"] < a for v in SC.values()) for a in (2, 4, 8)}, "r_med": rr,
                           "sp_rows_inside_bbox_mean": float(np.mean(insd)) if insd else None, "p_ge": {g: sum(v["p"] >= g for v in SC.values()) for g in (0.05, 0.1, 0.2, 0.3)}}
    log(f"Q3 sanity {res['sanity']['q3']}"); save()
    assert insd and np.mean(insd) > 0.99, "UTM of meta != UTM of rows"
    cf_ = []
    for r, v in SC.items():
        to = Transformer.from_crs(ocrs[r], "EPSG:3857", always_xy=True).transform
        g = stf(to, Point(*v["cen"]).buffer(rr, 32))
        cf_.append({"type": "Feature", "geometry": mapping(g), "properties": {"class_name": GOR, "confidence": 0.5, "region_name": r, "crs": "EPSG:3857",
                    "markup_type": "sp_or", "original_crs": ocrs[r], "threshold_group": "gor_swap"}})
    CE = b3._extra(cf_)[GOR]; assert len(CE["conf"]) == len(SC) and list(CE["reg"]) == list(SC), "centre feats order"
    res["sanity"]["q3_centre_hits"] = int(sum(1 for v in CE["pairs"].values() if v))
    def variant(prm):
        a, rho, smin, g = prm; keep = np.ones(len(A["conf"]), bool); add = dict(conf=[], reg=[], grp=[], pairs={})
        for k, (r, v) in enumerate(SC.items()):
            if v["mpx"] >= a: continue
            keep[v["rows"]] = False
            if v["p"] < g: continue
            ok = (A["conf"][v["rows"]] >= smin) & (v["d"] <= rho * v["half"])
            n = len(add["conf"]); add["conf"].append(v["p"]); add["reg"].append(r); add["grp"].append("gor_swap")
            if ok.any(): j = int(v["rows"][ok][np.argmax(A["conf"][v["rows"]][ok])]); prs = A["pairs"].get(j)
            else: prs = CE["pairs"].get(k)
            if prs: add["pairs"][n] = prs
        add = {k_: (np.array(v_) if k_ != "pairs" else v_) for k_, v_ in add.items()}
        if len(add["conf"]): add["conf"] = add["conf"].astype(np.float64); add["reg"] = add["reg"].astype(str); add["grp"] = add["grp"].astype(str)
        return {GOR: build(GOR, keep=keep, add=add)}
    GRID = [(a, rho, s, g) for a in (2, 4, 8) for rho in (0.2, 0.35) for s in (0.02, 0.1) for g in (0.05, 0.1, 0.2, 0.3)]
save()
# ---------------- nested selection ----------------
def cnt(curves, taus, regset):
    out = {}
    for c, cv in curves.items():
        per = {r: [0, 0, int(k)] for r, k in cv.gt_regions.items() if r in regset}
        for x, h, r, g in zip(cv.conf, cv.hit, cv.pred_region, cv.pred_group):
            if r in regset and x >= taus.get(g, {}).get(c, np.inf):
                l = per.setdefault(r, [0, 0, 0]); l[0 if h else 1] += 1
        out[c] = per
    return out
every = set().union(*folds)
t0 = cnt(cP, tB[0], folds[0])
for c in cP: assert all(t0[c].get(r, [0, 0, 0]) == nB[c].get(r, [0, 0, 0]) for r in set(t0[c]) | {r for r in nB[c] if r in folds[0]}), c
VC = {}
for prm in GRID: VC[str(prm)] = variant(prm); gc.collect()
log(f"{Q}: {len(VC)} variants built")
nV = {c: {r: list(v) for r, v in by.items()} for c, by in nB.items()}; chosen, trs = [], []
for k, held in enumerate(folds):
    train = every - held; base = cnt(cP, tB[k], train); best, bs, bh = None, M.score_counts(base)[0], None; row = {"off": bs}
    for key, CV in VC.items():
        _, _, tk = D.honest(CV, [held, train]); tc = dict(base); tc.update(cnt(CV, tk[0], train)); s = M.score_counts(tc)[0]; row[key] = s
        if s > bs + 1e-12: best, bs, bh = key, s, cnt(CV, tk[0], held)
    chosen.append(best); trs.append(row); log(f"fold {k} pick={best} train off {row['off']:.5f} best {bs:.5f}")
    if bh is not None:
        for c in bh:
            for r in list(nV[c]):
                if r in held: del nV[c][r]
            nV[c].update(bh[c])
res["pick_per_fold"] = chosen; res["train_scores"] = trs; save()
out = {}
for sl, s in slices.items():
    a, b = M.score_counts(nB, s)[0], M.score_counts(nV, s)[0]; bs = M.paired_bootstrap_counts(nV, nB, site_family, BOOT, 0, s)
    out[sl] = {"B": a, "V": b, "delta": b - a, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
pa, pb = M.score_counts(nB)[1], M.score_counts(nV)[1]
out["cls"] = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12} for c in pa if pa[c]["n_gt"] > 0}
g = {"delta": out["all"]["delta"] >= 0.003, "P": (out["all"]["P"] or 0) >= PMIN, "new34": out["new34"]["delta"] >= -1e-12, "nolidar": out["nolidar"]["delta"] >= -1e-12,
     "floors": all(v["ok"] for v in out["cls"].values())}
g["PASS"] = all(g.values()); out["gate"] = g; res["gate"] = out; save()
log(f"{Q} vs V3P: Δ {out['all']['delta']:+.6f} P {out['all']['P']} | new34 {out['new34']['delta']:+.6f} nolidar {out['nolidar']['delta']:+.6f} noKis {out['noKislovodsk']['delta']:+.6f} | "
    + " ".join(f"{c[:8]} {v['d']:+.5f}" for c, v in out["cls"].items()) + f" | picks {chosen} | {g}")
