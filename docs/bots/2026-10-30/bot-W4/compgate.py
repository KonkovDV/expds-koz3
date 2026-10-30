#!/usr/bin/env python3
"""W4 P-W4-compgate + v3_on+M_rp (PREREG docs/PREREG_w4_compgate.md). d3_gate imported unchanged; v3_gate helpers reused."""
import importlib.util, json, sys, time, gc, math, copy
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D, M, P, log, site_family = G.D, G.M, G.P, G.log, G.site_family
import shapely.geometry as sg
BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = Path("/workspace/koz3/orch/bot-W4/work/compgate.json")
RPC = ("gorodishcha", "arkhitektury", "object_poly", "fortifikatsii")
MODS = ("geo", "li", "sp_or", "ae", "or")
GMOD = {**{g: "li" for g in ("li", "li_dev", "li_mosaic", "li_coarse", "li_mask", "li_fort", "li_ae")},
        **{g: "sp_or" for g in ("sp_or", "sp_focal", "sp_focal2", "sp_arch", "sp_dfine")},
        "ae": "ae", "ae_fort": "ae", "or": "or", **{g: "geo" for g in ("maggr", "etgr", "terfl", "gor_phys")}}
TMOD = {"maggr": "geo", "etgr": "geo", "terfl": "geo", "li": "li", "sp_or": "sp_or", "ae": "ae", "or": "or"}
def gmod(g):
    if g in GMOD: return GMOD[g]
    for p, m in (("li", "li"), ("sp", "sp_or"), ("ae", "ae"), ("etgr", "geo"), ("terfl", "geo"), ("maggr", "geo"), ("gor_phys", "geo")):
        if g.startswith(p): return m
    if g == "or" or g.startswith("or_"): return "or"
    raise KeyError(g)
PUB = {"geo": 1441, "li": 964, "sp_or": 377, "ae": 75, "or": 50}
res = {"boot": BOOT, "sanity": {}, "cmp": {}}
def save(): OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))

stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
gtm = defaultdict(Counter)   # (cls, region) -> Counter(mod)
oof = Counter()
for f in stk.gt_all:
    q = f["properties"]; m = TMOD[q["markup_type"]]; gtm[(q["class_name"], q["region_name"])][m] += 1; oof[m] += 1
tot = sum(oof.values())
W = {"UNW": {m: 1.0 for m in MODS},
     "PUB": {m: (PUB[m] / sum(PUB.values())) / (oof[m] / tot) for m in MODS},
     "CLO": {m: 0.2 / (oof[m] / tot) for m in MODS}}
res["weights"] = W; res["oof_gt"] = dict(oof); log("weights", W)

cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10)

GSEEN = Counter()
def modsplit(curves):
    for c in curves.values(): GSEEN.update(c.pred_group)
    res["groups_seen"] = {g: [k, gmod(g)] for g, k in GSEEN.items()}
    n, info, taus = D.honest(curves, folds)
    per = {}
    for m in MODS:
        cm = {}
        for cls, c in curves.items():
            mk = np.array([gmod(g) == m for g in c.pred_group], bool)
            gr = Counter({r: gtm[(cls, r)][m] for r in c.gt_regions if gtm[(cls, r)][m]})
            cm[cls] = M.ClassCurve(conf=c.conf[mk], hit=c.hit[mk], pred_region=[r for r, k in zip(c.pred_region, mk) if k],
                                   n_gt=int(sum(gr.values())), gt_regions=gr, pred_group=[g for g, k in zip(c.pred_group, mk) if k])
        per[m] = D.honest(cm, folds, frozen=taus)[0]
    # C0: Σ_m == unweighted; GT check
    bad = 0
    for cls in n:
        for r, v in n[cls].items():
            s = [sum(per[m][cls].get(r, [0, 0, 0])[i] for m in MODS) for i in range(3)]
            bad += s != list(v)
    return n, per, bad, info["unseen_dropped"]

def wcounts(per, w):
    out = defaultdict(dict)
    for m in MODS:
        for cls, by in per[m].items():
            for r, (a, b, g) in by.items():
                o = out[cls].setdefault(r, [0.0, 0.0, 0.0]); o[0] += w[m] * a; o[1] += w[m] * b; o[2] += w[m] * g
    return dict(out)

def evalv(name, curves):
    n, per, bad, unseen = modsplit(curves)
    res["sanity"][f"C0_{name}_mismatch_cells"] = bad
    ws = {k: wcounts(per, w) for k, w in W.items()}
    res["sanity"][f"C1_{name}_unw_minus_honest"] = M.score_counts(ws["UNW"])[0] - M.score_counts(n)[0]
    log(f"{name}: honest {M.score_counts(n)[0]:.10f} C0 bad {bad} C1 {res['sanity'][f'C1_{name}_unw_minus_honest']:.2e} unseen {unseen}")
    return {"n": n, "w": ws}

def compare(name, B, R, rname, gate=False):
    out = {"vs": rname}
    for k, s in slices.items():
        a, b = M.score_counts(R["n"], s)[0], M.score_counts(B["n"], s)[0]
        bs = M.paired_bootstrap_counts(B["n"], R["n"], site_family, BOOT, 0, s)
        out[f"UNW_{k}"] = {"ref": a, "var": b, "delta": b - a, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    for k in ("PUB", "CLO"):
        a, b = M.score_counts(R["w"][k])[0], M.score_counts(B["w"][k])[0]
        bs = M.paired_bootstrap_counts(B["w"][k], R["w"][k], site_family, BOOT, 0, None)
        out[k] = {"ref": a, "var": b, "delta": b - a, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    pa, pb = M.score_counts(R["n"])[1], M.score_counts(B["n"])[1]
    out["floor"] = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12}
                    for c in pa if pa[c]["n_gt"] > 0}
    out["composition_ok"] = out["PUB"]["delta"] >= 0 and out["CLO"]["delta"] >= 0
    if gate:
        g = {"delta_ge_0.003": out["UNW_all"]["delta"] >= 0.003, "P_ge_0.9": (out["UNW_all"]["P"] or 0) >= 0.9,
             "new34_nolidar_ge0": out["UNW_new34"]["delta"] >= 0 and out["UNW_nolidar"]["delta"] >= 0,
             "floor_ok": all(v["ok"] for v in out["floor"].values()), "composition_ok": out["composition_ok"]}
        g["PASS"] = all(g.values()); out["gate"] = g
    res["cmp"][f"{name}_vs_{rname}"] = out; save()
    log(f"{name} vs {rname}: UNW {out['UNW_all']['delta']:+.6f} P {out['UNW_all']['P']} | new34 {out['UNW_new34']['delta']:+.6f} nolidar {out['UNW_nolidar']['delta']:+.6f} "
        f"| PUB {out['PUB']['delta']:+.6f} P {out['PUB']['P']} | CLO {out['CLO']['delta']:+.6f} P {out['CLO']['P']} | comp_ok {out['composition_ok']}"
        + (f" | PASS {out['gate']['PASS']} {out['gate']}" if gate else ""))

def drop_cls(t, drop_classes, drop=lambda g, r: False):
    t2 = copy.copy(t); t2.base = {}
    for c, b in t.base.items():
        keep = np.array([c not in drop_classes and not drop(g, r) for g, r in zip(b["grp"], b["reg"])], bool)
        idx = np.flatnonzero(keep); new = {int(o): n for n, o in enumerate(idx)}
        t2.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx], pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
    return t2

def disk(geom):
    g = sg.shape(geom); c = g.representative_point(); sc = max(math.cos(math.atan(math.sinh(c.y / 6378137.0))), 1e-6)
    return sg.mapping(c.buffer(0.05 / sc, quad_segs=8))

def mrp(feats, on=True):
    out = []
    for f in feats:
        if on and f["properties"].get("class_name") in RPC:
            f = {"type": "Feature", "geometry": disk(f["geometry"]), "properties": f["properties"]}
        out.append(f)
    return out

# RPC rows of the stack (split_arch, as W3 make_joint)
rows = []
for p in sorted(Path("/workspace/koz3/w1/d3/split_arch").glob("*.tsv")):
    for line in open(p, encoding="utf-8"):
        if not any(f'"{c}"' in line for c in RPC): continue
        f = json.loads(line.partition("\t")[2])
        if f["properties"].get("class_name") in RPC: rows.append(f)
nbase = {c: len(b["conf"]) for c, b in tab.base.items() if c in RPC}
res["sanity"]["rpc_rows_split_vs_table"] = {"split": dict(Counter(f["properties"]["class_name"] for f in rows)), "table": nbase}
log("rpc rows", res["sanity"]["rpc_rows_split_vs_table"])

no_et = G.drop_engine(tab, lambda g, r: g == "etgr")
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
dfx, dreg = G.dfine_feats()
SP = G.SP
EA = evalv("A", cA)
cB0, _ = no_et.curves(c19); EB0 = evalv("B0", cB0); del cB0
base3 = G.drop_engine(tab, lambda g, r: g == "etgr" or (g in SP and r in dreg))
cP, _ = base3.curves(c19 + dfx); EP = evalv("V3P", cP); del cP; gc.collect()
compare("V3P", EP, EB0, "B0")
# M0 sanity: RPC rows via extras without disk == A
noR = drop_cls(tab, set(RPC))
c, _ = noR.curves(mrp(rows, on=False)); n0 = D.honest(c, folds)[0]; del c
res["sanity"]["M0_A_via_extras_minus_A"] = M.score_counts(n0)[0] - M.score_counts(EA["n"])[0]; log("M0", res["sanity"]["M0_A_via_extras_minus_A"]); save()
c, _ = noR.curves(mrp(rows)); EM = evalv("MRP_A", c); del c, noR; gc.collect()
compare("MRP", EM, EA, "A")
grp = lambda f: M.threshold_group(f["properties"])
rows3 = [f for f in rows if not (grp(f) == "etgr" or (grp(f) in SP and f["properties"]["region_name"] in dreg))]
noR3 = drop_cls(base3, set(RPC)); del tab, base3, no_et; gc.collect()
c, _ = noR3.curves(mrp(rows3, on=False) + c19 + dfx); n3 = D.honest(c, folds)[0]; del c
res["sanity"]["M0_V3P_via_extras_minus_V3P"] = M.score_counts(n3)[0] - M.score_counts(EP["n"])[0]; log("M0 V3P", res["sanity"]["M0_V3P_via_extras_minus_V3P"]); save()
c, _ = noR3.curves(mrp(rows3) + mrp(c19) + mrp(dfx)); EPM = evalv("V3P_MRP", c); del c; gc.collect()
compare("V3P_MRP", EPM, EP, "V3P", gate=True)
compare("V3P_MRP", EPM, EB0, "B0")
compare("V3P_MRP", EPM, EA, "A")
compare("V3P", EP, EA, "A")
log("done"); save()
