#!/usr/bin/env python3
"""W1 joint gate J1/J2 (PREREG_W1 16:10) on top of W4 v3_on V3P; helpers copied from W4 v3_gate.py 7db123e unchanged.
Orig: W4 v3_on gate (PREREG docs/PREREG_w4_v3on.md): B0=v2clean vs V3P/V3F, honest-10, d3_gate machinery imported unchanged."""
import importlib.util, json, sys, time, gc
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
_s = importlib.util.spec_from_file_location("d3_gate", "/workspace/koz3/orch/bot-W1/work/d3_gate.py")
D = importlib.util.module_from_spec(_s); sys.modules["d3_gate"] = D; _s.loader.exec_module(D)
M, P, S, SPD, log, site_family, iter_features = D.M, D.P, D.S, D.SPD, D.log, D.site_family, D.iter_features
from shapely.geometry import shape, mapping

BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/koz3/w1/cf/joint/joint_gate.json")
SP = {"sp_or", "sp_focal", "sp_arch", "sp_focal2"}
OOF = [Path("/workspace/koz3/w4geo/spor_oof/full/oof_dfine_spor.geojson"), Path("/workspace/koz3/w4geo/spor_oof/rest/oof_dfine_spor.geojson")]
TAU = {"kurgany_povrezhdennye": 0.555265, "kurgany_tselye": 0.670285, "gorodishcha": 0.68847, "arkhitektury": 0.373015, "fortifikatsii": 0.869051}


def load_geo(p, pred):
    return [f for f, _ in iter_features(Path(p)) if pred(f["properties"])]


def drop_engine(tab, drop):   # copy of TableEngine base without rows where drop(grp, reg)
    import copy
    t2 = copy.copy(tab); t2.base = {}
    for c, b in tab.base.items():
        keep = np.array([not drop(g, r) for g, r in zip(b["grp"], b["reg"])], bool)
        idx = np.flatnonzero(keep); new = {int(o): n for n, o in enumerate(idx)}
        t2.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx],
                          pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
    return t2


def dfine_feats():
    by = defaultdict(list); fold_reg = defaultdict(set); n = 0
    for p in OOF:
        for f, _ in iter_features(p):
            q = f["properties"]
            if q.get("class_name") not in D.CLASSES5:
                continue
            n += 1; fold_reg[q.get("region_name")].add(q.get("fold"))
            by[q["region_name"]].append({"geom": shape(f["geometry"]), "class_name": q["class_name"], "confidence": float(q["confidence"]),
                                         "region_name": q["region_name"], "source": q.get("tif"), "original_crs": q.get("original_crs"), "model": 0})
    assert all(len(v) == 1 for v in fold_reg.values()), "region in several folds"
    out = []
    for reg in sorted(by):
        cl = SPD.cluster_circles(by[reg], 4.0, 4.0)
        for f in P.finalize(P.merge_within_markup(cl)):
            out.append({"type": "Feature", "geometry": mapping(f["geom"]),
                        "properties": {"class_name": f["class_name"], "confidence": round(float(f["confidence"]), 6), "region_name": f["region_name"],
                                       "crs": "EPSG:3857", "markup_type": "sp_or", "original_crs": f.get("original_crs"), "threshold_group": "sp_dfine"}})
        by[reg] = None
    log(f"D-FINE OOF: {n} ellipses, {len(fold_reg)} regions -> {len(out)} circles {dict(Counter(f['properties']['class_name'] for f in out))}")
    return out, set(fold_reg)


import pickle, shapely
from scipy.stats import rankdata
W5D = Path("/workspace/koz3/w1/cf/w5d/work")
RPC = {"gorodishcha", "arkhitektury", "object_poly", "fortifikatsii"}
RTAG = "w5cf"
ROUT = json.load(open("/workspace/koz3/w1/cf/ws_rr_results_w5cf.json"))
CNT = pickle.load(open("/workspace/koz3/w1/cf/ws_rr_counts_w5cf.pkl", "rb"))


def keep_src(tab, drop):   # like drop_engine, also stores 'src' = position in tab.base[c]
    t2 = drop_engine(tab, drop)
    for c, b in tab.base.items():
        t2.base[c]["src"] = np.flatnonzero(np.array([not drop(g, r) for g, r in zip(b["grp"], b["reg"])], bool))
    return t2


def rp_engine(tab, NP):   # M_rp: RPC classes -> W0 rp pairs for stack rows
    import copy
    t2 = copy.copy(tab); t2.base = {}
    for k, c in enumerate(tab.CL):
        b = dict(tab.base[c])
        if c in RPC:
            idx = tab._idx[c]; pos = {int(i): n for n, i in enumerate(idx)}
            P_, J_, S_ = NP["rp"].get((c, c), (np.zeros(0, int), np.zeros(0, int), np.zeros(0)))
            prs = defaultdict(list)
            for a_, b_, s_ in zip(P_, J_, S_):
                prs[pos[int(a_)]].append((int(b_), float(s_)))
            b["pairs"] = dict(prs)
        t2.base[c] = b
    t2._nosort = set(RPC)
    return t2


def rp_extra(tab, ex_plain, extra):   # recompute extras pairs for RPC classes by W0 rp rule
    pr = M._collect(tab.mod, tab.calc, S.fc(extra), False); M._UTM_GEOM_CACHE.clear()
    out = dict(ex_plain); rt = tab.calc.r_tolerance
    for c in RPC:
        if c not in ex_plain or c not in tab.trees:
            continue
        lst = pr[c]; assert len(lst) == len(ex_plain[c]["conf"])
        ok = [x[0].geom_type == "Polygon" for x in lst]
        pts = [x[0].representative_point() if o else None for x, o in zip(lst, ok)]
        qp, qg = tab.trees[c].query([p if p is not None else shapely.Point() for p in pts])
        prs = defaultdict(list)
        for a_, b_ in zip(qp.tolist(), qg.tolist()):
            p = pts[a_]
            if p is None:
                continue
            g = tab.G[c][b_]
            good = (p.distance(g) <= rt) if g.geom_type == "Point" else (g.contains(p) and g.exterior.distance(p) >= 0.05 * 0.71)
            if good:
                prs[a_].append((b_, 1.0))
        e = dict(ex_plain[c]); e["pairs"] = dict(prs); out[c] = e
    return out


def curves_ord(self, extra=()):
    """= D.TableEngine.curves, but for classes in self._nosort the stack-row pair lists keep their stored order (as ws_rr/W0 rp);
    extras and other classes: ascending GT index (official order), exactly as in d3_gate."""
    ex = self._extra(list(extra)); nos = getattr(self, "_nosort", set())
    out = {}
    for c in self.CL:
        b, e = self.base[c], ex.get(c)
        nb = len(b["conf"])
        conf = b["conf"] if e is None else np.concatenate([b["conf"], e["conf"]])
        reg = b["reg"] if e is None else np.concatenate([b["reg"], e["reg"]])
        grp = b["grp"] if e is None else np.concatenate([b["grp"], e["grp"]])
        pairs = dict(b["pairs"])
        if e is not None:
            pairs.update({nb + i: v for i, v in e["pairs"].items()})
        order = np.argsort(-conf, kind="stable")
        rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
        hit = np.zeros(len(order), bool); used = set()
        for i in sorted(pairs, key=lambda i: rank[i]):
            best, bj = float("-inf"), -1
            for j, s_ in (pairs[i] if (c in nos and i < nb) else sorted(pairs[i])):
                if j not in used and s_ > best + 1e-9:
                    best, bj = s_, j
            if bj >= 0:
                used.add(bj); hit[rank[i]] = True
        out[c] = M.ClassCurve(conf=conf[order], hit=hit, pred_region=reg[order].tolist(), n_gt=len(self.G[c]),
                              gt_regions=Counter(self.greg[c]), pred_group=grp[order].tolist())
    return out, {"touched_clusters": None, "overlap_warnings": [], "rebuild_equal": None}


D.TableEngine.curves = curves_ord


def main():
    t0 = time.perf_counter()
    stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
                  Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
    tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
    with open("/workspace/koz3/w5data/work/table.pkl", "rb") as fh:
        T = pickle.load(fh)
    cls = T["cls"].astype(int); conf = T["conf"].astype(float); mk = T["mk"].astype(str); treg = T["reg"].astype(str); N = len(cls)
    del T; gc.collect()
    tab._idx = IDX = {c: np.flatnonzero(cls == k) for k, c in enumerate(tab.CL)}
    for c in tab.CL:
        assert np.array_equal(tab.base[c]["reg"], treg[tab._idx[c]])
    slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
    cA, _ = tab.curves()
    regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
    folds = M.family_folds(regs, site_family, 10); K = len(folds)
    nA, _, _ = D.honest(cA, folds); A = M.score_counts(nA)[0]; res = {"A_honest": A, "boot": BOOT, "variants": {}, "sanity": {}}
    log(f"A honest {A:.10f} (ws_rr A {ROUT['A']:.10f})")
    fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in treg])
    z = np.load(W5D / f"nested_{RTAG}.npz"); OUTS = z["outer"]; INN = z["inner"]
    li = mk == "li"; RB = np.zeros(N); LIM = {}
    for k, c in enumerate(tab.CL):
        m = np.flatnonzero(li & (cls == k))
        if len(m):
            RB[m] = (rankdata(conf[m]) - 0.5) / len(m); LIM[c] = m

    def blend(k, a):   # = ws_rr.blend
        R = np.where(fid == k, OUTS, INN[k]); Sc = conf.copy()
        for c, m in LIM.items():
            Sc[m] = RB[m]; pm = m[R[m] >= 0]
            if len(pm) < 2:
                continue
            q = np.sort(RB[pm]); rt = q[(rankdata(R[pm], method="ordinal") - 1).astype(int)]
            Sc[pm] = a * rt + (1 - a) * RB[pm]
        return Sc

    def set_conf(eng, Sc):
        for c, b in eng.base.items():
            v = Sc[IDX[c]]
            b["conf"] = v[b["src"]] if "src" in b else v

    def per_fold(eng, extra):   # RR1s: per outer fold k, scores blend(k, alpha_k), honest resweep, keep fold-k counts
        tot = None
        for k in range(K):
            a = ROUT["picks"][k]["sw"]; set_conf(eng, blend(k, a))
            cv, _ = eng.curves(extra)
            rest = set().union(*[f for j, f in enumerate(folds) if j != k])
            n, _, _ = D.honest(cv, [folds[k], rest])
            del cv; gc.collect()
            log(f"  fold {k} a={a} RSS {D.rss_mb():.0f} MB")
            if tot is None:
                tot = {c: {} for c in n}
            for c, per in n.items():
                for r, v in per.items():
                    if r in folds[k]:
                        tot[c][r] = v
        set_conf(eng, conf)
        return tot

    NP = pickle.load(open(W5D / "w0_disk_pairs.pkl", "rb"))
    tabM = rp_engine(tab, NP)
    # S0 sanity on base A
    cM, _ = tabM.curves(); nM, _, _ = D.honest(cM, folds); sM = M.score_counts(nM)[0]
    wM = M.score_counts(CNT["M_rp"])[0]; res["sanity"]["S0a_M_rp"] = [sM, wM]
    log(f"S0a M_rp on A: {sM:.10f} vs ws_rr {wM:.10f} (Δ {sM - A:+.6f})")
    tabM2 = keep_src(tabM, lambda g, r: False)
    nR = per_fold(tabM2, []); sR = M.score_counts(nR)[0]; wR = M.score_counts(CNT["RR1s"])[0]
    res["sanity"]["S0b_RR1s_cf"] = [sR, wR]
    log(f"S0b RR1s-cf on A: {sR:.10f} vs ws_rr {wR:.10f} (Δ {sR - A:+.6f})")
    res["sanity"]["S0_ok"] = bool(abs(sM - wM) < 1e-6 and abs(sR - wR) < 1e-6)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    del tabM2, cM; gc.collect()

    c19 = load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
    dfx, dreg = dfine_feats(); extra = c19 + dfx
    drop = lambda g, r: g == "etgr" or (g in SP and r in dreg)
    base3 = drop_engine(tab, drop)
    cB, _ = base3.curves(extra); nB, _, _ = D.honest(cB, folds); sB = M.score_counts(nB)[0]
    res["S1_V3P"] = sB; log(f"S1 V3P honest {sB:.6f} (W4 0.333787)"); del cB; gc.collect()
    ex_plain = tab._extra(extra); EX = rp_extra(tab, ex_plain, extra)
    res["n_extra_rp_pairs"] = {c: [len(ex_plain[c]["pairs"]), len(EX[c]["pairs"])] for c in RPC if c in EX}
    log(f"extras pairs plain->rp {res['n_extra_rp_pairs']}")
    j1 = keep_src(tabM, drop); j1._extra = lambda e: EX
    del tab; gc.collect()

    def score(name, nV, nRef, refname):
        sc = {}
        for k, s in slices.items():
            sa, sb = M.score_counts(nRef, s)[0], M.score_counts(nV, s)[0]
            bs = M.paired_bootstrap_counts(nV, nRef, site_family, BOOT, 0, s) if BOOT else {}
            sc[k] = {"ref": sa, "V": sb, "delta": sb - sa, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
        pa, pb = M.score_counts(nRef)[1], M.score_counts(nV)[1]
        floor = {c: {"f1_ref": pa[c]["f1"], "f1_V": pb[c]["f1"], "d": pb[c]["f1"] - pa[c]["f1"],
                     "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12,
                     "tp": [pa[c]["tp"], pb[c]["tp"]], "fp": [pa[c]["fp"], pb[c]["fp"]]} for c in pa if pa[c]["n_gt"] > 0}
        g = {"delta_ge_0.003": sc["all"]["delta"] >= 0.003, "P_ge_0.9833": (sc["all"]["P"] or 0) >= 1 - 0.1 / 6,
             "floor_ok": all(v["ok"] for v in floor.values()), "new34_nolidar_not_down": sc["new34"]["delta"] >= 0 and sc["nolidar"]["delta"] >= 0}
        g["PASS"] = all(g.values()); g["P_ge_0.9857_x7"] = (sc["all"]["P"] or 0) >= 1 - 0.1 / 7
        res["variants"][name] = {"ref": refname, "scores": sc, "floor": floor, "gate": g}
        log(f"{name} vs {refname}: V {sc['all']['V']:.6f} Δall {sc['all']['delta']:+.6f} P {sc['all']['P']} | new34 {sc['new34']['delta']:+.6f} "
            f"| nolidar {sc['nolidar']['delta']:+.6f} | floor {g['floor_ok']} {[c[:6] for c, v in floor.items() if not v['ok']]} | PASS {g['PASS']}")
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))

    c1, _ = j1.curves(extra); n1, _, _ = D.honest(c1, folds); del c1; gc.collect()
    n2 = per_fold(j1, extra)
    score("J2_RR1s_cf_Mrp", n2, nB, "V3P")
    score("J1_Mrp", n1, nB, "V3P")
    score("J2_vs_J1", n2, n1, "J1")
    pickle.dump(dict(V3P=nB, J1=n1, J2=n2), open(OUT.with_suffix(".counts.pkl"), "wb"))
    res["sec"] = round(time.perf_counter() - t0)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str)); log("done", res["sec"], "s")


main()
