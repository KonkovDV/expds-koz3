#!/usr/bin/env python3
"""W4 independent honest-10 gate: dfine2-a (P-W2-13, folds 0+3 only) on top of v4 (= C1 centre, out_C1).
Sidecar of orch/bot-W4/comp/c1fp/c1fp_prep.py (gate machinery = v4build/gate/q6_prep_head.py, unchanged).
Member construction = PREREG P-W2-13 ENS8: per region SPD.cluster_circles(ref model0 + dfine2 model1, 4 m, merge 8 m, mean, 2)
 -> P.finalize(P.merge_within_markup(.)); applied ONLY on the 18 regions covered by dfine2-a (D-FINE folds 0,3); other 114 regions keep
ref sp_dfine circles (b folds not landed). threshold_group kept 'sp_dfine' (= PREREG report variant ens8_mean_as_sp_dfine) so the v4
[post.w4_sat] centre flag + nested sp_dfine τ apply exactly as in v4.
Variants: ID (ref-only through the same circle code: must reproduce C1 exactly), E8C1 (C1 recipe re-run nested: (d0,s) grid + sp_dfine τ),
E8C1f (C1's per-fold (d0,s) frozen, only sp_dfine τ re-swept)."""
import sys
sys.argv = sys.argv[:1]
exec(open("/workspace/koz3/v4build/gate/q6_prep_head.py").read())
OUT = Path("/workspace/koz3/orch/bot-W4/comp/dfine2a/out"); OUT.mkdir(parents=True, exist_ok=True)
C1S = json.load(open("/workspace/koz3/orch/bot-W4/comp/c1fp/out/prep_summary.json"))["variants"]["C1"]
D2P = [Path("/workspace/koz3/w2/oof/dfine2/dmitrijkv_koz3-w2-dfine2-a") / f"oof_dfine_f{k}.geojson" for k in (0, 3)]
P_, SPD_ = G.P, G.SPD


def raw(paths, model):
    by = defaultdict(list); fr = defaultdict(set)
    for p in paths:
        for f, _ in G.iter_features(p):
            q = f["properties"]
            if q.get("class_name") not in CL5: continue
            fr[q["region_name"]].add(q.get("fold"))
            by[q["region_name"]].append({"geom": shape(f["geometry"]), "class_name": q["class_name"], "confidence": float(q["confidence"]),
                                         "region_name": q["region_name"], "source": q.get("tif"), "original_crs": q.get("original_crs"), "model": model})
    assert all(len(v) == 1 for v in fr.values())
    return by, {r: next(iter(v)) for r, v in fr.items()}


ref, rf = raw(G.OOF, 0); d2, df = raw(D2P, 1)
bad = {r: (rf.get(r), f) for r, f in df.items() if rf.get(r) != f}; assert not bad, bad
AREG = set(d2); log("dfine2-a regions", len(AREG), "dets", sum(map(len, d2.values())), "folds", sorted(set(df.values())))


def circ(by_list, regs, merge_m, ens, nm):
    out = []
    for reg in sorted(regs):
        dets = [d for by in by_list for d in by.get(reg, [])]
        for f in P_.finalize(P_.merge_within_markup(SPD_.cluster_circles(dets, 4.0, merge_m, ens, nm))):
            out.append({"type": "Feature", "geometry": mapping(f["geom"]),
                        "properties": {"class_name": f["class_name"], "confidence": round(float(f["confidence"]), 6), "region_name": f["region_name"],
                                       "crs": "EPSG:3857", "markup_type": "sp_or", "original_crs": f.get("original_crs"), "threshold_group": "sp_dfine"}})
    return out


mk_ex = lambda fs: [as_ex(f, drel_of(shape(f["geometry"]), f["properties"]["region_name"])) for f in fs]
def splice(new):   # keep ex_dfx order (sorted regions, dfine_feats order); replace the AREG region blocks in place (tie order matters)
    byr = defaultdict(list)
    for e in new: byr[e["region_name"]].append(e)
    out, done = [], set()
    for e in ex_dfx:
        r = e["region_name"]
        if r not in AREG: out.append(e)
        elif r not in done: out += byr[r]; done.add(r)
    assert done == AREG
    return out
ex_ID = splice(mk_ex(circ([ref], AREG, 4.0, "max", 1)))
ex_E8 = splice(mk_ex(circ([ref, d2], AREG, 8.0, "mean", 2)))
assert [(e["class_name"], e["confidence"], e["region_name"]) for e in ex_ID] == [(e["class_name"], e["confidence"], e["region_name"]) for e in ex_dfx]
# identity: ref-only circles on AREG through this code == dfx circles there
_a = sorted((e["class_name"], e["confidence"], e["region_name"], json.dumps(e["geometry"], sort_keys=True)) for e in ex_dfx if e["region_name"] in AREG)
_b = sorted((e["class_name"], e["confidence"], e["region_name"], json.dumps(e["geometry"], sort_keys=True)) for e in ex_ID if e["region_name"] in AREG)
ID_EXACT = _a == _b; log("identity circles exact", ID_EXACT, len(_a), len(_b)); assert ID_EXACT
log("E8 circles on AREG", sum(e["region_name"] in AREG for e in ex_E8), "vs ref", len(_a), Counter(e["class_name"] for e in ex_E8 if e["region_name"] in AREG))

fin = lambda t: {g: {c: (None if not math.isfinite(v) else float(v)) for c, v in d.items()} for g, d in t.items()}
unfin = lambda t: {g: {c: (math.inf if v is None else float(v)) for c, v in d.items()} for g, d in t.items()}
TB = [unfin(t) for t in T["taus_V3P"]]
DF = "sp_dfine"
GAGRID = [(0.0, 0.0)] + [(d0, s) for d0 in (0.0, 0.25, 0.5, 0.75) for s in (0.25, 0.5, 0.75)]


class _Eng1:
    def __init__(self, c):
        self.CL = [c]; self.base = base3.base; self.G = base3.G; self.greg = base3.greg


def mult(dr, grp, d0, s):
    m = np.ones(len(dr))
    if s > 0:
        ok = np.isfinite(dr) & (dr > d0) & (grp == DF)
        m[ok] = np.maximum(0.0, 1.0 - s * np.minimum((dr[ok] - d0) / (1.0 - d0), 1.0))
    return m


def sweep(curve, cls, taus, regions):   # = c1fp_prep.sweep (AMEND1 metric/prod rule)
    gnames = curve.pred_group
    inr = np.array([r in regions for r in curve.pred_region], bool); isdf = np.array([g == DF for g in gnames], bool)
    tval = np.array([taus.get(g, {}).get(cls, math.inf) if g != DF else 0 for g in gnames], float)
    keep0 = inr & ~isdf & (curve.conf >= tval); tp0 = float(curve.hit[keep0].sum()); n0 = int(keep0.sum())
    G_ = sum(int(k) for r, k in curve.gt_regions.items() if r in regions)
    idx = np.flatnonzero(inr & isdf); o = np.argsort(-curve.conf[idx], kind="stable")
    keep, tp, cut = M._group_cuts(curve.conf[idx][o], curve.hit[idx][o])
    tt, nn = tp0 + tp, n0 + keep
    f = np.where(nn > 0, M._f1(tt, nn - tt, G_ - tt), 0.0)
    return round(float(cut[int(np.argmax(f))]), 6)


res = {"variants": {}, "areg": sorted(AREG), "identity_circles_exact": ID_EXACT, "d2_files": [str(p) for p in D2P]}
EXS = {"ID": ex_ID, "E8C1": ex_E8, "E8C1f": ex_E8}
for name, lst in EXS.items():
    ex = ex_of(ex_c19 + lst); cB = curves_from(ex); cache = {}
    def cv(c, pri):
        pri = tuple(p for p in pri if p[1] > 0)
        if not pri: return cB[c]
        if (c, pri) not in cache:
            e = ex[c]; m = np.ones(len(e["conf"]))
            for d0, s in pri: m = m * mult(e["drel"], e["grp"], d0, s)
            cache[(c, pri)] = curves_from({c: dict(e, conf=e["conf"] * m)}, eng=_Eng1(c))[c]
        return cache[(c, pri)]
    def evaluate(tb, tr, d0, s):
        t = {g: dict(d) for g, d in tb.items()}; cur = dict(cB)
        if s > 0:
            for c in CL5:
                cur[c] = cv(c, ((d0, s),)); t[DF][c] = sweep(cur[c], c, t, tr)
        return t, M.score_counts(counts_at(cur, t, tr))[0], cur
    # base (no centre, V3P taus) for this member set
    hb = {c: {} for c in cB}
    for fi in range(len(folds)):
        for c, d in counts_at(cB, TB[fi], folds[fi]).items(): hb[c].update(d)
    s_base = M.score_counts(hb)[0]
    chosen, taus, held = [], [], {c: {} for c in cB}
    for fi in range(len(folds)):
        tr = every - folds[fi]
        if name == "E8C1f":
            cch = C1S["chosen"][fi]; t, sc, cur = evaluate(TB[fi], tr, cch["d0"], cch["s"]); hp = (cch["d0"], cch["s"])
        else:
            best = None
            for d0, s in GAGRID:
                t_, sc_, cur_ = evaluate(TB[fi], tr, d0, s); key = (round(sc_, 12), -s, d0)
                if best is None or key > best[0]: best = (key, (d0, s), t_, cur_, sc_)
            _, hp, t, cur, sc = best
        assert all(t[g] == TB[fi][g] for g in TB[fi] if g != DF), "freeze violated"
        for c, d in counts_at(cur, t, folds[fi]).items(): held[c].update(d)
        chosen.append(dict(center=dict(d0=hp[0], s=hp[1]), train=sc)); taus.append(fin(t))
        log(f"{name} fold {fi}: d0,s {hp} τ_dfine {({c: round(v, 6) for c, v in t[DF].items()})}")
    sg_, per = M.score_counts(held)
    log(f"{name} gate-side nested {sg_:.6f} (no-centre base for this member set {s_base:.6f})")
    res["variants"][name] = dict(chosen=chosen, taus=taus, gate_nested=sg_, gate_base_nocentre=s_base, cls={c: x["f1"] for c, x in per.items()})
    if name == "ID":
        same = all(chosen[i]["center"] == {"d0": C1S["chosen"][i]["d0"], "s": C1S["chosen"][i]["s"]} for i in range(10)) and \
               all(taus[i] == C1S["taus"][i] for i in range(10))
        same = same and abs(sg_ - C1S["gate_nested"]) < 1e-9 and abs(s_base - T["V3P"]) < 1e-9
        res["ID_reproduces_C1"] = same; log("ID reproduces C1 chosen+taus:", same, "C1 gate_nested", C1S["gate_nested"]); assert same
res["taus_base"] = [fin(t) for t in TB]
pickle.dump(dict(res=res, ex={"ID": ex_c19 + ex_ID, "E8": ex_c19 + ex_E8}), open(OUT / "prep.pkl", "wb"))
json.dump(res, open(OUT / "prep_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
