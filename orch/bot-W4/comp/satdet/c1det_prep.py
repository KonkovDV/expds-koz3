#!/usr/bin/env python3
"""W4 independent honest-10 comparison: W1-SATDET Faster R-CNN OOF (sp_det2, PREREG 843d807 D1/D2 class sets) on top of v4 (= C1, out_C1).
Sidecar of orch/bot-W4/comp/dfine2a/dfine2a_prep.py (gate machinery = v4build/gate/q6_prep_head.py, unchanged).
Extras: tools/satdet/t1_prep.det_feats (cluster_circles 4/4 -> merge_within_markup -> finalize, group sp_det2, markup sp_or).
Variants (all C1-frozen: C1 per-fold centre (d0,s) + C1 taus for every existing group, incl. sp_dfine):
 ID     no extras -> must reproduce C1 chosen+taus+gate_nested exactly.
 C1D1f  + sp_det2 all 5 classes; only sp_det2 per-class tau fitted per held fold on the 9 training folds.
 C1D2f  + sp_det2 {gor, fort, arkh}; same.
NOT part of the PREREG family (W4 extra comparison vs v4)."""
import sys
DET = "/workspace/koz3/w1/satdet/pulled/oof_frcnn_spor.geojson"
sys.argv = sys.argv[:1]
exec(open("/workspace/koz3/v4build/gate/q6_prep_head.py").read())
OUT = Path("/workspace/koz3/orch/bot-W4/comp/satdet/outc1"); OUT.mkdir(parents=True, exist_ok=True)
C1S = json.load(open("/workspace/koz3/orch/bot-W4/comp/c1fp/out/prep_summary.json"))["variants"]["C1"]


def det_feats(path):   # = tools/satdet/t1_prep.py det_feats (843d807)
    by = defaultdict(list); fold_reg = defaultdict(set); n = 0
    for f, _ in D.iter_features(Path(path)):
        q = f["properties"]
        if q.get("class_name") not in CL5:
            continue
        n += 1; fold_reg[q["region_name"]].add(q.get("model_group"))
        by[q["region_name"]].append({"geom": shape(f["geometry"]), "class_name": q["class_name"], "confidence": float(q["confidence"]),
                                     "region_name": q["region_name"], "source": q.get("tif"), "original_crs": q.get("original_crs"), "model": 0})
    assert all(len(v) == 1 for v in fold_reg.values()), "region predicted by several models"
    assert set(fold_reg) <= dreg, set(fold_reg) - dreg
    out = []
    for reg in sorted(by):
        for f in G.P.finalize(G.P.merge_within_markup(G.SPD.cluster_circles(by[reg], 4.0, 4.0))):
            out.append({"type": "Feature", "geometry": mapping(f["geom"]),
                        "properties": {"class_name": f["class_name"], "confidence": round(float(f["confidence"]), 6), "region_name": f["region_name"],
                                       "crs": "EPSG:3857", "markup_type": "sp_or", "original_crs": f.get("original_crs"), "threshold_group": "sp_det2"}})
    log(f"det OOF: {n} ellipses, {len(fold_reg)} regions -> {len(out)} circles {dict(Counter(f['properties']['class_name'] for f in out))}")
    return out, n, len(fold_reg)


detf, n_ell, n_reg = det_feats(DET)
det = [as_ex(f) for f in detf]; assert all(e["threshold_group"] == "sp_det2" for e in det)
fin = lambda t: {g: {c: (None if not math.isfinite(v) else float(v)) for c, v in d.items()} for g, d in t.items()}
unfin = lambda t: {g: {c: (math.inf if v is None else float(v)) for c, v in d.items()} for g, d in t.items()}
TC1 = [unfin(t) for t in C1S["taus"]]
DF, DG = "sp_dfine", "sp_det2"


class _Eng1:
    def __init__(self, c):
        self.CL = [c]; self.base = base3.base; self.G = base3.G; self.greg = base3.greg


def mult(dr, grp, d0, s):
    m = np.ones(len(dr))
    if s > 0:
        ok = np.isfinite(dr) & (dr > d0) & (grp == DF)
        m[ok] = np.maximum(0.0, 1.0 - s * np.minimum((dr[ok] - d0) / (1.0 - d0), 1.0))
    return m


def sweep(curve, cls, taus, regions, grpname):   # = c1fp_prep.sweep generalised to one swept group
    gnames = curve.pred_group
    inr = np.array([r in regions for r in curve.pred_region], bool); isg = np.array([g == grpname for g in gnames], bool)
    tval = np.array([taus.get(g, {}).get(cls, math.inf) if g != grpname else 0 for g in gnames], float)
    keep0 = inr & ~isg & (curve.conf >= tval); tp0 = float(curve.hit[keep0].sum()); n0 = int(keep0.sum())
    G_ = sum(int(k) for r, k in curve.gt_regions.items() if r in regions)
    idx = np.flatnonzero(inr & isg)
    if idx.size == 0:
        return math.inf
    o = np.argsort(-curve.conf[idx], kind="stable")
    keep, tp, cut = M._group_cuts(curve.conf[idx][o], curve.hit[idx][o])
    tt, nn = tp0 + tp, n0 + keep
    f = np.where(nn > 0, M._f1(tt, nn - tt, G_ - tt), 0.0)
    return round(float(cut[int(np.argmax(f))]), 6)


res = {"variants": {}, "det": DET, "n_ellipses": n_ell, "n_regions": n_reg, "n_circles": len(det),
       "circles_by_class": dict(Counter(e["class_name"] for e in det))}
EXS = {"ID": [], "C1D1f": det, "C1D2f": [e for e in det if e["class_name"] in ("gorodishcha", "fortifikatsii", "arkhitektury")]}
for name, extra in EXS.items():
    ex = ex_of(ex_c19 + ex_dfx + extra); cB = curves_from(ex); cache = {}
    def cv(c, d0, s):
        if s <= 0: return cB[c]
        if (c, d0, s) not in cache:
            e = ex[c]
            cache[(c, d0, s)] = curves_from({c: dict(e, conf=e["conf"] * mult(e["drel"], e["grp"], d0, s))}, eng=_Eng1(c))[c]
        return cache[(c, d0, s)]
    chosen, taus, held = [], [], {c: {} for c in cB}
    for fi in range(len(folds)):
        tr = every - folds[fi]; cch = C1S["chosen"][fi]
        cur = {c: (cv(c, cch["d0"], cch["s"]) if c in CL5 else cB[c]) for c in cB}
        t = {g: dict(d) for g, d in TC1[fi].items()}
        if extra:
            t[DG] = {}
            for c in sorted({e["class_name"] for e in extra}):
                t[DG][c] = sweep(cur[c], c, t, tr, DG)
        sc = M.score_counts(counts_at(cur, t, tr))[0]
        for c, d in counts_at(cur, t, folds[fi]).items(): held[c].update(d)
        chosen.append(dict(center=dict(d0=cch["d0"], s=cch["s"]), train=sc)); taus.append(fin(t))
        log(f"{name} fold {fi}: d0,s ({cch['d0']},{cch['s']}) train {sc:.6f} τ_det2 {t.get(DG)}")
    sg_, per = M.score_counts(held)
    log(f"{name} gate-side nested {sg_:.6f} (C1 gate_nested {C1S['gate_nested']:.6f}, Δ {sg_ - C1S['gate_nested']:+.6f})")
    res["variants"][name] = dict(chosen=chosen, taus=taus, gate_nested=sg_, n_extra=len(extra), cls={c: x["f1"] for c, x in per.items()})
    if name == "ID":
        same = all(fin(TC1[i]) == taus[i] for i in range(10)) and abs(sg_ - C1S["gate_nested"]) < 1e-9
        res["ID_reproduces_C1"] = same; log("ID reproduces C1 taus + gate_nested:", same); assert same
pickle.dump(dict(res=res, ex={k: ex_c19 + ex_dfx + v for k, v in EXS.items()}), open(OUT / "prep.pkl", "wb"))
json.dump(res, open(OUT / "prep_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
