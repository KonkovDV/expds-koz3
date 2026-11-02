#!/usr/bin/env python3
"""W1-SATDET step 1 (PREREG docs/PREREG_W1_SATDET.md 843d807): gate-side nested selection for P1/P2 (scene-centre prior on
{gorodishcha, arkhitektury}, sp_dfine) and per-fold honest taus for D1/D2 (Faster R-CNN OOF, group sp_det2).
argv: [--det oof_frcnn_spor.geojson]. Output /workspace/koz3/w1/satdet/out/prep.pkl (+ prep_summary.json).
Gate machinery copied from tools/satrec2/s1b_prep.py (W1-SATREC2); curves_from conf_fn now receives the class."""
import importlib.util, json, math, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D = G.D; M = D.M
_s = importlib.util.spec_from_file_location("sat_recall", "/workspace/koz3/wt/w1sr/expds/sat_recall.py")
SR = importlib.util.module_from_spec(_s); _s.loader.exec_module(SR)
import shapely
from shapely import wkb as WKB
from shapely.geometry import mapping, shape
from pyproj import Transformer

OUT = Path("/workspace/koz3/w1/satrec/out")
OUT2 = Path("/workspace/koz3/w1/satdet/out"); OUT2.mkdir(parents=True, exist_ok=True)
DET = sys.argv[sys.argv.index("--det") + 1] if "--det" in sys.argv else None
T = json.load(open("/workspace/koz3/orch/bot-W1/work/v3prod/t_taus.json"))
folds = [set(f) for f in T["folds"]]; every = set().union(*folds); dreg = set(T["dreg"])
CL5 = D.CLASSES5
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
base3 = G.drop_engine(tab, lambda g, r: g == "etgr" or (g in G.SP and r in dreg)); del tab
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
dfx, dreg2 = G.dfine_feats(); assert dreg2 == dreg
FB = json.load(open("/workspace/koz3/repo/docs/bots/2026-10-29/bot-10/work11/frame_boxes.json"))["boxes"]


def drel_of(geom, reg):
    b = FB.get(reg, {}).get("sp_or")
    if not b:
        return None
    pt = geom.representative_point()
    best = None
    for x0, y0, x1, y1 in b:
        hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
        if hw <= 0 or hh <= 0 or not (x0 <= pt.x <= x1 and y0 <= pt.y <= y1):
            continue
        d = max(abs(pt.x - (x0 + hw)) / hw, abs(pt.y - (y0 + hh)) / hh)
        best = d if best is None else min(best, d)
    return best


def as_ex(f, drel=None):
    p = f["properties"]
    return dict(geometry=f["geometry"], class_name=p["class_name"], confidence=float(p["confidence"]), region_name=p["region_name"],
                markup_type=p.get("markup_type"), threshold_group=M.threshold_group(p), drel=drel)


ex_c19 = [as_ex(f) for f in c19]
ex_dfx = [as_ex(f, drel_of(shape(f["geometry"]), f["properties"]["region_name"])) for f in dfx]
log("c19", len(ex_c19), "dfx", len(ex_dfx), "dfx drel None", sum(e["drel"] is None for e in ex_dfx))

# ---------- gate machinery ----------
def ex_of(lst):
    ex = base3._extra([{"type": "Feature", "geometry": e["geometry"],
                        "properties": {k: v for k, v in e.items() if k not in ("drel", "geometry")}} for e in lst])
    assert sum(len(d["conf"]) for d in ex.values()) == len(lst), ("dropped extras", sum(len(d["conf"]) for d in ex.values()), len(lst))
    # attach drel in the same per-class order (_collect keeps input order of valid features)
    per = defaultdict(list)
    for e in lst:
        per[e["class_name"]].append(e)
    for c, d in ex.items():
        assert len(d["conf"]) == len(per[c]), (c, len(d["conf"]), len(per[c]))
        assert np.allclose(d["conf"], [e["confidence"] for e in per[c]])
        d["drel"] = np.array([np.nan if e["drel"] is None else e["drel"] for e in per[c]])
    return ex


def curves_from(ex, conf_fn=None, keep_fn=None):
    out = {}
    for c in base3.CL:
        b, e = base3.base[c], ex.get(c)
        nb = len(b["conf"])
        if e is not None:
            econf = e["conf"] if conf_fn is None else conf_fn(e, c)
            keep = np.ones(len(econf), bool) if keep_fn is None else keep_fn(e, econf)
            idx = np.flatnonzero(keep); rem = {int(o): n for n, o in enumerate(idx)}
            econf, ereg, egrp = econf[idx], e["reg"][idx], e["grp"][idx]
            epairs = {rem[i]: v for i, v in e["pairs"].items() if i in rem}
        conf = b["conf"] if e is None else np.concatenate([b["conf"], econf])
        reg = b["reg"] if e is None else np.concatenate([b["reg"], ereg])
        grp = b["grp"] if e is None else np.concatenate([b["grp"], egrp])
        pairs = dict(b["pairs"])
        if e is not None:
            pairs.update({nb + i: v for i, v in epairs.items()})
        order = np.argsort(-conf, kind="stable")
        rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
        hit = np.zeros(len(order), bool); used = set()
        for i in sorted(pairs, key=lambda i: rank[i]):
            best, bj = float("-inf"), -1
            for j, s_ in sorted(pairs[i]):
                if j not in used and s_ > best + 1e-9:
                    best, bj = s_, j
            if bj >= 0:
                used.add(bj); hit[rank[i]] = True
        out[c] = M.ClassCurve(conf=conf[order], hit=hit, pred_region=reg[order].tolist(), n_gt=len(base3.G[c]),
                              gt_regions=Counter(base3.greg[c]), pred_group=grp[order].tolist())
    return out


def counts_at(curves, taus, regions):
    counts = {cls: {} for cls in curves}
    for cls, c in curves.items():
        rnames, rcodes, gnames, gcodes = M._curve_codes(c)
        per = counts[cls]
        for r, k in c.gt_regions.items():
            if r in regions:
                per.setdefault(r, [0, 0, 0])[2] += int(k)
        lut = np.fromiter((r in regions for r in rnames), dtype=bool, count=len(rnames))
        m = lut[rcodes] if rcodes.size else np.zeros(0, bool)
        if not m.any():
            continue
        has = np.array([cls in taus.get(g, {}) for g in gnames], bool)
        tval = np.array([taus.get(g, {}).get(cls, 0.0) for g in gnames], np.float64)
        gh = gcodes[m]
        tau = np.where(~has[gh], np.inf, tval[gh])
        kept = c.conf[m] >= tau
        rk = rcodes[m][kept]; hk = c.hit[m][kept].astype(bool)
        tpc = np.bincount(rk[hk], minlength=len(rnames)); fpc = np.bincount(rk[~hk], minlength=len(rnames))
        for code in np.unique(rk).tolist():
            lst = per.setdefault(rnames[code], [0, 0, 0]); lst[0] += int(tpc[code]); lst[1] += int(fpc[code])
    return counts


def run(curves, fixed=frozenset()):
    n, _, ft = D.honest(curves, folds, fixed=fixed)
    tr = [M.score_counts(counts_at(curves, ft[fi], every - folds[fi]))[0] for fi in range(len(folds))]
    return n, ft, tr


fin = lambda t: {g: {c: (None if not math.isfinite(v) else float(v)) for c, v in d.items()} for g, d in t.items()}
res = {"folds": T["folds"], "variants": {}}

ex_base_l = ex_c19 + ex_dfx
ex_base = ex_of(ex_base_l)
cB = curves_from(ex_base)
nB, ftB, trB = run(cB)
sB = M.score_counts(nB)[0]
log(f"sanity gate base {sB:.7f} vs V3P {T['V3P']:.7f}")
assert abs(sB - T["V3P"]) < 1e-6, "gate base not reproduced"
res["gate_base"] = sB



# ======================= W1-SATDET =======================
GOR, ARKH = "gorodishcha", "arkhitektury"; L = (GOR, ARKH)
CLS = list(cB.keys()); assert set(L) <= set(CLS), CLS
REGS = sorted(every); RI = {r: i for i, r in enumerate(REGS)}
W = np.array([M.CLASS_WEIGHTS.get(c, 0.0) for c in CLS]); CI = {c: i for i, c in enumerate(CLS)}
log("classes", CLS)


def held_arr(n):
    a = np.zeros((len(CLS), len(REGS), 3), np.int32)
    for ci, c in enumerate(CLS):
        for r, v in n.get(c, {}).items():
            a[ci, RI[r]] = v
    return a


def agg(counts, regions):
    a = np.zeros((len(CLS), 3))
    for ci, c in enumerate(CLS):
        for r, v in counts.get(c, {}).items():
            if r in regions:
                a[ci] += v
    return a


def point(curves, fixed=frozenset()):
    n, _, ft = D.honest(curves, folds, fixed=fixed)
    tr = np.zeros((10, len(CLS), 3))
    for fi in range(10):
        trs = every - folds[fi]
        tr[fi] = agg(counts_at(curves, ft[fi], trs), trs)
    return dict(tr=tr, held=held_arr(n), ft=[fin(t) for t in ft], gate=M.score_counts(n)[0], n=n)


def f1(a):
    tp, fp, g = a[..., 0], a[..., 1], a[..., 2]
    den = 2 * tp + fp + (g - tp)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def sc(t):   # [C,3]
    f = f1(t); has = t[:, 2] > 0
    return float((W * f * has).sum() / (W * has).sum())


def slices_delta(hc):
    return {nm: M.score_counts(hc, sl)[0] - M.score_counts(nB, sl)[0] for nm, sl in (("new34", set(T["new"])), ("nolidar", set(T["optical"])))}


def hc_of(held):
    return {c: {REGS[ri]: held[ci, ri].tolist() for ri in range(len(REGS)) if held[ci, ri].any()} for ci, c in enumerate(CLS)}


# ---- Track 2: prior m(d) = max(0, 1 - s*clip((d-d0)/(1-d0),0,1)) on sp_dfine, classes in L only (same s for both in a point) ----
def prior_conf(d0, s):
    def fn(e, c):
        if s <= 0 or c not in L:
            return e["conf"]
        d = e["drel"]
        m = np.maximum(0.0, 1.0 - s * np.clip((np.nan_to_num(d, nan=0.0) - d0) / (1.0 - d0), 0.0, 1.0))
        off = np.isnan(d) | (e["grp"] != "sp_dfine")
        return e["conf"] * np.where(off, 1.0, m)
    return fn


D0S = (0.0, 0.25, 0.5); SS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
PTS = {}
def get(d0, s):
    if s == 0:
        d0 = 0.0
    if (d0, s) not in PTS:
        PTS[(d0, s)] = point(curves_from(ex_base, conf_fn=prior_conf(d0, s)))
        log(("P", d0, s), f"gate honest {PTS[(d0, s)]['gate']:.6f}")
    return PTS[(d0, s)]


base_pt = get(0.0, 0.0)
assert abs(base_pt["gate"] - sB) < 1e-9
OTHER = [CI[c] for c in CLS if c not in L]
for d0 in D0S:
    for s in SS:
        p = get(d0, s)
        # separability: classes outside L bit-identical to base
        assert np.array_equal(p["held"][OTHER], base_pt["held"][OTHER]) and np.array_equal(p["tr"][:, OTHER], base_pt["tr"][:, OTHER]), ("non-separable", d0, s)


def select_P(shared):
    combos = []
    for d0 in D0S:
        for sg in SS:
            for sa in (SS if not shared else (sg,)):
                if sg == 0 and sa == 0 and d0 != 0.0:
                    continue
                combos.append((d0, sg, sa))
    held = base_pt["held"].copy(); out = dict(chosen=[], taus=[], train_score=[])
    for fi in range(10):
        best = None
        for d0, sg, sa in combos:
            pg, pa = get(d0, sg), get(d0, sa)
            t = base_pt["tr"][fi].copy(); t[CI[GOR]] = pg["tr"][fi][CI[GOR]]; t[CI[ARKH]] = pa["tr"][fi][CI[ARKH]]
            key = (round(sc(t), 12), -sg, -sa, d0)
            if best is None or key > best[0]:
                best = (key, (d0, sg, sa))
        (s_tr, *_), (d0, sg, sa) = best
        pg, pa = get(d0, sg), get(d0, sa)
        idx = [RI[r] for r in folds[fi]]
        held[CI[GOR], idx] = pg["held"][CI[GOR], idx]; held[CI[ARKH], idx] = pa["held"][CI[ARKH], idx]
        t = {g: dict(d) for g, d in base_pt["ft"][fi].items()}
        for c, p in ((GOR, pg), (ARKH, pa)):
            for g, d in p["ft"][fi].items():
                if c in d:
                    t.setdefault(g, {})[c] = d[c]
        out["chosen"].append(None if sg == 0 and sa == 0 else dict(d0=d0, s_gor=sg, s_arkh=sa))
        out["taus"].append(t); out["train_score"].append(s_tr)
    hc = hc_of(held)
    out["gate_nested"] = M.score_counts(hc)[0]; out.update({"gate_d_" + k: v for k, v in slices_delta(hc).items()})
    out["hc"] = hc
    log("P shared" if shared else "P indep", "chosen", out["chosen"], f"nested gate {out['gate_nested']:.6f} Δ {out['gate_nested'] - sB:+.6f}",
        f"Δnew34 {out['gate_d_new34']:+.4f} Δnolidar {out['gate_d_nolidar']:+.4f}")
    return out


res["variants"]["P1"] = select_P(True)
res["variants"]["P2"] = select_P(False)
res["P_points"] = {f"{k[0]}_{k[1]}": v["gate"] for k, v in PTS.items()}


# ---- Track 1: Faster R-CNN OOF → same post as sp_dfine (cluster_circles 4/4, merge_within_markup, finalize) → sp_det2 ----
def det_feats(path):
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
    return out


if DET:
    det = [as_ex(f) for f in det_feats(DET)]
    assert all(e["threshold_group"] == "sp_det2" for e in det)
    for name, keep in (("D1", set(CL5)), ("D2", {"gorodishcha", "fortifikatsii", "arkhitektury"})):
        extra = [e for e in det if e["class_name"] in keep]
        p = point(curves_from(ex_of(ex_base_l + extra)))
        v = dict(chosen=[None] * 10, taus=p["ft"], extra=extra, gate_nested=p["gate"], hc=p["n"])
        v.update({"gate_d_" + k: x for k, x in slices_delta(p["n"]).items()})
        v["sp_det2_taus"] = [t.get("sp_det2", {}) for t in p["ft"]]
        res["variants"][name] = v
        log(name, len(extra), f"nested gate {v['gate_nested']:.6f} Δ {v['gate_nested'] - sB:+.6f} Δnew34 {v['gate_d_new34']:+.4f} Δnolidar {v['gate_d_nolidar']:+.4f}")

res["taus_base"] = [fin(t) for t in ftB]
res["nB"] = nB
tag = "_det" if DET else ""
pickle.dump(dict(res=res, ex_base=ex_base_l), open(OUT2 / f"prep{tag}.pkl", "wb"))
summ = {k: v for k, v in res.items() if k not in ("nB", "variants")}
summ["variants"] = {k: {a: b for a, b in v.items() if a not in ("extra", "hc")} for k, v in res["variants"].items()}
json.dump(summ, open(OUT2 / f"prep{tag}_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
