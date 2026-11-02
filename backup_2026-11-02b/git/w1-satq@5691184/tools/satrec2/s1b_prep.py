#!/usr/bin/env python3
"""W1-SATREC2 step 1 (PREREG docs/PREREG_W1_SATREC2.md 1ff6643), derived from SATREC s1 (PREREG docs/PREREG_W1_SATREC.md 140839a): gate-side nested selection for R1/R2/R3/C1/C2.
Output: /workspace/koz3/w1/satrec/out/prep.pkl = per variant: per-fold taus + per-fold hyperparams + extras list."""
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
OUT2 = Path("/workspace/koz3/w1/satrec2/out"); OUT2.mkdir(parents=True, exist_ok=True)
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
            econf = e["conf"] if conf_fn is None else conf_fn(e)
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



KT = "kurgany_tselye"
CLS = list(cB.keys()); assert KT in CLS, CLS
REGS = sorted(every); RI = {r: i for i, r in enumerate(REGS)}
NL = set(json.load(open("/workspace/koz3/orch/bot-W1/work/satrec2/nl_regions.json"))["nl_post"])
H = set(json.load(open("/workspace/koz3/orch/bot-W1/work/satrec2/H_regions.json"))["H"])
W = np.array([M.CLASS_WEIGHTS.get(c, 0.0) for c in CLS])
log("classes", CLS, "NL", len(NL), "H", len(H))


def agg(counts, regions):
    a = np.zeros((len(CLS), 3))
    for ci, c in enumerate(CLS):
        for r, v in counts.get(c, {}).items():
            if r in regions:
                a[ci] += v
    return a


def held_arr(n):
    a = np.zeros((len(CLS), len(REGS), 3), np.int32)
    for ci, c in enumerate(CLS):
        for r, v in n.get(c, {}).items():
            a[ci, RI[r]] = v
    return a


def point(curves, fixed=frozenset()):
    n, _, ft = D.honest(curves, folds, fixed=fixed)
    tr = np.zeros((10, 2, len(CLS), 3))
    for fi in range(10):
        trs = every - folds[fi]
        cnt = counts_at(curves, ft[fi], trs)
        tr[fi, 0] = agg(cnt, trs); tr[fi, 1] = agg(cnt, trs & H)
    return dict(tr=tr, held=held_arr(n), ft=[fin(t) for t in ft], gate=M.score_counts(n)[0])


def nlmask(e):
    return np.isin(e["reg"], list(NL))


def mult2(e, d0, s, rule):
    d = e["drel"]
    m = np.maximum(0.0, 1.0 - s * np.clip((np.nan_to_num(d, nan=0.0) - d0) / (1.0 - d0), 0.0, 1.0))
    off = np.isnan(d) | ~np.isin(e["grp"], ["sp_dfine", "sp_dfine_nl"])
    if rule:
        off |= ~nlmask(e)
    return e["conf"] * np.where(off, 1.0, m)


def curves2(d0, s, rule, tau=None, split=False):
    def conf_fn(e):
        return mult2(e, d0, s, rule) if s > 0 else e["conf"]
    def keep_fn(e, cf):
        if tau is None:
            return np.ones(len(cf), bool)
        sel = e["grp"] == ("sp_dfine_nl" if split else "sp_dfine")
        return ~sel | (cf >= tau)
    ex = ex_base
    if split:   # B2: sp_dfine in lidar-free regions → own group
        ex = {c: dict(d, grp=np.where((d["grp"] == "sp_dfine") & nlmask(d), "sp_dfine_nl", d["grp"]).astype(object)) for c, d in ex_base.items()}
    return curves_from(ex, conf_fn=conf_fn, keep_fn=keep_fn)


D0S = (0.0, 0.25, 0.5); SS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5); TAUS = [round(0.20 + 0.05 * k, 2) for k in range(11)]
PTS = {}
def get(fam, rule, tau, d0, s):
    if s == 0:
        d0 = 0.0
    k = (fam, rule, tau, d0, s)
    if k not in PTS:
        split = fam == "B" and rule
        fixed = frozenset({"sp_dfine_nl" if split else "sp_dfine"}) if tau is not None else frozenset()
        PTS[k] = point(curves2(d0, s, rule, tau, split), fixed)
        log(k, f"gate honest {PTS[k]['gate']:.6f}")
    return PTS[k]


base_pt = get("A", False, None, 0.0, 0.0)
assert abs(base_pt["gate"] - sB) < 1e-9
for fam, rule in (("A", False), ("A", True), ("B", False), ("B", True)):
    for tau in ([None] if fam == "A" else TAUS):
        for d0 in D0S:
            for s in SS:
                get(fam, rule, tau, d0, s)
pickle.dump(PTS, open(OUT2 / "points.pkl", "wb"))


def f1(a):   # a [..., 3] tp fp ngt
    tp, fp, g = a[..., 0], a[..., 1], a[..., 2]
    den = 2 * tp + fp + (g - tp)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def sc(per_cls_tuples):   # [C,3]
    f = f1(per_cls_tuples); has = per_cls_tuples[:, 2] > 0
    return float((W * f * has).sum() / (W * has).sum())


FLOOR = {c: (-0.002 if c in ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha") else -0.005) for c in CLS}
log("floors", FLOOR)
KI = CLS.index(KT)


def select(fam, rule, constrained):
    combos = []
    for tau in ([None] if fam == "A" else TAUS):
        for d0 in D0S:
            for so in SS:
                for sk in SS:
                    if so == 0 and sk == 0 and d0 != 0.0:
                        continue
                    combos.append((tau, d0, so, sk))
    out = dict(chosen=[], taus=[], admissible=[], train_score=[])
    held = np.zeros((len(CLS), len(REGS), 3), np.int32)
    for fi in range(10):
        bt, bh = base_pt["tr"][fi, 0], base_pt["tr"][fi, 1]
        bf = f1(bt); bhs = sc(bh)
        best, nadm = None, 0
        cands = combos + ([("base",)] if constrained and fam == "B" else [])
        for cb in cands:
            if cb == ("base",):
                pk = po = base_pt
            else:
                tau, d0, so, sk = cb
                po = get(fam, rule, tau, d0, so); pk = get(fam, rule, tau, d0, sk)
            t = po["tr"][fi].copy(); t[:, KI] = pk["tr"][fi][:, KI]
            s_tr = sc(t[0])
            if constrained:
                df = f1(t[0]) - bf
                ok = all(df[ci] >= FLOOR[c] - 1e-12 for ci, c in enumerate(CLS)) and sc(t[1]) - bhs >= -1e-12
                if not ok:
                    continue
            nadm += 1
            key = (round(s_tr, 12),) + ((1,) if cb == ("base",) else (0, -cb[3], -cb[2], cb[1], -cb[0] if cb[0] is not None else 0))
            if best is None or key > best[0]:
                best = (key, cb, po, pk, s_tr)
        _, cb, po, pk, s_tr = best
        idx = [RI[r] for r in folds[fi]]
        held[:, idx] = po["held"][:, idx]; held[KI, idx] = pk["held"][KI, idx]
        t = {g: dict(d) for g, d in po["ft"][fi].items()}
        for g, d in pk["ft"][fi].items():
            if KT in d:
                t.setdefault(g, {})[KT] = d[KT]
        if cb != ("base",) and cb[0] is not None:
            t["sp_dfine_nl" if rule else "sp_dfine"] = {c: cb[0] for c in CL5}
        out["chosen"].append(None if cb == ("base",) else dict(tau=cb[0], d0=cb[1], s_o=cb[2], s_kt=cb[3]))
        out["taus"].append(t); out["admissible"].append(nadm); out["train_score"].append(s_tr)
    hc = {c: {REGS[ri]: held[ci, ri].tolist() for ri in range(len(REGS)) if held[ci, ri].any() or held[ci, ri, 2]} for ci, c in enumerate(CLS)}
    out["gate_nested"] = M.score_counts(hc)[0]
    for nm, sl in (("new34", set(T["new"])), ("nolidar", set(T["optical"]))):
        out["gate_d_" + nm] = M.score_counts(hc, sl)[0] - M.score_counts(nB, sl)[0]
    out["rule"] = rule; out["family"] = fam; out["constrained"] = constrained
    log(fam, rule, constrained, "chosen", out["chosen"], "adm", out["admissible"], f"nested gate {out['gate_nested']:.6f} Δ {out['gate_nested']-sB:+.6f}",
        f"Δnew34 {out['gate_d_new34']:+.4f} Δnolidar {out['gate_d_nolidar']:+.4f}")
    return out


VAR = {"A1": ("A", False, False), "A2": ("A", True, False), "A3": ("A", False, True),
       "B1": ("B", False, False), "B2": ("B", True, False), "B3": ("B", False, True)}
res["variants"] = {k: select(*v) for k, v in VAR.items()}
res["taus_base"] = [fin(t) for t in ftB]
pickle.dump(dict(res=res, ex_base=ex_base_l), open(OUT2 / "prep2.pkl", "wb"))
json.dump(res, open(OUT2 / "prep2_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
