"""E2 (bot-2): заранее объявленные варианты из koz3_shared/bot-2/EXPERIMENTS.md (28.10.2026 20:14 МСК).

Сопоставление только внутри своего участка (режим region). Калибратор isotonic.
  nl_iso_wbf       — iso + WBF только на участках без лидара; отдельные пороги для них; лидарные = база;
  all_iso_wbf_noli — iso везде + WBF везде, группы markup li для WBF — одна группа (не сливаются);
  nl_iso           — как nl_iso_wbf, без WBF.
Приёмка — правило bot-10 П1–П4 (/workspace/koz3_shared/EXPERIMENTS.md); П5 — в отчёте.
Запуск: OMP_NUM_THREADS=2 nice -n 10 /workspace/rerank/venv/bin/python scripts/fusion_e2_eval.py
"""
import argparse, json, pickle, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/expds-koz3")
ap.add_argument("--prep", default="/workspace/koz3_shared/bot-2/work/fusion_prep.pkl")
ap.add_argument("--base", default="/workspace/rerank/work/base_honest.pkl")
ap.add_argument("--variants", default="base,nl_iso_wbf,all_iso_wbf_noli,nl_iso")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--out", default="/workspace/koz3_shared/bot-2/work/fusion_e2_eval.json")
a = ap.parse_args()
sys.path.insert(0, a.repo)
from expds.metric import ClassCurve, sweep_groups, curve_where, score_counts, paired_bootstrap_counts, family_folds
from expds.naming import site_family
from expds.constants import CLASS_WEIGHTS
from sklearn.isotonic import IsotonicRegression

MODE = "region"; MIN_POS = 10
R = pickle.load(open(a.prep, "rb")); B = pickle.load(open(a.base, "rb"))
CL = R["classes"]; folds10 = B["folds"]; every = set().union(*folds10)
ROOT = Path(a.repo)
boxes = set(json.loads((ROOT / "notes" / "las_boxes_59.json").read_text(encoding="utf-8")))
gt_all = {r for c in CL for r in R["gt_region"][c]}
optical = {r for r in gt_all if r} - boxes
new = {l.strip() for l in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
NL = {c: np.array([r not in boxes for r in R["P"][c]["region"]]) for c in CL}

def random_family_folds(regions, k, seed):      # копия bot-10/repo/scripts/honest_bootstrap.py
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    names = sorted(fams); rng = np.random.default_rng(seed); lab = rng.permutation(len(names)) % k
    return [set().union(*(fams[n] for n, l in zip(names, lab) if l == i)) for i in range(k)]

def greedy(c, ceff, active):
    P, pr = R["P"][c], R["pairs"][c]
    m = pr["same"]; pi, pj, ps = pr["i"][m], pr["j"][m], pr["s"][m]
    keep = active[pi]; pi, pj, ps = pi[keep], pj[keep], ps[keep]
    hit = np.zeros(len(P["conf"]), bool)
    if not len(pi): return hit
    o = np.lexsort((pj, pi)); pi, pj, ps = pi[o], pj[o], ps[o]
    starts = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]; ends = np.r_[starts[1:], len(pi)]
    preds = pi[starts]
    order = np.lexsort((P["ord"][preds], -P["conf"][preds], -ceff[preds]))
    matched = set()
    for k in order.tolist():
        best, bj = -np.inf, -1
        for q in range(starts[k], ends[k]):
            j = int(pj[q])
            if j in matched: continue
            if ps[q] > best + 1e-9: best, bj = ps[q], j
        if bj >= 0: matched.add(bj); hit[preds[k]] = True
    return hit

def curve(c, ceff, active, hit, sel=None):
    P = R["P"][c]; m = active if sel is None else (active & sel)
    idx = np.flatnonzero(m)
    o = idx[np.lexsort((P["ord"][idx], -P["conf"][idx], -ceff[idx]))]
    gtc = Counter(R["gt_region"][c])
    return ClassCurve(conf=ceff[o].astype(float), hit=hit[o], pred_region=P["region"][o].tolist(),
                      n_gt=sum(gtc.values()), gt_regions=gtc, pred_group=P["group"][o].tolist())

def calibrate_iso(c, fit_mask, apply_mask, label):
    P = R["P"][c]; out = P["conf"].copy(); info = Counter()
    for g in np.unique(P["group"][apply_mask]):
        gm = P["group"] == g; tm = gm & fit_mask; am = gm & apply_mask
        y = label[tm]; npos = int(y.sum()); nneg = int(len(y) - npos)
        if npos < MIN_POS or nneg < MIN_POS: info["identity"] += 1; continue
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(P["conf"][tm], y)
        out[am] = ir.predict(P["conf"][am]); info["fit"] += 1
    return out, info

T_CACHE = {}
def groups_present(c):
    if c not in T_CACHE:
        P = R["P"][c]; key = [f"{r}|{m}" for r, m in zip(P["region"], P["markup"])]
        T = {}
        for k, g in zip(key, P["group"].tolist()): T.setdefault(k, set()).add(g)
        T_CACHE[c] = np.array([len(T[k]) for k in key])
    return T_CACHE[c]

def fuse(c, ceff, cl):
    P = R["P"][c]; active = np.ones(len(cl), bool); ce = ceff.copy()
    idx = np.flatnonzero(cl >= 0)
    if not len(idx): return ce, active, 0
    o = idx[np.lexsort((P["ord"][idx], -ceff[idx], cl[idx]))]
    starts = np.r_[0, np.flatnonzero(np.diff(cl[o])) + 1]; ends = np.r_[starts[1:], len(o)]
    T = groups_present(c)
    for s, e in zip(starts.tolist(), ends.tolist()):
        mem = o[s:e]; best = mem[0]; gmax = {}
        for m_ in mem.tolist():
            g = P["group"][m_]; gmax[g] = max(gmax.get(g, -np.inf), ceff[m_])
        n = len(gmax); t = max(int(T[best]), n)
        ce[best] = float(np.mean(list(gmax.values()))) * min(n, t) / t; active[mem[1:]] = False
    return ce, active, len(starts)

BASE_HIT = {c: greedy(c, R["P"][c]["conf"], np.ones(len(R["P"][c]["conf"]), bool)) for c in CL}
BASE_CURVES = {c: curve(c, R["P"][c]["conf"], np.ones(len(R["P"][c]["conf"]), bool), BASE_HIT[c]) for c in CL}

def taus_of(curves, regions):
    res = sweep_groups({c: curve_where(cv, regions) for c, cv in curves.items()}); t = {}
    for c, v in res.items():
        for g, gv in v["groups"].items(): t.setdefault(g, {})[c] = gv["threshold"]
    return t

def add_counts(counts, curves, held, taus):
    for c, cv in curves.items():
        hc = curve_where(cv, held); per = counts[c]
        if hc.conf.size == 0: continue
        tau = np.array([taus.get(g, {}).get(c, np.inf) for g in hc.pred_group])
        kept = hc.conf >= tau
        for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
            per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1

def add_gt(counts, held):
    for c in CL:
        for r, n in Counter(R["gt_region"][c]).items():
            if r in held: counts[c].setdefault(r, [0, 0, 0])[2] += int(n)

def evaluate(variant, folds):
    counts = {c: {} for c in CL}; meta = {"fused": Counter(), "calib": Counter()}
    for held in folds:
        train = every - held
        add_gt(counts, held)
        if variant == "base":
            add_counts(counts, BASE_CURVES, held, taus_of(BASE_CURVES, train)); continue
        curves = {}
        for c in CL:
            P = R["P"][c]; tm = np.fromiter((r in train for r in P["region"]), bool, len(P["region"]))
            if variant.startswith("nl_"):
                ceff, info = calibrate_iso(c, tm & NL[c], NL[c], BASE_HIT[c])
                cl = np.where(NL[c], P["cluster"], -1)
            else:
                ceff, info = calibrate_iso(c, tm, np.ones(len(tm), bool), BASE_HIT[c])
                cl = np.where(P["markup"] == "li", -1, P["cluster"])
            meta["calib"].update(info)
            active = np.ones(len(ceff), bool)
            if variant.endswith("wbf") or variant.endswith("wbf_noli"):
                ceff, active, nf = fuse(c, ceff, cl); meta["fused"][c] += nf
            curves[c] = (ceff, active, greedy(c, ceff, active))
        if variant.startswith("nl_"):
            lid = {r for r in every if r in boxes}
            nlr = every - lid
            # лидарные участки — как в базе (сырые уверенности, пороги базы по всем train)
            add_counts(counts, BASE_CURVES, held & lid, taus_of(BASE_CURVES, train))
            cv_nl = {c: curve(c, *curves[c], sel=NL[c]) for c in CL}
            add_counts(counts, cv_nl, held & nlr, taus_of(cv_nl, train & nlr))
        else:
            cvs = {c: curve(c, *curves[c]) for c in CL}
            add_counts(counts, cvs, held, taus_of(cvs, train))
    return counts, {k: dict(v) for k, v in meta.items()}

def scores(counts):
    return score_counts(counts)[0], score_counts(counts, new)[0], score_counts(counts, optical)[0]

def wdelta(ca, cb):
    _, pa = score_counts(ca); _, pb = score_counts(cb)
    den = sum(CLASS_WEIGHTS.get(c, 0.0) for c in pa if pa[c]["n_gt"] > 0)
    return {c: CLASS_WEIGHTS.get(c, 0.0) * (pa[c]["f1"] - pb[c]["f1"]) / den for c in pa if pa[c]["n_gt"] > 0}

lofo = family_folds(every, site_family, None)
seed_folds = [random_family_folds(every, 10, s) for s in range(a.seeds)]
out = {"declared": "koz3_shared/bot-2/EXPERIMENTS.md E2, 28.10.2026 20:14:08 MSK", "mode": MODE,
       "n_lofo_folds": len(lofo), "n_new": len(new), "n_optical": len(optical), "variants": {}}
t0 = time.time(); base = {}
for v in a.variants.split(","):
    c10, meta = evaluate(v, folds10)
    cl_, _ = evaluate(v, lofo)
    cs = [evaluate(v, f)[0] for f in seed_folds]
    s10 = scores(c10); sl = score_counts(cl_)[0]; ss = [score_counts(x)[0] for x in cs]
    r = {"honest10": s10[0], "new34": s10[1], "no_lidar": s10[2], "lofo": sl, "seeds": ss, "meta": meta,
         "per_class": {c: {k: round(x, 5) for k, x in d.items()} for c, d in score_counts(c10)[1].items()}}
    if v == "base":
        base = {"c10": c10, "s10": s10, "sl": sl, "ss": ss}
    else:
        bs = paired_bootstrap_counts(c10, base["c10"], site_family, n=a.boot)
        d = s10[0] - base["s10"][0]; wd = wdelta(c10, base["c10"])
        sd = [x - y for x, y in zip(ss, base["ss"])]
        heavy = {c: wd.get(c, 0.0) for c in ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha")}
        P1 = d >= 0.002; P2 = bs.get("p_gt0", 0) >= 0.85
        P3 = (sl - base["sl"] > 0) and sum(x > 0 for x in sd) >= 4
        P4 = all(x >= -0.0005 for x in heavy.values()) and s10[1] >= base["s10"][1] - 0.001 and s10[2] >= base["s10"][2] - 0.001
        r.update(d_honest10=d, d_new34=s10[1] - base["s10"][1], d_no_lidar=s10[2] - base["s10"][2], boot=bs,
                 d_lofo=sl - base["sl"], seed_deltas=sd, weighted_class_delta=wd, heavy=heavy,
                 P1=bool(P1), P2=bool(P2), P3=bool(P3), P4=bool(P4), accept_P1_P4=bool(P1 and P2 and P3 and P4),
                 P5_package_only=bool(d < 0.006))
    out["variants"][v] = r
    print(v, round(s10[0], 6), round(s10[1], 6), round(s10[2], 6), "lofo", round(sl, 6),
          {k: r[k] for k in ("d_honest10", "P1", "P2", "P3", "P4") if k in r}, round(time.time() - t0), flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1, default=str)
print("done", round(time.time() - t0))
