"""E3: только отдельные пороги для участков без лидара; объявлено в
koz3_shared/EXPERIMENTS.md (28.10.2026 20:39:18 МСК).
E3n — вложенный выбор по классам (внутр. family_folds(train,5)), решающий; E3p — без выбора, диагностика.
Сопоставление global (как materials/compute_metrics.py: пулинг по всем источникам), region — диагностика.
Запуск: OMP_NUM_THREADS=2 nice -n 10 /workspace/rerank/venv/bin/python scripts/fusion_e3_eval.py
"""
import argparse, json, pickle, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/expds-koz3")
ap.add_argument("--prep", default="/workspace/koz3_shared/bot-2/work/fusion_prep.pkl")
ap.add_argument("--base", default="/workspace/rerank/work/base_honest.pkl")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--inner", type=int, default=5)
ap.add_argument("--out", default="/workspace/koz3_shared/bot-2/work/fusion_e3_eval.json")
a = ap.parse_args()
sys.path.insert(0, a.repo)
from expds.metric import ClassCurve, sweep_groups, curve_where, score_counts, paired_bootstrap_counts, family_folds
from expds.naming import site_family
from expds.constants import CLASS_WEIGHTS

R = pickle.load(open(a.prep, "rb")); B = pickle.load(open(a.base, "rb"))
CL = R["classes"]; folds_pkl = B["folds"]; every = set().union(*folds_pkl)
folds10 = family_folds(every, site_family, 10)
FOLDS_EQUAL = sorted(map(sorted, folds10)) == sorted(map(sorted, folds_pkl))
ROOT = Path(a.repo)
boxes = set(json.loads((ROOT / "notes" / "las_boxes_59.json").read_text(encoding="utf-8")))
NLR = {r for r in every if r not in boxes}
gt_all = {r for c in CL for r in R["gt_region"][c]}
optical = {r for r in gt_all if r} - boxes
new = {l.strip() for l in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}

def random_family_folds(regions, k, seed):      # копия repo/scripts/honest_bootstrap.py
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    names = sorted(fams); rng = np.random.default_rng(seed); lab = rng.permutation(len(names)) % k
    return [set().union(*(fams[n] for n, l in zip(names, lab) if l == i)) for i in range(k)]

def greedy(c, mode):
    P, pr = R["P"][c], R["pairs"][c]; ceff = P["conf"]
    m = pr["same"] if mode == "region" else np.ones(len(pr["i"]), bool)
    pi, pj, ps = pr["i"][m], pr["j"][m], pr["s"][m]
    hit = np.zeros(len(ceff), bool)
    if not len(pi): return hit
    o = np.lexsort((pj, pi)); pi, pj, ps = pi[o], pj[o], ps[o]
    starts = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]; ends = np.r_[starts[1:], len(pi)]
    preds = pi[starts]; order = np.lexsort((P["ord"][preds], -ceff[preds]))
    matched = set()
    for k in order.tolist():
        best, bj = -np.inf, -1
        for q in range(starts[k], ends[k]):
            j = int(pj[q])
            if j in matched: continue
            if ps[q] > best + 1e-9: best, bj = ps[q], j
        if bj >= 0: matched.add(bj); hit[preds[k]] = True
    return hit

def curves_for(mode):
    out = {}
    for c in CL:
        P = R["P"][c]; hit = greedy(c, mode)
        o = np.lexsort((P["ord"], -P["conf"])); gtc = Counter(R["gt_region"][c])
        out[c] = ClassCurve(conf=P["conf"][o].astype(float), hit=hit[o], pred_region=P["region"][o].tolist(),
                            n_gt=sum(gtc.values()), gt_regions=gtc, pred_group=P["group"][o].tolist())
    return out

def taus_of(curves, regions):
    res = sweep_groups({c: curve_where(cv, regions) for c, cv in curves.items()}); t = {}
    for c, v in res.items():
        for g, gv in v["groups"].items(): t.setdefault(g, {})[c] = gv["threshold"]
    return t

def add_counts(counts, curves, held, tb, tn, sel):
    """Лидарные участки: tb. Без лидара: tn для классов из sel (если нет — tb). unseen = drop."""
    for c, cv in curves.items():
        hc = curve_where(cv, held); per = counts[c]
        for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
        if hc.conf.size == 0: continue
        use_nl = c in sel
        tau = np.array([(tn.get(g, {}).get(c, tb.get(g, {}).get(c, np.inf)) if (use_nl and r in NLR)
                         else tb.get(g, {}).get(c, np.inf)) for g, r in zip(hc.pred_group, hc.pred_region)])
        kept = hc.conf >= tau
        for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
            per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1

def nested_select(curves, train):
    cb = {c: {} for c in CL}; cn = {c: {} for c in CL}
    for ih in family_folds(train, site_family, a.inner):
        it = train - ih; tb = taus_of(curves, it); tn = taus_of(curves, it & NLR)
        add_counts(cb, curves, ih, tb, tn, set()); add_counts(cn, curves, ih, tb, tn, set(CL))
    pb, pn = score_counts(cb)[1], score_counts(cn)[1]
    return {c for c in CL if c in pn and pn[c]["n_gt"] > 0 and pn[c]["f1"] > pb[c]["f1"]}

def evaluate(curves, folds, nested=True):
    out = {v: {c: {} for c in CL} for v in ("base", "E3n", "E3p")}; sels = Counter()
    for held in folds:
        train = every - held; tb = taus_of(curves, train); tn = taus_of(curves, train & NLR)
        add_counts(out["base"], curves, held, tb, tn, set())
        add_counts(out["E3p"], curves, held, tb, tn, set(CL))
        if nested:
            sel = nested_select(curves, train); sels.update(sel)
            add_counts(out["E3n"], curves, held, tb, tn, sel)
    return out, dict(sels)

def sc(counts): return [score_counts(counts)[0], score_counts(counts, new)[0], score_counts(counts, optical)[0]]
def wdelta(ca, cb):
    _, pa = score_counts(ca); _, pb = score_counts(cb)
    den = sum(CLASS_WEIGHTS.get(c, 0.0) for c in pa if pa[c]["n_gt"] > 0)
    return {c: CLASS_WEIGHTS.get(c, 0.0) * (pa[c]["f1"] - pb[c]["f1"]) / den for c in pa if pa[c]["n_gt"] > 0}

t0 = time.time()
res = {"declared": "koz3_shared/work/EXPERIMENTS.md E3, 28.10.2026 20:39:18 MSK", "folds10_equal_pkl": FOLDS_EQUAL,
       "n_regions": len(every), "n_nl_regions": len(NLR), "modes": {}}
for mode in ("global", "region"):
    curves = curves_for(mode)
    c10, sel10 = evaluate(curves, folds10)
    s = {v: sc(c10[v]) for v in c10}
    print(mode, "honest10", {v: [round(x, 6) for x in s[v]] for v in s}, "sel", sel10, round(time.time() - t0), flush=True)
    M = {"sel_counts_honest10": sel10, "variants": {}}
    if mode == "global":
        M["base_counts_equal_pkl"] = all({r: list(x) for r, x in c10["base"][c].items()} ==
                                         {r: list(x) for r, x in B["counts"][c].items()} for c in CL if c in B["counts"])
        cl, sell = evaluate(curves, family_folds(every, site_family, None))
        print(mode, "lofo", {v: round(score_counts(cl[v])[0], 6) for v in cl}, round(time.time() - t0), flush=True)
        cs = []
        for sd in range(a.seeds):
            x, _ = evaluate(curves, random_family_folds(every, 10, sd)); cs.append(x)
            print(mode, "seed", sd, {v: round(score_counts(x[v])[0], 6) for v in x}, round(time.time() - t0), flush=True)
        M["sel_counts_lofo"] = sell
    for v in ("base", "E3n", "E3p"):
        r = {"honest10": s[v][0], "new34": s[v][1], "no_lidar": s[v][2],
             "per_class": {c: {k: round(x, 5) for k, x in d.items()} for c, d in score_counts(c10[v])[1].items()}}
        if v != "base":
            bs = paired_bootstrap_counts(c10[v], c10["base"], site_family, n=a.boot)
            d = s[v][0] - s["base"][0]; wd = wdelta(c10[v], c10["base"])
            heavy = {c: wd.get(c, 0.0) for c in ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha")}
            r.update(d_honest10=d, d_new34=s[v][1] - s["base"][1], d_no_lidar=s[v][2] - s["base"][2], boot=bs,
                     weighted_class_delta=wd, heavy=heavy)
            if mode == "global":
                dl = score_counts(cl[v])[0] - score_counts(cl["base"])[0]
                sdl = [score_counts(x[v])[0] - score_counts(x["base"])[0] for x in cs]
                P1 = d >= 0.002; P2 = bs.get("p_gt0", 0) >= 0.85; P3 = dl > 0 and sum(x > 0 for x in sdl) >= 4
                P4 = all(x >= -0.0005 for x in heavy.values()) and s[v][1] >= s["base"][1] - 0.001 and s[v][2] >= s["base"][2] - 0.001
                r.update(lofo=score_counts(cl[v])[0], lofo_base=score_counts(cl["base"])[0], d_lofo=dl,
                         seed_base=[score_counts(x["base"])[0] for x in cs], seed_deltas=sdl,
                         seed_min=min(sdl), seed_std=float(np.std(sdl, ddof=1)), seeds_negative=[i for i, x in enumerate(sdl) if x < 0],
                         P1=bool(P1), P2=bool(P2), P3=bool(P3), P4=bool(P4), accept_P1_P4=bool(P1 and P2 and P3 and P4),
                         P5_package_only=bool(d < 0.006))
        M["variants"][v] = r
    res["modes"][mode] = M
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1, default=str)
print("done", round(time.time() - t0))
