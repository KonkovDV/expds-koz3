"""Честная офлайн-оценка: калибровка уверенности по группам (OOF по семьям) и WBF только для
кластеров из РАЗНЫХ групп порога (bot-2, п. 5 REPORT.md). Без переобучения моделей.

Схема на каждый фолд семей F (фолды — из base_honest.pkl):
  1) калибраторы (класс, группа) учатся на участках вне F; метка — попадание объекта
     при исходных уверенностях (официальное жадное сопоставление);
  2) калибровка применяется ко всем объектам; WBF (если включён) сливает кластеры;
  3) жадное сопоставление заново (порядок по новым уверенностям);
  4) пороги групп подбираются sweep_groups на участках вне F и применяются к F.
Режимы пар: region — эталон только своего участка (как официальная метрика, режущая по участкам);
            global — как expds.metric.build_curves (эталон любого участка того же класса).
Запуск: OMP_NUM_THREADS=2 nice -n 10 /workspace/rerank/venv/bin/python scripts/fusion_calib_eval.py
"""
import argparse, json, pickle, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/expds-koz3")
ap.add_argument("--prep", default="/workspace/koz3_shared/bot-2/work/fusion_prep.pkl")
ap.add_argument("--base", default="/workspace/rerank/work/base_honest.pkl")
ap.add_argument("--modes", default="global,region")
ap.add_argument("--variants", default="base,platt,iso,wbf_raw,platt_wbf,iso_wbf,platt_max")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--out", default="/workspace/koz3_shared/bot-2/work/fusion_calib_eval.json")
a = ap.parse_args()
sys.path.insert(0, a.repo)
from expds.metric import ClassCurve, sweep_groups, curve_where, score_counts, paired_bootstrap_counts
from expds.naming import site_family
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

R = pickle.load(open(a.prep, "rb"))
B = pickle.load(open(a.base, "rb"))
folds = B["folds"]; every = set().union(*folds)
ROOT = Path(a.repo)
boxes = json.loads((ROOT / "notes" / "las_boxes_59.json").read_text(encoding="utf-8"))
gt_regions_all = {r for c in R["classes"] for r in R["gt_region"][c]}
optical = {r for r in gt_regions_all if r} - set(boxes)
new = {l.strip() for l in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
CL = R["classes"]
MIN_POS = 10

def greedy(c, ceff, active, mode):
    P, pr = R["P"][c], R["pairs"][c]
    m = pr["same"] if mode == "region" else np.ones(len(pr["i"]), bool)
    pi, pj, ps = pr["i"][m], pr["j"][m], pr["s"][m]
    keep = active[pi]; pi, pj, ps = pi[keep], pj[keep], ps[keep]
    hit = np.zeros(len(P["conf"]), bool)
    if not len(pi):
        return hit
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
        if bj >= 0:
            matched.add(bj); hit[preds[k]] = True
    return hit

def curve(c, ceff, active, hit):
    P = R["P"][c]
    idx = np.flatnonzero(active)
    o = idx[np.lexsort((P["ord"][idx], -P["conf"][idx], -ceff[idx]))]
    gtc = Counter(R["gt_region"][c])
    return ClassCurve(conf=ceff[o].astype(float), hit=hit[o], pred_region=P["region"][o].tolist(),
                      n_gt=sum(gtc.values()), gt_regions=gtc, pred_group=P["group"][o].tolist())

def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))

def calibrate(c, kind, train_mask, label):
    P = R["P"][c]; out = P["conf"].copy(); info = {}
    for g in np.unique(P["group"]):
        gm = P["group"] == g; tm = gm & train_mask
        y = label[tm]; npos = int(y.sum()); nneg = int(len(y) - npos)
        if npos < MIN_POS or nneg < MIN_POS:
            info[g] = "identity"; continue
        x = P["conf"][tm]
        if kind == "platt":
            lr = LogisticRegression(C=1e4, max_iter=1000).fit(logit(x)[:, None], y)
            out[gm] = lr.predict_proba(logit(P["conf"][gm])[:, None])[:, 1]
        else:
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y)
            out[gm] = ir.predict(P["conf"][gm])
        info[g] = "fit"
    return out, info

T_CACHE = {}
def groups_present(c):
    if c not in T_CACHE:
        P = R["P"][c]
        key = np.char.add(np.char.add(P["region"].astype(str), "|"), P["markup"].astype(str))
        T = {}
        for k, g in zip(key.tolist(), P["group"].tolist()): T.setdefault(k, set()).add(g)
        T_CACHE[c] = np.array([len(T[k]) for k in key.tolist()])
    return T_CACHE[c]

def fuse(c, ceff, how):
    """Кластеры из разных групп -> один объект (геометрия лучшего члена, его группа)."""
    P = R["P"][c]; cl = P["cluster"]; active = np.ones(len(cl), bool); ce = ceff.copy()
    idx = np.flatnonzero(cl >= 0)
    if not len(idx): return ce, active, 0
    o = idx[np.lexsort((P["ord"][idx], -ceff[idx], cl[idx]))]          # по кластеру, лучший первым
    starts = np.r_[0, np.flatnonzero(np.diff(cl[o])) + 1]; ends = np.r_[starts[1:], len(o)]
    T = groups_present(c)
    for s, e in zip(starts.tolist(), ends.tolist()):
        mem = o[s:e]; best = mem[0]
        if how == "max":
            val = ceff[best]
        else:
            gmax = {}
            for m_ in mem.tolist():
                g = P["group"][m_]; gmax[g] = max(gmax.get(g, -np.inf), ceff[m_])
            n = len(gmax); t = max(int(T[best]), n)
            val = float(np.mean(list(gmax.values()))) * min(n, t) / t
        ce[best] = val; active[mem[1:]] = False
    return ce, active, len(starts)

def run(variant, mode):
    cal = {"platt": "platt", "iso": "isotonic", "platt_wbf": "platt", "iso_wbf": "isotonic", "platt_max": "platt"}.get(variant)
    fus = {"wbf_raw": "wbf", "platt_wbf": "wbf", "iso_wbf": "wbf", "platt_max": "max"}.get(variant)
    base_hit = {c: greedy(c, R["P"][c]["conf"], np.ones(len(R["P"][c]["conf"]), bool), mode) for c in CL}
    counts = {c: {} for c in CL}; meta = {"fused_clusters": {}, "calib": {}}
    static = cal is None
    cache = None
    for fi, held in enumerate(folds):
        train = every - held
        if not static or cache is None:
            curves = {}
            for c in CL:
                P = R["P"][c]; ceff = P["conf"].copy()
                if cal:
                    tm = np.fromiter((r in train for r in P["region"]), bool, len(P["region"]))
                    ceff, info = calibrate(c, cal, tm, base_hit[c]); meta["calib"].setdefault(c, Counter()).update(info.values())
                active = np.ones(len(ceff), bool)
                if fus:
                    ceff, active, nf = fuse(c, ceff, fus); meta["fused_clusters"][c] = nf
                hit = greedy(c, ceff, active, mode) if (cal or fus) else base_hit[c]
                curves[c] = curve(c, ceff, active, hit)
            cache = curves
        curves = cache
        res = sweep_groups({c: curve_where(cv, train) for c, cv in curves.items()})
        taus = {}
        for c, v in res.items():
            for g, gv in v["groups"].items(): taus.setdefault(g, {})[c] = gv["threshold"]
        for c, cv in curves.items():
            hc = curve_where(cv, held); per = counts[c]
            for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
            if hc.conf.size == 0: continue
            tau = np.array([taus.get(g, {}).get(c, np.inf) for g in hc.pred_group])
            kept = hc.conf >= tau
            for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
                per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    meta["calib"] = {c: dict(v) for c, v in meta["calib"].items()}
    return counts, meta

out = {"n_new": len(new), "n_optical": len(optical), "base_pkl": B["scores"], "modes": {}}
t0 = time.time()
for mode in a.modes.split(","):
    res = {}; cnts = {}
    for v in a.variants.split(","):
        counts, meta = run(v, mode); cnts[v] = counts
        s = [score_counts(counts)[0], score_counts(counts, new)[0], score_counts(counts, optical)[0]]
        per = score_counts(counts)[1]
        res[v] = {"honest": s[0], "new34": s[1], "no_lidar": s[2], "meta": meta,
                  "per_class": {c: {k: round(x, 5) for k, x in d.items()} for c, d in per.items()}}
        if v != "base":
            b = res["base"]
            res[v].update(d_honest=s[0] - b["honest"], d_new34=s[1] - b["new34"], d_no_lidar=s[2] - b["no_lidar"])
            bs = paired_bootstrap_counts(counts, cnts["base"], site_family, n=a.boot)
            res[v]["boot"] = bs
            res[v]["honest_accept"] = bool(res[v]["d_honest"] >= 0.002 and bs.get("p_gt0", 0) >= 0.8
                                           and res[v]["d_new34"] >= -0.001 and res[v]["d_no_lidar"] >= -0.001)
        print(mode, v, [round(x, 6) for x in s], {k: res[v][k] for k in ("d_honest", "honest_accept") if k in res[v]},
              res[v]["boot"].get("p_gt0") if "boot" in res[v] else "", round(time.time() - t0), flush=True)
    if mode == "global":
        res["base"]["pkl_counts_equal"] = all(
            {r: list(x) for r, x in cnts["base"][c].items()} == {r: list(x) for r, x in B["counts"][c].items()}
            for c in CL if c in B["counts"])
    out["modes"][mode] = res
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1, default=str)
print("done", round(time.time() - t0))
