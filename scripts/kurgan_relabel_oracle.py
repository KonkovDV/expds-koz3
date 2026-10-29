"""Сколько балла даёт правильный подкласс кургана (повреждённый 30 / целый 20) на OOF-стеке.

Честно: официальное жадное сопоставление (копия expds.metric.build_curves) по участкам,
пороги групп — honest_counts с фолдами из base_honest.pkl (leave-family-out).
Сценарии:
  base        — стек как есть (проверка: должен совпасть с курганными счётчиками base_honest.pkl);
  oracle_max  — как oracle, но уверенность = max по копиям места (не зависит от --floor);
  oracle      — у каждого места, совпавшего (cov>=0.5 и точка внутри) с эталонным курганом
                любого подкласса, оставляем одну копию верного подкласса (уверенность копии
                этого подкласса, иначе имеющейся); прочие места не трогаем;
  sim_soft_a  — классификатор с AUC=a (бинормальная модель): у близнецов conf_c *= p_c;
  sim_hard_a  — то же, но остаётся одна копия argmax (с учётом априорной доли 0.7).
Запуск: OMP_NUM_THREADS=2 nice -n 10 /workspace/rerank/venv/bin/python scripts/kurgan_relabel_oracle.py
"""
import argparse, json, os, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/expds-koz3")
ap.add_argument("--by-region", default="/workspace/koz3_shared/bot-2/work/by_region_003")
ap.add_argument("--gt", default="/workspace/rerank/gt_train_v2.geojson")
ap.add_argument("--base", default="/workspace/rerank/work/base_honest.pkl")
ap.add_argument("--shm", default="/workspace/koz3_shared/bot-2/work/shm")
ap.add_argument("--aucs", default="0.59,0.65,0.75,0.85,0.95")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--prior", type=float, default=0.7)
ap.add_argument("--out", default="/workspace/koz3_shared/bot-2/work/kurgan_relabel_oracle.json")
a = ap.parse_args()
sys.path.insert(0, a.repo)
import expds.metric as M
M._SHM_CACHE_DIR = Path(a.shm)          # кэш не в общий /dev/shm
from expds.metric import load_official, _collect, ClassCurve, honest_counts, score_counts, paired_bootstrap_counts
from expds.naming import site_family
from shapely.strtree import STRtree

D, I = "kurgany_povrezhdennye", "kurgany_tselye"
# группы, где лидар выдаёт обе копии в одной точке (см. work/by_region/_summary.json: all-счётчики D≈I)
TWIN_GROUPS = {"li", "li_dev", "li_mosaic"}
mod = load_official(str(Path(a.repo) / "materials" / "compute_metrics.py"))
calc = mod.F1Calculator()

gt_by = defaultdict(list)
for f in json.loads(Path(a.gt).read_bytes())["features"]:
    p = f.get("properties") or {}
    if mod.normalize_class_name(p.get("class_name", "")) in (D, I):
        gt_by[p.get("region_name")].append(f)
regions = sorted(set(gt_by) | {p.stem for p in Path(a.by_region).glob("*.jsonl")})

def greedy(p, g):
    """p: [(geom, conf, reg, grp)], g: [geom] -> (conf, hit, reg, grp) по убыванию conf (как build_curves)."""
    order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True)
    hits = np.zeros(len(p), dtype=bool)
    if p and g:
        tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in g])
        pg_ord = [p[i][0] for i in order]
        q_pred, q_gt = tree.query(pg_ord)
        if len(q_pred):
            splits = np.flatnonzero(np.diff(q_pred)) + 1
            matched = set()
            for rank, cand in zip(q_pred[np.r_[0, splits]], np.split(q_gt, splits)):
                pg = pg_ord[int(rank)]
                if pg.geom_type != "Polygon":
                    continue
                best, bj = float("-inf"), -1
                for j in sorted(int(k) for k in cand):
                    if j in matched:
                        continue
                    ok, s = calc._match_quality(pg, g[j])
                    if ok and s > best + 1e-9:
                        best, bj = s, j
                if bj >= 0:
                    matched.add(bj); hits[int(rank)] = True
    return (np.array([p[i][1] for i in order]), hits, [p[i][2] for i in order], [p[i][3] for i in order])

aucs = [float(x) for x in a.aucs.split(",")]
scen = ["base", "oracle", "oracle_max"] + [f"sim_{m}_{x}_s{s}" for x in aucs for s in range(a.seeds) for m in ("soft", "hard")]
acc = {s: {c: [[], [], [], []] for c in (D, I)} for s in scen}
ngt = {c: Counter() for c in (D, I)}
stats = Counter()
t0 = time.time()
for ri, reg in enumerate(regions):
    fp = Path(a.by_region) / f"{reg}.jsonl"
    feats = [json.loads(l) for l in open(fp)] if fp.exists() else []
    P = _collect(mod, calc, {"features": feats}, False)
    G = _collect(mod, calc, {"features": gt_by.get(reg, [])}, True)
    g = {c: [x[0] for x in G.get(c, [])] for c in (D, I)}
    for c in (D, I):
        ngt[c][reg] += len(g[c])
    trees = {c: STRtree(g[c]) if g[c] else None for c in (D, I)}
    # места: ключ (группа, центроид 0.1 м)
    loc = {}
    dup = Counter()
    items = []
    for c in (D, I):
        for geom, conf, r, grp in P.get(c, []):
            ct = geom.centroid
            k0 = (grp, round(ct.x, 1), round(ct.y, 1))
            n = dup[(k0, c)]; dup[(k0, c)] += 1        # дубли одного класса в одной точке не теряем
            loc.setdefault(k0 + (n,), {})[c] = (geom, conf, r, grp)
    rng_keys = sorted(loc)
    truth = {}
    for k in rng_keys:
        geom = next(iter(loc[k].values()))[0]
        best, bc = float("-inf"), None
        for c in (D, I):
            if trees[c] is None or geom.geom_type != "Polygon":
                continue
            for j in trees[c].query(geom):
                ok, s = calc._match_quality(geom, g[c][int(j)])
                if ok and s > best:
                    best, bc = s, c
        truth[k] = bc
        stats["loc"] += 1; stats["twin"] += len(loc[k]) == 2
        stats[f"truth_{bc}"] += 1
        if bc and len(loc[k]) == 2: stats[f"twin_truth_{bc}"] += 1
    for s in scen:
        lists = {D: [], I: []}
        if s == "base":
            for k in rng_keys:
                for c, v in loc[k].items(): lists[c].append(v)
        elif s == "oracle":
            for k in rng_keys:
                v, t = loc[k], truth[k]
                if t is None:
                    for c, x in v.items(): lists[c].append(x)
                else:
                    x = v.get(t)
                    if x is None and next(iter(v.values()))[3] not in TWIN_GROUPS:
                        x = next(iter(v.values()))        # одноклассовая группа: просто меняем подкласс
                    if x is not None:                    # у близнецовой группы копия ниже --floor => отброшена
                        lists[t].append(x)
        elif s == "oracle_max":                         # место -> одна копия верного подкласса с max conf копий
            for k in rng_keys:
                v, t = loc[k], truth[k]
                if t is None:
                    for c, x in v.items(): lists[c].append(x)
                else:
                    x = max(v.values(), key=lambda q: q[1]); lists[t].append(x)
        else:
            _, m, x, sd = s.split("_"); x = float(x); sd = int(sd[1:])
            dd = np.sqrt(2) * norm.ppf(min(x, 0.9999))
            rng = np.random.default_rng([sd, ri])
            ys = rng.random(len(rng_keys)); zs = rng.standard_normal(len(rng_keys))
            for k, u, z in zip(rng_keys, ys, zs):
                v, t = loc[k], truth[k]
                if len(v) < 2:
                    for c, q in v.items(): lists[c].append(q)
                    continue
                y = (t == D) if t else (u < a.prior)
                zz = z + (dd / 2 if y else -dd / 2)
                lo = dd * zz                                    # log-odds при равных априорных
                if m == "soft":
                    pd = 1 / (1 + np.exp(-lo))
                    for c, q in v.items():
                        lists[c].append((q[0], q[1] * (pd if c == D else 1 - pd), q[2], q[3]))
                else:
                    c = D if lo + np.log(a.prior / (1 - a.prior)) > 0 else I
                    lists[c].append(v[c])
        for c in (D, I):
            cf, h, rr, gg = greedy(lists[c], g[c])
            A = acc[s][c]; A[0].append(cf); A[1].append(h); A[2] += rr; A[3] += gg
    if ri % 10 == 0:
        print(ri, len(regions), reg, len(feats), round(time.time() - t0), flush=True)

base = pickle.load(open(a.base, "rb"))
folds = base["folds"]
res = {"stats": dict(stats), "base_pkl_score": base["scores"][0], "scenarios": {}}
cnts = {}
for s in scen:
    curves = {}
    for c in (D, I):
        A = acc[s][c]
        conf = np.concatenate(A[0]) if A[0] else np.zeros(0); hit = np.concatenate(A[1]) if A[1] else np.zeros(0, bool)
        o = np.argsort(-conf, kind="stable")
        curves[c] = ClassCurve(conf=conf[o], hit=hit[o], pred_region=[A[2][i] for i in o], n_gt=sum(ngt[c].values()),
                               gt_regions=Counter(ngt[c]), pred_group=[A[3][i] for i in o])
    kc, info = honest_counts(curves, folds)
    full = dict(base["counts"]); full.update(kc); cnts[s] = full
    tot, per = score_counts(full)
    sm = {c: np.array(list(kc[c].values())).sum(0).tolist() for c in (D, I)}
    res["scenarios"][s] = {"score": tot, "delta_vs_pkl": tot - base["scores"][0], "kurgan_counts_tp_fp_ngt": sm,
                           "f1": {c: per[c] for c in (D, I)} if isinstance(per, dict) else None}
    print(s, round(tot, 5), sm, flush=True)
for s in ("oracle", "oracle_max") + tuple(x for x in scen if x.endswith("_s0")):
    try:
        res["scenarios"][s]["bootstrap_vs_base"] = paired_bootstrap_counts(cnts[s], cnts["base"], site_family)
    except Exception as e:
        res["scenarios"][s]["bootstrap_vs_base"] = repr(e)
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1, default=str)
print("done", round(time.time() - t0))
