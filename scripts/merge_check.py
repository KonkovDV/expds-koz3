""", merge-check: сколько стоит post.merge_across_sources (expds/post.py:42-79) на честном A.

Стек табло (scripts/scoreboard.py build_stack) НЕ проходит merge_within_markup, а сдача проходит
(expds/watchdog.py:66-70: пороги -> merge). Здесь слияние моделируется поверх честных порогов:
в каждом внешнем фолде пороги (группа, класс) — с train-семей (как base_honest.pkl), на held —
объекты выше порога, склейка как в post.merge_across_sources, затем жадное сопоставление по всем
кандидатам класса (как build_curves) с заменой склеенных членов на части объединения.
Источник (source = имя файла) в стеке не сохранён -> прокси по листингу S3 (work/merge/s3_sources.json):
склейка возможна только на участках, где у markup >1 входного файла;
  U: ребро между любыми двумя пересекающимися (площадь>0) объектами (верхняя оценка);
  L: только внутри одной группы порога (дубли одной модели с разных снимков).
Режим сопоставления global — основной (база 0.314615), region — диагностика (0.312067).
Запуск: OMP_NUM_THREADS=1 nice -n 10 /workspace/rerank/venv/bin/python scripts/merge_check.py
"""
import argparse, json, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/expds-koz3")
ap.add_argument("--prep", default="/workspace/koz3_shared/bot-2/work/fusion_prep.pkl")
ap.add_argument("--base", default="/workspace/rerank/work/base_honest.pkl")
ap.add_argument("--subset", default="/workspace/koz3_shared/bot-2/work/merge/subset.pkl")
ap.add_argument("--gt", default="/workspace/rerank/gt_train_v2.geojson")
ap.add_argument("--variants", default="U,L")
ap.add_argument("--full", action="store_true", help="LOFO + 5 раскладок + бутстреп (для приёмки)")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--out", default="/workspace/koz3_shared/bot-2/work/merge/merge_check.json")
a = ap.parse_args()
sys.path.insert(0, a.repo)
import shapely
from shapely.ops import unary_union
from shapely.strtree import STRtree
import expds.metric as M
from expds.metric import (ClassCurve, sweep_groups, curve_where, score_counts, paired_bootstrap_counts,
                          family_folds, load_official, _collect)
from expds.naming import site_family
from expds.constants import CLASS_WEIGHTS
from expds.post import polygons_of

t0 = time.time()
mod = load_official(str(Path(a.repo) / "materials" / "compute_metrics.py")); calc = mod.F1Calculator()
R = pickle.load(open(a.prep, "rb")); B = pickle.load(open(a.base, "rb")); S = pickle.load(open(a.subset, "rb"))
CL = R["classes"]; folds_pkl = B["folds"]; every = set().union(*folds_pkl)
folds10 = family_folds(every, site_family, 10)
assert sorted(map(sorted, folds10)) == sorted(map(sorted, folds_pkl))
ROOT = Path(a.repo)
boxes = set(json.loads((ROOT / "notes" / "las_boxes_59.json").read_text(encoding="utf-8")))
gt_all = {r for c in CL for r in R["gt_region"][c]}
optical = {r for r in gt_all if r} - boxes
new = {l.strip() for l in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
SEP = "\x00"

# эталон — тем же _collect, что в fusion_calib_prep.py (индексы j совпадают)
gtfc = json.loads(Path(a.gt).read_bytes()); G = _collect(mod, calc, gtfc, True); del gtfc
M._UTM_GEOM_CACHE.clear()
gt_geom = {c: [x[0] for x in G.get(c, [])] for c in CL}
assert all(len(gt_geom[c]) == len(R["gt_region"][c]) for c in CL)
gt_reg = R["gt_region"]
gt_tree = {c: STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in gt_geom[c]])
           if gt_geom[c] else None for c in CL}
gt_area = {c: np.array([g.area for g in gt_geom[c]]) for c in CL}

# объекты подмножества в метрической СК (как у метрики)
feats = S["features"]; mk_of = {}
for f in feats:
    p = f["properties"]; mk_of[p["_ord"]] = str(p.get("markup_type"))
    p["region_name"] = f'{p["region_name"]}{SEP}{p["_ord"]}'
SUB = _collect(mod, calc, {"features": feats}, False); del feats, S
M._UTM_GEOM_CACHE.clear()
idx_of = {c: {int(o): i for i, o in enumerate(R["P"][c]["ord"])} for c in CL}
sub = {}   # c -> arrays
for c in CL:
    it = SUB.get(c, [])
    if not it: continue
    ords = np.array([int(x[2].rsplit(SEP, 1)[1]) for x in it])
    sub[c] = {"geom": np.array([x[0] for x in it], object), "ord": ords,
              "idx": np.array([idx_of[c][o] for o in ords]), "conf": np.array([x[1] for x in it], float),
              "region": np.array([x[2].rsplit(SEP, 1)[0] for x in it], object),
              "group": np.array([x[3] for x in it], object), "markup": np.array([mk_of[o] for o in ords], object)}
    assert np.allclose(R["P"][c]["conf"][sub[c]["idx"]], sub[c]["conf"])
print("subset", {c: len(v["ord"]) for c, v in sub.items()}, round(time.time() - t0), flush=True)

# кандидатные рёбра (пересечение площадью > 0) внутри (участок, markup, класс) — один раз
EDGES = {}
for c, s in sub.items():
    tree = STRtree(s["geom"]); qi, qj = tree.query(s["geom"], predicate="intersects")
    m = (qi < qj) & (s["region"][qi] == s["region"][qj]) & (s["markup"][qi] == s["markup"][qj])
    qi, qj = qi[m], qj[m]
    if len(qi):
        ar = shapely.area(shapely.intersection(s["geom"][qi], s["geom"][qj])); k = ar > 0; qi, qj = qi[k], qj[k]
    EDGES[c] = (qi, qj, s["group"][qi] == s["group"][qj])
print("edges", {c: len(v[0]) for c, v in EDGES.items()}, round(time.time() - t0), flush=True)

def random_family_folds(regions, k, seed):      # копия repo/scripts/honest_bootstrap.py
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    names = sorted(fams); rng = np.random.default_rng(seed); lab = rng.permutation(len(names)) % k
    return [set().union(*(fams[n] for n, l in zip(names, lab) if l == i)) for i in range(k)]

def pairs_for(c, geoms, regions):
    """Пары (новый объект, эталон) по официальному _match_quality."""
    out = []
    if gt_tree[c] is None or not geoms: return out
    qp, qg = gt_tree[c].query(geoms)
    for i, j in zip(qp.tolist(), qg.tolist()):
        if geoms[i].geom_type != "Polygon": continue
        ok, s_ = calc._match_quality(geoms[i], gt_geom[c][j])
        if ok: out.append((i, j, float(s_), gt_reg[c][j] == regions[i]))
    return out

def greedy_hits(conf, ordk, pi, pj, ps):
    hit = np.zeros(len(conf), bool)
    if not len(pi): return hit
    o = np.lexsort((pj, pi)); pi, pj, ps = pi[o], pj[o], ps[o]
    st = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]; en = np.r_[st[1:], len(pi)]
    preds = pi[st]; order = np.lexsort((ordk[preds], -conf[preds])); matched = set()
    for k in order.tolist():
        best, bj = -np.inf, -1
        for q in range(st[k], en[k]):
            j = int(pj[q])
            if j in matched: continue
            if ps[q] > best + 1e-9: best, bj = ps[q], j
        if bj >= 0: matched.add(bj); hit[preds[k]] = True
    return hit

def base_pairs(c, mode):
    pr = R["pairs"][c]; m = pr["same"] if mode == "region" else np.ones(len(pr["i"]), bool)
    return pr["i"][m], pr["j"][m], pr["s"][m]

BASE_HIT = {}
def curves_for(mode):
    out = {}
    for c in CL:
        P = R["P"][c]; hit = greedy_hits(P["conf"], P["ord"], *base_pairs(c, mode)); BASE_HIT[(mode, c)] = hit
        o = np.lexsort((P["ord"], -P["conf"])); gtc = Counter(R["gt_region"][c])
        out[c] = ClassCurve(conf=P["conf"][o].astype(float), hit=hit[o], pred_region=P["region"][o].tolist(),
                            n_gt=sum(gtc.values()), gt_regions=gtc, pred_group=P["group"][o].tolist())
    return out

def taus_of(curves, regions):
    res = sweep_groups({c: curve_where(cv, regions) for c, cv in curves.items()}); t = {}
    for c, v in res.items():
        for g, gv in v["groups"].items(): t.setdefault(g, {})[c] = gv["threshold"]
    return t

def merge_groups(c, held, tau, variant, rule=None):
    """Группы склейки на held при порогах tau. Возвращает список списков локальных индексов sub[c]."""
    s = sub[c]; qi, qj, same_g = EDGES[c]
    th = np.array([tau.get(g, {}).get(c, np.inf) for g in s["group"]])
    kept = (s["conf"] >= th) & np.array([r in held for r in s["region"]])
    m = kept[qi] & kept[qj]
    if variant == "L": m &= same_g
    qi, qj = qi[m], qj[m]
    if not len(qi): return []
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(qi.tolist(), qj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj
    gr = defaultdict(list)
    for x in list(parent): gr[find(x)].append(x)
    out = [sorted(v) for v in gr.values() if len(v) > 1]
    if rule is not None: out = [g for g in out if rule(c, g)]
    return out

def add_counts(counts, c, held, tau, conf, region, group, hit, gt_regions):
    per = counts[c]
    for r, n in gt_regions.items():
        if r in held: per.setdefault(r, [0, 0, 0])[2] += int(n)
    th = np.array([tau.get(g, {}).get(c, np.inf) for g in group])
    hm = np.array([r in held for r in region]); kept = hm & (conf >= th)
    for r, h in zip(region[kept], hit[kept]): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1

GDIAG = []   # (variant, mode, class, region, size, n_parts, n_gt_cov, n_gt_members, d_tp)
def evaluate(curves, folds, mode, variants, diag=False, rule=None):
    out = {v: {c: {} for c in CL} for v in ["base"] + variants}
    gtc = {c: Counter(R["gt_region"][c]) for c in CL}
    for held in folds:
        tau = taus_of(curves, every - held)
        for c in CL:
            P = R["P"][c]; reg = P["region"]; hit0 = BASE_HIT[(mode, c)]
            add_counts(out["base"], c, held, tau, P["conf"], reg, P["group"], hit0, gtc[c])
            for v in variants:
                grs = merge_groups(c, held, tau, v, rule) if c in sub else []
                if not grs:
                    add_counts(out[v], c, held, tau, P["conf"], reg, P["group"], hit0, gtc[c]); continue
                s = sub[c]; drop = np.zeros(len(P["conf"]), bool); ng, nc, nr, ngp, no = [], [], [], [], []
                info = []
                for g in grs:
                    drop[s["idx"][g]] = True
                    best = g[int(np.argmax(s["conf"][g]))]
                    parts = polygons_of(unary_union(list(s["geom"][g])))
                    info.append((g, len(ng), len(parts)))
                    for part in parts:
                        ng.append(part); nc.append(s["conf"][best]); nr.append(s["region"][best])
                        ngp.append(s["group"][best]); no.append(s["ord"][best])
                keep = np.flatnonzero(~drop); K = len(keep)
                conf = np.r_[P["conf"][keep], nc]; ordk = np.r_[P["ord"][keep], no]
                region = np.r_[reg[keep], np.array(nr, object)]; group = np.r_[P["group"][keep], np.array(ngp, object)]
                pi, pj, ps = base_pairs(c, mode); remap = -np.ones(len(P["conf"]), np.int64); remap[keep] = np.arange(K)
                mm = remap[pi] >= 0; pi, pj, ps = remap[pi[mm]], pj[mm], ps[mm]
                npairs = [x for x in pairs_for(c, ng, nr) if mode == "global" or x[3]]
                if npairs:
                    pi = np.r_[pi, [K + x[0] for x in npairs]]; pj = np.r_[pj, [x[1] for x in npairs]]
                    ps = np.r_[ps, [x[2] for x in npairs]]
                hit = greedy_hits(conf, ordk, pi, pj, ps)
                add_counts(out[v], c, held, tau, conf, region, group, hit, gtc[c])
                if diag:
                    pr = R["pairs"][c]; same = pr["same"]
                    for g, k0, npart in info:
                        u = unary_union(list(s["geom"][g])); r0 = s["region"][g[0]]
                        cov = 0
                        if gt_tree[c] is not None:
                            for j in gt_tree[c].query(u).tolist():
                                if gt_reg[c][j] == r0 and gt_area[c][j] > 0 and u.intersection(gt_geom[c][j]).area >= 0.5 * gt_area[c][j]:
                                    cov += 1
                                elif gt_reg[c][j] == r0 and gt_geom[c][j].geom_type == "Point" and u.distance(gt_geom[c][j]) <= calc.r_tolerance:
                                    cov += 1
                        idxs = set(s["idx"][g].tolist()); sel = np.isin(pr["i"], list(idxs)) & same
                        n_mem = len(set(pr["j"][sel].tolist()))
                        tp_before = int(hit0[s["idx"][g]].sum()); tp_after = int(hit[K + k0:K + k0 + npart].sum())
                        GDIAG.append((v, mode, c, r0, len(g), npart, cov, n_mem, tp_after - tp_before,
                                      sorted(set(s["group"][g].tolist()))))
    return out

def sc(counts): return [score_counts(counts)[0], score_counts(counts, new)[0], score_counts(counts, optical)[0]]
def wdelta(ca, cb):
    _, pa = score_counts(ca); _, pb = score_counts(cb)
    den = sum(CLASS_WEIGHTS.get(c, 0.0) for c in pa if pa[c]["n_gt"] > 0)
    return {c: CLASS_WEIGHTS.get(c, 0.0) * (pa[c]["f1"] - pb[c]["f1"]) / den for c in pa if pa[c]["n_gt"] > 0}

VARS = a.variants.split(",")
res = {"modes": {}, "subset_sizes": {c: int(len(v["ord"])) for c, v in sub.items()},
       "edges": {c: int(len(v[0])) for c, v in EDGES.items()}}
RULE = None   # правило варианта объявляется в EXPERIMENTS.md до запуска (см. --rule)
for mode in ("global", "region"):
    curves = curves_for(mode)
    c10 = evaluate(curves, folds10, mode, VARS, diag=True, rule=RULE)
    s = {v: sc(c10[v]) for v in c10}
    print(mode, "honest10", {v: [round(x, 6) for x in s[v]] for v in s}, round(time.time() - t0), flush=True)
    Mo = {"variants": {}}
    if mode == "global":
        Mo["base_counts_equal_pkl"] = all({r: list(x) for r, x in c10["base"][c].items()} ==
                                          {r: list(x) for r, x in B["counts"][c].items()} for c in CL if c in B["counts"])
    for v in c10:
        r = {"honest10": s[v][0], "new34": s[v][1], "no_lidar": s[v][2]}
        if v != "base":
            d = s[v][0] - s["base"][0]; wd = wdelta(c10[v], c10["base"]); r["delta"] = d
            r["wdelta"] = {k: round(x, 6) for k, x in wd.items() if abs(x) > 1e-9}
            bs = paired_bootstrap_counts(c10[v], c10["base"], site_family, n=a.boot); r["boot"] = bs
        Mo["variants"][v] = r
    res["modes"][mode] = Mo
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1, default=float)

# диагностика групп
res["groups"] = [list(map(lambda x: x if not isinstance(x, (np.integer, np.floating)) else x.item(), g)) for g in GDIAG]
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1, default=float)
print("done", round(time.time() - t0), flush=True)
