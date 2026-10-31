"""bot-8: откуда 939/2825 (kurgan_relabel_oracle, base) против 918/2770 (fusion_calib, region).

По каждому участку из bot-2/work/by_region_003 (курганы conf>=0.003, порядок стека):
  A  stack — объекты в порядке стека; сверка: hits == build_curves(scope="site") этого участка;
  B  loc   — объекты в порядке sorted(loc) как в kurgan_relabel_oracle (ключ (группа, cx, cy, n)),
             затем тот же жадный алгоритм.
Затем honest_counts с фолдами base_honest.pkl, суммарные TP/FP/n_gt для D и I.
"""
import argparse, json, pickle, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/koz3_shared/bot-8/repo")
ap.add_argument("--by-region", default="/workspace/koz3_shared/bot-2/work/by_region_003")
ap.add_argument("--gt", default="/workspace/rerank/gt_train_v2.geojson")
ap.add_argument("--base", default="/workspace/rerank/work/base_honest.pkl")
ap.add_argument("--out", default="/workspace/koz3_shared/bot-8/bench/kurgan_site_check.json")
a = ap.parse_args()
sys.path.insert(0, a.repo)
import expds.metric as M
M._SHM_CACHE_DIR = Path("/workspace/koz3_shared/bot-8/shm_cache")
from shapely.strtree import STRtree

D, I = "kurgany_povrezhdennye", "kurgany_tselye"
mod = M.load_official(str(Path(a.repo) / "materials" / "compute_metrics.py"))
calc = mod.F1Calculator()
gt_by = {}
for f in json.loads(Path(a.gt).read_bytes())["features"]:
    p = f.get("properties") or {}
    if mod.normalize_class_name(p.get("class_name", "")) in (D, I):
        gt_by.setdefault(p.get("region_name"), []).append(f)
regions = sorted(set(gt_by) | {p.stem for p in Path(a.by_region).glob("*.jsonl")})


def greedy(p, g):   # копия kurgan_relabel_oracle.greedy (стабильная сортировка по conf)
    order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True)
    hits = np.zeros(len(p), dtype=bool)
    if p and g:
        tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in g])
        pg = [p[i][0] for i in order]
        qp, qg = tree.query(pg)
        if len(qp):
            sp = np.flatnonzero(np.diff(qp)) + 1
            used = set()
            for rank, cand in zip(qp[np.r_[0, sp]], np.split(qg, sp)):
                x = pg[int(rank)]
                if x.geom_type != "Polygon":
                    continue
                best, bj = float("-inf"), -1
                for j in sorted(int(k) for k in cand):
                    if j in used:
                        continue
                    ok, s = calc._match_quality(x, g[j])
                    if ok and s > best + 1e-9:
                        best, bj = s, j
                if bj >= 0:
                    used.add(bj); hits[int(rank)] = True
    return np.array([p[i][1] for i in order]), hits, [p[i][2] for i in order], [p[i][3] for i in order]


acc = {v: {c: [[], [], [], []] for c in (D, I)} for v in ("stack", "loc")}
ngt = {c: Counter() for c in (D, I)}
mism = 0; ties_regions = Counter(); t0 = time.time()
for ri, reg in enumerate(regions):
    fp = Path(a.by_region) / f"{reg}.jsonl"
    feats = [json.loads(l) for l in open(fp, encoding="utf-8")] if fp.exists() else []
    P = M._collect(mod, calc, {"features": feats}, False)
    G = M._collect(mod, calc, {"features": gt_by.get(reg, [])}, True)
    g = {c: [x[0] for x in G.get(c, [])] for c in (D, I)}
    for c in (D, I):
        ngt[c][reg] += len(g[c])
    loc, dup = {}, Counter()
    for c in (D, I):
        for geom, conf, r, grp in P.get(c, []):
            ct = geom.centroid; k0 = (grp, round(ct.x, 1), round(ct.y, 1))
            n = dup[(k0, c)]; dup[(k0, c)] += 1
            loc.setdefault(k0 + (n,), {})[c] = (geom, conf, r, grp)
    by_loc = {D: [], I: []}
    for k in sorted(loc):
        for c, v in loc[k].items():
            by_loc[c].append(v)
    # сверка A с build_curves(scope="site") на этом участке
    M._UTM_GEOM_CACHE.clear()
    cur = M.build_curves({"type": "FeatureCollection", "features": feats},
                         {"type": "FeatureCollection", "features": gt_by.get(reg, [])},
                         str(Path(a.repo) / "materials" / "compute_metrics.py"), scope="site")
    for c in (D, I):
        for v, lst in (("stack", P.get(c, [])), ("loc", by_loc[c])):
            cf, h, rr, gg = greedy(lst, g[c])
            A = acc[v][c]; A[0].append(cf); A[1].append(h); A[2] += rr; A[3] += gg
            if v == "stack":
                bc = cur.get(c)
                if bc is None:
                    mism += int(len(cf) > 0)
                elif not (np.array_equal(bc.conf, cf) and np.array_equal(bc.hit, h)):
                    mism += 1
            else:
                prev = acc["stack"][c][1][-1]
                if int(prev.sum()) != int(h.sum()) or not np.array_equal(prev, h):
                    ties_regions[c] += 1
    del feats, P, G, cur, loc, by_loc
    if ri % 20 == 0:
        print(ri, len(regions), reg, round(time.time() - t0), flush=True)

base = pickle.load(open(a.base, "rb"))
res = {"mismatch_vs_build_curves_site": mism, "regions_where_loc_order_changes_hits": dict(ties_regions)}
for v in ("stack", "loc"):
    curves = {}
    for c in (D, I):
        A = acc[v][c]
        conf = np.concatenate(A[0]); hit = np.concatenate(A[1]); o = np.argsort(-conf, kind="stable")
        curves[c] = M.ClassCurve(conf=conf[o], hit=hit[o], pred_region=[A[2][i] for i in o],
                                 n_gt=sum(ngt[c].values()), gt_regions=Counter(ngt[c]),
                                 pred_group=[A[3][i] for i in o])
    kc, _ = M.honest_counts(curves, base["folds"])
    full = dict(base["counts"]); full.update(kc)
    tot, _ = M.score_counts(full)
    res[v] = {"honest_tp_fp_ngt": {c: np.array(list(kc[c].values())).sum(0).tolist() for c in (D, I)},
              "all_tp": {c: int(curves[c].hit.sum()) for c in (D, I)},
              "n_pred": {c: int(curves[c].conf.size) for c in (D, I)},
              "score_with_pkl_other_classes": tot}
    print(v, res[v], flush=True)
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
print("done", round(time.time() - t0))
