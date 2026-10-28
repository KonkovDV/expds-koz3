"""Подготовка к офлайн-оценке калибровки и WBF (bot-2): один проход по участкам.

Для каждого объекта стека: класс, уверенность, группа порога, markup, участок, порядок в файле.
Пары (объект, эталон), прошедшие официальный _match_quality (cov ≥ 0.5 и точка внутри),
с признаком «эталон того же участка» (официальная метрика режет по участкам; build_curves в репо — нет).
Кластеры для WBF: пересекающиеся (площадь > 0) объекты одного (участок, markup, класс)
из РАЗНЫХ групп порога, union-find.
Читает work/stack_regions/*.jsonl (потоковая разбивка стека), эталон — целиком (он мал).
"""
import argparse, json, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/expds-koz3")
ap.add_argument("--regions-dir", default="/workspace/koz3_shared/bot-2/work/stack_regions")
ap.add_argument("--gt", default="/workspace/rerank/gt_train_v2.geojson")
ap.add_argument("--shm", default="/workspace/koz3_shared/bot-2/work/shm")
ap.add_argument("--out", default="/workspace/koz3_shared/bot-2/work/fusion_prep.pkl")
a = ap.parse_args()
sys.path.insert(0, a.repo)
import shapely
from shapely.strtree import STRtree
import expds.metric as M
M._SHM_CACHE_DIR = Path(a.shm)
from expds.metric import load_official, _collect

mod = load_official(str(Path(a.repo) / "materials" / "compute_metrics.py"))
calc = mod.F1Calculator()
CLASSES = list(calc.target_classes)
SEP = "\x00"

gtfc = json.loads(Path(a.gt).read_bytes())
n_sub = sum(1 for f in gtfc["features"] if (f.get("properties") or {}).get("sub_region_name"))
G = _collect(mod, calc, gtfc, True)
gt_geom = {c: [x[0] for x in G.get(c, [])] for c in CLASSES}
gt_reg = {c: [x[2] for x in G.get(c, [])] for c in CLASSES}
gt_tree = {c: STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in gt_geom[c]])
           if gt_geom[c] else None for c in CLASSES}
del gtfc
M._UTM_GEOM_CACHE.clear()

P = {c: defaultdict(list) for c in CLASSES}          # c -> поле -> список
pairs = {c: [] for c in CLASSES}                     # (local_idx, gt_idx, score, same_region)
n_cluster = 0; stats = defaultdict(int)
t0 = time.time()
files = sorted(Path(a.regions_dir).glob("*.jsonl"))
for fi, fp in enumerate(files):
    feats = [json.loads(l) for l in open(fp)]
    markup_of = {}
    for f in feats:
        p = f["properties"]; markup_of[p["_ord"]] = str(p.get("markup_type") or "")
        p["region_name"] = f'{p.get("region_name", "default_source")}{SEP}{p["_ord"]}'
    out = _collect(mod, calc, {"features": feats}, False)
    M._UTM_GEOM_CACHE.clear(); del feats
    for c in CLASSES:
        items = out.get(c, [])
        if not items:
            continue
        base = len(P[c]["conf"])
        geoms = [x[0] for x in items]
        for g, conf, rr, grp in items:
            reg, o = rr.rsplit(SEP, 1); o = int(o)
            P[c]["conf"].append(conf); P[c]["ord"].append(o); P[c]["region"].append(reg)
            P[c]["group"].append(grp); P[c]["markup"].append(markup_of[o])
        # пары с эталоном (по всем участкам, отмечаем «свой» участок)
        if gt_tree[c] is not None:
            qp, qg = gt_tree[c].query(geoms)
            for i, j in zip(qp.tolist(), qg.tolist()):
                pg = geoms[i]
                if pg.geom_type != "Polygon":
                    continue
                ok, s = calc._match_quality(pg, gt_geom[c][j])
                if ok:
                    same = gt_reg[c][j] == P[c]["region"][base + i]
                    pairs[c].append((base + i, j, float(s), same)); stats["pairs_same" if same else "pairs_cross"] += 1
        # кластеры из разных групп в пределах (markup) — участок и класс уже общие
        cl = np.full(len(items), -1, dtype=np.int64)
        grp_arr = np.array([x[3] for x in items], dtype=object)
        mk_arr = np.array([markup_of[int(x[2].rsplit(SEP, 1)[1])] for x in items], dtype=object)
        if len(set(grp_arr.tolist())) > 1:
            garr = np.array(geoms, dtype=object)
            tree = STRtree(garr)
            qi, qj = tree.query(garr, predicate="intersects")
            m = (qi < qj) & (grp_arr[qi] != grp_arr[qj]) & (mk_arr[qi] == mk_arr[qj])
            qi, qj = qi[m], qj[m]
            if len(qi):
                area = shapely.area(shapely.intersection(garr[qi], garr[qj]))
                keep = area > 0
                qi, qj = qi[keep], qj[keep]
                parent = np.arange(len(items))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]; x = parent[x]
                    return x
                for i, j in zip(qi.tolist(), qj.tolist()):
                    ri, rj = find(i), find(j)
                    if ri != rj: parent[ri] = rj
                roots = np.array([find(i) for i in range(len(items))])
                cnt = np.bincount(roots, minlength=len(items))
                multi = np.flatnonzero(cnt[roots] > 1)
                if len(multi):
                    uniq, inv = np.unique(roots[multi], return_inverse=True)
                    cl[multi] = n_cluster + inv; n_cluster += len(uniq)
                    stats[f"clustered|{c}"] += len(multi)
        P[c]["cluster"].extend(cl.tolist())
    if fi % 20 == 0:
        print(fi, len(files), fp.stem, round(time.time() - t0), flush=True)

res = {"classes": CLASSES, "n_gt_sub_region": n_sub, "stats": dict(stats), "n_cluster": n_cluster,
       "gt_region": gt_reg, "P": {}, "pairs": {}}
for c in CLASSES:
    d = P[c]
    res["P"][c] = {"conf": np.array(d["conf"], float), "ord": np.array(d["ord"], np.int64),
                   "region": np.array(d["region"], object), "group": np.array(d["group"], object),
                   "markup": np.array(d["markup"], object), "cluster": np.array(d["cluster"], np.int64)}
    pr = pairs[c]
    res["pairs"][c] = {"i": np.array([x[0] for x in pr], np.int64), "j": np.array([x[1] for x in pr], np.int64),
                       "s": np.array([x[2] for x in pr], float), "same": np.array([x[3] for x in pr], bool)}
pickle.dump(res, open(a.out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
print("done", round(time.time() - t0), dict(stats), n_cluster, "gt_sub", n_sub)
