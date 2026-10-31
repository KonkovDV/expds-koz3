"""Задача 13 (диагностика, не эксперимент): какая доля GT kurgany_povrezhdennye в лидарных регионах не находится
принятым li-выходом стека A и сколько из пропущенных имеют «кольцевую» сигнатуру (яма в центре + вал).

Входы (read-only): /workspace/rerank/gt_train_v2.geojson, /workspace/rerank/stack_arch.geojson (стек A),
bot-4/work/cropverif/li_kurgan_candidates.npz (stack_idx ↔ регион/центр), bot-2/morph/morph_li_kurgan.parquet.
Матчинг ≈ официальный критерий сценария А (cov=area(P∩G)/min ≥ 0,5 И representative_point(P) внутри G),
но БЕЗ жадного 1:1 (верхняя оценка находок). Регионы 021/102/036 исключены.
Кольцо (пред-декларировано в EXPERIMENTS.md): r_peak ≥ 1,5 м И ring ≥ 0,15 м И hmax ≥ 0,10 м.
"""
import json, sys, time
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo/materials")
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo/scripts")
from compute_metrics import _to_utm, _polygon_coverage  # noqa
from split_stack_by_family import iter_features  # noqa
from shapely.geometry import shape
from shapely.strtree import STRtree
from pathlib import Path

SKIP = ("021", "102", "036")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
KEPT = {"ae": (0.99053, 0.802255), "li": (0.06415, 0.19422), "li_ae": (0.993061, 1.000001),
        "li_mosaic": (0.959005, 0.17042), "li_dev": (0.99131, 0.99957), "sp_focal": (0.364475, 0.372525),
        "sp_or": (0.364475, 0.372525)}
LI = ("li", "li_mosaic", "li_dev")
OUT = Path(sys.argv[1])

C = np.load("/workspace/koz3_shared/bot-4/work/cropverif/li_kurgan_candidates.npz")
regs = sorted(r for r in set(C["region"].tolist()) if not r.startswith(SKIP))
cidx = {int(s): i for i, s in enumerate(C["stack_idx"])}

gt = defaultdict(list)  # region -> [(cls, poly_utm, markup)]
for f in json.load(open("/workspace/rerank/gt_train_v2.geojson"))["features"]:
    p = f["properties"]
    if p["class_name"] in K and p["region_name"] in regs:
        gt[p["region_name"]].append((p["class_name"], _to_utm(shape(f["geometry"]), "auto"), p.get("markup_type")))
trees = {r: STRtree([g[1].buffer(15) for g in v]) for r, v in gt.items()}
print("gt", {k: len(v) for k, v in gt.items()}, flush=True)

preds = defaultdict(list)  # region -> [(idx, group, cls, conf, poly)]
t0 = time.time(); n = 0
for i, (ft, _raw) in enumerate(iter_features(Path("/workspace/rerank/stack_arch.geojson"))):
    p = ft["properties"]; r = p.get("region_name"); n += 1
    if p.get("class_name") not in K or r not in trees:
        continue
    g = _to_utm(shape(ft["geometry"]), "auto")
    if len(trees[r].query(g, predicate="intersects")) == 0:
        continue
    grp = p.get("threshold_group") or p.get("markup_type")
    preds[r].append((i, grp, p["class_name"], float(p.get("confidence") or 0), g))
print("stack", n, "kept_near", sum(len(v) for v in preds.values()), round(time.time() - t0), "s", flush=True)

import pyarrow.parquet as pq, csv
_cols = ["morph_r_peak", "morph_ring", "morph_hmax", "morph_h0", "morph_tpi2", "morph_tpi5", "morph_valid"]
_T = pq.read_table("/workspace/koz3_shared/bot-2/morph/morph_li_kurgan.parquet", columns=["stack_idx"] + _cols)
_need = {q[0] for v in preds.values() for q in v}
_si = _T.column("stack_idx").to_numpy()
_cv = {c: _T.column(c).to_numpy() for c in _cols}
M = {int(s): {c: float(_cv[c][k]) for c in _cols} for k, s in enumerate(_si) if int(s) in _need}
del _T, _cv, _si

rows = []; chk = []
for r in regs:
    P = preds.get(r, [])
    ptree = STRtree([q[4] for q in P]) if P else None
    for gi, (gcls, G, gmk) in enumerate(gt.get(r, [])):
        hits = []
        if ptree is not None:
            for j in ptree.query(G, predicate="intersects"):
                idx, grp, cls, conf, Pg = P[j]
                if _polygon_coverage(Pg, G) >= 0.5 and G.covers(Pg.representative_point()):
                    hits.append((idx, grp, cls, conf, Pg))
        def acc(h, same=True, li_only=False):
            idx, grp, cls, conf, _ = h
            if li_only and grp not in LI: return False
            if same and cls != gcls: return False
            if not same and cls == gcls: return False
            t = KEPT.get(grp)
            return t is not None and conf >= t[K.index(cls)]
        li_any = [h for h in hits if h[1] in LI or h[1] == "li_coarse"]
        best = max(li_any, key=lambda h: h[3]) if li_any else None
        # признаки рельефа: li-кандидат, лучше всего центрированный на GT (min расстояние центра до rep.point GT)
        rp = G.representative_point(); feat = {}
        cands = [h for h in li_any if h[0] in cidx]
        if cands:
            h = min(cands, key=lambda h: h[4].centroid.distance(G.centroid))
            ci = cidx[h[0]]; d = float(np.hypot(C["cx"][ci] - h[4].centroid.x, C["cy"][ci] - h[4].centroid.y))
            chk.append(d)
            if h[0] in M:
                feat = dict(M[h[0]]); feat["off_gt"] = float(h[4].centroid.distance(G.centroid))
        rows.append(dict(region=r, cls=gcls, markup=gmk, area=G.area, gx=rp.x, gy=rp.y,
                         acc_li=any(acc(h, li_only=True) for h in hits), acc_all=any(acc(h) for h in hits),
                         acc_cross=any(acc(h, same=False) for h in hits),
                         li_anyconf_same=any(h[2] == gcls for h in li_any), li_anyconf=bool(li_any),
                         best_li_conf=best[3] if best else np.nan, **feat))
keys = list(dict.fromkeys(k for r_ in rows for k in r_))
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("center check (cands cx vs stack centroid) max m:", round(max(chk), 3) if chk else None, "n", len(chk))
