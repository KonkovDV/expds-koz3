"""Аудит: где теряются эталоны по (класс, markup_type эталона) на OOF-стеке arch.
Приближение метрики для кругов: pred засчитан, если его центр внутри GT-полигона
(для finds_points: ≤2 м). Жадно по уверенности, один-к-одному, по региону и классу.
Пороги — из /workspace/rerank/stack_arch_base.json (как в сдаче).
Выход: JSON со счётчиками ceil (любой кандидат), kept_cover, tp_greedy, и симуляция
'dup' — второй копии каждого оставленного кургана (одна и та же точка, ещё один TP,
если внутри есть второй ещё не сопоставленный эталон)."""
import json, math, os, sys, glob, collections
import numpy as np
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
FAM = '/workspace/koz3_shared/bot-10/work/fam'
base = json.load(open('/workspace/rerank/stack_arch_base.json'))['per_class']
TH = {(c, g): v['threshold'] for c, d in base.items() for g, v in d['groups'].items()}
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson'))['features']
G = collections.defaultdict(list)  # (region, class) -> [(geom, mt)]
for f in gt:
    p = f['properties']; g = shape(f['geometry'])
    G[(p['region_name'], p['class_name'])].append((g, p['markup_type']))
del gt
trees = {k: STRtree([g for g, _ in v]) for k, v in G.items()}
S = collections.Counter()
def key(*a): return '|'.join(map(str, a))
for fn in sorted(glob.glob(FAM + '/*.jsonl')):
    P = collections.defaultdict(list)
    for line in open(fn):
        f = json.loads(line); p = f['properties']; c = p['class_name']
        grp = p.get('threshold_group') or p['markup_type']
        g = f['geometry']
        if g['type'] == 'Point': x, y = g['coordinates'][:2]
        else:
            ring = np.asarray(g['coordinates'][0] if g['type'] == 'Polygon' else g['coordinates'][0][0])
            x, y = ring[:-1, 0].mean(), ring[:-1, 1].mean()
        thr = TH.get((c, grp))
        kept = thr is not None and p['confidence'] >= thr
        P[(p['region_name'], c)].append((p['confidence'], x, y, grp, kept))
    for (reg, c), preds in P.items():
        gl = G.get((reg, c), [])
        if not gl: 
            for conf, x, y, grp, kept in preds:
                if kept: S[key('fp_noGT', c, grp)] += 1
            continue
        tr = trees[(reg, c)]
        scale = math.cos(math.atan(math.sinh(preds[0][2] / 6378137.0)))
        def hits(x, y):
            pt = Point(x, y)
            if c == 'finds_points':
                r = 2.0 / scale
                return [int(i) for i in tr.query(pt.buffer(r)) if gl[int(i)][0].centroid.distance(pt) <= r]
            return [int(i) for i in tr.query(pt) if gl[int(i)][0].contains(pt)]
        ceil = set(); kc = set()
        preds.sort(key=lambda t: -t[0])
        matched = set(); tpg = collections.Counter(); dup_extra = 0
        for conf, x, y, grp, kept in preds:
            h = hits(x, y)
            ceil.update(h)
            if not kept: continue
            kc.update(h)
            free = [i for i in h if i not in matched]
            if free:
                matched.add(free[0]); S[key('tp', c, grp)] += 1
                S[key('tp_gtmt', c, gl[free[0]][1])] += 1
                free2 = [i for i in free[1:]]
                if free2: S[key('dup_possible', c, grp)] += 1
            else:
                S[key('fp', c, grp)] += 1
                if h: S[key('fp_onTP', c, grp)] += 1   # попал в уже взятый эталон
        for i, (g, mt) in enumerate(gl):
            S[key('gt', c, mt)] += 1
            if i in ceil: S[key('ceil', c, mt)] += 1
            if i in kc: S[key('kept_cover', c, mt)] += 1
    print(os.path.basename(fn), file=sys.stderr, flush=True)
json.dump(dict(S), open('/workspace/koz3_shared/audit/work/layer_recall.json', 'w'), ensure_ascii=False, indent=0, sort_keys=True)
