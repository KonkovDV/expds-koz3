"""Точная честная оценка пост-фильтра «курган внутри городища» ПОСЛЕ порогов (по образцу bot-10 work6/twins_eval.py).
Пороги фолда — из кривых A на остальных семьях (bot-10 work6/taus_A.json); на удержанных: conf >= порога (unseen -> drop);
вариант KIG: удалить kurgany_* с центроидом внутри объединения прошедших порог gorodishcha того же региона
(PREREG_kig.md). KIGgt — диагностика-оракул с эталонными городищами. Сопоставление — expds.metric bot-10 по семьям.
    python kig_eval.py <out.pkl>"""
import sys, os, json, pickle, time, gc
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds import metric as M
from expds.naming import site_family
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import prepared
M._SHM_CACHE_DIR = Path('/proc/expds_no_cache')
OUT = sys.argv[1]
os.environ['EXPDS_MATCH_SCOPE'] = 'global'
W = Path('/workspace/koz3_shared/bot-10')
TAUS = json.load(open(W / 'work6/taus_A.json'))
mod = M.load_official(None); calc = mod.F1Calculator()
KCL = ('kurgany_povrezhdennye', 'kurgany_tselye')
mn = {}
for fl in TAUS['layouts'].values():
    for f in fl:
        for g, d in f['taus'].items():
            for c, t in d.items():
                mn[(g, c)] = min(mn.get((g, c), float('inf')), t)
feats = {}; t = time.time(); n = 0
for p in sorted((W / 'work/fam').glob('*.jsonl')):
    lst = []
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            f = json.loads(line); pr = f['properties']
            t0 = mn.get((M.threshold_group(pr), pr.get('class_name')))
            if t0 is None or float(pr.get('confidence', 0)) < t0: continue
            f['_g'] = M.threshold_group(pr); lst.append(f)
    feats[p.stem] = lst; n += len(lst)
print('prefilter', n, round(time.time() - t), 's', flush=True); gc.collect()
gt_by_fam = defaultdict(list); GTGOR = defaultdict(list); GTK = defaultdict(list)
for f in json.load(open('/workspace/rerank/gt_train_v2.geojson', encoding='utf-8'))['features']:
    pr = f['properties']; r = str(pr.get('region_name', 'default_source'))
    gt_by_fam[site_family(r)].append(f)
    if pr.get('class_name') == 'gorodishcha': GTGOR[r].append(shape(f['geometry']).buffer(0))
    if pr.get('class_name') in KCL: GTK[r].append(shape(f['geometry']))
GTGOR = {r: prepared.prep(unary_union(v)) for r, v in GTGOR.items()}
GTKU = {r: prepared.prep(unary_union([g.buffer(0) for g in v])) for r, v in GTK.items()}
UNIT = {}
for members in json.load(open(W / 'work/family_clusters.json', encoding='utf-8'))['clusters'].values():
    for m in members: UNIT[m] = '+'.join(sorted(members))
MEM = defaultdict(list)
for fam in set(gt_by_fam) | set(feats): MEM[UNIT.get(fam, fam)].append(fam)
GTC = {u: M._collect(mod, calc, {'type': 'FeatureCollection', 'features': [f for m in ms for f in gt_by_fam.get(m, [])]}, True) for u, ms in MEM.items()}
print('gt collected', flush=True)
_C = {}
def cen(f):
    k = id(f)
    if k not in _C: _C[k] = shape(f['geometry']).centroid
    return _C[k]

def kig(kept, gt=False):
    """Удаляет курганы внутри городищ. Возвращает (оставленные, диагностика)."""
    by = defaultdict(list)
    if not gt:
        for f in kept:
            if f['properties']['class_name'] == 'gorodishcha':
                by[f['properties'].get('region_name')].append(shape(f['geometry']).buffer(0))
        U = {r: prepared.prep(unary_union(v)) for r, v in by.items()}
    else:
        U = GTGOR
    out = []; st = defaultdict(int)
    for f in kept:
        pr = f['properties']
        if pr['class_name'] in KCL and pr.get('region_name') in U:
            r = pr.get('region_name'); c = cen(f)
            real = r in GTKU and GTKU[r].contains(c)
            if U[r].contains(c):
                st['in'] += 1; st['in_real'] += real; continue
            st['out_gorreg'] += 1; st['out_gorreg_real'] += real
        out.append(f)
    return out, st
VARS = {'A': None, 'KIG': False, 'KIGgt': True}
R = {}
for L in TAUS['layouts']:
    t = time.time()
    for v, sp in VARS.items():
        counts = {c: {} for c in calc.target_classes}; st = defaultdict(int); fam_st = defaultdict(lambda: defaultdict(int))
        for fold in TAUS['layouts'][L]:
            held = set(fold['held']); taus = fold['taus']
            for unit in sorted({UNIT.get(site_family(r), site_family(r)) for r in held}):
                kept = [f for m in MEM[unit] for f in feats.get(m, []) if f['properties'].get('region_name') in held
                        and float(f['properties'].get('confidence', 0)) >= taus.get(f['_g'], {}).get(f['properties']['class_name'], float('inf'))]
                if sp is not None:
                    kept, s = kig(kept, sp)
                    for k, x in s.items(): st[k] += x; fam_st[unit][k] += x
                preds = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': kept}, False)
                cv = M.match_collected(calc, preds, GTC.get(unit, {}))
                for c, cc in cv.items():
                    per = counts[c]
                    for r, nn in cc.gt_regions.items():
                        if r in held: per.setdefault(r, [0, 0, 0])[2] += int(nn)
                    for r, h in zip(cc.pred_region, cc.hit): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
        R[(v, L)] = (counts, dict(st), {u: dict(x) for u, x in fam_st.items()})
        print(L, v, round(M.score_counts(counts)[0], 6), dict(st), flush=True)
    print(L, 'time', round(time.time() - t), 's', flush=True)
    pickle.dump(R, open(OUT, 'wb'))
print('done', flush=True)
