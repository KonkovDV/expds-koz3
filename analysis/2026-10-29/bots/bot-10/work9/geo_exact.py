"""Точная честная оценка ПОСЛЕ порогов (порядок как в watchdog.finish v2) для геофизических классes
finds_points/object_poly: варианты A, S (PIXGBM), s6 (stack6 sep после порогов), Ss6 (pix -> пороги -> stack6).
Фолд: пороги (группа, класс) по кривым на остальных участках (A: bot-10 work/curves, S: bot-10 work7/curves_g_S),
политика unseen: keep (порог -inf) или drop (+inf); затем post.multipole_object_poly + post.filter_finds_points_regimes
из ветки bot7-candidate-v2 (только s6/Ss6); затем официальное сопоставление по семьям (match_collected).
merge_within_markup не моделируется (как и в оценках bot-10). Δ итогового балла = Σ w_c ΔF1_c / Σ w_c(все классы с эталоном).
    python geo_exact.py <out.pkl>
Пути — переменные окружения B10 (корень с repo/), V2_EXPDS, FAM_A, FAM_S, CURV_A, CURV_S, GT, CLUSTERS; по умолчанию — коробка."""
import sys, os, json, pickle, time, gc, importlib.util
from collections import defaultdict
from pathlib import Path
E = os.environ.get
B10 = Path(E('B10', '/workspace/koz3_shared/bot-10'))
sys.path.insert(0, str(B10 / 'repo')); sys.path.insert(0, str(B10 / 'repo' / 'scripts'))
from expds import metric as M
from expds.naming import site_family
from honest_bootstrap import load_curves, regions_of, random_family_folds
from shapely.geometry import shape, mapping
M._SHM_CACHE_DIR = Path('/proc/expds_no_cache')
R7 = Path(E('V2_EXPDS', '/workspace/koz3_shared/bot-7/repo/expds'))   # expds ветки bot7-candidate-v2
spec = importlib.util.spec_from_file_location('expds7', R7 / '__init__.py', submodule_search_locations=[str(R7)])
e7 = importlib.util.module_from_spec(spec); sys.modules['expds7'] = e7; spec.loader.exec_module(e7)
import expds7.post as P7
os.environ.setdefault('EXPDS_MATCH_SCOPE', 'global')
OUT = sys.argv[1]
CL = ('finds_points', 'object_poly')
FAMS = ('Гнёздово', 'Семибратнее', 'Судьбищи')
STACK = {'A': Path(E('FAM_A', B10 / 'work/fam')), 'S': Path(E('FAM_S', B10 / 'work7/fam_S'))}
CURV = {'A': Path(E('CURV_A', B10 / 'work/curves')), 'S': Path(E('CURV_S', B10 / 'work7/curves_g_S'))}
VARS = {'A': ('A', False), 'S': ('S', False), 's6': ('A', True), 'Ss6': ('S', True)}

# пороги по фолдам
cvA_all = load_curves(CURV['A']); regions = regions_of(cvA_all); den_classes = {c for c, v in cvA_all.items() if sum(v.gt_regions.values()) > 0}
DEN = sum(M.CLASS_WEIGHTS.get(c, 0.0) for c in den_classes)
assert regions and DEN > 0, f'пустые кривые: {CURV}'
print('regions', len(regions), 'den classes', sorted(den_classes), 'DEN', DEN, flush=True)
lay = {'k10': M.family_folds(regions, site_family, 10), 'lofo': M.family_folds(regions, site_family, None)}
for s in range(5): lay[f'seed{s}'] = random_family_folds(regions, 10, s)
TAUS = {}
for sk in STACK:
    cv = {c: v for c, v in load_curves(CURV[sk]).items() if c in CL}
    assert regions_of(cv) <= regions, sk
    for L, folds in lay.items():
        every = set().union(*folds); fl = []
        for held in folds:
            res = M.sweep_groups({c: M.curve_where(v, every - held) for c, v in cv.items()})
            taus = {}
            for c, v in res.items():
                for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
            fl.append({'held': set(held), 'taus': taus})
        TAUS[(sk, L)] = fl
del cvA_all; gc.collect(); print('taus done', flush=True)

feats = {}
for sk, d in STACK.items():
    feats[sk] = {}
    for fam in FAMS:
        lst = []
        with open(d / f'{fam}.jsonl', encoding='utf-8') as fh:
            for line in fh:
                f = json.loads(line); pr = f['properties']
                if pr.get('class_name') not in CL: continue
                f['_g'] = M.threshold_group(pr); lst.append(f)
        feats[sk][fam] = lst
    print('loaded', sk, {k: len(v) for k, v in feats[sk].items()}, flush=True)
mod = M.load_official(None); calc = M_calc = mod.F1Calculator()
gt_by_fam = defaultdict(list); gt_other = defaultdict(int)
for f in json.load(open(E('GT', '/workspace/rerank/gt_train_v2.geojson'), encoding='utf-8'))['features']:
    pr = f['properties']
    if pr.get('class_name') not in CL: continue
    fam = site_family(str(pr.get('region_name', 'default_source')))
    if fam in FAMS: gt_by_fam[fam].append(f)
    else: gt_other[fam] += 1
print('gt', {k: len(v) for k, v in gt_by_fam.items()}, 'gt outside geo fams', dict(gt_other), flush=True)
assert not gt_other
UNIT = {}
for members in json.load(open(E('CLUSTERS', B10 / 'work/family_clusters.json'), encoding='utf-8'))['clusters'].values():
    for m in members: UNIT[m] = '+'.join(sorted(members))
assert all(f not in UNIT for f in FAMS)
GTC = {fam: M._collect(mod, calc, {'type': 'FeatureCollection', 'features': gt_by_fam[fam]}, True) for fam in FAMS}

def post_v2(kept):
    dd = []
    for f in kept:
        d = dict(f['properties']); d['geom'] = shape(f['geometry']); dd.append(d)
    dd = P7.multipole_object_poly(dd, None)
    dd = P7.filter_finds_points_regimes(dd, None)
    out = []
    for d in dd:
        g = d.pop('geom'); out.append({'type': 'Feature', 'geometry': mapping(g), 'properties': d})
    return out

R = {}
for pol in ('keep', 'drop'):
    miss = float('-inf') if pol == 'keep' else float('inf')
    for L in lay:
        t = time.time()
        for v, (sk, do_post) in VARS.items():
            counts = {c: {} for c in CL}; st = defaultdict(int)
            for fold in TAUS[(sk, L)]:
                held = fold['held']; taus = fold['taus']
                for fam in FAMS:
                    kept = [f for f in feats[sk][fam] if f['properties'].get('region_name') in held
                            and float(f['properties'].get('confidence', 0)) >= taus.get(f['_g'], {}).get(f['properties']['class_name'], miss)]
                    if not kept and not any(r in held for r in {g['properties'].get('region_name') for g in gt_by_fam[fam]}): continue
                    st['in'] += len(kept)
                    if do_post: kept = post_v2(kept)
                    st['out'] += len(kept)
                    cv = M.match_collected(calc, M._collect(mod, calc, {'type': 'FeatureCollection', 'features': kept}, False), GTC[fam])
                    for c, cc in cv.items():
                        if c not in counts: continue
                        per = counts[c]
                        for r, n in cc.gt_regions.items():
                            if r in held: per.setdefault(r, [0, 0, 0])[2] += int(n)
                        for r, h in zip(cc.pred_region, cc.hit): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
            R[(pol, v, L)] = (counts, dict(st))
            _, per = M.score_counts(counts)
            part = sum(M.CLASS_WEIGHTS.get(c, 0) * per[c]['f1'] for c in CL if per[c]['n_gt'] > 0) / DEN
            print(pol, L, v, 'geo_part', round(part, 6), {c: (round(per[c]['f1'], 4), int(per[c]['tp']), int(per[c]['fp']), int(per[c]['n_gt'])) for c in CL}, dict(st), flush=True)
        print(pol, L, 'time', round(time.time() - t), 's', flush=True)
        pickle.dump({'R': R, 'DEN': DEN}, open(OUT, 'wb'))
print('done', flush=True)
