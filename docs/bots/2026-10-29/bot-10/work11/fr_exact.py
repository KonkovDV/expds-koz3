"""bot-10 work11 Доп. 5: FR1h_exact (копия nb_exact.py; стек FR = work11/fam_fr1h, множитель кадра уже применён — он не зависит от порогов). Ниже исходный docstring NB_exact:
 — neigh аудита 6e26062 (post.kurgan_neighbour_prior, параметры по умолчанию) в порядке сдачи.
    python nb_exact.py <global|site> <taus_A.json> <taus_NB.json> <out.pkl>
Варианты: A, TO (twins 78add0e) — пороги A по фолду; NB, TONB — на удержанных семьях опора соседей = СЫРАЯ conf >= фолдового
порога NB своей группы (как watchdog._tau(cfg) с новыми порогами), множитель на li/li_coarse/li_mosaic, затем фолдовые пороги NB
(unseen=drop), затем (TONB) twins. Сопоставление — M.match_collected, как gm_exact.py / work9/twins9.py."""
import sys, os, json, pickle, time, gc, importlib.util
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds import metric as M
from expds.naming import site_family
from shapely.geometry import shape
M._SHM_CACHE_DIR = Path('/proc/expds_no_cache')
# bot-6 post как отдельный пакет expds6
B6 = Path('/workspace/koz3_shared/bot-10/work8/b6_78/expds')  # bot6-twins-only 78add0e
spec = importlib.util.spec_from_file_location('expds6', B6 / '__init__.py', submodule_search_locations=[str(B6)])
e6 = importlib.util.module_from_spec(spec); sys.modules['expds6'] = e6; spec.loader.exec_module(e6)
import expds6.post as P6
import inspect, textwrap
# Исправленная копия: масштаб cos φ фиксируется на регион (в оригинале xy=(x*s(y), y*s(y)) с s от самой точки ->
# производная y*s(y) по y ~0.04 на 55° с.ш., т.е. расстояния север-юг сжаты в ~16 раз).
_src = inspect.getsource(P6.kurgan_cross_modal_twins).replace('s = _ground_scale(c.y)', 's = _S0.setdefault(reg, _ground_scale(c.y))')
_src = _src.replace('def kurgan_cross_modal_twins(', 'def kurgan_cross_modal_twins_fixed(')
_ns = dict(vars(P6)); _ns['_S0'] = {}
exec(compile(_src, 'twins_fixed', 'exec'), _ns)
TW_FIXED = _ns['kurgan_cross_modal_twins_fixed']
pass  # 78add0e: исходный патч bot-10 не применяется (код уже исправлен) → TOfix ≡ TO
scope = sys.argv[1]; TAUSF = sys.argv[2]; TAUSNB = sys.argv[3]; OUT = sys.argv[4]
os.environ['EXPDS_MATCH_SCOPE'] = scope
W = Path('/workspace/koz3_shared/bot-10')
STACK = {'A': W / 'work11/fam0', 'NB': W / 'work11/fam_fr1h'}  # NB = FR1h
TAUS = {'A': json.load(open(TAUSF)), 'NB': json.load(open(TAUSNB))}
VARS = {'A': ('A', None), 'TO': ('A', {'dedup_ae_vs_li_m': 0.0}), 'FR': ('NB', {'_tw': False}), 'TOFR': ('NB', {'dedup_ae_vs_li_m': 0.0})}
LIG = {'li', 'li_coarse', 'li_mosaic'}
NB_SPEC = {'groups': ['li', 'li_coarse', 'li_mosaic'], 'count_groups': None, 'radius_m': 200.0, 'k': 3.0, 'floor': 0.5, 'slope': 0.5, 'cap': 2.0, 'support_conf': None}
mod = M.load_official(None); calc = mod.F1Calculator()
KCL = {'kurgany_povrezhdennye', 'kurgany_tselye'}

def minimal_taus(T):
    mn = {}
    for fl in T['layouts'].values():
        for f in fl:
            for g, d in f['taus'].items():
                for c, t in d.items():
                    k = (g, c); mn[k] = min(mn.get(k, float('inf')), t)
    return mn

# предфильтр: признаки с conf >= минимального порога по всем фолдам/раскладкам
feats = {}   # stack -> fam -> list of feature dicts (+ _g)
for sk, d in STACK.items():
    mn = minimal_taus(TAUS[sk]); t = time.time(); n = 0; feats[sk] = {}
    for p in sorted(d.glob('*.jsonl')):
        lst = []
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                f = json.loads(line); pr = f['properties']
                t0 = mn.get((M.threshold_group(pr), pr.get('class_name')))
                if False and t0 is not None and M.threshold_group(pr) in LIG and pr.get('class_name') in KCL: t0 = t0 / 1.5
                if t0 is None or float(pr.get('confidence', 0)) < t0: continue
                f['_g'] = M.threshold_group(pr); lst.append(f)
        feats[sk][p.stem] = lst; n += len(lst)
    print('prefilter', sk, n, round(time.time() - t), 's', flush=True); gc.collect()
gt_by_fam = defaultdict(list)
for f in json.load(open('/workspace/rerank/gt_train_v2.geojson', encoding='utf-8'))['features']:
    gt_by_fam[site_family(str(f['properties'].get('region_name', 'default_source')))].append(f)
UNIT = {}
for members in json.load(open(W / 'work/family_clusters.json', encoding='utf-8'))['clusters'].values():
    for m in members: UNIT[m] = '+'.join(sorted(members))
MEM = defaultdict(list)
for fam in set(gt_by_fam) | set(feats['A']): MEM[UNIT.get(fam, fam)].append(fam)
GTC = {u: M._collect(mod, calc, {'type': 'FeatureCollection', 'features': [f for m in ms for f in gt_by_fam.get(m, [])]}, True) for u, ms in MEM.items()}
print('gt collected', flush=True)
TWIN_SPEC = {}
spa = importlib.util.spec_from_file_location('expdsA', W / 'work11/audit2_src/expds/__init__.py', submodule_search_locations=[str(W / 'work11/audit2_src/expds')])
eA = importlib.util.module_from_spec(spa); sys.modules['expdsA'] = eA; spa.loader.exec_module(eA)
import expdsA.post as PA
GM_SPEC = {'min_gor_conf': 0.0, 'buffer_m': 0.0, 'groups': None, 'gor_groups': None, 'min_area_m2': 500.0, 'max_area_m2': 400_000.0}
def apply_gm(kk, gors):
    dd = [{'_i': i, 'class_name': f['properties']['class_name'], 'threshold_group': f['properties'].get('threshold_group'),
           'markup_type': f['properties'].get('markup_type'), 'confidence': float(f['properties'].get('confidence', 0)),
           'region_name': f['properties'].get('region_name'), 'geom': f['_shp']} for i, f in enumerate(kk)]
    gd = [{'_i': -1, 'class_name': 'gorodishcha', 'threshold_group': g['properties'].get('threshold_group'), 'markup_type': g['properties'].get('markup_type'),
           'confidence': float(g['properties'].get('confidence', 0)), 'region_name': g['properties'].get('region_name'), 'geom': g['_shp']} for g in gors]
    res = PA.kurgans_outside_hillforts(dd + gd, dict(GM_SPEC))
    keep = {r['_i'] for r in res if r['_i'] >= 0}
    return [f for i, f in enumerate(kk) if i in keep]
def apply_post(kept, sp):
    if sp is None: return kept, 0, 0
    dd = [{'_i': i, 'class_name': f['properties']['class_name'], 'threshold_group': f['properties'].get('threshold_group'),
           'markup_type': f['properties'].get('markup_type'), 'confidence': float(f['properties'].get('confidence', 0)),
           'region_name': f['properties'].get('region_name'), 'geom': f['_shp']} for i, f in enumerate(kept)]
    fn = TW_FIXED if sp.get('_fix') else P6.kurgan_cross_modal_twins
    res = fn(dd, dict(sp))
    out = []; ntw = 0
    for r in res:
        f = kept[r['_i']]
        if r['threshold_group'] in ('ae_twin', 'sp_focal_twin'):
            ntw += 1
            out.append({'type': 'Feature', 'geometry': f['geometry'],
                        'properties': dict(f['properties'], class_name='kurgany_povrezhdennye', threshold_group=r['threshold_group'])})
        else: out.append(f)
    return out, ntw, len(kept) + ntw - len(out)
def apply_nb(cands, taus):
    """cands — признаки удержанных семей (сырая conf). Возвращает список с новой conf у курганов li*."""
    idx = [i for i, f in enumerate(cands) if f['properties']['class_name'] in KCL and f['_g'] in LIG]
    for i in idx:
        if '_shp' not in cands[i]: cands[i]['_shp'] = shape(cands[i]['geometry'])
    dd = [{'_i': i, 'class_name': cands[i]['properties']['class_name'], 'threshold_group': cands[i]['_g'], 'markup_type': cands[i]['properties'].get('markup_type'),
           'confidence': float(cands[i]['properties'].get('confidence', 0)), 'region_name': cands[i]['properties'].get('region_name'), 'geom': cands[i]['_shp']} for i in idx]
    tau = lambda f: taus.get(f['threshold_group'], {}).get(f['class_name'], float('inf'))
    res = PA.kurgan_neighbour_prior(dd, dict(NB_SPEC), tau)
    out = list(cands); nup = ndn = 0
    for r in res:
        f = cands[r['_i']]; c0 = float(f['properties'].get('confidence', 0))
        if r['confidence'] != c0:
            nup += r['confidence'] > c0; ndn += r['confidence'] < c0
            out[r['_i']] = dict(f, properties=dict(f['properties'], confidence=r['confidence']))
    return out, nup, ndn
R = {}
for L in TAUS['A']['layouts']:
    t = time.time()
    for v, (sk, sp) in VARS.items():
        counts = {c: {} for c in calc.target_classes}; st = defaultdict(int)
        for fold in TAUS[sk]['layouts'][L]:
            assert TAUS[sk]['layouts'][L].index(fold) >= 0
            held = set(fold['held']); taus = fold['taus']
            for unit in sorted({UNIT.get(site_family(r), site_family(r)) for r in held}):
                kept = [f for m in MEM[unit] for f in feats[sk].get(m, []) if f['properties'].get('region_name') in held]
                if sp is not None and sp.get('_nb'):
                    kept, nup, ndn = apply_nb(kept, taus); st['nb_up'] += nup; st['nb_down'] += ndn
                kept = [f for f in kept if float(f['properties'].get('confidence', 0)) >= taus.get(f['_g'], {}).get(f['properties']['class_name'], float('inf'))]
                if sp is not None:
                    for f in kept:
                        if '_shp' not in f and f['properties']['class_name'] in KCL | {'gorodishcha'}: f['_shp'] = shape(f['geometry'])
                    kk = [f for f in kept if f['properties']['class_name'] in KCL]
                    rest = [f for f in kept if f['properties']['class_name'] not in KCL]
                    if sp.get('_tw', True):
                        tsp = {k: v for k, v in sp.items() if not k.startswith('_')}
                        kk, ntw, ndrop = apply_post(kk, tsp); st['twins'] += ntw; st['dropped'] += ndrop
                    kept = rest + kk
                preds = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': kept}, False)
                gts = GTC.get(unit, {})
                cv = M.match_collected(calc, preds, gts)
                for c, cc in cv.items():
                    per = counts[c]
                    for r, n in cc.gt_regions.items():
                        if r in held: per.setdefault(r, [0, 0, 0])[2] += int(n)
                    for r, h in zip(cc.pred_region, cc.hit): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
        R[(v, L)] = (counts, dict(st))
        _, per = M.score_counts(counts)
        print('   ', v, {c: (per[c]['tp'], per[c]['fp']) for c in KCL if c in per}, flush=True)
        print(L, v, round(M.score_counts(counts)[0], 6), dict(st), flush=True)
    print(L, 'time', round(time.time() - t), 's', flush=True)
    pickle.dump(R, open(OUT, 'wb'))
print('done', flush=True)
