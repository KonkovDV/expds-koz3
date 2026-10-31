"""Честная проверка object_poly etgr/terfl, совместимая с honest-10 (bot-10 / expds.metric.honest_counts).

Протокол «honest-10 + LOSO для невиданных групп» (unseen="loso"):
  фолды — те же family_folds(…,10) из /workspace/rerank/work/base_honest.pkl. На фолде H пороги групп, которые есть на
  обучающих участках (every−H), — штатно sweep_groups по every−H. Для группы g, которой в every−H нет (etgr/terfl:
  все их участки — семья Гнёздово, она целиком в одном фолде), порог для каждого удержанного участка r подбирается
  sweep_groups на every−H ∪ (H∖{r}); берётся только порог g. Эталон участка r в подбор не входит никогда.
  Если g больше нигде нет (одиночный участок) → drop. Так же (внутри того же обучающего множества) выбираются
  геометрия и мин. площадь в режиме «nested».
Остальные классы — счётчики базы honest-10 (base_honest.pkl, 0,314615). Попадания — как в honest_counts
(жадное сопоставление по всем кандидатам, официальный _match_quality).

Источники кандидатов: stack (кэш stack_arch) и pixgbm (bot-3/pixgbm/oof/<region>_<markup>.geojson при READY_<region>;
поля class/score/region/markup_type, EPSG:3857, Polygon). Режимы pixgbm: S — вместо стека той же (участок, группа),
U — объединение.
Запуск: OMP_NUM_THREADS=2 nice -n 10 /workspace/rerank/venv/bin/python geo_val.py [--pix S|U] [--out geo_val.json]
"""
import sys, os, json, pickle, glob, argparse, resource, copy, time
resource.setrlimit(resource.RLIMIT_AS, (3_000_000_000, 3_000_000_000))
sys.path.insert(0, '/workspace/rerank')
import numpy as np, shapely
from shapely.strtree import STRtree
from collections import Counter, defaultdict
from common import *
from expds.metric import ClassCurve, sweep_groups, curve_where, score_counts, paired_bootstrap_counts
from expds import metric as M
from expds.naming import site_family
ap = argparse.ArgumentParser(); ap.add_argument('--pix', default=''); ap.add_argument('--out', default='geo_val.json')
ap.add_argument('--oof', default='/workspace/koz3_shared/bot-3/pixgbm/oof'); ap.add_argument('--boot', type=int, default=1000)
ap.add_argument('--tpix', type=float, default=None, help='один уровень t_pix pixgbm (компоненты 0.3/0.5/0.7 вложены)')
ap.add_argument('--markups', default='etgr,terfl', help='какие методы pixgbm брать')
A = ap.parse_args()
mod = M.load_official(METRIC); calc = mod.F1Calculator()
D = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opoly.pkl'), 'rb'))
B = pickle.load(open(W/'work/base_honest.pkl', 'rb')); FOLDS = B['folds']; EVERY = set().union(*FOLDS)
GT = shapely.from_wkb(D['gt_wkb']); GREG = np.array(D['greg']); NGT = D['n_gt']
TREE = STRtree(GT)
GEO = ('etgr', 'terfl', 'etgr_pix', 'terfl_pix')  # pixgbm получает свои группы порога: шкала score другая
# ---- кандидаты
geom = list(shapely.from_wkb(D['wkb'])); conf = list(D['conf']); reg = list(D['reg']); grp = list(D['grp']); src = ['stack'] * len(geom)
pix_info = {}
if A.pix:
    ready = sorted(os.path.basename(p)[6:] for p in glob.glob(os.path.join(A.oof, 'READY_*')))
    feats = []
    for r in ready:
        for p in sorted(glob.glob(os.path.join(A.oof, f'{r}_*.geojson'))):
            if p.rsplit('_', 1)[1].split('.')[0] not in A.markups.split(','): continue
            fc = json.load(open(p, encoding='utf-8-sig'))
            for f in fc['features']:
                q = f['properties']
                if A.tpix is not None and abs(float(q.get('t_pix', -1)) - A.tpix) > 1e-6: continue
                f['properties'] = dict(class_name=q.get('class', q.get('class_name', 'object_poly')), confidence=q.get('score', q.get('confidence')),
                                       region_name=q.get('region', q.get('region_name')), markup_type=q.get('markup_type'))
                feats.append(f)
    pix_info = dict(ready=ready, n=len(feats), tpix=A.tpix, markups=A.markups)
    if feats:
        pr = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': feats}, False).get('object_poly', [])
        assert len(pr) == len(feats), (len(pr), len(feats))
        pkeys = {(f['properties']['region_name'], f['properties']['markup_type']) for f in feats}
        if A.pix == 'S':
            keep = [i for i in range(len(geom)) if (reg[i], grp[i]) not in pkeys]
            geom, conf, reg, grp, src = ([x[i] for i in keep] for x in (geom, conf, reg, grp, src))
        for f, x in zip(feats, pr):
            geom.append(x[0]); conf.append(float(x[1])); reg.append(f['properties']['region_name']); grp.append(f['properties']['markup_type'] + '_pix'); src.append('pixgbm')
    pix_info['keys'] = sorted(map(list, pkeys)) if feats else []
conf = np.array(conf, float); reg = np.array(reg, object); grp = np.array(grp, object); AREA = shapely.area(np.array(geom, object))
N = len(geom); ISGEO = np.isin(grp, GEO)
# ---- геометрия
def variant(g, v):
    if v == 'orig': return g
    if v.startswith('simp'): return g.simplify(float(v[4:]), preserve_topology=True)
    if v.startswith('pole'):
        from shapely.ops import polylabel
        try: p = polylabel(g, tolerance=0.02)
        except Exception: p = g.representative_point()
        return p.buffer(float(v[4:]), 16)
    if v == 'hull': return g.convex_hull
    raise ValueError(v)
VARS = ['orig', 'simp0.1', 'simp0.25', 'simp0.5', 'hull']  # круг в полюсе отклонён ботом «1» (игра с нормировкой)
AMIN = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
_pairs_cache = {}
def pairs_for(v):
    """Список пар (j, cov) с hit для каждого кандидата; вариант v применяется только к etgr/terfl."""
    if v in _pairs_cache: return _pairs_cache[v]
    out = [None] * N
    gg = [variant(geom[i], v) if ISGEO[i] else geom[i] for i in range(N)]
    qp, qg = TREE.query(gg)
    for a, b in zip(qp, qg):
        pg = gg[a]
        if pg.geom_type != 'Polygon' or pg.is_empty: continue
        ok, s = calc._match_quality(pg, GT[b])
        if ok: (out.__setitem__(a, []) if out[a] is None else None); out[a].append((int(b), float(s)))
    out = [sorted(x) if x else None for x in out]
    _pairs_cache[v] = out; return out
ORDER = np.argsort(-conf, kind='stable')
def curve(vmap, amap):
    """vmap/amap: {группа: вариант/мин.площадь}; кандидаты etgr/terfl ниже мин. площади выбрасываются."""
    alive = np.ones(N, bool)
    for g in GEO: alive &= ~((grp == g) & (AREA < amap.get(g, 0.0)))
    P = {v: pairs_for(v) for v in set(vmap.values()) | {'orig'}}
    o = ORDER[alive[ORDER]]; hit = np.zeros(len(o), bool); used = set()
    for k, i in enumerate(o):
        lst = P[vmap.get(grp[i], 'orig') if ISGEO[i] else 'orig'][i]
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in lst:
            if j not in used and s > best + 1e-9: best, bj = s, j
        if bj >= 0: used.add(bj); hit[k] = True
    return ClassCurve(conf=conf[o], hit=hit, pred_region=list(reg[o]), n_gt=NGT, gt_regions=Counter(D['greg']), pred_group=list(grp[o]))
def taus(c, regions):
    r = sweep_groups({'object_poly': curve_where(c, regions)})
    return {g: v['threshold'] for g, v in r.get('object_poly', {}).get('groups', {}).items()}
def train_f1(c, regions, g):
    """F1 класса на regions при порогах, подобранных на них же (для выбора геометрии/площади группы g)."""
    r = sweep_groups({'object_poly': curve_where(c, regions)}); return r.get('object_poly', {}).get('f1', 0.0)
def honest(unseen='drop', vmap=None, amap=None, nested=False, fixed=None):
    """Счётчики object_poly по фолдам. unseen: drop | keep | loso | fixed (порог из fixed={группа: τ})."""
    vmap = vmap or {}; amap = amap or {}
    cache = {}
    def C(vm, am):
        k = (tuple(sorted(vm.items())), tuple(sorted(am.items())))
        if k not in cache: cache[k] = curve(vm, am)
        return cache[k]
    base_c = C(vmap, amap)
    cnt = {}; chosen = []
    for H in FOLDS:
        train = EVERY - H; t_fold = taus(base_c, train)
        for r in sorted(H):
            n = base_c.gt_regions.get(r, 0)
            if n: cnt.setdefault(r, [0, 0, 0])[2] += int(n)
        hregs = sorted({x for x in reg if x in H})
        for r in hregs:
            t = dict(t_fold); vm, am = dict(vmap), dict(amap)
            for g in GEO:
                if g in t or not np.any((reg == r) & (grp == g)): continue
                if unseen == 'drop': continue
                if unseen == 'keep': t[g] = -np.inf; continue
                if unseen == 'fixed':
                    if g in fixed: t[g] = fixed[g]
                    continue
                tr = train | (H - {r})
                if not np.any(np.isin(reg, list(tr)) & (grp == g)): continue  # группы больше нигде нет → drop
                if nested:  # выбор геометрии и мин. площади группы g по обучающему множеству
                    best = None
                    for v in VARS:
                        for a in AMIN:
                            f = train_f1(C({**vm, g: v}, {**am, g: a}), tr, g)
                            if best is None or f > best[0] + 1e-12: best = (f, v, a)
                    vm[g], am[g] = best[1], best[2]; chosen.append(dict(region=r, group=g, var=best[1], amin=best[2], train_f1=round(best[0], 4)))
                tt = taus(C(vm, am), tr)
                if g in tt: t[g] = tt[g]
            c = C(vm, am); sel = np.array([x == r for x in c.pred_region])
            for gi, h, cf in zip(np.array(c.pred_group, object)[sel], c.hit[sel], c.conf[sel]):
                if gi in t and cf >= t[gi]: cnt.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return cnt, chosen
def full(cnt):
    c = copy.deepcopy(B['counts']); c['object_poly'] = cnt; return c
def summarize(name, cnt, ref, chosen=None):
    s, per = score_counts(full(cnt)); s0 = score_counts(full(ref))[0]
    geo_regs = {r for r in reg[ISGEO]}
    row = dict(name=name, honest10=round(s, 6), delta=round(s - s0, 6), opoly=dict(tp=per['object_poly']['tp'], fp=per['object_poly']['fp'], f1=round(per['object_poly']['f1'], 4)),
               geo_regions={r: cnt.get(r) for r in sorted(geo_regs)}, op_counts={r: list(v) for r, v in sorted(cnt.items())})
    if A.boot and name != 'A_drop':
        fam = paired_bootstrap_counts(full(cnt), full(ref), site_family, A.boot)
        site = paired_bootstrap_counts(full(cnt), full(ref), lambda r: r if site_family(r) == 'Гнёздово' or r in geo_regs else site_family(r), A.boot)
        row['boot_family'] = {k: round(v, 6) if isinstance(v, float) else v for k, v in fam.items()}
        row['boot_site'] = {k: round(v, 6) if isinstance(v, float) else v for k, v in site.items()}
    if chosen: row['chosen'] = chosen
    print(json.dumps(row, ensure_ascii=False), flush=True); return row
t0 = time.time(); rows = []
print('pairs_orig', sum(len(x) for x in pairs_for('orig') if x), 'cands', N, Counter(zip(grp, src)), 'pix', {k: v for k, v in pix_info.items() if k != 'keys'}, flush=True)
ref, _ = honest('drop'); rows.append(summarize('A_drop', ref, ref))
rows.append(summarize('keep_all', honest('keep')[0], ref))
rows.append(summarize('fixed_cfg_etgr0.096635_terfl=keep', honest('fixed', fixed={'etgr': 0.096635, 'terfl': -np.inf})[0], ref))
rows.append(summarize('LOSO_orig_a0', honest('loso')[0], ref))
for v in VARS:
    for a in AMIN:
        if v == 'orig' and a == 0: continue
        rows.append(summarize(f'LOSO_{v}_a{a}', honest('loso', {g: v for g in GEO}, {g: a for g in GEO})[0], ref))
cnt, ch = honest('loso', nested=True); rows.append(summarize('LOSO_nested(var,amin,τ)', cnt, ref, ch))
json.dump(dict(pix=pix_info, rows=rows, sec=round(time.time() - t0, 1)), open(A.out, 'w'), ensure_ascii=False, indent=1, default=float)
print('sec', round(time.time() - t0, 1), 'maxrss MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
