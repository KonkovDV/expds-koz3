"""Оценка PIXGBM (bot-3). (1) адресная: восстановление пропусков стека (корзины gap_poly) и finds-пропусков;
(2) honest-10, 4 режима (scope global|site × unseen drop|keep), П1–П5 из geo_gap/evlib.criteria.
Варианты (заявлены в VARIANTS_DECLARED.md до замера): S, U, AU. Вход: pix_lofo.geojson (модели без своей семьи)."""
import os, sys, json, collections
GG = '/workspace/koz3_shared/bot-3/geo_gap'; W = '/workspace/koz3_shared/bot-3/pixgbm'
sys.path.insert(0, GG); os.chdir(GG)
sys.argv = ['x']
exec(open('eval_r.py').read().split('FP = stack_preds')[0])   # build('R3'), как в eval_pkg.py
from evlib import *
from shapely.strtree import STRtree
PIXF = sys.argv[1] if len(sys.argv) > 1 else f'{W}/pix_lofo.geojson'
TAG = os.environ.get('TAG', 'lofo')
A2 = file_preds('a01/A2.geojson')
RUN = {r for r in REGIONS if r[:3] in [str(n) for n in range(201, 212)]}
CL = ['finds_points', 'object_poly']
PIX = {c: [(g, cf, r, grp, 'pix') for g, cf, r, grp, _ in v] for c, v in file_preds(PIXF).items()}
for c in CL: PIX.setdefault(c, [])
print('pix', {c: collections.Counter(p[3] for p in PIX[c]) for c in CL}, flush=True)
GEOREG = {r for r in REGIONS if r[:3].isdigit() and 200 <= int(r[:3]) <= 217}
STACK = {c: stack_preds(c) for c in CL}
A2R3 = {'object_poly': build('R3'), 'finds_points': swap('finds_points', RUN, A2.get('finds_points', []))}
os.chdir(W)
mt_of = lambda grp: grp.replace('_pix', '')

def variant_S(c, base):
    """замена: кандидаты стека того же класса и метода на участках, где pix дал кандидатов, — на pix."""
    cov = {(p[2], mt_of(p[3])) for p in PIX[c]}
    return [p for p in base if (p[2], p[3]) not in cov] + PIX[c]

def variant_U(c, base):
    """объединение: pix-кандидат отбрасывается, если его центр внутри полигона базы (poly) или ≤1 м от точки базы (finds)."""
    by = collections.defaultdict(list)
    for p in base: by[p[2]].append(p[0])
    trees = {r: (STRtree(v), v) for r, v in by.items()}
    out = list(base)
    for p in PIX[c]:
        t = trees.get(p[2]); ce = p[0].centroid
        if t is not None:
            if c == 'object_poly':
                hit = any(t[1][int(j)].contains(ce) for j in t[0].query(ce))
            else:
                hit = any(t[1][int(j)].centroid.distance(ce) <= 1.0 for j in t[0].query(ce.buffer(1.0)))
            if hit: continue
        out.append(p)
    return out

VARS = {'stack': STACK, 'A2R3': A2R3,
        'S': {c: variant_S(c, STACK[c]) for c in CL},
        'U': {c: variant_U(c, STACK[c]) for c in CL},
        'AU': {c: variant_U(c, A2R3[c]) for c in CL}}
# разбор S по классам (доп., не новый вариант): замена только finds / только object_poly
VARS['Sf'] = {'finds_points': VARS['S']['finds_points'], 'object_poly': STACK['object_poly']}
VARS['Sp'] = {'finds_points': STACK['finds_points'], 'object_poly': VARS['S']['object_poly']}
VARS['Sp+A2f'] = {'finds_points': A2R3['finds_points'], 'object_poly': VARS['S']['object_poly']}
VARS['Sf+R3'] = {'finds_points': VARS['S']['finds_points'], 'object_poly': A2R3['object_poly']}
# H (объявлен 28.10 23:00): правило S вне Гнёздова (200–207), правило U в Гнёздове (208–217)
_gn = lambda r: r[:3].isdigit() and 208 <= int(r[:3]) <= 217
VARS['H'] = {c: [p for p in VARS['S'][c] if not _gn(p[2])] + [p for p in VARS['U'][c] if _gn(p[2])] for c in CL}
PART = os.environ.get('PART', 'targeted,honest')
print('n', {k: {c: len(v[c]) for c in CL} for k, v in VARS.items()}, flush=True)

# ───────── (1) адресная ─────────
def greedy(P, c):
    """жадное сопоставление как в gap_poly.py (по убыванию conf, лучший свободный эталон своего участка)."""
    G = GTS[c]; gg = [x[0] for x in G]; tree = STRtree(gg)
    order = sorted(range(len(P)), key=lambda i: P[i][1], reverse=True)
    matched, used = {}, set()
    for i in order:
        pg = P[i][0]
        if c == 'object_poly' and pg.geom_type != 'Polygon': continue
        q = pg if c == 'object_poly' else pg.centroid.buffer(calc.r_tolerance)
        best, bj = -np.inf, -1
        for j in sorted(int(k) for k in tree.query(q)):
            if j in used or G[j][2] != P[i][2]: continue
            ok, s = calc._match_quality(pg, gg[j])
            if ok and s > best + 1e-9: best, bj = s, j
        if bj >= 0: used.add(bj); matched[i] = bj
    return matched

def kept_mask(P, c, unseen, scope):
    cv = curves_of({c: P}, scope)[c]
    keep = np.zeros(len(P), bool); every = set().union(*FOLDS)
    for held in FOLDS:
        res = M.sweep_groups({c: M.curve_where(cv, every - held)})
        taus = {g: gv['threshold'] for g, gv in res[c]['groups'].items()}
        for i in range(len(P)):
            if P[i][2] in held:
                t = taus.get(P[i][3])
                keep[i] = (P[i][1] >= t) if t is not None else (unseen == 'keep')
    return keep

def outcome(P, c, unseen, scope):
    m = greedy(P, c); k = kept_mask(P, c, unseen, scope)
    tp = {j for i, j in m.items() if k[i]}
    fp = [i for i in range(len(P)) if k[i] and i not in m]
    return tp, fp, k, m

def buckets_poly():
    """корзины gap_poly.py (unseen keep): A — нет кандидата стека в 5 м; Dboth — ближайший не проходит ни cov, ни точку."""
    c = 'object_poly'; P = STACK[c]; G = GTS[c]; gg = [x[0] for x in G]
    tree = STRtree(gg); order = sorted(range(len(P)), key=lambda i: P[i][1], reverse=True)
    matched, used = {}, set()
    for i in order:                       # дословно gap_poly.py (без ограничения участком)
        pg = P[i][0]
        if pg.geom_type != 'Polygon': continue
        best, bj = -np.inf, -1
        for j in sorted(int(k) for k in tree.query(pg)):
            if j in used: continue
            ok, s = calc._match_quality(pg, gg[j])
            if ok and s > best + 1e-9: best, bj = s, j
        if bj >= 0: used.add(bj); matched[i] = bj
    keep = kept_mask(P, c, 'keep', 'global')
    tp_g = {j for i, j in matched.items() if keep[i]}
    okpair = collections.defaultdict(list)
    for i, p in enumerate(P):
        for j in tree.query(p[0]):
            if calc._match_quality(p[0], gg[j])[0]: okpair[int(j)].append(i)
    ptree = STRtree([p[0] for p in P]); A, D = [], []
    for j in range(len(G)):
        if j in tp_g: continue
        near = [int(i) for i in ptree.query(gg[j].buffer(5.0))]
        if not near: A.append(j); continue
        if okpair.get(j): continue
        ds = [gg[j].distance(P[i][0]) for i in near]; pg = P[near[int(np.argmin(ds))]][0]
        mn = min(pg.area, gg[j].area); cov = pg.intersection(gg[j]).area / mn if mn > 0 else 0
        rp = gg[j].buffer(1e-10).contains(pg.representative_point())
        if cov < 0.5 and not rp: D.append(j)
    return {'A_нет_кандидата_в_5м': A, 'D_оба_не_проходят': D}

def buckets_finds():
    """пропуски finds стека (honest-10, unseen keep, scope site): A — нет кандидата стека ≤5 м; B — есть ≤5 м, но нет ≤2 м;
    C — есть ≤2 м (ниже порога / занят)."""
    c = 'finds_points'; P = STACK[c]; G = GTS[c]
    tp, _, _, _ = outcome(P, c, 'keep', 'site')
    by = collections.defaultdict(list)
    for p in P: by[p[2]].append(p[0].centroid)
    trees = {r: (STRtree(v), v) for r, v in by.items()}
    out = {'A_нет_кандидата_в_5м': [], 'B_нет_в_2м': [], 'C_есть_в_2м': []}
    for j, (g, _, r, _) in enumerate(G):
        if j in tp: continue
        t = trees.get(r); d = np.inf
        if t is not None:
            near = [int(i) for i in t[0].query(g.buffer(5.0))]
            if near: d = min(g.distance(t[1][i]) for i in near)
        out['A_нет_кандидата_в_5м' if d > 5 else ('B_нет_в_2м' if d > 2 else 'C_есть_в_2м')].append(j)
    return out

if 'targeted' in PART:
  gn = lambda c, j: site_family(GTS[c][j][2]) == 'Гнёздово'
  T = {}
  for c, bk in [('object_poly', buckets_poly()), ('finds_points', buckets_finds())]:
      T[c] = {'buckets_n': {b: len(v) for b, v in bk.items()},
              'buckets_n_gnezdovo': {b: sum(gn(c, j) for j in v) for b, v in bk.items()}}
      print(c, 'корзины', T[c], flush=True)
      for scope in ['site', 'global']:
          tp0, fp0, _, _ = outcome(STACK[c], c, 'keep', scope)
          for vn in ['S', 'U', 'AU', 'A2R3']:
              P = VARS[vn][c]; tp, fp, k, m = outcome(P, c, 'keep', scope)
              rec = {}
              for b, ids in bk.items():
                  ids = set(ids)
                  rec[b] = {'recovered': len(ids & tp), 'recovered_gnezdovo': sum(gn(c, j) for j in ids & tp),
                            'recovered_by_pix': sum(1 for i, j in m.items() if k[i] and j in ids and P[i][4] == 'pix')}
              pixfp = [i for i in fp if P[i][4] == 'pix']
              T[c][f'{vn}|{scope}|keep'] = dict(
                  per_bucket=rec, tp=len(tp), fp=len(fp), d_tp_vs_stack=len(tp) - len(tp0), d_fp_vs_stack=len(fp) - len(fp0),
                  lost_vs_stack=len(tp0 - tp), gained_vs_stack=len(tp - tp0),
                  pix_kept=int(sum(k[i] for i in range(len(P)) if P[i][4] == 'pix')), pix_fp=len(pixfp),
                  pix_fp_gnezdovo=sum(site_family(P[i][2]) == 'Гнёздово' for i in pixfp),
                  d_tp_gnezdovo=sum(gn(c, j) for j in tp) - sum(gn(c, j) for j in tp0))
              print(c, vn, scope, json.dumps(T[c][f'{vn}|{scope}|keep'], ensure_ascii=False), flush=True)
  json.dump(T, open(f'{W}/targeted_{TAG}.json', 'w'), ensure_ascii=False, indent=1, default=float)

# ───────── (2) honest-10 ─────────
out = {}
PAIRS = [tuple(x.split(':')) for x in os.environ.get('PAIRS', 'S:stack,U:stack,AU:stack,AU:A2R3,A2R3:stack').split(',')]
for scope in (['global', 'site'] if 'honest' in PART else []):
    cv = {k: curves_of(v, scope) for k, v in VARS.items() if any(k in pr for pr in PAIRS)}
    for a, b in PAIRS:
        for unseen in ['drop', 'keep']:
            r = criteria(cv[a], cv[b], unseen); key = f'{a}_vs_{b}|{scope}|{unseen}'; out[key] = r
            print(key, 'base', [round(x, 6) for x in r['base']], 'var', [round(x, 6) for x in r['var']],
                  'd10 %+.6f lofo %+.6f rnd %s P(>0) %.3f' % (r['d_honest10'], r['d_lofo'], [round(x, 5) for x in r['d_rand']], r['boot'].get('p_gt0', 0)),
                  'P1-4', r['P1'], r['P2'], r['P3'], r['P4'], 'P5<MDE', r['P5_below_MDE'],
                  '| opoly', {k: r['per_class_var']['object_poly'][k] for k in ('tp', 'fp', 'f1')},
                  '| finds', {k: r['per_class_var']['finds_points'][k] for k in ('tp', 'fp', 'f1')}, flush=True)
if out: json.dump(out, open(f'{W}/honest_{TAG}.json', 'w'), ensure_ascii=False, indent=1, default=float)
