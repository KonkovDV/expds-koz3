"""bot-5. Шаг (2) NOLIDAR.md: «городище в сцене» без растров + привязка к кандидату.
Растров SpOr на коробке нет (5 из 175), поэтому «сцена» = участок с SpOr-снимком, признаки —
только из стека кандидатов (/workspace/rerank/work/table.pkl) и размеров tif из s3_listing.jsonl
(файлы разметки из листинга НЕ используются — это утечка эталона).
Классификатор: логрегрессия L2 (numpy, C=1, гиперпараметры заданы заранее, не подбирались).
Строгая вложенность: для каждого внешнего фолда p на обучающих участках — из внутреннего OOF по
остальным фолдам раскладки, p на отложенном — модель на всех обучающих; порог новой группы
gor_scene выбирает sweep_groups только на обучающих участках (как honest_counts, unseen=drop).
Меняется только класс gorodishcha -> Δ к A и к A+R (rerank_arch меняет только arch) совпадает."""
import sys, json, pickle, collections, time, math
import numpy as np, shapely
from pathlib import Path
ROOT = Path('/workspace/koz3_shared/bot-5/repo'); sys.path.insert(0, str(ROOT))
from expds import metric as M
from expds.metric import ClassCurve, family_folds, honest_counts, score_counts, sweep_groups, curve_where, paired_bootstrap_counts
from expds.naming import site_family
M._SHM_CACHE_DIR = Path('/workspace/koz3_shared/bot-5/tmp/shm_off/x')
OUT = Path('/workspace/koz3_shared/bot-5/analysis/hillfort'); OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; cls, conf, reg, grp, mk = T['cls'], T['conf'], T['reg'], T['grp'], T['mk']
G = 'gorodishcha'; kg = CL.index(G)
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py')); calc = mod.F1Calculator()
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson'))
gmk = {c: [] for c in CL}
for f in gt['features']:
    p = f.get('properties') or {}; c = mod.normalize_class_name(p.get('class_name', 'unknown'))
    if c in gmk and f.get('geometry'): gmk[c].append(str(p.get('markup_type')))
del gt
regions_gt = {r for c in CL for r in T['greg'][c]}
new = {l.strip() for l in (ROOT / 'notes' / 'regions_las_new.txt').read_text(encoding='utf-8').splitlines() if l.strip()}
boxes = set(json.loads((ROOT / 'notes' / 'las_boxes_59.json').read_text(encoding='utf-8')))
GG = shapely.from_wkb(pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))[G]); greg = np.array(T['greg'][G]); gm = np.array(gmk[G])
log('loaded', len(cls), 'gor GT', len(GG), collections.Counter(gm))

# ---------- сцены: участки с SpOr tif (только размеры файлов) ----------
tifs = collections.defaultdict(list)
for line in open('/workspace/data/s3_listing.jsonl', encoding='utf-8'):
    d = json.loads(line); k = d['key']
    if d.get('type') == 'file' and k.lower().endswith(('.tif', '.tiff')) and '_SpOr_' in k:
        tifs[k.split('/')[3]].append(int(d['size']))
sp_rows = np.flatnonzero(mk == 'sp_or')
sp_regs = set(reg[sp_rows])
SCN = sorted(set(tifs) & sp_regs)
log('SpOr tif regions', len(tifs), 'with sp_or candidates', len(SCN))
y = np.array([int(np.any((greg == r) & (gm == 'sp_or'))) for r in SCN])
log('positive scenes', int(y.sum()), 'of', len(SCN))

# ---------- признаки сцены ----------
by = collections.defaultdict(list)
for i in sp_rows: by[reg[i]].append(i)
FN = []; X = []; best = {}; centre = {}
OTH = [c for c in CL if c != G]
for r in SCN:
    ii = np.array(by[r]); f = {}
    g = ii[cls[ii] == kg]; cg = np.sort(conf[g])[::-1] if len(g) else np.zeros(0)
    f['g_n'] = math.log1p(len(g)); f['g_max'] = cg[0] if len(cg) else 0.0
    f['g_top3'] = cg[:3].mean() if len(cg) else 0.0
    f['g_n05'] = math.log1p((cg >= 0.5).sum()); f['g_n09'] = math.log1p((cg >= 0.9).sum())
    if len(g):
        b = g[np.argmax(conf[g])]; best[r] = int(b); f['g_best_area'] = math.log1p(T['area'][b])
        f['g_area05'] = math.log1p(T['area'][g[conf[g] >= 0.5]].sum())
    else: f['g_best_area'] = 0.0; f['g_area05'] = 0.0
    for c in OTH:
        j = ii[cls[ii] == CL.index(c)]; cc = conf[j]
        f[f'{c}_max'] = cc.max() if len(cc) else 0.0; f[f'{c}_n05'] = math.log1p((cc >= 0.5).sum())
    x0, x1 = (T['cx'][ii] - T['bw'][ii] / 2).min(), (T['cx'][ii] + T['bw'][ii] / 2).max()
    y0, y1 = (T['cy'][ii] - T['bh'][ii] / 2).min(), (T['cy'][ii] + T['bh'][ii] / 2).max()
    centre[r] = ((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 1.0), max(y1 - y0, 1.0))
    f['fp_logw'] = math.log(centre[r][2]); f['fp_logh'] = math.log(centre[r][3])
    s = np.array(tifs[r]); f['tif_n'] = len(s); f['tif_logmax'] = math.log(s.max()); f['tif_logsum'] = math.log(s.sum())
    f['has_las'] = float(r in boxes)
    if best.get(r) is not None:
        b = best[r]; f['g_best_dcen'] = math.hypot(T['cx'][b] - centre[r][0], T['cy'][b] - centre[r][1]) / (0.5 * min(centre[r][2], centre[r][3]))
    else: f['g_best_dcen'] = 2.0
    FN = list(f); X.append([float(f[k]) for k in FN])
X = np.array(X); log('features', FN)

def lr_fit(Xt, yt, C=1.0, it=50):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9; Z = np.c_[np.ones(len(Xt)), (Xt - mu) / sd]
    if yt.min() == yt.max(): return ('const', float(yt.mean()))
    w = np.zeros(Z.shape[1]); lam = np.r_[0, np.ones(Z.shape[1] - 1)] / C
    for _ in range(it):  # Ньютон
        p = 1 / (1 + np.exp(-Z @ w)); gr = Z.T @ (p - yt) + lam * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(lam) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, gr); w -= step
        if np.abs(step).max() < 1e-8: break
    return ('lr', mu, sd, w)
def lr_pred(m, Xq):
    if m[0] == 'const': return np.full(len(Xq), m[1])
    _, mu, sd, w = m; Z = np.c_[np.ones(len(Xq)), (Xq - mu) / sd]; return 1 / (1 + np.exp(-Z @ w))
def auc(s, t):
    t = np.asarray(t); s = np.asarray(s); P, N = s[t == 1], s[t == 0]
    if not len(P) or not len(N): return None
    return float(((P[:, None] > N[None]).sum() + 0.5 * (P[:, None] == N[None]).sum()) / (len(P) * len(N)))
sidx = {r: k for k, r in enumerate(SCN)}

# ---------- строки gor ----------
gor_rows = np.flatnonzero(cls == kg)
PP, JJ, SS = T['pairs'][(G, G)]
pairs_by = collections.defaultdict(list)
for a, b, s in zip(PP, JJ, SS): pairs_by[int(a)].append((int(b), float(s)))
# новые строки: id >= N0, у каждой свои пары
N0 = len(cls); new_rows = {}   # variant -> list of (region, pairs)
R_CIRC = 10.0  # м, задано заранее
def circle_pairs(r, x, yy):
    poly = shapely.Point(x, yy).buffer(R_CIRC, 32); out = []
    for j in np.flatnonzero(greg == r):
        g = GG[j]; gg = g.buffer(calc.r_tolerance) if g.geom_type == 'Point' else g
        if not poly.intersects(gg): continue
        ok, s = calc._match_quality(poly, g)
        if ok: out.append((int(j), float(s)))
    return out
V = {'snap': [], 'centre': [], 'csnap': []}
for r in SCN:
    b = best.get(r)
    if b is not None: V['snap'].append((r, pairs_by.get(b, [])))
    cx, cy, w, h = centre[r]; cp = circle_pairs(r, cx, cy); V['centre'].append((r, cp))
    near = b is not None and math.hypot(T['cx'][b] - cx, T['cy'][b] - cy) <= 0.2 * 0.5 * min(w, h)
    V['csnap'].append((r, pairs_by.get(b, []) if near else cp))
for k, v in V.items(): log('variant', k, 'rows', len(v), 'rows with an OK pair', sum(1 for _, p in v if p))

def greedy_hits(order_ids, pair_of):
    used = set(); hit = np.zeros(len(order_ids), bool)
    for n, i in enumerate(order_ids):
        bj, bs = -1, -np.inf
        for j, s in sorted(pair_of(i)):
            if j in used: continue
            if s > bs + 1e-9: bs, bj = s, j
        if bj >= 0: used.add(bj); hit[n] = True
    return hit
def build_curve(var, p_of, gate=False):
    """gor-кривая: базовые строки (+ новые строки варианта с conf=p сцены)."""
    ids = list(map(int, gor_rows)); cf = list(conf[gor_rows].astype(float)); rg = list(reg[gor_rows]); gp = list(grp[gor_rows])
    if gate:
        for n, i in enumerate(ids):
            if mk[i] == 'sp_or' and rg[n] in sidx: cf[n] = cf[n] * p_of[rg[n]]
    extra = {}
    if var:
        for k, (r, pr) in enumerate(V[var]):
            nid = N0 + k; extra[nid] = pr; ids.append(nid); cf.append(float(p_of[r])); rg.append(r); gp.append('gor_scene')
    cf = np.array(cf); o = np.argsort(-cf, kind='stable'); ids = [ids[k] for k in o]
    hit = greedy_hits(ids, lambda i: extra[i] if i >= N0 else pairs_by.get(i, []))
    return ClassCurve(conf=cf[o], hit=hit, pred_region=[rg[k] for k in o], n_gt=T['n_gt'][G],
                      gt_regions=collections.Counter(T['greg'][G]), pred_group=[gp[k] for k in o])

def gor_counts(folds, var, gate=False, collect_p=None):
    every = set().union(*folds); per = {}
    for i, held in enumerate(folds):
        train = every - held
        p_of = {}
        if var or gate:
            tr = [r for r in SCN if r in train]
            for j, fj in enumerate(folds):  # внутренний OOF по остальным фолдам раскладки
                if j == i: continue
                q = [r for r in tr if r in fj]
                if not q: continue
                fit = [r for r in tr if r not in fj]
                m = lr_fit(X[[sidx[r] for r in fit]], y[[sidx[r] for r in fit]])
                for r, pv in zip(q, lr_pred(m, X[[sidx[r] for r in q]])): p_of[r] = float(pv)
            m = lr_fit(X[[sidx[r] for r in tr]], y[[sidx[r] for r in tr]])
            hq = [r for r in SCN if r in held]
            if hq:
                for r, pv in zip(hq, lr_pred(m, X[[sidx[r] for r in hq]])):
                    p_of[r] = float(pv)
                    if collect_p is not None: collect_p[r] = float(pv)
        cv = build_curve(var, p_of, gate)
        res = sweep_groups({G: curve_where(cv, train)})
        taus = {g: gv['threshold'] for g, gv in res[G]['groups'].items()}
        hc = curve_where(cv, held)
        for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
        tau = np.array([taus.get(g, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
        for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
            per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return per

# ---------- базовые кривые всех классов (A) ----------
def greedy_base(c):
    P, J, S = T['pairs'][(c, c)]; byc = collections.defaultdict(list)
    for a, b, s in zip(P, J, S): byc[int(a)].append((int(b), float(s)))
    idx = np.flatnonzero(cls == CL.index(c)); o = idx[np.argsort(-conf[idx], kind='stable')]
    hit = greedy_hits(list(map(int, o)), lambda i: byc.get(i, []))
    return ClassCurve(conf=conf[o].astype(np.float64), hit=hit, pred_region=list(reg[o]), n_gt=T['n_gt'][c],
                      gt_regions=collections.Counter(T['greg'][c]), pred_group=list(grp[o]))
curves = {c: greedy_base(c) for c in CL}
regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
optical = {r for r in regions if r} - boxes
folds10 = family_folds(regions, site_family, 10)
base_full, _ = honest_counts(curves, folds10)
A = score_counts(base_full)[0]; log('honest-10 A', A)
del curves
chk = gor_counts(folds10, None)
assert all(chk.get(r, [0,0,0]) == v for r, v in base_full[G].items() if v != [0,0,0]), 'gor counts mismatch'
log('own gor routine == honest_counts gor: OK')

def rand_folds(seed, k=10):  # как bot-10/repo/scripts/honest_bootstrap.py:random_family_folds
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    names = sorted(fams); lab = np.random.default_rng(seed).permutation(len(names)) % k
    return [set().union(*(fams[n] for n, l in zip(names, lab) if l == i)) for i in range(k)]
def gsc(per):
    tp = sum(v[0] for v in per.values()); fp = sum(v[1] for v in per.values()); n = sum(v[2] for v in per.values())
    return dict(tp=tp, fp=fp, n_gt=n, f1=float(M._f1(tp, fp, n - tp)))
layouts = {'fam10': folds10, 'LOFO': family_folds(regions, site_family, None)}
for s in range(5): layouts[f'seed{s}'] = rand_folds(s)
VARS = [('snap', False), ('centre', False), ('csnap', False), ('csnap', True)]
R = {'definition': '/workspace/koz3_shared/EXPERIMENTS.md:13-20 (П1-П6, bot-10, 28.10)', 'A_honest10': A,
     'scenes': len(SCN), 'positive_scenes': int(y.sum()), 'features': FN, 'variants': {}}
base_g = {nm: gor_counts(f, None) for nm, f in layouts.items()}
for nm in base_g: log('base gor', nm, gsc(base_g[nm]))
for var, gate in VARS:
    name = var + ('+gate' if gate else ''); pc = {}; out = {}
    for nm, f in layouts.items():
        per = gor_counts(f, var, gate, collect_p=pc if nm == 'fam10' else None)
        b, v = gsc(base_g[nm]), gsc(per)
        out[nm] = dict(base=b, var=v, delta=15 * (v['f1'] - b['f1']) / 100)
        if nm == 'fam10':
            vf = dict(base_full); vf[G] = per
            out['fam10']['score'] = score_counts(vf)[0]; out['fam10']['delta_exact'] = score_counts(vf)[0] - A
            out['fam10']['delta_new34'] = score_counts(vf, new)[0] - score_counts(base_full, new)[0]
            out['fam10']['delta_optical'] = score_counts(vf, optical)[0] - score_counts(base_full, optical)[0]
            out['fam10']['bootstrap'] = paired_bootstrap_counts(vf, base_full, site_family, n=1000, seed=0)
            out['fam10']['auc_oof'] = auc([pc[r] for r in SCN], y)
        log(name, nm, out[nm]['delta'], v)
    d = out['fam10']; seeds = [out[f'seed{s}']['delta'] for s in range(5)]
    out['acceptance'] = {
        'P1_delta10>=0.002': [d['delta_exact'], d['delta_exact'] >= 0.002],
        'P2_boot_p_gt0>=0.85': [d['bootstrap'].get('p_gt0'), (d['bootstrap'].get('p_gt0') or 0) >= 0.85],
        'P3_lofo>0_and_4of5_seeds': [out['LOFO']['delta'], seeds, out['LOFO']['delta'] > 0 and sum(x > 0 for x in seeds) >= 4],
        'P4_povr_tselye_gor>=-0.0005_new34_optical>=-0.001': [{'povr': 0.0, 'tselye': 0.0, 'gor': d['delta_exact']},
            d['delta_new34'], d['delta_optical'], d['delta_exact'] >= -0.0005 and d['delta_new34'] >= -0.001 and d['delta_optical'] >= -0.001],
        'P5_below_MDE_0.006_package_only': d['delta_exact'] < 0.006}
    R['variants'][name] = out
    json.dump(R, open(OUT / 'gor_scene.json', 'w'), ensure_ascii=False, indent=1, default=float)
log('done')
import resource; log('peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
