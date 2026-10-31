"""bot-6 ORGATE G — финальная модель (все семьи) + рабочая точка из вложенной процедуры k10.
Логика фолдов/OOF/порогов — копия bot-6/work/orgate/gate_eval.py (вариант G, k10, global), но gate_eval не трогается
и в его res/ ничего не пишется. Сверка: счётчики arch k10 G должны совпасть с res/res_k10_grp.json (54/327).
Рабочая точка = медиана по 10 фолдам порогов sweep_groups (каждый — на OOF обучающих семей своего фолда).
Пишет сюда: orgate_g.txt (LightGBM), orgate_g.json (meta), fold_taus.json, offline_decisions.json."""
import sys, json, pickle, time, hashlib
sys.path.insert(0, '/workspace/rerank')
from common import *
import fast, numpy as np, lightgbm as lgb
from expds.metric import ClassCurve, curve_where, sweep_groups
OUT = Path('/workspace/koz3_shared/bot-6/work/orgate_final')
T = fast.T; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']; N = len(cls)
X2 = np.load(W/'work/X2.npy', mmap_mode='r'); n2 = json.load(open(W/'work/X2_names.json')); pool = np.load(W/'work/pool.npy'); y = np.load(W/'work/first.npy')
A = 'arkhitektury'; C = fast.cid[A]; arch = cls == C; ap_idx = np.flatnonzero(arch & pool)
fg = np.load('/workspace/koz3_shared/bot-6/work/orgate/feats_gate.npz'); ng = list(fg['names'])
row = np.full(N, -1); row[fg['stack_idx']] = np.arange(len(fg['stack_idx'])); assert (row[ap_idx] >= 0).all()
XA = np.c_[np.asarray(X2[ap_idx]), fg['X'][row[ap_idx]]].astype(np.float32); names = n2 + ng
ya = y[ap_idx].astype(float); rega = reg[ap_idx]
las_all = np.zeros(N, bool); las_all[fg['stack_idx']] = fg['X'][:, ng.index('r_las')] > 0
base_cv = fast.curves()
regions = {r for c in base_cv.values() for r in c.pred_region} | {r for c in base_cv.values() for r in c.gt_regions}
folds = family_folds(regions, site_family, 10)
fold_of = {r: i for i, f in enumerate(folds) for r in f}
fid_a = np.array([fold_of[r] for r in rega]); fid_all_arch = np.array([fold_of[r] for r in reg[arch]]); fam_a = np.array([site_family(r) for r in rega])
PL = fast.PL[(A, A)]
def arch_curve(score):
    idx = np.flatnonzero(arch); o = idx[np.argsort(-score[idx], kind='stable')]; hit = np.zeros(len(o), bool); matched = set()
    for r, p in enumerate(o):
        lst = PL.get(int(p))
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in lst:
            if j in matched: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: matched.add(bj); hit[r] = True
    return ClassCurve(conf=score[o].astype(np.float64), hit=hit, pred_region=list(T['reg'][o]), n_gt=T['n_gt'][A], gt_regions=fast.GREG[A], pred_group=list(T['grp'][o]))
cols = names; ci = list(range(len(names))); mono = [1 if n in ('conf', 'logit') else 0 for n in cols]; cat = [cols.index(n) for n in ('cls', 'grp', 'mk')]
P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, verbose=-1, num_threads=2, seed=0)
def fit(tr): return lgb.train(P, lgb.Dataset(XA[tr], ya[tr], categorical_feature=cat), 300)
def train_pred(tr, te):
    if ya[tr].sum() < 5 or not te.any(): return None
    return fit(tr).predict(XA[te])
t0 = time.time(); fold_taus = []; tpfp = [0, 0]
for k, held in enumerate(folds):
    if not (fid_all_arch == k).any(): fold_taus.append(None); continue
    S = np.full(N, -1.0); trm = fid_a != k; te = fid_a == k
    p = train_pred(trm, te)
    if p is not None: S[ap_idx[te]] = p
    tf = sorted(set(fam_a[trm])); rng = np.random.default_rng(k); lab = dict(zip(tf, rng.permutation(len(tf)) % 5)); inner = np.array([lab.get(f, -1) for f in fam_a])
    for j in range(5):
        q = train_pred(trm & (inner != j), trm & (inner == j))
        if q is not None: S[ap_idx[trm & (inner == j)]] = q
    S[las_all & arch] = -1
    cv = arch_curve(S); sw = sweep_groups({A: curve_where(cv, regions - held)})
    taus = {g: float(gv['threshold']) for g, gv in sw[A]['groups'].items()}; fold_taus.append(taus)
    hc = curve_where(cv, held); tau = np.array([taus.get(g, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
    tpfp[0] += int(hc.hit[kept].sum()); tpfp[1] += int((~hc.hit[kept]).sum())
    print('fold', k, taus, f'{time.time()-t0:.0f}s', flush=True)
ref = json.load(open('/workspace/koz3_shared/bot-6/work/orgate/res/res_k10_grp.json'))['G']['arch_tp_fp']
print('k10 G arch TP/FP', tpfp, 'ref', ref, 'REPRO_OK' if tpfp == list(ref) else 'REPRO_MISMATCH', flush=True)
groups = sorted({g for t in fold_taus if t for g in t})
med = {g: float(np.median([t[g] for t in fold_taus if t and g in t])) for g in groups}
json.dump(dict(folds=[sorted(f) for f in folds], fold_taus=fold_taus, median=med, k10_arch_tp_fp=tpfp, ref=ref), open(OUT/'fold_taus.json', 'w'), ensure_ascii=False, indent=1)
print('median taus', med, flush=True)
# финальная модель на всех семьях (тот же пул, метка, гиперпараметры)
bst = fit(np.ones(len(ap_idx), bool)); bst.save_model(str(OUT/'orgate_g.txt'))
rmeta = json.load(open('/workspace/koz3_shared/bot-6/repo/models/rerank_arch_lgbm.json'))
assert rmeta['names'] == n2
sha = hashlib.sha256((OUT/'orgate_g.txt').read_bytes()).hexdigest()
meta = dict(variant='G', names=names, x2_names=n2, gate_names=ng, classes=rmeta['classes'], grp_vocab=rmeta['grp_vocab'], mk_vocab=rmeta['mk_vocab'],
            thresholds=med, threshold_rule='median of 10 k10 fold thresholds (sweep_groups on inner-OOF of training families, LAS rule applied)',
            fold_thresholds=fold_taus, las_rule='arch in region with LAS input -> dropped (score -1)', arch_group_alias={'sp_or': 'sp_arch'},
            train=dict(rows=int(len(ap_idx)), positives=int(ya.sum()), rounds=300, params=P, model_sha256=sha,
                       table='/workspace/rerank/work/table.pkl', feats='bot-6/work/orgate/feats_gate.npz'))
json.dump(meta, open(OUT/'orgate_g.json', 'w'), ensure_ascii=False, indent=1, default=float)
# офлайн-решения финальной модели на всех arch-кандидатах (для сверки с пайплайном)
S = np.full(N, -1.0); S[ap_idx] = bst.predict(XA); S[las_all & arch] = -1
grp = T['grp']; tau_all = np.array([med.get(g, np.inf) for g in grp])
keep = arch & (S >= tau_all)
per = {}
for r in np.unique(reg[arch]):
    ia = np.flatnonzero(arch & (reg == r))
    per[str(r)] = dict(n_arch=int(ia.size), n_pool=int(pool[ia].sum()), las=bool(las_all[ia].any()), n_or=int((grp[ia] == 'or').sum()),
                       kept=[int(i) for i in ia[keep[ia]]], removed=[int(i) for i in ia[~keep[ia]]],
                       score_kept=[float(S[i]) for i in ia[keep[ia]]])
json.dump(per, open(OUT/'offline_decisions.json', 'w'), ensure_ascii=False)
print('model sha256', sha, 'size', (OUT/'orgate_g.txt').stat().st_size, 'kept total', int(keep.sum()), 'of', int(arch.sum()), f'{time.time()-t0:.0f}s')
