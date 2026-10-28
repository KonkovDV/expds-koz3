"""THR (bot-4): вложенный выбор стратегии порогов по классу (S0 max / S1 plateau / S2 no-lidar) поверх реранкера arch.
Объявление: koz3_shared/bot-4/THR.md. Запуск: OMP_NUM_THREADS=2 nice -n 10 venv/bin/python work/thr/thr_eval.py [--layouts canon,seed0..]"""
import sys, json, time, pickle, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, '/workspace/rerank'); sys.path.insert(0, '/workspace/expds-koz3')
import fast
from common import site_family, family_folds, score_counts, paired_bootstrap_counts, sets, load, W
from expds.metric import sweep_groups, curve_where, CLASS_WEIGHTS
ap = argparse.ArgumentParser(); ap.add_argument('--layouts', default='canon,seed0,seed1,seed2,seed3,seed4,lofo'); ap.add_argument('--boot', type=int, default=1000)
a = ap.parse_args()
OUT = Path(__file__).parent
T = fast.T; CL = fast.CL; ARCH = fast.cid['arkhitektury']; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']
gt = load(W / 'gt_train_v2.geojson'); new, optical = sets(gt); del gt
boxes = set(json.loads(Path('/workspace/expds-koz3/notes/las_boxes_59.json').read_text(encoding='utf-8')))
base_cv = fast.curves()
regions = {r for c in base_cv.values() for r in c.pred_region} | {r for c in base_cv.values() for r in c.gt_regions}
NLR = {r for r in regions if r not in boxes}
def rff(regions, k, seed):
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    nm = sorted(fams); rng = np.random.default_rng(seed); lab = rng.permutation(len(nm)) % k
    return [set().union(*(fams[n] for n, l in zip(nm, lab) if l == i)) for i in range(k)]
canon = family_folds(regions, site_family, 10)
# arch-only curve for a given score vector
arch_idx = np.flatnonzero(cls == ARCH)
def arch_curve(score):
    keep = np.zeros(len(cls), bool); keep[arch_idx] = True
    return fast.curves(score=score, keep=keep)['arkhitektury']
def mixed(outer, train_scores, trainmask):
    S = conf.copy(); m = cls == ARCH
    S[m & trainmask] = train_scores[m & trainmask]; S[m & ~trainmask] = outer[m & ~trainmask]
    return S
def taus(curves, regs, pick='max'):
    res = sweep_groups({c: curve_where(cv, regs) for c, cv in curves.items()}, pick=pick); t = {}
    for c, v in res.items():
        for g, gv in v['groups'].items(): t.setdefault(c, {})[g] = gv['threshold']
    return t
def strat_taus(curves, train):
    tb = taus(curves, train)
    return {'S0': (tb, None), 'S1': (taus(curves, train, 'plateau'), None), 'S2': (tb, taus(curves, train & NLR))}
def add(counts, curves, held, ST, choice):
    for c, cv in curves.items():
        hc = curve_where(cv, held); per = counts[c]
        for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
        if hc.conf.size == 0: continue
        tb, tn = ST[choice.get(c, 'S0')]; tbc = tb.get(c, {}); tnc = (tn or {}).get(c, {})
        tau = np.array([(tnc.get(g, tbc.get(g, np.inf)) if (tn is not None and r in NLR) else tbc.get(g, np.inf))
                        for g, r in zip(hc.pred_group, hc.pred_region)])
        kept = hc.conf >= tau
        for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
            per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
def nested_choice(curves, train, inner_folds):
    cnt = {s: {c: {} for c in CL} for s in ('S0', 'S1', 'S2')}
    for ih in inner_folds:
        ih = ih & train
        if not ih: continue
        ST = strat_taus(curves, train - ih)
        for s in cnt: add(cnt[s], curves, ih, ST, {c: s for c in CL})
    pf = {s: score_counts(cnt[s])[1] for s in cnt}; ch = {}
    for c in CL:
        f0 = pf['S0'][c]['f1']; best = max(('S1', 'S2'), key=lambda s: (pf[s][c]['f1'], s == 'S1'))
        ch[c] = best if pf[best][c]['f1'] > f0 else 'S0'
    return ch
def load_rr(layout):
    if layout == 'canon':
        z = np.load(W / 'work/nested_v2.npz'); return z['outer'], (lambda k: z['inner'][k])
    if layout == 'lofo':
        z = np.load(W / 'work/v3/nested_lofo.npz'); zc = np.load(W / 'work/nested_v2.npz')['outer']; return z['outer'], (lambda k: zc)
    z = np.load(W / f'work/v3/nested_{layout}.npz'); inner = z['inner']; return z['outer'], (lambda k: inner[k])
def run_layout(layout):
    folds = canon if layout == 'canon' else (family_folds(regions, site_family, None) if layout == 'lofo' else rff(regions, 10, int(layout[4:])))
    outer, inner_of = load_rr(layout)
    fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in reg])
    V = ('A', 'R', 'N', 'RN', 'P1', 'P2'); counts = {v: {c: {} for c in CL} for v in V}; chosen = {'N': [], 'RN': []}
    every = set().union(*folds)
    for k, held in enumerate(folds):
        train = every - held
        if layout == 'lofo':
            inner_folds = [f - held for f in canon if f - held]
        else:
            inner_folds = [f for j, f in enumerate(folds) if j != k]
        cv0 = base_cv
        cvR = dict(base_cv); cvR['arkhitektury'] = arch_curve(mixed(outer, inner_of(k), fid != k))
        for name, cv in (('A', cv0), ('R', cvR)):
            ST = strat_taus(cv, train)
            if name == 'A':
                add(counts['A'], cv, held, ST, {}); add(counts['P1'], cv, held, ST, {c: 'S1' for c in CL}); add(counts['P2'], cv, held, ST, {c: 'S2' for c in CL})
                ch = nested_choice(cv, train, inner_folds); chosen['N'].append(ch); add(counts['N'], cv, held, ST, ch)
            else:
                add(counts['R'], cv, held, ST, {})
                ch = nested_choice(cv, train, inner_folds); chosen['RN'].append(ch); add(counts['RN'], cv, held, ST, ch)
        if layout == 'lofo' and k % 10 == 0: print(layout, k, f'{time.time()-t0:.0f}s', flush=True)
    return counts, chosen
WT = CLASS_WEIGHTS
def summary(cnt, ref):
    s = [score_counts(cnt)[0], score_counts(cnt, new)[0], score_counts(cnt, optical)[0]]
    r = [score_counts(ref)[0], score_counts(ref, new)[0], score_counts(ref, optical)[0]]
    pa, pb = score_counts(cnt)[1], score_counts(ref)[1]
    wd = {c: WT[c] * (pa[c]['f1'] - pb[c]['f1']) / 100 for c in CL if abs(pa[c]['f1'] - pb[c]['f1']) > 1e-12}
    return dict(score=s, d=s[0] - r[0], d_new=s[1] - r[1], d_opt=s[2] - r[2], wdelta=wd)
t0 = time.time(); res = {'n_regions': len(regions), 'n_nl': len(NLR)}
for lay in a.layouts.split(','):
    cnt, chosen = run_layout(lay); L = {'chosen': chosen}
    for v in cnt: L[v] = summary(cnt[v], cnt['A'])
    L['RN_vs_R'] = summary(cnt['RN'], cnt['R'])
    if lay == 'canon':
        for v in ('R', 'N', 'RN', 'P1', 'P2'): L[v]['boot'] = paired_bootstrap_counts(cnt[v], cnt['A'], site_family, a.boot)
        L['RN_vs_R']['boot'] = paired_bootstrap_counts(cnt['RN'], cnt['R'], site_family, a.boot)
        pickle.dump(cnt, open(OUT / 'counts_canon.pkl', 'wb'))
    res[lay] = L
    print(lay, {v: (round(L[v]['score'][0], 6), round(L[v]['d'], 5)) for v in cnt}, 'RN-R', round(L['RN_vs_R']['d'], 5), f'{time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(OUT / f'res_thr_{a.layouts.replace(",", "_")}.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('done', time.time() - t0)
