"""Сводка work/eval_geom.pkl: таблица вариантов, бутстреп по семьям против base и
вложенный (nested) выбор варианта: для каждого фолда семей вариант класса выбирается по
F1 на ОСТАЛЬНЫХ фолдах, а его счётчики берутся на удержанном фолде — выбор варианта
тоже не видит удержанных семей. Запуск: ../venv/bin/python summarize.py [eval_geom.pkl]"""
from common7 import *
from expds.metric import CLASS_WEIGHTS, _f1
db = pickle.load(open(W / (sys.argv[1] if len(sys.argv) > 1 else 'eval_geom.pkl'), 'rb'))
gt = load('/workspace/rerank/gt_train_v2.geojson'); new, opt = sets(gt)
regions = {r for v in db['base']['counts'].values() for r in v} | {g['properties']['region_name'] for g in gt['features']}
folds = family_folds(regions, site_family, 10)
FAM = {'kur': ['kurgany_povrezhdennye', 'kurgany_tselye'], 'gor': ['gorodishcha'], 'obj': ['object_poly'], 'fort': ['fortifikatsii']}
def tot(counts, c, regs=None):
    v = counts.get(c, {}); s = [0, 0, 0]
    for r, x in v.items():
        if regs is None or r in regs:
            for i in range(3): s[i] += x[i]
    return s
def wf1(counts, cls, regs):
    num = den = 0
    for c in cls:
        tp, fp, n = tot(counts, c, regs)
        if n == 0: continue
        num += CLASS_WEIGHTS[c] * _f1(tp, fp, n - tp); den += CLASS_WEIGHTS[c]
    return num / den if den else 0
out = {'variants': {}, 'nested': {}}
L = []
b = db['base']['counts']; b0 = db['base']['exact_honest']
L.append(f"{'вариант':20s} {'all':>8s} {'Δall':>8s} {'new34':>8s} {'optical':>8s} {'P(Δ>0)':>7s} {'Δ p5..p95':>18s}  F1 затронутых классов (tp/fp)")
for n, v in db.items():
    s = v['exact_honest']
    bs = paired_bootstrap_counts(v['counts'], b, site_family, n=1000) if n != 'base' else {}
    fam = n.split('_')[0]
    cls = FAM.get(fam, [])
    per = score_counts(v['counts'])[1]
    ctxt = ' '.join(f"{c[:10]} {per[c]['f1']:.4f}({tot(v['counts'],c)[0]}/{tot(v['counts'],c)[1]})" for c in cls)
    L.append(f"{n:20s} {s[0]:8.5f} {s[0]-b0[0]:+8.5f} {s[1]:8.5f} {s[2]:8.5f} {bs.get('p_gt0', float('nan')):7.3f} "
             f"{bs.get('p5', 0):+8.5f}..{bs.get('p95', 0):+8.5f}  {ctxt}")
    out['variants'][n] = {'exact_honest': s, 'insample': v['ins'], 'approx_honest': v['approx_honest'], 'bootstrap_vs_base': bs,
                          'per_class': {c: {'f1': per[c]['f1'], 'tp_fp_ngt': tot(v['counts'], c)} for c in per}}
# вложенный выбор
comb = {c: dict(r) for c, r in b.items()}
comb = {c: {} for c in b}
picks = {f: [] for f in FAM}
for held in folds:
    train = regions - held
    chosen = {}
    for f, cls in FAM.items():
        cand = [n for n in db if n == 'base' or n.split('_')[0] == f]  # combo не участвует
        best = max(cand, key=lambda n: (wf1(db[n]['counts'], cls, train), n == 'base'))
        picks[f].append(best)
        for c in cls: chosen[c] = best
    for c in b:
        src = db[chosen.get(c, 'base')]['counts'].get(c, {})
        for r, x in src.items():
            if r in held: comb[c][r] = list(x)
s = [score_counts(comb)[0], score_counts(comb, new)[0], score_counts(comb, opt)[0]]
bs = paired_bootstrap_counts(comb, b, site_family, n=1000)
per = score_counts(comb)[1]
L.append('')
L.append(f"NESTED (выбор варианта по 9 фолдам, счёт на 10-м): all {s[0]:.5f} (Δ {s[0]-b0[0]:+.5f}) new34 {s[1]:.5f} optical {s[2]:.5f}; "
         f"P(Δ>0) {bs['p_gt0']:.3f}, Δ p5..p95 {bs['p5']:+.5f}..{bs['p95']:+.5f}")
L.append('  per-class F1: ' + ' '.join(f"{c[:10]} {per[c]['f1']:.4f}" for c in per))
from collections import Counter
for f, p in picks.items(): L.append(f"  выбор {f}: {dict(Counter(p))}")
out['nested'] = {'exact_honest': s, 'bootstrap_vs_base': bs, 'picks': picks, 'per_class': {c: per[c]['f1'] for c in per}}
txt = '\n'.join(L); print(txt)
stem = (sys.argv[1] if len(sys.argv) > 1 else 'eval_geom.pkl').replace('.pkl', '')
open(W / f'{stem}_summary.txt', 'w').write(txt + '\n'); json.dump(out, open(W / f'{stem}_summary.json', 'w'), ensure_ascii=False, indent=1)
