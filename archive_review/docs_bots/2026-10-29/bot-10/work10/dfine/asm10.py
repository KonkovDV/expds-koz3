"""bot-10 work10: каталоги кривых (симлинки / слияние по классам) для D-FINE C.
 c5_Cg = work/curves, затронутые юниты -> curves_C_g; c5_Cs = work4/curves_site, затронутые -> curves_C_s;
 PKC_g = work7/curves_g_Ss6 (пакет S+stack6) с классами курганов/gor/arch/fort из c5_Cg (finds/object_poly из Ss6).
 Сверка: база, пересчитанная здесь (curves_base_g/_s), == work/curves и work4/curves_site; D-FINE не трогает finds/object_poly."""
import json, os, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
W = Path('/workspace/koz3_shared/bot-10'); D = Path(__file__).resolve().parent; os.chdir(D)
T = json.load(open('touched.json')); units = set(T['units'])
A = {'g': W / 'work/curves', 's': W / 'work4/curves_site'}
def eq(x, y):
    if set(x) != set(y): return False
    return all(np.array_equal(np.sort(x[k].conf), np.sort(y[k].conf)) and int(np.sum(x[k].hit)) == int(np.sum(y[k].hit)) and x[k].gt_regions == y[k].gt_regions for k in x)
rep = {}
for sc in ('g', 's'):
    allu = {p.stem for p in A[sc].glob('*.pkl')}
    bad = [u for u in sorted(units) if not eq(pickle.load(open(A[sc] / f'{u}.pkl', 'rb')), pickle.load(open(D / f'curves_base_{sc}' / f'{u}.pkl', 'rb')))]
    rep[f'base_check_{sc}'] = {'units': len(units), 'differ': bad, 'not_in_A': sorted(units - allu)}
    d = D / f'c5_C{sc}'; d.mkdir(exist_ok=True)
    for p in d.glob('*.pkl'): p.unlink()
    for u in sorted(allu):
        os.symlink((D / f'curves_C_{sc}' / f'{u}.pkl') if u in units else (A[sc] / f'{u}.pkl'), d / f'{u}.pkl')
    # D-FINE не меняет finds/object_poly
    fo = [u for u in sorted(units) if not all(np.array_equal(pickle.load(open(A[sc] / f'{u}.pkl', 'rb'))[k].conf, pickle.load(open(D / f'curves_C_{sc}' / f'{u}.pkl', 'rb'))[k].conf)
                                              for k in ('finds_points', 'object_poly') if k in pickle.load(open(A[sc] / f'{u}.pkl', 'rb')))]
    rep[f'finds_op_changed_{sc}'] = fo
P = W / 'work7/curves_g_Ss6'; d = D / 'PKC_g'; d.mkdir(exist_ok=True)
for p in d.glob('*.pkl'): p.unlink()
mixed = []
for p in sorted(P.glob('*.pkl')):
    u = p.stem; s6 = pickle.load(open(p, 'rb'))
    if u not in units: os.symlink(p.resolve(), d / p.name); continue
    c = pickle.load(open(D / 'curves_C_g' / p.name, 'rb'))
    out = {k: (s6[k] if k in ('finds_points', 'object_poly') else c[k]) for k in set(s6) | set(c)}
    pickle.dump(out, open(d / p.name, 'wb')); mixed.append(u)
rep['PKC_mixed_units'] = mixed
json.dump(rep, open('asm10.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False))
