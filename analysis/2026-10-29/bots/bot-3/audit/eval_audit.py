"""Аудит bot-3 (29.10): honest-10 / LOFO, 4 режима, П1–П5 (geo_gap/evlib.criteria) для вариантов
VARIANTS_DECLARED.md против S (pixgbm/pix_lofo.geojson, правило замены variant_S из pixgbm/eval_pix.py).
argv: имена вариантов (audit/var/<имя>/pix_lofo.geojson). Выход audit/out/honest_audit_<TAG>.json."""
import os, sys, json, collections
GG = '/workspace/koz3_shared/bot-3/geo_gap'; A = '/workspace/koz3_shared/bot-3/audit'
NAMES = sys.argv[1:]
sys.path.insert(0, GG); os.chdir(GG)
sys.argv = ['x']
exec(open('eval_r.py').read().split('FP = stack_preds')[0])
from evlib import *
TAG = os.environ.get('TAG', 'v1')
CL = ['finds_points', 'object_poly']
STACK = {c: stack_preds(c) for c in CL}
mt_of = lambda grp: grp.replace('_pix', '')

def load_pix(path):
    P = {c: [(g, cf, r, grp, 'pix') for g, cf, r, grp, _ in v] for c, v in file_preds(path).items()}
    for c in CL: P.setdefault(c, [])
    return P

def variant_S(PIX, c, base):            # дословно pixgbm/eval_pix.py
    cov = {(p[2], mt_of(p[3])) for p in PIX[c]}
    return [p for p in base if (p[2], p[3]) not in cov] + PIX[c]

VARS = {'S': (lambda P: {c: variant_S(P, c, STACK[c]) for c in CL})(load_pix('/workspace/koz3_shared/bot-3/pixgbm/pix_lofo.geojson'))}
for n in NAMES:
    VARS[n] = (lambda P: {c: variant_S(P, c, STACK[c]) for c in CL})(load_pix(f'{A}/var/{n}/pix_lofo.geojson'))
print('n', {k: {c: len(v[c]) for c in CL} for k, v in VARS.items()}, flush=True)
out = {}
for scope in ['global', 'site']:
    cv = {k: curves_of(v, scope) for k, v in VARS.items()}
    for a in NAMES:
        for unseen in ['drop', 'keep']:
            r = criteria(cv[a], cv['S'], unseen); key = f'{a}_vs_S|{scope}|{unseen}'; out[key] = r
            print(key, 'base', [round(x, 6) for x in r['base']], 'var', [round(x, 6) for x in r['var']],
                  'd10 %+.6f lofo %+.6f rnd %s P(>0) %.3f' % (r['d_honest10'], r['d_lofo'], [round(x, 5) for x in r['d_rand']], r['boot'].get('p_gt0', 0)),
                  'P1-4', r['P1'], r['P2'], r['P3'], r['P4'], 'P5<MDE', r['P5_below_MDE'],
                  '| opoly', {k: r['per_class_var']['object_poly'][k] for k in ('tp', 'fp', 'f1')}, 'vs', {k: r['per_class_base']['object_poly'][k] for k in ('tp', 'fp', 'f1')},
                  '| finds', {k: r['per_class_var']['finds_points'][k] for k in ('tp', 'fp', 'f1')}, 'vs', {k: r['per_class_base']['finds_points'][k] for k in ('tp', 'fp', 'f1')}, flush=True)
json.dump(out, open(f'{A}/out/honest_audit_{TAG}.json', 'w'), ensure_ascii=False, indent=1, default=float)
