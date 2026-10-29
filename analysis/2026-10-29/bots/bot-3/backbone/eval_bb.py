"""honest-10 (4 режима, П1–П5 geo_gap/evlib.criteria) для бэкбон-вариантов (bot-3, backbone/VARIANTS_DECLARED.md).
База S = pixgbm/pix_lofo.geojson через variant_S (как в pixgbm/eval_pix.py); Bx = backbone/pix_lofo_Bx.geojson тем же правилом.
env PAIRS (по умолч. B1:S,B2:S,B1:stack,B2:stack,S:stack), TAG."""
import os, sys, json, collections
BBW = '/workspace/koz3_shared/bot-3/backbone'
src = open('/workspace/koz3_shared/bot-3/pixgbm/eval_pix.py').read().split('VARS = {')[0]
exec(src)                                   # GTS, STACK, PIX (S), variant_S, curves_of, criteria … ; cwd → pixgbm
S_PIX = PIX
def load_pix(fn):
    p = {c: [(g, cf, r, grp, 'pix') for g, cf, r, grp, _ in v] for c, v in file_preds(fn).items()}
    for c in CL: p.setdefault(c, [])
    return p
def var_S_with(pix, c, base):
    cov = {(p[2], mt_of(p[3])) for p in pix[c]}
    return [p for p in base if (p[2], p[3]) not in cov] + pix[c]
VARS = {'stack': STACK, 'S': {c: var_S_with(S_PIX, c, STACK[c]) for c in CL}}
for v in os.environ.get('BBV', 'B1,B2').split(','):
    fn = f'{BBW}/pix_lofo_{v}.geojson'
    if os.path.exists(fn):
        px = load_pix(fn)
        print(v, {c: collections.Counter(p[3] for p in px[c]) for c in CL}, flush=True)
        VARS[v] = {c: var_S_with(px, c, STACK[c]) for c in CL}
print('n', {k: {c: len(v[c]) for c in CL} for k, v in VARS.items()}, flush=True)
TAG = os.environ.get('TAG', 'bb')
PAIRS = [tuple(x.split(':')) for x in os.environ.get('PAIRS', 'B1:S,B2:S,B1:stack,B2:stack,S:stack').split(',')]
PAIRS = [p for p in PAIRS if p[0] in VARS and p[1] in VARS]
out = {}
for scope in ['global', 'site']:
    cv = {k: curves_of(v, scope) for k, v in VARS.items() if any(k in pr for pr in PAIRS)}
    for a, b in PAIRS:
        for unseen in ['drop', 'keep']:
            r = criteria(cv[a], cv[b], unseen); key = f'{a}_vs_{b}|{scope}|{unseen}'; out[key] = r
            print(key, 'base', [round(x, 6) for x in r['base']], 'var', [round(x, 6) for x in r['var']],
                  'd10 %+.6f lofo %+.6f rnd %s P(>0) %.3f' % (r['d_honest10'], r['d_lofo'], [round(x, 5) for x in r['d_rand']], r['boot'].get('p_gt0', 0)),
                  'P1-4', r['P1'], r['P2'], r['P3'], r['P4'], 'P5<MDE', r['P5_below_MDE'],
                  '| wd opoly %+.6f finds %+.6f' % (r['w_delta']['object_poly'], r['w_delta']['finds_points']),
                  '| opoly', {k: r['per_class_var']['object_poly'][k] for k in ('tp', 'fp', 'f1')},
                  '| finds', {k: r['per_class_var']['finds_points'][k] for k in ('tp', 'fp', 'f1')}, flush=True)
json.dump(out, open(f'{BBW}/honest_{TAG}.json', 'w'), ensure_ascii=False, indent=1, default=float)
