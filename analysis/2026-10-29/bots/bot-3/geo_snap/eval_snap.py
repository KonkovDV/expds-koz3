"""geo_snap шаг 3 (bot-3): honest-10, 4 режима, П1–П5 — P1 (peaks r=1,0) и P2 (r в фолде из {0,5;1,0;2,0}) против S.
finds = finds S. Кандидаты: cand/<mode>_all.geojson (predict_peaks.py, LOFO-модели)."""
from common import *
C = f'{W}/cand'
cand = {k: [(g, cf, r, grp, 'pix') for g, cf, r, grp, _ in file_preds(f'{C}/{k}_all.geojson').get('object_poly', [])]
        for k in ('pole', 'pk05', 'pk10', 'pk20')}
c = 'object_poly'
cnt = lambda v: collections.Counter((p[2], p[3]) for p in v)
rep = {'pole_eq_pix_lofo': cnt(cand['pole']) == cnt(PIX[c]), 'n': {k: len(v) for k, v in cand.items()}, 'n_pix_lofo': len(PIX[c])}
print(rep, flush=True)
FIN = SV['finds_points']
VAR = {'S': SV, 'S_rerun': {'finds_points': FIN, c: variant_S(c, STACK[c], cand['pole'])}}
for k in ('pk05', 'pk10', 'pk20'): VAR[k] = {'finds_points': FIN, c: variant_S(c, STACK[c], cand[k])}
# P2: r для отложенной геосемьи — по честному баллу object_poly на участках двух других геосемей (без её участков вообще)
GEOREG = {p[2] for p in PIX[c]} | {g[2] for g in GTS[c]}
FAMS = sorted({site_family(r) for r in GEOREG})
fam_regs = {F: {r for r in REGIONS if site_family(r) == F} for F in FAMS}
sel = {}
for scope in ['global', 'site']:
    cvk = {k: curves_of({c: VAR[k][c]}, scope)[c] for k in ('pk05', 'pk10', 'pk20')}
    sel[scope] = {}
    for F in FAMS:
        drop = fam_regs[F]; folds = [f - drop for f in FOLDS if f - drop]
        sc = {}
        for k, cv in cvk.items():
            cs, _ = M.honest_counts({c: M.curve_where(cv, REGIONS - drop)}, folds, unseen='keep')
            regs = (GEOREG - drop)
            tp = sum(v[0] for r, v in cs[c].items() if r in regs); fp = sum(v[1] for r, v in cs[c].items() if r in regs)
            ng = sum(v[2] for r, v in cs[c].items() if r in regs)
            sc[k] = 2 * tp / (2 * tp + fp + (ng - tp)) if tp else 0.0
        sel[scope][F] = dict(f1_other_geo=sc, r=max(sc, key=lambda k: (sc[k], k == 'pk10')))
        print('P2 выбор', scope, F, sel[scope][F], flush=True)
rep['P2_select'] = sel
out = {'check': rep}
for scope in ['global', 'site']:
    rf = {F: sel[scope][F]['r'] for F in FAMS}
    covset = {(q[2], mt_of(q[3])) for q in cand['pole']}
    P2 = [p for p in STACK[c] if (p[2], p[3]) not in covset]
    for F in FAMS: P2 += [p for p in cand[rf[F]] if site_family(p[2]) == F]
    VAR['P2'] = {'finds_points': FIN, c: P2}
    cv = {k: curves_of(VAR[k], scope) for k in ('S', 'S_rerun', 'pk10', 'P2', 'pk05', 'pk20')}
    for a, b in [('S_rerun', 'S'), ('pk10', 'S'), ('P2', 'S'), ('pk05', 'S'), ('pk20', 'S')]:
        for unseen in ['drop', 'keep']:
            r = criteria(cv[a], cv[b], unseen); key = f'{a}_vs_{b}|{scope}|{unseen}'; out[key] = r
            print(key, 'base', [round(x, 6) for x in r['base']], 'var', [round(x, 6) for x in r['var']],
                  'd10 %+.6f lofo %+.6f rnd %s P(>0) %.3f' % (r['d_honest10'], r['d_lofo'], [round(x, 5) for x in r['d_rand']], r['boot'].get('p_gt0', 0)),
                  'P1-4', r['P1'], r['P2'], r['P3'], r['P4'], 'P5<MDE', r['P5_below_MDE'],
                  '| opoly var', {k: r['per_class_var'][c][k] for k in ('tp', 'fp', 'f1')},
                  'base', {k: r['per_class_base'][c][k] for k in ('tp', 'fp', 'f1')}, flush=True)
json.dump(out, open(f'{W}/honest_snap.json', 'w'), ensure_ascii=False, indent=1, default=float)
