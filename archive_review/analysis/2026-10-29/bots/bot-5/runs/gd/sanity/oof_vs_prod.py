"""bot-5 (A): e2e-санити gor_dense. Прод (time_phase.log: GD.apply на 132 участках SpOr трейна, голова на всём трейне)
против OOF (те же пики/ворота, что PFnoD бота 10: rows_of(fr,'all+cp',gate_of(train,held)), τ на фолд = bot-10
work14/pkg_gor2_counts.json chosen[*]['PFnoD']). Метрики: доля участков с ≥1 точкой conf ≥ τ; медиана conf выданных точек."""
import os, re, json, statistics as S, sys
from pathlib import Path
W = Path('/workspace/koz3_shared/bot-5/work/gd'); TAU = 0.824066
# ---- прод ----
rx = re.compile(r"INFO expds\.gor_dense: (\S+): gor_dense (\d+) точек \(conf ([0-9., ]+)\)")
prod = {m.group(1): [float(x) for x in m.group(3).split(',')] for m in map(rx.search, open(W / 'time_phase.log', encoding='utf-8')) if m}
def stats(per, tau):   # per: {регион: [conf…]}
    kept = [c for v in per.values() for c in v if c >= tau]
    return dict(n_regions=len(per), share_regions_det=round(sum(any(c >= tau for c in v) for v in per.values()) / max(len(per), 1), 4),
                n_kept=len(kept), median_kept=round(S.median(kept), 4) if kept else None,
                median_all=round(S.median([c for v in per.values() for c in v]), 4))
RES = {'tau_fixed': TAU, 'prod': {'source': 'work/gd/time_phase.log (conf округлён до 3 знаков в логе)', **stats(prod, TAU)},
     'prod_borderline_0.824': sum(1 for v in prod.values() for c in v if abs(c - TAU) < 0.0006)}
# ---- OOF ----
os.environ['MODE'] = 'none'
src = W / 'oofsrc/scripts/gor_dense/gor_dense_diag.py'   # = repo 6df3e79 (машинерия, которую exec-ит bot-10 pkg_gor.py)
_d = src.read_text(encoding='utf-8').split("\nif MODE == 'diag':")[0]
__file__ = str(src); exec(compile(_d, 'gor_dense_diag_head', 'exec'))
CH = json.load(open('/workspace/koz3_shared/bot-10/work14/pkg_gor2_counts.json', encoding='utf-8'))['chosen']
RES['oof'] = {}
for nm in [x for x in ('fam10', 'LOFO', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4') if x in PK['layouts']]:
    folds = layouts[nm]; every = set().union(*folds); pf, pfix = {}, {}
    for k, (fr, held) in enumerate(zip(PK['layouts'][nm], folds)):
        gp_ = gate_of(every - held, held); rw = rows_of(fr, 'all+cp', gp_)
        per = {r: [] for r in held if r in SCNS}
        for row in rw: per.setdefault(row[0], []).append(float(row[1]))
        t = CH.get(nm, {}).get('PFnoD', [None] * len(folds))[k]
        for r, v in per.items(): pf[r] = (v, t)
    by_t = {r: [c for c in v if t is not None and c >= t] for r, (v, t) in pf.items()}
    kept = [c for v in by_t.values() for c in v]
    RES["oof"][nm] = {'tau_per_fold_bot10': stats({r: v for r, (v, t) in pf.items()}, -1) | {
                        'share_regions_det': round(sum(bool(v) for v in by_t.values()) / len(by_t), 4), 'n_kept': len(kept),
                        'median_kept': round(S.median(kept), 4) if kept else None},
                    'tau_fixed': stats({r: v for r, (v, t) in pf.items()}, TAU)}
    print(nm, json.dumps(RES["oof"][nm], ensure_ascii=False), flush=True)
print('PROD', json.dumps(RES['prod'], ensure_ascii=False))
json.dump(RES, open(W / 'sanity/oof_vs_prod.json', 'w'), ensure_ascii=False, indent=1)
