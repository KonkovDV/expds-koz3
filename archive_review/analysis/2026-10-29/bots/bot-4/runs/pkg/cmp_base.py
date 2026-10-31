"""Пакет bot-7 (без crop/twins, коробка) против базы stack_arch: canon counts + бутстреп семей."""
import sys, pickle, json
sys.path.insert(0, '/workspace/rerank')
from common import score_counts, paired_bootstrap_counts, site_family
from expds.metric import CLASS_WEIGHTS as WT
b = pickle.load(open('/workspace/koz3_shared/bot-4/work/thr/counts_canon.pkl', 'rb'))
p = pickle.load(open('/workspace/koz3_shared/bot-4/work/pkg/thr/counts_canon.pkl', 'rb'))
out = {}
for name, x, y in [('PKG_A-A', p['A'], b['A']), ('PKG_R-A', p['R'], b['A']), ('PKG_R-R', p['R'], b['R'])]:
    sx, sy = score_counts(x), score_counts(y)
    bt = paired_bootstrap_counts(x, y, site_family, 1000)
    out[name] = dict(score=sx[0], ref=sy[0], d=sx[0] - sy[0], boot={k: bt[k] for k in ('mean', 'p5', 'p95', 'p_gt0')},
                     dF1={c: sx[1][c]['f1'] - sy[1][c]['f1'] for c in sx[1]}, wd={c: WT[c] * (sx[1][c]['f1'] - sy[1][c]['f1']) / 100 for c in sx[1]})
    print(name, json.dumps(out[name], default=float)[:900])
json.dump(out, open('cmp_base_canon.json', 'w'), indent=1, default=float)
