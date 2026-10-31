"""bot-6: пороги групп на всех участках (sweep_groups) по кривым bot-10 work9/curves_orsp_g (fusion) и work/curves (A)."""
import sys, json
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo'); sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from expds import metric as M
from honest_bootstrap import load_curves
out = {}
for name, src in (('orsp', '/workspace/koz3_shared/bot-10/work9/curves_orsp_g'), ('A', '/workspace/koz3_shared/bot-10/work/curves')):
    res = M.sweep_groups(load_curves(Path(src)))
    out[name] = {g: {c: float(gv['threshold']) for c, v in res.items() for gg, gv in v['groups'].items() if gg == g}
                 for g in ('sp_focal', 'sp_focal2', 'sp_or')}
    print(name, out[name], flush=True)
json.dump(out, open('taus_all.json', 'w'), ensure_ascii=False, indent=1)
