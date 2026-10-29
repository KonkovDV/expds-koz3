"""bot-10 work11: in-sample пороги («табло» аудита) — sweep_groups по ВСЕМ регионам кривых. Формат {"thresholds": {group: {class: tau}}}.
    python taus_insample.py <curves_dir> <out.json>"""
import sys, json
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo'); sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from expds import metric as M
from honest_bootstrap import load_curves
cv = load_curves(Path(sys.argv[1]))
res = M.sweep_groups(cv)
th = {}
for c, v in res.items():
    for g, gv in v['groups'].items(): th.setdefault(g, {})[c] = float(gv['threshold'])
json.dump({'thresholds': th, 'src': sys.argv[1]}, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print(json.dumps(th, ensure_ascii=False))
