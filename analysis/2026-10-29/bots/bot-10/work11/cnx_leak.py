"""bot-10 work11 C2: утечка ConvNeXt — делит ли k5-раскладка бота 9 (region_fold.json) семью/юнит honest-10 между фолдами."""
import json, sys, collections
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds.naming import site_family
RF = json.load(open('/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f01_out/region_fold.json', encoding='utf-8'))['region_fold']
UNIT = {}
for ms in json.load(open('/workspace/koz3_shared/bot-10/work/family_clusters.json', encoding='utf-8'))['clusters'].values():
    for m in ms: UNIT[m] = '+'.join(sorted(ms))
u2f = collections.defaultdict(set); u2r = collections.defaultdict(list)
for r, f in RF.items():
    u = UNIT.get(site_family(r), site_family(r)); u2f[u].add(f); u2r[u].append(r)
split = {u: sorted(f) for u, f in u2f.items() if len(f) > 1}
fold_n = collections.Counter(RF.values())
out = {'n_sites': len(RF), 'fold_sizes': dict(sorted(fold_n.items())), 'n_units': len(u2f), 'split_units': {u: {'folds': split[u], 'sites': {r: RF[r] for r in u2r[u]}} for u in split},
       'clean_sites': sorted(r for u, rs in u2r.items() if u not in split for r in rs)}
print(json.dumps({k: v for k, v in out.items() if k != 'clean_sites'}, ensure_ascii=False, indent=1)); print('clean', len(out['clean_sites']))
json.dump(out, open('cnx_leak.json', 'w'), ensure_ascii=False, indent=1)
