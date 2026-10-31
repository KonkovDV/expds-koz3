"""bot-10 work11: по стеку work/fam для курганов — по (группа, класс): число, мин/квантили conf, доля < 0.05,
число участков с >=30 объектов conf>=0.05 (min_n site_calibration). Для проверки пола выдачи (neigh, calib)."""
import json, collections, sys
import numpy as np
from pathlib import Path
KUR = {'kurgany_povrezhdennye', 'kurgany_tselye'}
C = collections.defaultdict(list); RC = collections.defaultdict(collections.Counter)
for p in sorted(Path('/workspace/koz3_shared/bot-10/work/fam').glob('*.jsonl')):
    for ln in open(p, encoding='utf-8'):
        i = ln.find('"properties"'); pr = json.loads(ln)['properties']
        if pr.get('class_name') not in KUR: continue
        g = pr.get('threshold_group') or pr.get('markup_type'); c = float(pr.get('confidence', 1)); C[(g, pr['class_name'])].append(c)
        if c >= 0.05: RC[(g, pr['class_name'])][pr.get('region_name')] += 1
out = {}
for k, v in sorted(C.items()):
    v = np.asarray(v); r = RC[k]
    out['|'.join(k)] = {'n': int(len(v)), 'min': float(v.min()), 'q01': float(np.quantile(v, .01)), 'q10': float(np.quantile(v, .1)), 'q50': float(np.quantile(v, .5)),
        'frac_lt_0.05': float((v < 0.05).mean()), 'regions': len(r), 'regions_n_ge_30': sum(1 for x in r.values() if x >= 30)}
    print(k, out['|'.join(k)], flush=True)
json.dump(out, open('fam_conf_scan.json', 'w'), indent=1)
