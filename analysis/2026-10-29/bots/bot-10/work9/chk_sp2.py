"""bot-10: есть ли в стеке оценки (work/fam) кандидаты sp_focal2 ниже пола выдачи пайплайна (0,501645/0,645505)
и в ≤8/12 м от sp_focal — т. е. то, чего pipeline._wide_without_lidar (floors + drop_near_kurgans 12 м) не выдаёт."""
import json, sys, math
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import shape
FL = {'kurgany_povrezhdennye': 0.501645, 'kurgany_tselye': 0.645505}
st = Counter(); confs = []
for fp in sorted(Path('/workspace/koz3_shared/bot-10/work/fam').glob('*.jsonl')):
    if not Path('fam_orsp', fp.name).is_symlink():
        pts = {}
        for ln in open(fp, encoding='utf-8'):
            if '"sp_focal' not in ln: continue
            f = json.loads(ln); p = f['properties']; g = p.get('threshold_group') or p.get('markup_type')
            if g not in ('sp_focal', 'sp_focal2') or p['class_name'] not in FL: continue
            c = shape(f['geometry']).centroid; s = math.cos(math.atan(math.sinh(c.y / 6378137.0)))
            pts.setdefault((p['region_name'], g), []).append((c.x * s, c.y * s, float(p['confidence']), p['class_name']))
        for (r, g), v in pts.items():
            if g != 'sp_focal2': continue
            st['sp2'] += len(v); st['sp2_below_floor'] += sum(cf < FL[k] for _, _, cf, k in v); confs += [cf for _, _, cf, _ in v]
            a = pts.get((r, 'sp_focal'))
            if a:
                t = cKDTree(np.array([(x, y) for x, y, _, _ in a]))
                d, _ = t.query(np.array([(x, y) for x, y, _, _ in v]))
                st['sp2_within8_of_sp'] += int((d <= 8).sum()); st['sp2_within12_of_sp'] += int((d <= 12).sum())
                st['sp2_above_floor_within8'] += int(sum(dd <= 8 and cf >= FL[k] for dd, (_, _, cf, k) in zip(d, v)))
            else:
                st['sp2_regions_without_sp'] += 1
            st['regions_sp2'] += 1
print(dict(st)); print('sp2 conf quantiles', np.quantile(confs, [0, .05, .5, .95]).round(4).tolist() if confs else None)
