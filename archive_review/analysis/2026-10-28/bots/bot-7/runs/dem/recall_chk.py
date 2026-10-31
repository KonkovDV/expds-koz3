# bot-7: добавляют ли кандидаты DEM новые эталоны городищ в полноту кандидатов (по сравнению с базой A)?
import sys, json, glob, pickle, collections
from pathlib import Path
import shapely
ROOT = Path('/workspace/expds-koz3'); sys.path.insert(0, str(ROOT))
from expds import metric as M
M._SHM_CACHE_DIR = Path('/workspace/koz3_shared/bot-7/work/utm_cache')
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb')); GOR = 'gorodishcha'
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py')); calc = mod.F1Calculator()
G = shapely.from_wkb(pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))[GOR])
P, J, S = T['pairs'][(GOR, GOR)]; base_gt = set(int(j) for j in J)
res = {}
for tag, dirs in [('bot4_25', ['/workspace/koz3_shared/bot-4/work/dem/out']), ('bot7_34', [str(Path(__file__).parent / 'out')])]:
    feats = [f for d in dirs for p in sorted(glob.glob(d + '/*.geojson')) for f in json.load(open(p))['features']]
    new = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': feats}, False).get(GOR, [])
    tree = shapely.STRtree(G); hit = collections.defaultdict(list)
    for i, (g, cf, rg, grp) in enumerate(new):
        for j in tree.query(g):
            ok, s = calc._match_quality(g, G[j])
            if ok: hit[int(j)].append((rg, round(float(cf), 3)))
    res[tag] = {'cands': len(new), 'gt_hit': len(hit), 'gt_hit_new_vs_base': sorted(j for j in hit if j not in base_gt),
                'hits': {str(j): v for j, v in hit.items()}}
res['base_gt_with_candidate'] = len(base_gt); res['gor_gt'] = len(G)
print(json.dumps(res, ensure_ascii=False)); json.dump(res, open(Path(__file__).parent / 'res_recall.json', 'w'), ensure_ascii=False, indent=1)
