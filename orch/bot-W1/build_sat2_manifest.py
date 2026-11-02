#!/usr/bin/env python3
"""CPU-only: sat2 dataset manifest (region->family->fold, SpOr tifs, sp_or GT counts). Folds = nn_oof recipe:
candidates.family_folds over cache_spor_r1_t128_s25 index.csv families, weight = positive tiles, k=5."""
import csv, json, math, os, re, collections
from pathlib import Path
K = Path('/workspace/koz3'); RAW = K/'w5data/sat_raw'; CACHE = K/'w5data/cache_spor_r1_t128_s25'
def site_family(d):  # expds.naming.site_family: first word of name after NNN_
    name = re.sub(r'^\d+_', '', d); return re.split(r'[_\s]', name, maxsplit=1)[0]
rows = list(csv.DictReader(open(CACHE/'index.csv', encoding='utf-8')))
tiles = collections.Counter(r['family'] for r in rows); pos = collections.Counter(r['family'] for r in rows if r['positive'] == '1')
load = [0]*5; fold = {}
for f in sorted(tiles, key=lambda f: (-pos[f], f)):
    i = min(range(5), key=lambda j: load[j]); fold[f] = i; load[i] += pos[f]
cache_regions = collections.Counter(r['region'] for r in rows); cache_pos = collections.Counter(r['region'] for r in rows if r['positive'] == '1')
def area_m2(geom):
    def ring(c): return abs(sum(c[i][0]*c[i+1][1]-c[i+1][0]*c[i][1] for i in range(len(c)-1)))/2
    polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates'] if geom['type'] == 'MultiPolygon' else []
    a = sum(ring(p[0]) - sum(ring(h) for h in p[1:]) for p in polys)
    y = (polys[0][0][0][1] if polys else 0); lat = math.atan(math.sinh(y/6378137.0))
    return a*math.cos(lat)**2  # EPSG:3857 -> ground m2
gt = json.load(open(K/'w5data/gt_train_v2.geojson'))
G = collections.defaultdict(collections.Counter)
for f in gt['features']:
    p = f['properties']
    if p.get('markup_type') != 'sp_or': continue
    c = p['class_name']; G[p['region_name']][c] += 1
    if c.startswith('kurgany') and f.get('geometry') and area_m2(f['geometry']) >= 1000: G[p['region_name']][c+'_ge1k'] += 1
out = []
regs = sorted(set(os.listdir(RAW)) | set(G))
for r in regs:
    d = RAW/r; tifs = sorted(str(t.relative_to(K)) for t in d.rglob('*.tif')) if d.is_dir() else []
    utm = json.load(open(d/'UTM.json')).get('crs') if (d/'UTM.json').exists() else None
    fam = site_family(r)
    out.append({'region': r, 'family': fam, 'fold': fold.get(fam), 'in_cache': r in cache_regions,
                'cache_tiles': cache_regions.get(r, 0), 'cache_pos_tiles': cache_pos.get(r, 0),
                'n_spor_tif': len(tifs), 'tifs': tifs, 'utm_crs': utm, 'gt_sp_or': dict(G.get(r, {}))})
f0 = json.load(open(K/'out/sp_large-f0/fold0_split.json'))
chk = sorted(x['region'] for x in out if x['fold'] == 0 and x['n_spor_tif'])
summ = {'n_regions': len(out), 'with_tif': sum(1 for x in out if x['n_spor_tif']), 'n_tif': sum(x['n_spor_tif'] for x in out),
        'fold_load_pos_tiles': load, 'families_per_fold': {k: sorted(f for f in fold if fold[f] == k) for k in range(5)},
        'regions_per_fold': {str(k): sum(1 for x in out if x['fold'] == k) for k in range(5)},
        'regions_no_fold': sum(1 for x in out if x['fold'] is None),
        'fold0_matches_sp_large_split': set(f0['fold0_regions']) <= set(x['region'] for x in out if x['fold'] == 0) and f0['fold0_families'] == sorted(f for f in fold if fold[f] == 0),
        'gt_sp_or_total': dict(sum((collections.Counter(x['gt_sp_or']) for x in out), collections.Counter())),
        'gt_kurgans_per_fold': {str(k): sum(x['gt_sp_or'].get('kurgany_povrezhdennye', 0)+x['gt_sp_or'].get('kurgany_tselye', 0) for x in out if x['fold'] == k) for k in range(5)},
        'gt_kurgans_no_fold': sum(x['gt_sp_or'].get('kurgany_povrezhdennye', 0)+x['gt_sp_or'].get('kurgany_tselye', 0) for x in out if x['fold'] is None),
        'gt_regions_without_local_tif': sorted(x['region'] for x in out if x['gt_sp_or'] and not x['n_spor_tif'])}
json.dump({'built': '01.11 MSK bot-D/W1', 'fold_recipe': 'nn_oof family_folds k=5 on cache_spor_r1_t128_s25 positive tiles', 'summary': summ, 'regions': out},
          open(K/'orch/bot-W1/sat2_dataset_manifest.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))
