"""Официальный _extract_geometry vs repo gt._polygons по каждому объекту 456 файлов разметки (без пересчёта CRS)."""
import sys, csv, json, collections, importlib.util
from shapely.geometry import shape
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
from expds.gt import _polygons
spec = importlib.util.spec_from_file_location('cm', '/workspace/koz3_shared/bot-6/repo/materials/compute_metrics.py'); cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
R = list(csv.DictReader(open('../types/gt_repo_stats_files.csv', encoding='utf-8-sig'), delimiter=';'))
agg = collections.Counter(); bycls = collections.defaultdict(collections.Counter); ex = []
for r in R:
    d = json.load(open(r['file'], encoding='utf-8-sig'))
    for f in d.get('features') or []:
        g = f.get('geometry')
        off = cm._extract_geometry(f)
        try: rp = _polygons(shape(g)) if g else []
        except Exception: rp = []
        n_rep = len(rp); n_off = 0 if off is None else 1
        key = (r['class_name'], r['markup_type'])
        if g is None or not g.get('coordinates'): k = 'empty'
        elif n_off == 0: k = 'official_drops_nonempty'
        elif n_rep > 1: k = 'multipart_repo_splits'
        elif n_rep == 0: k = 'repo_drops_official_keeps'
        else: k = 'same'
        agg[k] += 1; bycls[key][k] += 1; bycls[key]['extra_parts_repo'] += max(0, n_rep - n_off)
        if off is not None and off.geom_type not in ('Polygon', 'Point'): agg['official_nonpoly_' + off.geom_type] += 1
        if k in ('official_drops_nonempty', 'repo_drops_official_keeps') and len(ex) < 10: ex.append((r['region'], key, k, (g or {}).get('type')))
print(dict(agg))
for k, v in sorted(bycls.items()):
    if any(v[x] for x in v if x != 'same'): print(k, dict(v))
print(ex)
