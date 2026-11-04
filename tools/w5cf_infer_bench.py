"""Time / peak RSS of expds.w5cf_infer on one site, as in the pipeline: region_features(LAS) + score(all objects).
Objects are streamed from a GeoJSON (EPSG:3857, e.g. the training stack) for --region; no train table in memory.
  python tools/w5cf_infer_bench.py --geojson stack_arch.geojson --region 190_Муравлево --las 190.las --utm 190_UTM.json --infer-dir DIR
"""
import argparse, json, resource, sys, time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ijson, shapely.geometry  # noqa: E402
from expds import w5cf_infer as W  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--geojson', required=True); ap.add_argument('--region', required=True)
ap.add_argument('--las', nargs='*', default=[]); ap.add_argument('--utm'); ap.add_argument('--infer-dir', required=True)
a = ap.parse_args(); cfg = {'w5cf': {'infer_dir': a.infer_dir}}
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
feats = []
with open(a.geojson, 'rb') as fh:
    for f in ijson.items(fh, 'features.item', use_float=True):
        p = f['properties']
        if str(p.get('region_name')) == a.region:
            o = {k: p.get(k) for k in ('class_name', 'confidence', 'markup_type', 'threshold_group', 'region_name')}
            o['geom'] = shapely.geometry.shape(f['geometry']); feats.append(o)
r0 = rss(); W.load_artifacts(cfg)
li = [f for f in feats if f.get('markup_type') == 'li']
t = time.time(); W.region_features(SimpleNamespace(name=a.region, las=[Path(x) for x in a.las], utm_json=Path(a.utm) if a.utm else None), li, cfg)
t1 = time.time() - t; r1 = rss(); t = time.time(); s = W.score(feats, cfg); t2 = time.time() - t
las_mb = sum(Path(x).stat().st_size for x in a.las) / 2 ** 20
print(json.dumps(dict(region=a.region, objects=len(feats), li=len(li), las_mb=round(las_mb), cf_rows=sum(W.KEY in f for f in feats),
                      scored=sum(v is not None for v in s), sec_region_features=round(t1, 1), sec_score=round(t2, 1),
                      maxrss_mb_after_load=r0, maxrss_mb_after_cf=r1, maxrss_mb=rss()), ensure_ascii=False))
