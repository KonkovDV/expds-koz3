import sys, time, json, cProfile, pstats, io, os, tomllib
sys.path[:0] = ['/workspace/koz3/w0/pylib', os.environ.get('W0_SRC', '/workspace/koz3/wt/w0')]
from shapely.geometry import shape
from expds import watchdog, writer, fastjson
from pathlib import Path
cfg = tomllib.load(open(os.environ.get('W0_SRC', '/workspace/koz3/wt/w0') + '/configs/default.toml', 'rb'))
srcs = ['/workspace/koz3/w2/oof/merged/oof_cnx_c3-sc_s0.geojson', 'ae03.geojson']
t = time.time(); feats = []
for s in srcs:
    for f in fastjson.load_json(s)['features']:
        p = dict(f['properties']); p['geom'] = shape(f['geometry']); feats.append(p)
print('load+shape', round(time.time() - t, 2), len(feats), flush=True)
pr = cProfile.Profile(); t = time.time(); pr.enable()
out = watchdog.finish(feats, cfg)
pr.disable(); tf = time.time() - t
t = time.time(); writer.write_geojson(out, Path('out.geojson')); tw = time.time() - t
print(json.dumps({'finish': round(tf, 2), 'write': round(tw, 2), 'n_out': len(out)}))
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(35); print(s.getvalue()[:9000])
