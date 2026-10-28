"""Официальная _extract_geometry по одному объекту vs fastgeom.extract_many (Rust + from_ragged_array)."""
import json, sys, time, statistics as S, gc
from pathlib import Path
import shapely
sys.path.insert(0, "/workspace/koz3_shared/bot-8/repo")
import expds.metric as M
from expds import fastgeom
m = M.load_official("/workspace/koz3_shared/bot-8/repo/materials/compute_metrics.py")
feats = json.loads(Path(sys.argv[1]).read_bytes())["features"] + json.loads(Path(sys.argv[2]).read_bytes())["features"]
reps = int(sys.argv[3]); gc.disable()
tr, tf = [], []
for _ in range(reps):
    t = time.perf_counter(); a = [m._extract_geometry(f) for f in feats]; tr.append(time.perf_counter() - t)
    t = time.perf_counter(); b = fastgeom.extract_many(m, feats); tf.append(time.perf_counter() - t)
same = all((x is None and y is None) or (x is not None and y is not None and shapely.to_wkb(x) == shapely.to_wkb(y)) for x, y in zip(a, b))
r = {"n": len(feats), "ref": tr, "fast": tf, "ref_med": S.median(tr), "fast_med": S.median(tf), "speedup": S.median(tr) / S.median(tf), "identical_wkb": same}
print(json.dumps(r)); Path(sys.argv[4]).write_text(json.dumps(r, indent=1))
