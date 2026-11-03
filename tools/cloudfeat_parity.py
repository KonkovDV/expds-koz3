"""Паритет expds.cloudfeat с признаками обучения RR1s-cf (box_out/cf_<site>.csv, ядро koz3-w5-cloudfeat).
python tools/cloudfeat_parity.py --cands cands_cf.csv --ref box_out --site 186 --las 186.las --utm 186_UTM.json
Те же кандидаты (cx, cy EPSG:3857, w, h), тот же LAS; сравнение после округления '%.5g'. Офлайн."""
import argparse, csv, json, resource, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expds import cloudfeat as cf
from expds.catalog import read_utm_json

ap = argparse.ArgumentParser()
ap.add_argument('--cands', required=True); ap.add_argument('--ref', required=True); ap.add_argument('--site', required=True)
ap.add_argument('--las', nargs='+', required=True); ap.add_argument('--utm', required=True); ap.add_argument('--json-out')
a = ap.parse_args()
ids, B = [], []
with open(a.cands, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['region'].split('_')[0] == a.site:
            ids.append(int(r['cid'])); B.append([float(r['cx']), float(r['cy']), float(r['w']), float(r['h'])])
B = np.asarray(B, float).reshape(-1, 4)
ref = {}
with open(Path(a.ref) / f'cf_{a.site}.csv') as f:
    rd = csv.reader(f); hdr = next(rd); assert tuple(hdr[1:]) == cf.NAMES, hdr
    for r in rd: ref[int(r[0])] = [float(v) if v else np.nan for v in r[1:]]
Rf = np.array([ref[i] for i in ids], np.float32)
t = time.time(); st = {}
F = cf.compute(a.las, read_utm_json(Path(a.utm)), B, stats=st)
sec = time.time() - t
nan_eq = bool((np.isnan(F) == np.isnan(Rf)).all())
fin = np.isfinite(F) & np.isfinite(Rf)
d = np.abs(F.astype(np.float64) - Rf)
rel = d / np.maximum(np.abs(Rf.astype(np.float64)), 1e-6)
res = dict(site=a.site, cands=len(ids), nan_pattern_equal=nan_eq, nan_mismatch=int((np.isnan(F) != np.isnan(Rf)).sum()),
           finite_vals=int(fin.sum()), max_abs=float(d[fin].max()) if fin.any() else 0.0,
           max_rel=float(rel[fin].max()) if fin.any() else 0.0, exact_frac=float((d[fin] == 0).mean()) if fin.any() else 1.0,
           per_feature_max_abs={n: float(d[:, k][fin[:, k]].max()) if fin[:, k].any() else 0.0 for k, n in enumerate(cf.NAMES)},
           sec=round(sec, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), **{("las_" + k): v for k, v in st.items()})
print(json.dumps(res, ensure_ascii=False))
if a.json_out: Path(a.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
sys.exit(0 if nan_eq and res['max_rel'] <= 1e-4 else 1)
