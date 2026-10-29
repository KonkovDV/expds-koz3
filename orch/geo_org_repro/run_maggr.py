"""Organizer MagGr baseline (01_baseline_maggr_anomalies.ipynb), headless.
1) original holdout-207 + LOSO numbers (cells verbatim);
2) OOF on KOZ3 geo folds F1..F5 and on family folds -> platform GeoJSON (EPSG:3857), all post-NMS candidates >= FLOOR.
Organizer code: allowed by rules, declared in ai_declaration."""
import json, sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from runnb import run
g = {"__name__": "nb"}
import os, ast
if os.environ.get("LEAN"):  # holdout/LOSO already reproduced (out_maggr.log); defs only from 15/19
    run("01_baseline_maggr_anomalies.ipynb", [1, 2, 4, 10, 13], g)
    _cs = json.load(open("01_baseline_maggr_anomalies.ipynb"))["cells"]
    for _i in (15, 17, 19):
        _t = ast.parse("".join(_cs[_i]["source"]))
        if _i != 17:
            _t.body = [n for n in _t.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) or
                       (isinstance(n, ast.Assign) and all(isinstance(x, ast.Name) and x.id in ("THRS", "POLY_FEATURES") for x in n.targets))]
        exec(compile(_t, f"cell{_i}", "exec"), g)
else:
    run("01_baseline_maggr_anomalies.ipynb", [1, 2, 4, 10, 13, 15, 17, 19, 22, 25], g)
exec("""
import time
from pathlib import Path
from rasterio.warp import transform_geom
from shapely.geometry import mapping, Point
FLOOR = 0.02
FOLDSETS = {
 'geo5': {'F1': ['200'], 'F2': [str(i) for i in range(201, 208)], 'F3': ['208','209','210','215'],
          'F4': ['211','212','216'], 'F5': ['213','214','217']},
 'fam': {'Sud': ['200'], 'Sem': [str(i) for i in range(201, 208)], 'Gne': [str(i) for i in range(208, 218)]},
}
def feats_for(d, pts_s, poly_s):
    out = []
    rn = f"{d['region']}_{d['site']}"
    k = nms_points(d['pt_xy'], pts_s, FLOOR)
    for i in k:
        gm = transform_geom(d['crs'], 'EPSG:3857', mapping(Point(*d['pt_xy'][i])))
        out.append({'type': 'Feature', 'geometry': gm, 'properties': {'class_name': 'finds_points',
                    'region_name': rn, 'markup_type': 'maggr', 'confidence': round(float(pts_s[i]), 5)}})
    k = nms_polys(d['cand_polys'], poly_s, FLOOR)
    for i in k:
        gm = transform_geom(d['crs'], 'EPSG:3857', mapping(d['cand_polys'][i]))
        out.append({'type': 'Feature', 'geometry': gm, 'properties': {'class_name': 'object_poly',
                    'region_name': rn, 'markup_type': 'maggr', 'confidence': round(float(poly_s[i]), 5)}})
    return out
for name, folds in FOLDSETS.items():
    t = time.time(); feats = []
    for fk, regs in folds.items():
        test = [d for d in DS if d['region'] in regs]
        train = [d for d in DS if d['region'] not in regs]
        if not test: continue
        pc, qc = fit_points(train), fit_polys(train)
        for d in test:
            feats += feats_for(d, pc.predict_proba(d['pt_X'])[:, 1], qc.predict_proba(d['cand_X'])[:, 1])
    Path('../out').mkdir(exist_ok=True)
    json.dump({'type': 'FeatureCollection', 'features': feats}, open(f'../out/maggr_oof_{name}.geojson', 'w'), ensure_ascii=False)
    print(name, len(feats), f'{time.time()-t:.0f}s', flush=True)
""", g)
