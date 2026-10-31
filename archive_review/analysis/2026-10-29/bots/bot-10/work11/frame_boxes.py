"""bot-10 work11 C3: рамки сцен ae/sp_or/or по участкам трейна (как oof_transform.py frame @d4edf2f: scan_root + map_footprint), только заголовки растров.
    python frame_boxes.py → frame_boxes.json {region: {markup: [[x0,y0,x1,y1], ...]}}"""
import json, sys, logging
sys.path.insert(0, '/workspace/koz3_shared/bot-10/work11/audit2_src')
from expds.catalog import read_utm_json, scan_root
from expds.geophys import map_footprint
import expds; assert 'audit2_src' in expds.__file__
out = {}; n = 0; fail = []
for reg in scan_root('/workspace/data/train-expds-3'):
    boxes = {}
    for mt in ('ae', 'sp_or', 'or'):
        for path in reg.rasters.get(mt, []):
            try:
                fp = map_footprint(path, read_utm_json(reg.utm_json))
            except Exception as e:
                fail.append([reg.name, str(path), repr(e)[:200]]); continue
            if fp is not None:
                boxes.setdefault(mt, []).append(list(fp.bounds)); n += 1
    if boxes: out[reg.name] = boxes
json.dump({'boxes': out, 'n_boxes': n, 'fail': fail}, open('frame_boxes.json', 'w'), ensure_ascii=False, indent=1)
print('regions with boxes', len(out), 'boxes', n, 'fail', len(fail))
