# bot-10 work14. Круги городищ D-FINE (вариант R пакета v3) из 5-фолдового OOF bot-9 — ТОЛЬКО класс gorodishcha.
# Склейка как bot-9 prep5.py fam C (= sp_dfine.convert circle: radius 4 м, merge 4 м, cover>=0.5), по (регион, класс).
# Вход (только чтение): bot-9/kaggle/dfine_spor/full_rest_out/oof_dfine_5f.geojson; выход: work14/dfine_gor.json
import sys, json, math, collections, hashlib
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds.open_ideas import mercator_scale
from shapely.geometry import shape, Point
SRC = Path('/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_rest_out/oof_dfine_5f.geojson')
cand = collections.defaultdict(list); regs_all = collections.Counter(); folds = {}; n = 0
with open(SRC, encoding='utf-8') as fh:
    for line in fh:
        line = line.strip().rstrip(',')
        if not line.startswith('{"type":"Feature"'): continue
        n += 1
        i = line.find('"region_name":"'); r = line[i + 15:line.index('"', i + 15)]; regs_all[r] += 1
        if '"class_name":"gorodishcha"' not in line: continue
        f = json.loads(line); p = f['properties']; x0, y0, x1, y1 = shape(f['geometry']).bounds
        folds[r] = p['fold']; cand[r].append(((x0 + x1) / 2, (y0 + y1) / 2, float(p['confidence']), (x0, y0, x1, y1)))
def cover(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    sm = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])); return w * h / sm if sm > 0 else 0.0
CELL = 100.0; MERGE = 4.0; R = 4.0
def cells_of(bx, pad):
    return {(i, j) for i in range(int(math.floor((bx[0] - pad) / CELL)), int(math.floor((bx[2] + pad) / CELL)) + 1)
            for j in range(int(math.floor((bx[1] - pad) / CELL)), int(math.floor((bx[3] + pad) / CELL)) + 1)}
out = []; n_in = 0
for reg, items in sorted(cand.items()):
    items.sort(key=lambda c: -c[2]); grid = collections.defaultdict(list); clusters = []
    for c in items:
        n_in += 1; x, y, sc, bx = c; hit = None; seen = set()
        for cell in cells_of(bx, MERGE * mercator_scale(y)):
            for k in grid.get(cell, ()):
                if k in seen: continue
                seen.add(k)
                for m in clusters[k]:
                    if math.hypot(x - m[0], y - m[1]) / mercator_scale((y + m[1]) / 2) <= MERGE or cover(bx, m[3]) >= 0.5:
                        hit = k if hit is None else min(hit, k); break
        if hit is None: clusters.append([c]); hit = len(clusters) - 1
        else: clusters[hit].append(c)
        for cell in cells_of(bx, 0):
            if hit not in grid[cell]: grid[cell].append(hit)
    for cl in clusters:
        b = cl[0]; geo = Point(b[0], b[1]).buffer(R * mercator_scale(b[1]), quad_segs=4)
        out.append(dict(region=reg, conf=round(b[2], 6), x=b[0], y=b[1], views=len(cl), coords=[[round(u, 2), round(v, 2)] for u, v in geo.exterior.coords]))
rep = dict(src=str(SRC), src_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(), features_total=n, regions_total=len(regs_all),
           regions=sorted(regs_all), gor_boxes_in=n_in, gor_circles_out=len(out), gor_regions=len(cand), fold_by_region=folds)
json.dump(dict(report=rep, circles=out), open(Path(__file__).with_name('dfine_gor.json'), 'w'), ensure_ascii=False)
print(json.dumps({k: v for k, v in rep.items() if k not in ('regions', 'fold_by_region')}, ensure_ascii=False))
