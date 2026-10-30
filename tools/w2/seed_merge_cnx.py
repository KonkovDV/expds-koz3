"""W2 P-W2-2: ансамбль сидов c3-sc (s0+s1) усреднением OOF-карт до декодирования; декодер = decode() ядра bot-9 (вендор).
usage: seed_merge_cnx.py <maps_root> <out_geojson> [--seeds s0,s1] [--check]   (--check: декод одного сида, сверка с OOF ядра)
Карты: hm (f32 prob) и mask (u8/255) — среднее вероятностей; wh/off (f16 raw) — среднее. valid — пересечение."""
import sys, types, json, glob, argparse
from pathlib import Path
import numpy as np
sys.modules.setdefault('torch', types.ModuleType('torch')); sys.modules['torch'].nn = types.SimpleNamespace(Module=object)
sys.modules.setdefault('torch.nn', sys.modules['torch'].nn)
V = '/workspace/koz3/w2/vendor'; sys.path.insert(0, V)
import cnx_head as HD
from scipy import ndimage as ndi
from shapely.geometry import Point
from shapely.ops import transform as stf
from pyproj import Transformer
ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('out'); ap.add_argument('--seeds', default='s0,s1'); ap.add_argument('--report', required=True)
A = ap.parse_args()
REP = json.load(open(A.report, encoding='utf-8')); CFG = REP['cfg']
HEAD = HD.get_head('centerhead', CFG); CIRCLE_R = HEAD.circle_r_m or CFG['circle_r_m']
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
seeds = A.seeds.split(',')
def index(seed):
    d = {}
    for p in glob.glob(f'{A.root}/{seed}/**/maps_cnx_*.npz', recursive=True):
        z = np.load(p, allow_pickle=False); m = json.loads(str(z['meta'])); d[m['region']] = p
    return d
IDX = {s: index(s) for s in seeds}
regions = sorted(set.intersection(*[set(v) for v in IDX.values()]))
print('seeds', {s: len(v) for s, v in IDX.items()}, 'common regions', len(regions), flush=True)
def load(p):
    z = np.load(p, allow_pickle=False); m = json.loads(str(z['meta']))
    mp = {'mask': z['mask'].astype(np.float32) / 255.0, '_valid': z['valid'].astype(bool), '_valid_h': z['valid_h'].astype(bool)}
    for k in z.files:
        if k not in ('mask', 'valid', 'valid_h', 'meta'): mp[k] = z[k].astype(np.float32)
    return mp, m
def decode(maps, m):
    st = m['step']; dets = []; mk = maps['mask']; gor = mk[0] > CFG['mask_thr']
    for c, mn in ((2, CFG['gor_min_px']), (3, CFG['fort_min_px']), (4, CFG['arch_min_px'])):
        pc = mk[c - 2]; lab, n = ndi.label(pc > CFG['mask_thr'])
        if not n: continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]; conf = ndi.mean(pc, lab, np.arange(1, n + 1)); boxes = ndi.find_objects(lab)
        for i in np.where(sizes >= mn)[0]:
            sl = boxes[i]; comp = lab[sl] == i + 1
            dt = ndi.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]; yy, xx = np.unravel_index(np.argmax(dt), dt.shape)
            dets.append((sl[1].start + xx, sl[0].start + yy, c, float(conf[i])))
    dets = [(m['left'] + (x + 0.5) * st, m['top'] - (y + 0.5) * st, c, s) for x, y, c, s in dets]
    sup = HD.block_mean(gor, HEAD.stride) > 0.5
    t = HEAD.decode({k: v for k, v in maps.items() if k != 'mask' and not k.startswith('_')}, m['left'], m['top'], st * HEAD.stride, maps['_valid_h'], sup)
    for X, Y, c, s in zip(t['X'], t['Y'], t['cls'], t['score']): dets.append((float(X), float(Y), int(c), float(s)))
    to3857 = Transformer.from_crs(m['crs'], 'EPSG:3857', always_xy=True).transform; feats = []
    for ux, uy, c, s in dets:
        g = stf(to3857, Point(ux, uy).buffer(CIRCLE_R, quad_segs=4)); ring = [[round(a, 2), round(b, 2)] for a, b in g.exterior.coords]
        feats.append(dict(type='Feature', geometry=dict(type='Polygon', coordinates=[ring]), properties=dict(
            class_name=CLASSES[c], confidence=round(min(1.0, max(0.0, s)), 5), region_name=m['region'], crs='EPSG:3857', markup_type='li',
            threshold_group='li_cnx', original_crs=m['crs'], fold=int(m['fold']))))
    return feats
allf = []; stats = {}
for r in regions:
    ms = [load(IDX[s][r]) for s in seeds]; m0 = ms[0][1]
    assert all(x[1]['fold'] == m0['fold'] and x[1]['left'] == m0['left'] and x[1]['H'] == m0['H'] for x in ms), r
    avg = {}
    for k in ms[0][0]:
        if k.startswith('_'): avg[k] = np.logical_and.reduce([x[0][k] for x in ms])
        else: avg[k] = np.mean([x[0][k] for x in ms], axis=0)
    f = decode(avg, m0); stats[r] = len(f); allf.extend(f)
for i, f in enumerate(allf): f['properties'] = dict(fid=i, **f['properties'])
json.dump(dict(type='FeatureCollection', name=Path(A.out).stem, crs=dict(type='name', properties=dict(name='urn:ogc:def:crs:EPSG::3857')), features=allf),
          open(A.out, 'w', encoding='utf-8'), ensure_ascii=False)
json.dump(dict(seeds=seeds, n=len(allf), per_region=stats), open(A.out + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', A.out, len(allf))
