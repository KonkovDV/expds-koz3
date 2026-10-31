"""Compact table of all candidates + exact official-match pairs (pred, gt, quality)."""
from common import *
import numpy as np, shapely
from shapely.strtree import STRtree
from expds import metric as M
mod = M.load_official(METRIC); calc = mod.F1Calculator()
import ijson
M._SHM_CACHE_DIR = W/'work'/'nocache_dir_disabled'
M._SHM_CACHE_DIR = Path('/nonexistent_ro/x')
gt = load(W/'gt_train_v2.geojson')
CL = list(calc.target_classes); cid = {c:i for i,c in enumerate(CL)}
cls = []; conf = []; reg = []; grp = []; mk = []; geom = []
def flush(chunk):
    fc = {'type':'FeatureCollection','features':chunk}
    pr = M._collect(mod, calc, fc, False); M._UTM_GEOM_CACHE.clear()
    it = {c: iter(v) for c, v in pr.items()}
    for f in chunk:
        p = f['properties']; c = mod.normalize_class_name(p.get('class_name','unknown'))
        cls.append(cid.get(c, -1)); conf.append(mod._get_confidence(f)); reg.append(str(p.get('region_name','default_source')))
        grp.append(M.threshold_group(p)); mk.append(str(p.get('markup_type') or ''))
        geom.append(next(it[c])[0] if c in cid else None)
    for v in it.values(): assert next(v, None) is None
chunk = []
with open(W/'stack_arch.geojson','rb') as fh:
    for f in ijson.items(fh, 'features.item', use_float=True):
        chunk.append(f)
        if len(chunk) == 50000: flush(chunk); chunk = []; print(len(cls), flush=True)
flush(chunk)
N = len(cls); cls = np.array(cls, np.int16); conf = np.array(conf); geom = np.array(geom, dtype=object)
gts = M._collect(mod, calc, gt, True); del gt
print('geoms', flush=True)
G = {c: [x[0] for x in gts.get(c, [])] for c in CL}
greg = {c: [x[2] for x in gts.get(c, [])] for c in CL}
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']
pairs = {}
for pc in CL:
    targets = KUR if pc in KUR else [pc]
    idx = np.flatnonzero(cls == cid[pc])
    for gc_ in targets:
        gg = G[gc_]
        if not len(idx) or not gg: continue
        tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == 'Point' else x for x in gg])
        qp, qg = tree.query(list(geom[idx]))
        P, J, S = [], [], []
        for a, b in zip(qp, qg):
            pg = geom[idx[a]]
            if pg.geom_type != 'Polygon': continue
            ok, s = calc._match_quality(pg, gg[b])
            if ok: P.append(idx[a]); J.append(b); S.append(s)
        pairs[(pc, gc_)] = (np.array(P, np.int64), np.array(J, np.int64), np.array(S))
        print(pc, gc_, len(P), flush=True)
cen = shapely.centroid(geom); 
out = dict(CL=CL, cls=cls, conf=conf, reg=np.array(reg), grp=np.array(grp), mk=np.array(mk),
           area=shapely.area(geom), perim=shapely.length(geom), cx=shapely.get_x(cen), cy=shapely.get_y(cen),
           nv=shapely.get_num_coordinates(geom), pairs=pairs, n_gt={c: len(G[c]) for c in CL}, greg=greg,
           gx={c: shapely.get_x(shapely.centroid(G[c])) if G[c] else np.zeros(0) for c in CL},
           gy={c: shapely.get_y(shapely.centroid(G[c])) if G[c] else np.zeros(0) for c in CL})
b = shapely.bounds(geom); out['bw'] = b[:,2]-b[:,0]; out['bh'] = b[:,3]-b[:,1]
pickle.dump(out, open(W/'work/table.pkl','wb'), protocol=5)
shapely.to_wkb(geom).dump if False else None
pickle.dump(shapely.to_wkb(geom), open(W/'work/geom_wkb.pkl','wb'), protocol=5)
pickle.dump({c: shapely.to_wkb(G[c]) for c in CL}, open(W/'work/gt_wkb.pkl','wb'), protocol=5)
print('done')
