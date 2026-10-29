"""bot-6 ORGATE: признаки без растров для arch-кандидатов (строки table.pkl = stack_idx). Пишет feats_gate.npz."""
import sys, json, pickle; sys.path.insert(0, '/workspace/rerank')
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import pyproj
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb')); CL = T['CL']; cid = {c: i for i, c in enumerate(CL)}
cls = T['cls'].astype(int); conf = T['conf'].astype(np.float32); reg = T['reg']; grp = T['grp']; mk = T['mk']
XY = np.c_[T['cx'], T['cy']]
A = cid['arkhitektury']; ai = np.flatnonzero(cls == A)
boxes = set(json.loads(Path('/workspace/expds-koz3/notes/las_boxes_59.json').read_text(encoding='utf-8')))
names = ['sh_pp', 'sh_rect',
         'ch_oth5_max', 'ch_oth5_n', 'ch_oth15_max', 'ch_oth15_n',
         'kur15_n', 'kur30_n', 'kur15_max', 'gor30_max', 'fort30_max', 'li15_n',
         'r_las', 'r_fli', 'r_fae', 'r_fgeo', 'r_ncls', 'r_km2', 'r_arch_dens', 'r_arch_hi', 'lat', 'lon']
F = np.zeros((ai.size, len(names)), np.float32); col = {n: i for i, n in enumerate(names)}
ar = np.maximum(T['area'][ai], 1e-6)
F[:, 0] = T['perim'][ai] / np.sqrt(ar); F[:, 1] = T['bw'][ai] * T['bh'][ai] / ar
pos = {p: k for k, p in enumerate(ai)}
KUR = [cid['kurgany_povrezhdennye'], cid['kurgany_tselye']]; G = cid['gorodishcha']; FO = cid['fortifikatsii']
def qmax(tree_idx, pts, r):
    if tree_idx.size == 0: return np.zeros(len(pts)), np.zeros(len(pts))
    tr = cKDTree(XY[tree_idx]); L = tr.query_ball_point(pts, r)
    return (np.array([conf[tree_idx[l]].max() if l else 0 for l in L], np.float32), np.array([len(l) for l in L], np.float32))
for r in np.unique(reg[ai]):
    ir = np.flatnonzero(reg == r); ia = ir[cls[ir] == A]; rows = np.array([pos[p] for p in ia]); P = XY[ia]
    for g in np.unique(grp[ia]):
        sel = grp[ia] == g; oth = ia[grp[ia] != g]
        for rad, a, b in ((5, 'ch_oth5_max', 'ch_oth5_n'), (15, 'ch_oth15_max', 'ch_oth15_n')):
            m, n = qmax(oth, P[sel], rad); F[rows[sel], col[a]] = m; F[rows[sel], col[b]] = n
    ik = ir[np.isin(cls[ir], KUR)]
    F[rows, col['kur15_max']], F[rows, col['kur15_n']] = qmax(ik, P, 15); F[rows, col['kur30_n']] = qmax(ik, P, 30)[1]
    F[rows, col['gor30_max']] = qmax(ir[cls[ir] == G], P, 30)[0]; F[rows, col['fort30_max']] = qmax(ir[cls[ir] == FO], P, 30)[0]
    F[rows, col['li15_n']] = qmax(ir[mk[ir] == 'li'], P, 15)[1]
    F[rows, col['r_las']] = float(r in boxes)
    F[rows, col['r_fli']] = np.mean(mk[ir] == 'li'); F[rows, col['r_fae']] = np.mean(mk[ir] == 'ae')
    F[rows, col['r_fgeo']] = np.mean(np.isin(mk[ir], ['maggr', 'etgr', 'terfl'])); F[rows, col['r_ncls']] = len(np.unique(cls[ir][cls[ir] >= 0]))
    xy = XY[ir]; km2 = max((np.ptp(xy[:, 0]) * np.ptp(xy[:, 1])) / 1e6, 1e-3)
    F[rows, col['r_km2']] = km2; F[rows, col['r_arch_dens']] = ia.size / km2; F[rows, col['r_arch_hi']] = np.mean(conf[ia] >= 0.2)
    u = Path(f'/workspace/data/train-expds-3/{r}/UTM.json')
    if u.exists() and '4326' not in u.read_text():
        crs = json.loads(u.read_text())['crs'].split('::')[-1]
        lon, lat = pyproj.Transformer.from_crs(f'EPSG:{crs}', 'EPSG:4326', always_xy=True).transform(*XY[ia].mean(0))
        F[rows, col['lat']] = lat; F[rows, col['lon']] = lon
    else: F[rows, col['lat']] = np.nan; F[rows, col['lon']] = np.nan
np.savez_compressed('feats_gate.npz', stack_idx=ai.astype(np.int32), X=F, names=np.array(names))
print('rows', ai.size, 'nan lat', int(np.isnan(F[:, col['lat']]).sum()), 'las rows', int(F[:, col['r_las']].sum()))
print({n: [float(np.nanmin(F[:, i])), float(np.nanmedian(F[:, i])), float(np.nanmax(F[:, i]))] for i, n in enumerate(names)})
