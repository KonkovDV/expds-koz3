"""Label-free: final-model li scores with full train context vs production-truncated context (conf >= production floor)."""
import os, sys, json, pickle, time, tomllib
import numpy as np
os.chdir('/workspace/koz3/w4geo/x_v4on'); sys.path.insert(0, '.')
from expds import w5cf_infer as W, w5_cloudfeat as WC
import shapely.wkb
from scipy.stats import spearmanr
D = '/workspace/koz3/w1/cf/w5d/work/'
OUT = '/workspace/koz3/orch/bot-W1/work/j2nc/trunc'
cfg = tomllib.load(open('configs/default.toml', 'rb'))
T = pickle.load(open(D + 'table.pkl', 'rb')); G = pickle.load(open(D + 'geom_wkb.pkl', 'rb'))
X4 = np.load(D + 'X4.npy', mmap_mode='r'); names = json.load(open(D + 'X4_names.json'))
pool = np.load(D + 'pool.npy')
CL = T['CL']; cls = np.asarray(T['cls']); conf = np.asarray(T['conf'], float); reg = np.asarray(T['reg']).astype(str)
grp = np.asarray(T['grp']).astype(str); mk = np.asarray(T['mk']).astype(str)
tbm = cfg.get('thresholds_by_markup', {}); thr0 = cfg['thresholds']
groups = set(cfg['w5cf']['groups']); FL = {}
def floor(g, c):
    k = (g, c)
    if k not in FL:
        t = tbm.get(g, {}).get(c, thr0.get(c, 0.0))
        FL[k] = WC.li_floor(cfg, g, c, t) if g in groups else t
    return FL[k]
fl = np.array([floor(g, CL[c]) for g, c in zip(grp, cls)])
keep = conf >= fl
json.dump({f'{g}|{c}': v for (g, c), v in FL.items()}, open(OUT + '_floors.json', 'w'), indent=1)
art = W.load_artifacts(cfg)
gqi = names.index('gq'); cfi = [i for i, n in enumerate(names) if n.startswith('cf_')]
s_full = np.full(len(conf), np.nan); s_tr = np.full(len(conf), np.nan)
li_any = (mk == 'li') & pool & keep
regs = sorted(set(reg[li_any]))
print('regions', len(regs), 'li pool survivors', int(li_any.sum()), flush=True)
g0, p0 = W.gq_of, W.in_pool
t0 = time.time()
for n_, rn in enumerate(regs):
    ii_all = np.flatnonzero(reg == rn)
    for mode in ('full', 'tr'):
        ii = ii_all if mode == 'full' else ii_all[keep[ii_all]]
        feats = []
        for j in ii:
            f = {'class_name': CL[cls[j]], 'confidence': float(conf[j]), 'markup_type': mk[j], 'threshold_group': grp[j],
                 'region_name': rn, 'geom': shapely.wkb.loads(G[j])}
            v = np.asarray(X4[j, cfi], np.float32)
            if mk[j] == 'li' and np.isfinite(v).any(): f[W.KEY] = v
            feats.append(f)
        W.reset()
        if X4[ii[0], names.index('has_las')] == 1: W._STATE['las_regions'].add(rn)
        W.gq_of = lambda g, c, cf_, a_, _q=np.array(X4[ii, gqi]): _q.copy()
        W.in_pool = lambda g, c, cf_, a_, margin=0.0, _p=pool[ii]: _p.copy()
        try:
            s = np.array([np.nan if v is None else v for v in W.score(feats, cfg)], float)
        finally:
            W.gq_of, W.in_pool = g0, p0
        (s_full if mode == 'full' else s_tr)[ii] = s
    if n_ % 10 == 0:
        print(n_, rn, len(ii_all), f'{time.time()-t0:.0f}s', flush=True)
        np.savez(OUT + '.npz', s_full=s_full, s_tr=s_tr, keep=keep, done=n_)
np.savez(OUT + '.npz', s_full=s_full, s_tr=s_tr, keep=keep, done=len(regs))
m = li_any & np.isfinite(s_full) & np.isfinite(s_tr)
res = {}
for c in range(len(CL)):
    mm = m & (cls == c)
    if mm.sum() < 5: continue
    d = np.abs(s_full[mm] - s_tr[mm])
    res[CL[c]] = dict(n=int(mm.sum()), spearman=float(spearmanr(s_full[mm], s_tr[mm])[0]), mad=float(d.mean()),
                      p90=float(np.quantile(d, .9)), mean_shift=float((s_tr[mm] - s_full[mm]).mean()))
print(json.dumps(res, indent=1)); json.dump(res, open(OUT + '.json', 'w'), indent=1)
