from common import *
import fast, numpy as np
from scipy.spatial import cKDTree
T = fast.T; cls=T['cls']; conf=T['conf']; grp=T['grp']; reg=T['reg']; CL=fast.CL
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
twin = np.load(W/'work/twin.npy')
okc = np.zeros((len(cls), len(CL)), bool)
for (pc, gc), (P, J, S) in T['pairs'].items(): okc[P, fast.cid[gc]] = True
own = okc[np.arange(len(cls)), np.maximum(cls,0)] & (cls>=0)
anyk = okc[:, 0] | okc[:, 1]
# near GT of same class (any kurgan for kurgans) within 15 m centroid, not ok -> geometry
near = np.zeros(len(cls), bool)
for c in CL:
    gx, gy, gr = T['gx'][c], T['gy'][c], np.array(T['greg'][c])
    cs = [c] if c not in fast.KUR else fast.KUR
    m = np.isin(cls, [fast.cid[x] for x in cs])
    for r in np.unique(reg[m]):
        ii = np.flatnonzero(m & (reg == r)); gm = gr == r
        if not gm.any(): continue
        d, _ = cKDTree(np.c_[gx[gm], gy[gm]]).query(np.c_[T['cx'][ii], T['cy'][ii]])
        near[ii] |= d < 15
base = fast.curves(); bs = honest(base, new, optical)
def run(name, keep=None, assign=None):
    s = honest(fast.curves(keep=keep, assign=assign), new, optical)
    per = score_counts(s[1])[1]
    print(f"{name:40s} {s[0][0]:.6f} ({s[0][0]-bs[0][0]:+.4f}) new {s[0][1]:.4f} opt {s[0][2]:.4f} | povr {per['kurgany_povrezhdennye']['f1']:.4f} tsel {per['kurgany_tselye']['f1']:.4f}", flush=True)
    return s
print('base', bs[0])
K = (cls==0)|(cls==1)
# oracle 1: kurgan confusion: drop copies that are ok only for the other subclass
conf_wrong = K & ~own & anyk
run('oracle drop confused kurgan copies', keep=~conf_wrong)
# oracle 1b: reassign to right subclass (no new candidates)
asg = cls.copy(); flip = conf_wrong & (twin<0); asg[flip] = 1-cls[flip]
run('oracle confused: drop twin / flip single', keep=~(conf_wrong & (twin>=0)), assign=asg)
# oracle 2: kurgan background FP (not near any kurgan GT)
run('oracle drop kurgan background', keep=~(K & ~anyk & ~near))
run('oracle drop kurgan geometry misses', keep=~(K & ~anyk & near))
for c in range(2, len(CL)):
    m = (cls==c)
    run(f'oracle drop bg {CL[c]}', keep=~(m & ~own & ~near))
    run(f'oracle drop geom-miss {CL[c]}', keep=~(m & ~own & near))
