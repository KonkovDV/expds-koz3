import sys, time, numpy as np, collections
sys.path.insert(0, "/workspace/koz3_shared/bot-2/work/wt_morphfast/scripts"); sys.path.insert(0, "/workspace/koz3_shared/bot-2/work/wt_morphfast"); sys.path.insert(0,".")
import inst_src as MF
from expds import accel
from expds.lidar import fill_interior_holes
C = dict(np.load("/workspace/koz3_shared/bot-4/work/cropverif/li_kurgan_candidates.npz"))
r = sys.argv[1]; m = np.flatnonzero(C["region"] == r)[:4096]
laz = MF.ML.find_laz("/workspace/data/train-expds-3", r)
dem, gi = accel.las_grid(str(laz), 0.5, 2, 60_000_000); dem = fill_interior_holes(dem, method="pyramid")
acc = collections.Counter()
for s in range(0, m.size, 512):
    j = m[s:s+512]; w = MF.windows(dem, gi["left"], gi["top"], C["cx"][j], C["cy"][j]); MF._T.clear(); MF.features(*w)
    T = MF._T
    for (k0,t0),(k1,t1) in zip(T[:-1],T[1:]): acc[k1]+=t1-t0
lines=open("inst_src.py").read().split("\n")
tot=sum(acc.values())
for k,v in acc.most_common(12):
    i=[n for n,l in enumerate(lines) if "_T.append((%d,"%k in l][0]
    print(round(1000*v/m.size,4), lines[i-1].strip()[:110])
print("total", 1000*tot/m.size)
