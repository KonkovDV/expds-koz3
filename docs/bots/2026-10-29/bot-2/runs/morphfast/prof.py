import sys, time, numpy as np
sys.path.insert(0, "/workspace/koz3_shared/bot-2/work/wt_morphfast/scripts"); sys.path.insert(0, "/workspace/koz3_shared/bot-2/work/wt_morphfast")
import morph_fast as MF
from expds import accel
from expds.lidar import fill_interior_holes
C = dict(np.load("/workspace/koz3_shared/bot-4/work/cropverif/li_kurgan_candidates.npz"))
r = sys.argv[1]; m = np.flatnonzero(C["region"] == r)[:4096]
laz = MF.ML.find_laz("/workspace/data/train-expds-3", r)
dem, gi = accel.las_grid(str(laz), 0.5, 2, 60_000_000); dem = fill_interior_holes(dem, method="pyramid")
B = int(sys.argv[2]) if len(sys.argv) > 2 else 512
tw = tf = 0
import cProfile, pstats
pr = cProfile.Profile()
for s in range(0, m.size, B):
    j = m[s:s+B]; t0 = time.perf_counter(); w = MF.windows(dem, gi["left"], gi["top"], C["cx"][j], C["cy"][j]); t1 = time.perf_counter()
    MF.features(*w); tf += time.perf_counter() - t1; tw += t1 - t0
print("ms/cand windows", 1000*tw/m.size, "features", 1000*tf/m.size)
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
