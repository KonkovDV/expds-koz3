# bot-8 micro-benchmark: own numpy re-implementation of the openness/SVF loop structure
# (same op pattern as EhimenNathan koz3/relief.py:33-53: n_dir directions, k = 1..K step max(1,K//10)).
# Synthetic DEM, float32, single process. Not their code.
import os, sys, time, json
import numpy as np

def openness(z, r, radius_m, n_dir=8, full=False):
    K = max(2, int(round(radius_m / r)))
    H, W = z.shape
    zp = np.pad(z, K, mode="edge")
    pos = np.zeros_like(z); neg = np.zeros_like(z); svf = np.zeros_like(z)
    step = 1 if full else max(1, K // 10)
    n = 0
    for a in np.arange(n_dir) * 2 * np.pi / n_dir:
        ca, sa = np.cos(a), np.sin(a)
        up = np.full_like(z, -np.pi / 2); dn = np.full_like(z, -np.pi / 2)
        for k in range(1, K + 1, step):
            dy, dx = int(round(k * sa)), int(round(k * ca))
            d = np.hypot(dx, dy) * r
            if d == 0: continue
            zs = zp[K + dy:K + dy + H, K + dx:K + dx + W]
            ang = np.arctan((zs - z) / np.float32(d))
            np.maximum(up, ang, out=up); np.maximum(dn, -ang, out=dn); n += 1
        pos += np.pi / 2 - up; neg += np.pi / 2 - dn; svf += np.sin(np.clip(up, 0, None))
    return n

rng = np.random.default_rng(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
z = np.cumsum(np.cumsum(rng.normal(0, 0.02, (N, N)).astype(np.float32), 0), 1).astype(np.float32)
res = []
for radius, full in [(10, False), (20, False), (40, False), (10, True), (20, True)]:
    t = time.perf_counter(); n = openness(z, 0.5, radius, 8, full); dt = time.perf_counter() - t
    res.append(dict(N=N, res_m=0.5, radius_m=radius, n_dir=8, every_pixel=full, samples=n,
                    sec=round(dt, 3), sec_per_Mpx=round(dt / (N * N / 1e6), 3)))
    print(res[-1], flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), f"svf_bench_{N}.json"), "w"), indent=1)
