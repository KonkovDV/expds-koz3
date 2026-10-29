import sys, time, numpy as np, torch
sys.path.insert(0, "tests"); sys.path.insert(0, "repo")
from test_slots import Fake, coords_of
torch.set_num_threads(2)
for (h, w, t, k) in [(2048, 2048, 128, 1), (4352, 4352, 128, 2), (2000, 2000, 384, 2)]:
    img = np.random.default_rng(0).integers(0, 256, (3, h, w), dtype=np.uint8)
    m = Fake(t, k); m._forward = lambda xb, k=k: xb[:, :k].sigmoid()   # почти бесплатный «проход»
    ramp = np.minimum(np.arange(t) + 1, np.arange(t)[::-1] + 1).astype(np.float32)
    wgt = np.minimum.outer(ramp, ramp); wgt /= wgt.max()
    ys, xs = coords_of(h, w, t); step = int(t * .75)
    r = {}
    for name, f in [("loop", lambda: m._predict_on_device(img, [(y, x) for y in ys for x in xs], wgt, 64)),
                    ("slots", lambda: m._predict_on_device_slots(img, ys, xs, wgt, 64, step))]:
        f(); t0 = time.perf_counter(); out = f(); r[name] = (time.perf_counter() - t0, out)
    eq = np.array_equal(r["loop"][1], r["slots"][1])
    print(f"{h}x{w} t={t} K={k} tiles={len(ys)*len(xs)}: loop {r['loop'][0]:.2f}s slots {r['slots'][0]:.2f}s x{r['loop'][0]/r['slots'][0]:.1f} equal={eq}", flush=True)
