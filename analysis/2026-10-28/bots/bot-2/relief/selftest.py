import time, numpy as np, sys
sys.path.insert(0, "/workspace/koz3_shared/bot-2/repo"); sys.path.insert(0, ".")
import relief_ext as R
from expds import accel
rng = np.random.default_rng(0); h = w = 1000; step = 0.5
yy, xx = np.mgrid[:h, :w] * step
z = 100 + 0.02 * xx + rng.normal(0, 0.03, (h, w))
for cy, cx in [(100, 100), (300, 250), (400, 420)]:
    z += 1.2 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0 ** 2))   # «курганы» r≈8 м
z[:50, :80] = np.nan; z = z.astype(np.float32)
print("accel backend:", getattr(accel, "backend", lambda: "?")())
t = time.time(); img, st = R.extra_channels(z, step, 0.0, h * step, None, accel, tags=R.EXTRA_TAGS[:7]); t1 = time.time() - t
print("accel", img.shape, img.dtype, f"{t1:.1f}s", {k: tuple(round(x, 4) for x in v) for k, v in st.items()})
r, c = int(round(h - 1 - 300 / step)), int(round(250 / step))
r = int(300 / step); print("center vs bg:", img[:, r, c].tolist(), img[:, 800, 800].tolist())
t = time.time(); img2, _ = R.extra_channels(z, step, 0.0, h * step, None, None, stats=st, tags=R.EXTRA_TAGS[:3]); print("rvt", f"{time.time()-t:.1f}s",
      "max|diff| vs accel:", [int(np.abs(img2[i].astype(int) - img[i].astype(int))[200:800, 200:800].max()) for i in range(3)])
