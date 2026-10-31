"""Сквозной прогон pipeline.run на выборке под cProfile (prefetch=0 — всё в главном потоке).
usage: prof_run.py REPO SAMPLE OUT_PREFIX [cfg_override_toml_text]
"""
import cProfile, json, logging, os, pstats, sys, time, tomllib
from pathlib import Path
repo, sample, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo / "inference" / "lib")); sys.path.insert(0, str(repo))
import torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                    stream=open(str(out) + ".log", "w"))
text = (repo / "configs" / "default.toml").read_text(encoding="utf-8")
text = text.replace("prefetch = 2", "prefetch = " + os.environ.get("PREFETCH", "0"))
if os.environ.get("NO_WATCHDOG", "1") == "1":
    text = text.replace("watchdog = true", "watchdog = false")
cfgp = Path(str(out) + ".toml"); cfgp.write_text(text, encoding="utf-8")
tomllib.loads(text)
from expds import pipeline
if os.environ.get("STUB_FORWARD", "1") == "1":
    # Дешёвая детерминированная замена сети: сигмоида сглаженного входа.
    # Нужна для замера CPU-части (на площадке сеть идёт на 4090); обе версии кода получают одно и то же.
    from expds import infer_nn
    import torch.nn.functional as F
    def _stub(self, xb):
        k = len(self.classes)
        g = F.avg_pool2d(xb.mean(1, keepdim=True), 4, 4)
        loc = g - F.avg_pool2d(g, 7, 1, 3, count_include_pad=False)
        shifts = torch.linspace(0.0, 0.6, k, dtype=xb.dtype).view(1, k, 1, 1)
        p = torch.sigmoid(loc * 6.0 - 3.0 - shifts)
        return F.interpolate(p, scale_factor=4, mode="nearest")
    infer_nn.Segmenter._forward = _stub
if os.environ.get("DEVICE_PATH", "1") == "1":
    # Путь как на площадке (CUDA): окна и накопление в torch, а не в numpy.
    from expds import infer_nn
    _orig_auto = infer_nn.Segmenter._auto_batch
    def _pred(self, img, overlap=0.25, batch=None):
        import numpy as np
        c, h, w = img.shape
        t = self.tile
        batch = 64 if batch is None else batch
        step = max(1, int(t * (1 - overlap)))
        ys = list(range(0, max(h - t, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - t, 0) + 1, step)) or [0]
        if ys[-1] + t < h: ys.append(h - t)
        if xs[-1] + t < w: xs.append(w - t)
        x_all = np.pad(img, ((0, 0), (0, max(0, t - h)), (0, max(0, t - w))))
        ramp = np.minimum(np.arange(t) + 1, np.arange(t)[::-1] + 1).astype(np.float32)
        weight = np.minimum.outer(ramp, ramp); weight /= weight.max()
        coords = [(y, x) for y in ys for x in xs]
        return self._predict_on_device(x_all, coords, weight, batch)[:, :h, :w]
    if not hasattr(infer_nn.Segmenter, "_speedup_native_predict"):
        infer_nn.Segmenter.predict = _pred
prof = cProfile.Profile()
t = time.perf_counter()
if os.environ.get("NOPROF") != "1":
    prof.enable()
s = pipeline.run(sample, Path(str(out) + ".geojson"), cfgp, repo)
prof.disable()
s["wall"] = time.perf_counter() - t
json.dump(s, open(str(out) + ".summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(s, ensure_ascii=False))
if os.environ.get("NOPROF") != "1":
    prof.dump_stats(str(out) + ".prof")
    with open(str(out) + ".txt", "w") as f:
        st = pstats.Stats(prof, stream=f)
        st.sort_stats("cumulative").print_stats(70)
        st.sort_stats("tottime").print_stats(50)
