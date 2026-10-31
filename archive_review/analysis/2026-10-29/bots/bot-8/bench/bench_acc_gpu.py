"""§7.12 (5) bench, clean-room branch bot8-cpu-a-clean (expds.lidar_a): tile blending + ensemble blend on the host
vs on the model's device.

Runs on CPU torch (box) or CUDA (bot 9, T4). 4 members as deployed_v20 (li_0 0.5, li_1 0.5, li_2 1.0 on 9 ch;
li_x_0 1.0 on 12 ch). Modes: host (= tiling.infer_map + runner.ensemble), device_exact (device tiles, host ensemble),
device (device tiles + device running sum) = runner.blend_stacks(accumulate=...). Reports wall (cuda-synced), process
CPU, peak VRAM, and equality vs host: bytes, max |Δ|, pixels flipping at the decode thresholds, and local-maximum peaks
(a simple max-pool NMS defined here, not the production decoder).

  python bench_acc_gpu.py --repo /path/to/wt-cpua-clean --dev cuda --px 5480 5480 --net stub --tta d8 --out acc_t4_stub.json
  --net stub: z[:, :7] (no network cost: isolates the host/device blending, as §7.11); --net unet: random-init smp
  Unet(resnet34) (realistic GPU overlap, needs segmentation_models_pytorch). --npz D.npz: x9/x12 from a cached DTM.
"""
import argparse
import json
import os
import resource
import statistics
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wt-cpua-clean"))
ap.add_argument("--dev", default="cpu")
ap.add_argument("--px", type=int, nargs=2, default=[1300, 1300])
ap.add_argument("--npz", default=None)
ap.add_argument("--net", default="stub", choices=["stub", "unet"])
ap.add_argument("--tta", default="1", help="1 | 3 (flips) | d8")
ap.add_argument("--tile", type=int, default=1024)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--modes", default="host,device_exact,device")
ap.add_argument("--out", default=None)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.repo))
import torch  # noqa: E402
from expds.lidar_a import runner, terrain, tiling  # noqa: E402

N_HM, N_MK = 3, 4          # 3 heat-map + 4 mask output channels (the deployed scenario-(a) heads)

tta = {"1": False, "3": True, "d8": "d8"}[a.tta]
dev = torch.device(a.dev)
cuda = dev.type == "cuda"


class Stub(torch.nn.Module):
    def forward(self, z):
        return z[:, :N_HM + N_MK]


def net(cin, seed):
    if a.net == "stub":
        return Stub()
    import segmentation_models_pytorch as smp
    torch.manual_seed(seed)
    return smp.Unet("resnet34", encoder_weights=None, in_channels=cin, classes=N_HM + N_MK).to(dev).eval()


if a.npz:
    z = np.load(a.npz)
    x12, vv = terrain.relief_stack_windowed(z["d"], z["v"], 0.25, extra=True, dev=a.dev)
else:
    H, W = a.px
    rng = np.random.default_rng(0)
    x12 = rng.normal(size=(12, H, W)).astype(np.float32)
    vv = np.ones((H, W), bool)
xs = {"": np.ascontiguousarray(x12[terrain.BASE_OF_EXTRA]), "x": x12}
models = [(net(9, 1), "", 0.5), (net(9, 2), "", 0.5), (net(9, 3), "", 1.0), (net(12, 4), "x", 1.0)]
ov = min(192, a.tile // 4)


def host():
    ps = [tiling.infer_map(m, xs[k], N_HM + N_MK, tile=a.tile, overlap=ov, tta=tta, dev=dev) for m, k, _ in models]
    return runner.ensemble(ps, models)


def mode(m):
    if m == "host":
        return host
    return lambda: runner.blend_stacks(xs, models, N_HM + N_MK, tta, dev, print, accumulate=m)


def peaks(prob, valid, thr=0.05, win=11):
    """Local maxima ≥ thr of each heat-map channel inside valid: a pixel equal to the max of its win×win window."""
    t = torch.from_numpy(np.ascontiguousarray(prob))
    mx = torch.nn.functional.max_pool2d(t[None], win, stride=1, padding=win // 2)[0]
    keep = ((t == mx) & (t >= thr) & torch.from_numpy(valid)[None]).nonzero().tolist()
    return [(c, y, x, 0, float(prob[c, y, x])) for c, y, x in keep]


def _peaks(pk, pr):
    a_, b_ = {q[:3]: q[4] for q in pk}, {q[:3]: q[4] for q in pr}
    common = a_.keys() & b_.keys()
    return {"peaks": len(pr), "peaks_only_new": len(a_.keys() - b_.keys()), "peaks_only_ref": len(b_.keys() - a_.keys()),
            "peak_score_max_abs": max((abs(a_[k] - b_[k]) for k in common), default=0.0)}


res = {"dev": str(dev), "gpu": torch.cuda.get_device_name(0) if cuda else None, "torch": torch.__version__,
       "px": list(x12.shape[1:]), "net": a.net, "tta": a.tta, "tile": a.tile, "threads": torch.get_num_threads(),
       "load1": os.getloadavg()[0], "modes": {}}
ref = None
with torch.no_grad():
    for m in a.modes.split(","):
        fn = mode(m)
        fn()                                            # warm-up (cuDNN, allocator)
        walls, cpus = [], []
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for _ in range(a.reps):
            c0 = time.process_time(); w0 = time.perf_counter()
            p = fn()
            if cuda:
                torch.cuda.synchronize()
            walls.append(time.perf_counter() - w0); cpus.append(time.process_time() - c0)
        r = {"wall": walls, "cpu": cpus, "wall_med": statistics.median(walls), "cpu_med": statistics.median(cpus),
             "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None}
        if ref is None:
            ref = p
            pk_ref = peaks(ref[:N_HM], vv)
        else:
            d = np.abs(p - ref)
            r.update(byte_equal=p.tobytes() == ref.tobytes(), max_abs=float(d.max()), px_diff=int((d > 0).sum()),
                     flips={str(t): int(((p > t) != (ref > t)).sum()) for t in (0.05, 0.1, 0.3, 0.35, 0.5)},
                     **_peaks(peaks(p[:N_HM], vv), pk_ref))
            r["x_wall"] = res["modes"][a.modes.split(",")[0]]["wall_med"] / r["wall_med"]
            r["x_cpu"] = res["modes"][a.modes.split(",")[0]]["cpu_med"] / r["cpu_med"]
        res["modes"][m] = r
        print(m, {k: (float(f'{v:.4g}') if isinstance(v, float) else v) for k, v in r.items() if k not in ("wall", "cpu")}, flush=True)
        del p
res["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
if a.out:
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
