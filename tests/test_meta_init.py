"""Модель, собранная на meta + assign, совпадает с обычной загрузкой: те же тензоры и тот же выход."""
import os, sys, glob, time
import numpy as np, torch
REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
from expds import infer_nn


def test_meta_equal():
    paths = sorted(glob.glob(os.path.join(REPO, "models", "*.pt")))
    assert paths
    x = torch.from_numpy(np.random.default_rng(0).normal(size=(2, 3, 128, 128)).astype(np.float32))
    for p in paths:
        if "classes" not in torch.load(p, map_location="cpu", weights_only=False):
            continue
        os.environ["EXPDS_META_INIT"] = "0"; t0 = time.perf_counter(); a = infer_nn.Segmenter(p, "cpu"); ta = time.perf_counter() - t0
        os.environ["EXPDS_META_INIT"] = "1"; t0 = time.perf_counter(); b = infer_nn.Segmenter(p, "cpu"); tb = time.perf_counter() - t0
        sa, sb = a.model.state_dict(), b.model.state_dict()
        assert sa.keys() == sb.keys()
        for k in sa:
            assert sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]), k
        assert [n for n, _ in a.model.named_buffers()] == [n for n, _ in b.model.named_buffers()]
        c = int(a.meta.get("in_channels") or 3)
        xi = x[:, :1].repeat(1, c, 1, 1) if c != 3 else x
        t = int(a.tile); xi = torch.nn.functional.interpolate(xi, size=(t, t)) if t != 128 else xi
        with torch.inference_mode():
            assert torch.equal(a.model(xi), b.model(xi))
        print(f"{os.path.basename(p)}: обычная {ta:.2f}s meta {tb:.2f}s")
