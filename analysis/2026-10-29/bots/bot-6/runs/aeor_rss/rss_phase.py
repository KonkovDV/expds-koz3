"""bot-6: пик RSS фазы dfine_aeor на одном растре с заглушкой детектора (flip включён).
python rss_phase.py <repo> <tif> <mk> [overview]"""
import json, sys, time, logging
from pathlib import Path
from types import SimpleNamespace
repo, tif, mk = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(repo))
import numpy as np, torch
torch.set_num_threads(2)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from expds import dfine_aeor as A


class StubModel:
    """Выход как у DFineForObjectDetection: logits [N,300,5], pred_boxes cxcywh 0..1."""
    def __init__(self):
        self.g = torch.Generator().manual_seed(0)
        self.calls = 0

    def __call__(self, pixel_values):
        self.calls += 1
        n = pixel_values.shape[0]
        m = pixel_values.mean()   # трогаем вход, как сеть
        logits = torch.randn(n, 300, 5, generator=self.g) - 3 + 0 * m
        boxes = torch.rand(n, 300, 4, generator=self.g) * 0.05 + torch.tensor([0.5, 0.5, 0.01, 0.01])
        return SimpleNamespace(logits=logits, pred_boxes=boxes)


model = StubModel()
cfg = {"dfine_aeor": {"enabled": True, "weights": {mk: ["stub"]}, "markups": [mk], "flip_tta": True,
                      "tile_s": 1e-6, "budget_s": 1e9, "require_thresholds": False, "device": "cpu"}}
reg = SimpleNamespace(name=tif.parent.parent.name, rasters={mk: [str(tif)]}, utm_json=None)
cache = {"device": torch.device("cpu")}
t0 = time.monotonic()
feats = A.region_feats(reg, cfg, None, cache, repo, loader=lambda st, m, root, c: ([model], list(A.MODEL_CLASSES)))
print("RSS_JSON=" + json.dumps({"tif": str(tif), "bytes": tif.stat().st_size, "n_feats": len(feats),
                                 "model_calls": model.calls, "s": round(time.monotonic() - t0, 1)}, ensure_ascii=False))
