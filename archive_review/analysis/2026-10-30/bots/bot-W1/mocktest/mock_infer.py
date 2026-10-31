# Мок-тест infer_region из ядер W1 (без rasterio/HF): фейковая модель и сцена, проверка 3 вариантов и свойств.
import sys, re, math, time, os, collections, json, types
import numpy as np, torch
from affine import Affine
path, tile, ov = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
s = open(path, encoding="utf-8").read()
a = s.index("def _valid(g)"); b = s.index("# ── Шаг 8.")
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
g = dict(np=np, torch=torch, math=math, time=time, os=os, collections=collections, CLASSES=CLASSES, MARKUP_TYPE="x", SP_GROUP="x_dfine",
         FOLD={"R": 0}, log=lambda *a: print("LOG", *a),
         CFG=dict(tile=tile, overlap=ov, bs=4, max_windows=0, max_det=100, min_score=0.05, nms_iou=0.5))
class Scene:
    def __init__(self, inv, key, size):
        self.W, self.H, self.crs = 2070, 1883, "EPSG:32638"; self.T = Affine(0.5, 0, 500000, 0, -0.5, 4800000)
    def read(self, x, y, n):
        im = np.full((n, n, 3), 100, np.uint8); return im
    def close(self): pass
g["Scene"] = Scene
exec(s[a:b], g)
class Out: pass
class M(torch.nn.Module):
    def forward(self, pixel_values):
        torch.manual_seed(int(pixel_values.sum().item()) % 1000 + int(pixel_values[0, 0, 0, :5].sum().item() * 10))
        B = pixel_values.shape[0]; o = Out()
        o.logits = torch.randn(B, 300, 5) * 2; o.pred_boxes = torch.rand(B, 300, 4) * torch.tensor([1, 1, .2, .2]) + torch.tensor([0, 0, .01, .01])
        return o
t = time.time()
out = g["infer_region"](M(), {"region": "R", "tifs": [("a/b/x.tif", 1)]}, torch.device("cpu"))
print({k: len(v) for k, v in out.items()}, f"{time.time()-t:.1f}s")
f = out["flipwbf"][0]; print(json.dumps(f["properties"], ensure_ascii=False)); print(f["geometry"]["coordinates"][0][0])
for k, v in out.items():
    c = [x["properties"]["confidence"] for x in v]; assert all(0.2 <= x <= 1 for x in c), k
print("OK")
