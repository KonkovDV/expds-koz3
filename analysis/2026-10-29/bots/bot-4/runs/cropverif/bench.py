"""Per-crop CPU cost of candidate embedders, 2 threads, batch 32. bot-4 CROPVERIF."""
import os, time, json, torch, timm, resource
torch.set_num_threads(2)
from transformers import AutoModel
import segmentation_models_pytorch as smp
res = {}
def bench(name, m, px, n=64, bs=32):
    m.eval(); x = torch.randn(bs, 3, px, px)
    with torch.inference_mode():
        m(x[:4])
        t = time.perf_counter(); k = 0
        while k < n: m(x); k += bs
        dt = (time.perf_counter() - t) / k
    res[f"{name}@{px}"] = round(dt * 1000, 2); print(name, px, f"{dt*1000:.1f} ms/crop", flush=True)
enc = smp.encoders.get_encoder("resnet18", in_channels=3, depth=5, weights=None)
bench("resnet18_smp(sp_crop)", enc, 96)
cn = timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=0)
for px in (96, 128, 224): bench("convnext_tiny.fb_in22k", cn, px)
for r in ("facebook/dinov2-with-registers-small", "facebook/dinov2-with-registers-base"):
    m = AutoModel.from_pretrained(r)
    f = lambda x, m=m: m(pixel_values=x).pooler_output
    class W(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return s.m(pixel_values=x).pooler_output
    for px in (112, 224): bench(r.split('/')[-1], W(m), px)
res["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
res["threads"] = torch.get_num_threads(); res["torch"] = torch.__version__
json.dump(res, open("bench.json", "w"), indent=1); print(res)
