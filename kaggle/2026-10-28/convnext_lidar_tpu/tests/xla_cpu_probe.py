# Проба torch_xla 2.8 на CPU-устройстве (PJRT_DEVICE=CPU): импорт, устройство, autocast bf16, шаг обучения ConvNeXt-T U-Net + centerhead.
import os, sys, time, json
os.environ.setdefault("PJRT_DEVICE", "CPU")
import torch, torch_xla, torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
torch.set_num_threads(2)
print("torch", torch.__version__, "torch_xla", torch_xla.__version__, "device_type", xr.device_type(), flush=True)
dev = torch_xla.device(); print("device", dev, flush=True)
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/vendor"]
from centerhead.head import build_unet_centernet
from centerhead import cfg_with
import timm, segmentation_models_pytorch as smp
print("timm", timm.__version__, "smp", smp.__version__, flush=True)
m = build_unet_centernet(5, "tu-convnext_tiny", encoder_weights=None, cfg=cfg_with(stride=4)).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
S = int(os.environ.get("P_S", 64)); B = 2
x = torch.randn(B, 5, S, S).to(dev)
for it in range(3):
    t = time.time()
    with torch.autocast("xla", dtype=torch.bfloat16):
        o = m(x)
    loss = sum(v.float().mean() for v in o.values())
    opt.zero_grad(); loss.backward(); opt.step(); xm.mark_step()
    xm.wait_device_ops(); print("it", it, "dt", round(time.time() - t, 2), {k: (tuple(v.shape), str(v.dtype)) for k, v in o.items()}, flush=True)
print("loss", float(loss.cpu()))
