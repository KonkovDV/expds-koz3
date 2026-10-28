# Где появляются aten::_local_scalar_dense (синхронизации хоста) в шаге обучения на XLA (PJRT CPU, convnext_atto, 128 px)
import os, sys, json, pathlib, types, math, time, numpy as np, torch
H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H.parent / "convnext_lidar/src/vendor"))
import torch_xla, torch_xla.debug.metrics as met; import cnx_head as HD
dev = torch_xla.device()
CFG = dict(crop=128, bs=2, circle_r_m=1.0, default_diam_m=8.0, wd=1e-4)
HEAD = HD.get_head("centerhead", CFG)
ns = dict(os=os, np=np, math=math, torch=torch, json=json, time=time, ARGS=types.SimpleNamespace(allow_cpu=True), HEAD=HEAD, XLA_ON=True, CFG=CFG, TPU_CHIP=None)
exec(compile((H / "src/tpu_train.py").read_text(), "tpu_train.py", "exec"), ns)
c = lambda: int(met.counter_value("aten::_local_scalar_dense") or 0)
m = HEAD.build(3, "tu-convnext_atto", None).to(dev); opt = ns["XAdamW"](m.parameters(), 1e-4, dev)
tg = HEAD.targets(np.array([[20.3, 30.7, 12.0, 0], [60.0, 90.0, 8.0, 1]]), 128); tb = {k: torch.from_numpy(np.stack([v, v])).to(dev) for k, v in tg.items()}
x = torch.randn(2, 3, 128, 128).to(dev); val = torch.ones(2, 128, 128).to(dev); m3 = torch.zeros(2, 3, 128, 128).to(dev)
F = torch.nn.functional
def mask_loss(logit, gt, valid):
    v = valid[:, None].float(); g = gt.float(); lg = logit.float()
    bce = (F.binary_cross_entropy_with_logits(lg, g, reduction="none") * v).sum() / (v.sum() * lg.shape[1]).clamp(min=1)
    p = torch.sigmoid(lg) * v; inter = (p * g).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + (g * v).sum((0, 2, 3))
    return bce + (1 - (2 * inter + 1) / (den + 1)).mean()
r = {}
for step in range(2):
    a = c()
    with torch.autocast("xla", dtype=torch.bfloat16): o = m(x)
    r[f"s{step}_fwd"] = c() - a; a = c()
    lh, parts = ns["head_loss_nosync"](o, tb, val); lm = mask_loss(o["mask"], m3, val); loss = lh + lm
    r[f"s{step}_loss"] = c() - a; a = c()
    opt.zero_grad(); loss.backward(); r[f"s{step}_bwd"] = c() - a; a = c()
    opt.step(torch.tensor(1e-4).to(dev)); r[f"s{step}_opt"] = c() - a; a = c()
    torch_xla.sync(); r[f"s{step}_sync"] = c() - a
print("SYNC_PROBE", json.dumps(r), "fallbacks", {k: met.counter_value(k) for k in met.counter_names() if k.startswith("aten::")})
