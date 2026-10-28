# Юнит-тесты TPU-частей (бокс, CPU): head_loss_nosync == HEAD.loss (bot-2 center_loss); XAdamW == torch.optim.AdamW;
# _pad_bs; при XLA (CNX_TEST_XLA=1, PJRT_DEVICE=CPU) — те же проверки на устройстве xla. Источник блока: src/tpu_train.py.
import json, math, os, sys, time, types, pathlib
import numpy as np, torch
H = pathlib.Path(__file__).resolve().parents[1]; V = H.parent / "convnext_lidar" / "src" / "vendor"
sys.path.insert(0, str(V)); import cnx_head as HD
torch.manual_seed(0); np.random.seed(0)
CFG = dict(crop=64, bs=4, iters=10, lr=3e-4, wd=1e-4, warmup=2, ema=0.999, workers=0, amp=True, log_every=5, infer_bs=8,
           min_iter_frac=0.25, infer_reserve_h=0.0, seed=0, circle_r_m=1.0, default_diam_m=10.0)
HEAD = HD.get_head("centerhead", CFG)
XLA = os.environ.get("CNX_TEST_XLA") == "1"
ns = dict(os=os, np=np, math=math, torch=torch, json=json, time=time, ARGS=types.SimpleNamespace(allow_cpu=True), HEAD=HEAD,
          XLA_ON=XLA, CFG=CFG, TPU_CHIP=None, build_model=lambda ch, init: None)
exec(compile((H / "src" / "tpu_train.py").read_text(encoding="utf-8"), "tpu_train.py", "exec"), ns)
dev = torch.device("cpu")
if XLA:
    import torch_xla; dev = torch_xla.device()
res = {}
# 1) лосс без синхронизаций == HEAD.loss (float-части)
m = HEAD.build(3, os.environ.get("CNX_ENCODER_TEST", "tu-convnext_atto"), None).eval()
x = torch.randn(2, 3, 64, 64)
with torch.no_grad(): o = {k: v for k, v in m(x).items() if v is not None}
objs = np.array([[20.3, 30.7, 12.0, 0], [40.0, 10.0, 8.0, 1], [50.5, 50.5, 20.0, 0]])
tg = HEAD.targets(objs, 64); tb = {k: torch.from_numpy(np.stack([v, v])) for k, v in tg.items()}
valid = torch.ones(2, 64, 64); valid[1, :, 40:] = 0
l_ref, p_ref = HEAD.loss(o, tb, valid)
od = {k: v.to(dev) for k, v in o.items()}; tbd = {k: v.to(dev) for k, v in tb.items()}
l_ns, p_ns = ns["head_loss_nosync"](od, tbd, valid.to(dev))
d_tot = abs(float(l_ref) - float(l_ns)); d_parts = {k: abs(float(p_ref[k]) - float(p_ns[k])) for k in p_ref}
res["loss"] = dict(ref=float(l_ref), nosync=float(l_ns), abs_diff=d_tot, parts_ref={k: float(v) for k, v in p_ref.items()},
                   parts_abs_diff=d_parts, parts_are_tensors=all(torch.is_tensor(v) for v in p_ns.values()))
assert set(p_ref) == set(p_ns), (p_ref, p_ns); assert d_tot < 1e-5 and max(d_parts.values()) < 1e-5, res["loss"]
# 2) XAdamW == torch.optim.AdamW (5 шагов, разные lr — как косинус)
torch.manual_seed(1); a = torch.nn.Sequential(torch.nn.Linear(7, 5), torch.nn.GELU(), torch.nn.Linear(5, 3))
b = torch.nn.Sequential(torch.nn.Linear(7, 5), torch.nn.GELU(), torch.nn.Linear(5, 3)); b.load_state_dict(a.state_dict()); b.to(dev)
oa = torch.optim.AdamW(a.parameters(), lr=3e-4, weight_decay=1e-4); ob = ns["XAdamW"](b.parameters(), 1e-4, dev)
xs = torch.randn(5, 4, 7)
for i, lr in enumerate([1e-4, 3e-4, 2e-4, 5e-5, 1e-3]):
    for g in oa.param_groups: g["lr"] = lr
    oa.zero_grad(); a(xs[i]).pow(2).mean().backward(); oa.step()
    ob.zero_grad(); b(xs[i].to(dev)).pow(2).mean().backward(); ob.step(torch.tensor(lr).to(dev))
    if XLA: torch_xla.sync()
dmax = max(float((pa - pb.cpu()).abs().max()) for pa, pb in zip(a.parameters(), b.parameters()))
res["xadamw_max_abs_diff"] = dmax; assert dmax < 1e-6, dmax
# 3) _pad_bs
z = ns["_pad_bs"](np.ones((3, 2, 4, 4), np.float32), 8); assert z.shape == (8, 2, 4, 4) and z[3:].sum() == 0 and z[:3].sum() == 96
res["pad_bs"] = list(z.shape); res["device"] = str(dev)
print("TEST_TPU_PARTS", json.dumps(res, ensure_ascii=False)); print("OK")
