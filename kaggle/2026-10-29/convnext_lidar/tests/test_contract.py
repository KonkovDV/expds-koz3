"""CPU-тест контракта весов bot-8: исполняет РЕАЛЬНЫЙ блок «Шаг 9б» из src/convnext_main.py (contract_dict/save_contract),
проверяет torch.load(weights_only=True), сборку сети из arch (src/cnx_contract.py) и равенство выходов исходной модели."""
import hashlib, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np, torch
H = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(H / "src/vendor"), str(H / "src")]
torch.manual_seed(0); torch.set_num_threads(2)
import cnx_head as HD, cnx_contract as CC
src = (H / "src/convnext_main.py").read_text(encoding="utf-8")
blk = src[src.index("# ── Шаг 9б."):src.index("# ── Шаг 10.")]
VMAN = json.load(open(H / "src/vendor/vendor_manifest.json", encoding="utf-8"))
RELIEF_SHA = {k: v for k, v in VMAN["sha256"].items() if k.startswith("expds") or k == "relief_ext.py"}
CFG = dict(res=0.5, crop=512, infer_stride=384, tta=8, min_score=0.05, peak_k=11, mask_thr=0.5, gor_min_px=600, fort_min_px=30,
           arch_min_px=30, max_det=20000, circle_r_m=1.0)
def _ver(m):
    import importlib.metadata as md
    try: return md.version(m)
    except Exception: return None
res = {}
for head_name in ("centerhead", "heat"):
    HEAD = HD.get_head(head_name, CFG)
    g = dict(torch=torch, np=np, json=json, hashlib=hashlib, HEAD=HEAD, HD=HD, CFG=CFG, TAGS5=("lrm_s", "lrm_l", "slope", "svf_r", "opp_r"),
             VMAN=VMAN, RELIEF_SHA=RELIEF_SHA, _ver=_ver, TIMM_NAME="convnext_tiny.fb_in1k", CIRCLE_R=0.5, OUT=Path(tempfile.mkdtemp()))
    exec(compile(blk, "convnext_main.py:Шаг9б", "exec"), g)
    for ch, init in ((3, "sc"), (5, "in")):
        t = time.time()
        model = HEAD.build(ch, "tu-convnext_tiny", None).eval()
        with torch.no_grad():                         # случайные, но не нулевые смещения голов
            for p in model.parameters(): p.add_(torch.randn_like(p) * 1e-2)
        sd = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in model.state_dict().items()}   # как в ядре
        job = dict(variant=f"c{ch}-{init}", ch=ch, init=init, fold=1); tag = f"{job['variant']}_f1"
        name = g["save_contract"](model, sd, job, tag); p = g["OUT"] / name
        c = torch.load(p, map_location="cpu", weights_only=True)
        net = CC.build_from_arch(c["arch"]); miss = net.load_state_dict({k: v.float() for k, v in c["state_dict"].items()}, strict=True); net.eval()
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})   # те же fp16-округлённые веса
        x = torch.randn(2, ch, 96, 128)
        with torch.no_grad():
            a = model(x); b = net(x); fl = CC.CnxFlat(net, c)(x)
        a = a if isinstance(a, dict) else {"y": a}; b = b if isinstance(b, dict) else {"hm": b[:, :2], "mask": b[:, 2:]}
        d = {k: float((a[k] - b[k]).abs().max()) for k in a if a[k] is not None and k in b}
        try:
            with torch.no_grad(): net(torch.randn(1, ch, 100, 100)); hw = "100×100 прошёл"
        except Exception as e: hw = "100×100 (не кратно 32): " + type(e).__name__
        res[f"{head_name}/{tag}"] = dict(file=name, bytes=p.stat().st_size, weights_only_load=True, keys=sorted(k for k in c if k != "state_dict"),
            n_tensors=len(c["state_dict"]), dtypes=sorted({str(v.dtype) for v in c["state_dict"].values()}), arch=c["arch"], in_kind=c["in_kind"],
            n_out=c["n_out"], out_names=c["out_names"], out_stride=c["out_stride"], prep_code_sha=c["prep_code_sha"],
            max_abs_vs_kernel_model=d, flat_shape=list(fl.shape), hw_check=hw, t_s=round(time.time() - t, 1))
        print(head_name, tag, json.dumps({k: res[f"{head_name}/{tag}"][k] for k in ("bytes", "n_tensors", "dtypes", "max_abs_vs_kernel_model", "flat_shape", "hw_check")}, ensure_ascii=False), flush=True)
        del model, net, c
import segmentation_models_pytorch as smp
k1 = list(smp.Unet("tu-convnext_tiny", encoder_weights=None, in_channels=3, classes=3).state_dict())
k2 = list(smp.Unet("tu-convnext_tiny.fb_in1k", encoder_weights=None, in_channels=3, classes=3).state_dict())
res["in_vs_sc_keys_equal"] = k1 == k2; print("ключи tu-convnext_tiny == tu-convnext_tiny.fb_in1k:", k1 == k2)
res["versions"] = {m: _ver(m) for m in ("torch", "timm", "segmentation_models_pytorch")}
Path(H / "logs/test_contract.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
