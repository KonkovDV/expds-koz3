"""CPU-проверка точности прямого прохода CnxUnetCenter: fp32 vs autocast bf16 vs autocast fp16 (+ чистый .half(), если CPU умеет).
Вход — реальные окна prep (ours_016_Хога.npz, выход cnx_prep, = prep-ядро побайтно). 2 потока. Пишет bf16_check.json."""
import json, resource, sys, time
from pathlib import Path
import numpy as np, torch
H = Path(__file__).resolve().parent
sys.path[:0] = [str(H.parent / "kaggle/convnext_lidar/src")]
import cnx_contract as CC
torch.set_num_threads(2); torch.manual_seed(0)
CK = H.parent / "kaggle/convnext_lidar/_scratch/dry/out_ch"
u8 = np.load(H / "ours_016_Хога.npz")["u8"]
SIZES = [int(s) for s in (sys.argv[1:] or ["256", "512"])]

def window(ch, n):
    _, Hh, W = u8.shape; y0, x0 = (Hh - n) // 2, (W - n) // 2
    return torch.from_numpy((u8[:ch, y0:y0 + n, x0:x0 + n].astype(np.float32) - 128) / 127)[None]

def metr(a, b):
    a, b = a.double(), b.double(); d = (a - b).abs()
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), rel_l2=float(d.norm() / a.norm().clamp(min=1e-12)),
                rel_max=float(d.max() / a.abs().max().clamp(min=1e-12)), ref_absmax=float(a.abs().max()))

def run(net, x, mode):
    with torch.no_grad():
        if mode == "fp32": o = net(x)
        elif mode in ("bf16", "fp16"):
            with torch.autocast("cpu", dtype=torch.bfloat16 if mode == "bf16" else torch.float16): o = net(x)
        elif mode == "half": o = net.half()(x.half()); net.float()
    return {k: v.float() for k, v in o.items()}

res = dict(torch=torch.__version__, threads=torch.get_num_threads(), cpu_bf16=torch.backends.mkldnn.is_available(), cases={})
for ch in (3, 5):
    for wname in ("random", "ckpt"):
        net = CC.CnxUnetCenter(in_channels=ch).eval()
        if wname == "ckpt":
            p = CK / ("ckpt_cnx_c3-sc_f3.pt" if ch == 3 else "ckpt_cnx_c5-in_f4.pt")
            c = torch.load(p, map_location="cpu", weights_only=False)
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in c["state_dict"].items()}, strict=True)
            wname = f"ckpt:{p.name}(dry-run {c['job'].get('variant')})"
        for n in SIZES:
            x = window(ch, n); t = time.time(); ref = run(net, x, "fp32"); case = {"fp32_s": round(time.time() - t, 2)}
            for mode in ("bf16", "fp16", "half"):
                try:
                    t = time.time(); o = run(net, x, mode)
                except Exception as e:
                    case[mode] = f"ERR {type(e).__name__}: {str(e)[:120]}"; continue
                m = {k: metr(ref[k], o[k]) for k in ("hm", "wh", "off", "mask")}
                sr, so = torch.sigmoid(ref["hm"]), torch.sigmoid(o["hm"])
                m["sigmoid_hm"] = metr(sr, so)
                m["sigmoid_hm"]["flip@0.05"] = int(((sr > 0.05) != (so > 0.05)).sum()); m["sigmoid_hm"]["n_ref>0.05"] = int((sr > 0.05).sum())
                sm, om = torch.sigmoid(ref["mask"]), torch.sigmoid(o["mask"])
                m["sigmoid_mask"] = metr(sm, om); m["sigmoid_mask"]["flip@0.5"] = int(((sm > .5) != (om > .5)).sum())
                m["finite"] = all(bool(torch.isfinite(v).all()) for v in o.values()); m["s"] = round(time.time() - t, 2)
                case[mode] = m
            res["cases"][f"c{ch}|{wname}|{n}"] = case
            print(f"c{ch} {wname[:30]} {n}", {k: (v if isinstance(v, (str, float)) else {kk: f"{vv['max_abs']:.3g}/{vv['rel_l2']:.2g}" for kk, vv in v.items() if isinstance(vv, dict)}) for k, v in case.items()}, flush=True)
        del net
res["maxrss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
json.dump(res, open(H / f"bf16_check_{'_'.join(map(str, SIZES))}.json", "w"), indent=1, ensure_ascii=False)
print("maxrss_MB", res["maxrss_MB"])
