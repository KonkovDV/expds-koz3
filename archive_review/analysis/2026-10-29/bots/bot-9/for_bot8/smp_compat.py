"""Совместимость CnxUnetCenter: эталон в нашем venv (smp 0.6.0/timm 1.0.29) vs venv bot-8 (smp 0.5.0/timm 1.0.30): ключи + выход fp32."""
import json, sys
from pathlib import Path
import numpy as np, torch
H = Path(__file__).resolve().parent
sys.path[:0] = [str(H.parent / "kaggle/convnext_lidar/src")]
import cnx_contract as CC, segmentation_models_pytorch as smp, timm
torch.set_num_threads(2)
u8 = np.load(H / "ours_016_Хога.npz")["u8"]; out = {}
REF = H / "_smp_ref.pt"; ref = torch.load(REF) if REF.exists() else {}
for ch, f in ((3, "ckpt_cnx_c3-sc_f3.pt"), (5, "ckpt_cnx_c5-in_f4.pt")):
    c = torch.load(H.parent / "kaggle/convnext_lidar/_scratch/dry/out_ch" / f, map_location="cpu", weights_only=False)
    net = CC.CnxUnetCenter(in_channels=ch).eval()
    keys = sorted(net.state_dict()); r = net.load_state_dict({k: v.float() for k, v in c["state_dict"].items()}, strict=True)
    x = torch.from_numpy((u8[:ch, 100:356, 200:456].astype(np.float32) - 128) / 127)[None]
    with torch.no_grad(): o = net(x)
    if f"c{ch}" in ref:
        out[f"c{ch}"] = dict(keys_eq=keys == ref[f"c{ch}"]["keys"], **{k: float((o[k] - ref[f"c{ch}"]["o"][k]).abs().max()) for k in o})
    else:
        ref[f"c{ch}"] = dict(keys=keys, o=o); out[f"c{ch}"] = "ref saved"
if not REF.exists(): torch.save(ref, REF)
print(json.dumps(dict(smp=smp.__version__, timm=timm.__version__, torch=torch.__version__, res=out)))
