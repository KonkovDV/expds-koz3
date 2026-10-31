# bot-5: CUDA check of DINOv2 with the same code as the gor_dense phase (copy of chk from koz3-b5-gordense-timing.py). python cuda_check.py <repo_root>
import sys, json, time, tomllib, torch
from pathlib import Path
R = Path(sys.argv[1]); sys.path.insert(0, str(R / "inference" / "lib")); sys.path.insert(0, str(R))
from expds import gor_dense as GD, gor_gate
st = GD.settings(tomllib.load(open(R / "configs/default.toml", "rb"))); dev = gor_gate._device(st)
m = gor_gate.load_backbone(st, R, dev); m = m.half() if dev.type == "cuda" else m
x = torch.zeros(1, 3, 518, 518, device=dev, dtype=next(m.parameters()).dtype); t = time.time()
with torch.no_grad(): m(pixel_values=x)
torch.cuda.synchronize() if dev.type == "cuda" else None
print("CUDA_CHECK", json.dumps(dict(device=str(dev), param_device=str(next(m.parameters()).device), dtype=str(next(m.parameters()).dtype),
      cuda=torch.cuda.is_available(), name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, fwd_s=round(time.time() - t, 3))))
