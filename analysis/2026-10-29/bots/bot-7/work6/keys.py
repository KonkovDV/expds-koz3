import torch
from safetensors.torch import load_file
from transformers import DFineConfig, DFineForObjectDetection
p = "/workspace/kaggle_out/koz3-dfine-spor-full/dfine_f1"
m = DFineForObjectDetection(DFineConfig.from_json_file(p + "/config.json"))
sd = load_file(p + "/model.safetensors")
r = m.load_state_dict(sd, strict=False)
print(len(sd), len(m.state_dict()), len(r.unexpected_keys), len(r.missing_keys))
import re
def g(ks): return sorted({re.sub(r"\.\d+\.", ".N.", k) for k in ks})
print("UNEXP", g(r.unexpected_keys)[:40]); print("MISS", g(r.missing_keys)[:40])
