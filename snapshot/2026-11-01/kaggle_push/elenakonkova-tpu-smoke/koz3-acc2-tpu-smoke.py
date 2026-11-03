# koz3-acc2-tpu-smoke — verify TPU availability on elenakonkova
import json, os, platform, sys, time
from pathlib import Path
T0 = time.time()
out = {"ok": False, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
def log(*a):
    print(f"[{time.time()-T0:5.1f}s]", *a, flush=True)
log("start", platform.platform())
out["env_tpu"] = {k: os.environ.get(k) for k in sorted(os.environ) if any(s in k.upper() for s in ("TPU","XLA","PJRT","ISTPU"))}
tpu = {"ok": False}
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch
    d = xm.xla_device()
    x = torch.randn(64, 64, device=d)
    y = (x @ x).sum()
    xm.mark_step()
    tpu["torch_xla"] = getattr(torch_xla, "__version__", "?")
    tpu["device"] = str(d)
    tpu["matmul_sum"] = float(y.item())
    tpu["ok"] = True
    log("TPU ok", tpu)
except Exception as e:
    tpu["error"] = type(e).__name__ + ":" + str(e)[:400]
    log("TPU FAIL", tpu["error"])
out["tpu"] = tpu
out["ok"] = bool(tpu.get("ok"))
out["elapsed_s"] = round(time.time()-T0, 2)
Path("/kaggle/working/tpu_smoke_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
Path("/kaggle/working/STATUS.txt").write_text("DONE\n" if out["ok"] else "FAILED\n", encoding="utf-8")
log("RESULT", json.dumps(out))
sys.exit(0 if out["ok"] else 1)
