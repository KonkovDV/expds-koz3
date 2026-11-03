# koz3-acc2-gpu-smoke — verify GPU + internet on elenakonkova (private)
import json, os, platform, socket, ssl, subprocess, sys, time, urllib.request
from pathlib import Path

T0 = time.time()
out = {"ok": False, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

def log(*a):
    print(f"[{time.time()-T0:5.1f}s]", *a, flush=True)

log("start", platform.platform(), sys.version.split()[0])

# Internet probe
inet = {"ok": False}
try:
    with urllib.request.urlopen("https://www.kaggle.com", timeout=20) as r:
        inet["status"] = r.status
        inet["ok"] = r.status == 200
    log("internet ok", inet["status"])
except Exception as e:
    inet["error"] = type(e).__name__ + ":" + str(e)[:200]
    log("internet FAIL", inet["error"])
out["internet"] = inet

# GPU probe
gpu = {"ok": False}
try:
    import torch
    gpu["torch"] = torch.__version__
    gpu["cuda_available"] = bool(torch.cuda.is_available())
    gpu["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        gpu["devices"] = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpu["devices"].append({"name": p.name, "total_mem_gb": round(p.total_memory/2**30, 2)})
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        gpu["matmul_ok"] = True
        gpu["matmul_sum"] = float(y)
        gpu["ok"] = True
        log("GPU ok", gpu["devices"])
    else:
        log("CUDA not available")
except Exception as e:
    gpu["error"] = type(e).__name__ + ":" + str(e)[:300]
    log("GPU FAIL", gpu["error"])
out["gpu"] = gpu

# nvidia-smi
try:
    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
    out["nvidia_smi"] = r.stdout.strip() or r.stderr.strip()
except Exception as e:
    out["nvidia_smi_error"] = repr(e)

out["ok"] = bool(inet.get("ok")) and bool(gpu.get("ok"))
out["elapsed_s"] = round(time.time() - T0, 2)
Path("/kaggle/working/smoke_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
Path("/kaggle/working/STATUS.txt").write_text("DONE\n" if out["ok"] else "FAILED\n", encoding="utf-8")
log("RESULT", json.dumps(out))
sys.exit(0 if out["ok"] else 1)
