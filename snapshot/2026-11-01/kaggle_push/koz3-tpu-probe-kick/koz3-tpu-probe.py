# koz3-tpu-probe — tiny hardware/software probe for Kaggle TPU VM (no training).
import json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

info = {
    "hostname": platform.node(),
    "python": sys.version,
    "platform": platform.platform(),
    "executable": sys.executable,
    "cwd": os.getcwd(),
    "env_keys_tpu": {k: os.environ.get(k) for k in sorted(os.environ) if any(s in k.upper() for s in ("TPU", "XLA", "PJRT", "ISTPU", "KAGGLE"))},
}

try:
    info["nproc"] = os.cpu_count()
    info["cpuinfo_model"] = None
    p = Path("/proc/cpuinfo")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("model name") or line.startswith("Hardware"):
                info["cpuinfo_model"] = line.split(":",1)[-1].strip(); break
    try:
        info["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[:40]
    except Exception as e:
        info["lscpu_error"] = repr(e)
except Exception as e:
    info["cpu_error"] = repr(e)

try:
    mem = Path("/proc/meminfo").read_text().splitlines()
    info["meminfo"] = {l.split(":")[0]: l.split(":")[1].strip() for l in mem if l.startswith(("MemTotal","MemAvailable","SwapTotal","SwapFree"))}
except Exception as e:
    info["mem_error"] = repr(e)

try:
    du = {}
    for path in ("/", "/kaggle", "/kaggle/working", "/tmp", "/kaggle/input"):
        try:
            u = shutil.disk_usage(path); du[path] = dict(total_gb=round(u.total/2**30,2), used_gb=round(u.used/2**30,2), free_gb=round(u.free/2**30,2))
        except Exception as e:
            du[path] = repr(e)
    info["disk"] = du
except Exception as e:
    info["disk_error"] = repr(e)

try:
    info["dev_accel"] = sorted(str(p) for p in Path("/dev").glob("accel*"))
    info["dev_vfio"] = sorted(str(p) for p in Path("/dev/vfio").glob("*")) if Path("/dev/vfio").exists() else []
except Exception as e:
    info["dev_error"] = repr(e)

def ver(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None

for pkg in ("torch","torchvision","torch_xla","jax","jaxlib","libtpu","libtpu-nightly","numpy","lightgbm","xgboost","scikit-learn","pandas","pyarrow","polars"):
    info[f"pkg_{pkg}"] = ver(pkg)

PROBE = r'''
import json, os, time
t=time.time()
out={"ok":False}
try:
    import torch
    out["torch"]=torch.__version__
    out["cuda_available"]=torch.cuda.is_available()
except Exception as e:
    out["torch_error"]=repr(e)
try:
    import torch_xla, torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    d = torch_xla.device()
    v = float(((torch.ones(8, device=d)*2).sum()).cpu())
    out.update(torch_xla=torch_xla.__version__, device_type=xr.device_type(),
               n_dev=len(torch_xla.devices()), global_devices=xr.global_runtime_device_count(),
               ok=(v==16.0), t_s=round(time.time()-t,2),
               accel_type=os.environ.get("TPU_ACCELERATOR_TYPE"), istpuvm=os.environ.get("ISTPUVM"),
               pjrt=os.environ.get("PJRT_DEVICE"))
    try: out["hw"]=xm.xla_device_hw(d)
    except Exception as e: out["hw"]=repr(e)
    try:
        import torch_xla.tpu as T; out["tpu_version"]=T.version()
    except Exception as e: out["tpu_version"]=repr(e)
except Exception as e:
    out["xla_error"]=repr(e)
try:
    import jax, jaxlib
    out["jax"]=jax.__version__; out["jaxlib"]=jaxlib.__version__
    try:
        out["jax_devices"]=[str(d) for d in jax.devices()]
    except Exception as e:
        out["jax_devices_error"]=repr(e)
except Exception as e:
    out["jax_error"]=repr(e)
print("PROBE_JSON "+json.dumps(out), flush=True)
'''
os.environ.setdefault("PJRT_DEVICE", "TPU")
try:
    p = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=600)
    info["probe_rc"] = p.returncode
    info["probe_stdout_tail"] = p.stdout[-4000:]
    info["probe_stderr_tail"] = p.stderr[-4000:]
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            info["xla_probe"] = json.loads(line[len("PROBE_JSON "):])
except Exception as e:
    info["probe_error"] = repr(e)

try:
    import lightgbm as lgb
    import numpy as np
    X = np.random.randn(5000, 32).astype("float32"); y = (X[:,0] > 0).astype("float32")
    t1 = time.time()
    lgb.train({"objective":"binary","verbosity":-1,"num_threads":os.cpu_count() or 1},
              lgb.Dataset(X,y), num_boost_round=50)
    info["lightgbm_smoke_s"] = round(time.time()-t1, 3)
    info["lightgbm_ok"] = True
except Exception as e:
    info["lightgbm_ok"] = False
    info["lightgbm_error"] = repr(e)

out_path = Path("/kaggle/working/tpu_probe.json")
out_path.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
log("WROTE", out_path)
log("SUMMARY", json.dumps({
    "nproc": info.get("nproc"),
    "mem": info.get("meminfo"),
    "disk_working": (info.get("disk") or {}).get("/kaggle/working"),
    "xla": info.get("xla_probe"),
    "pkgs": {k: info[k] for k in info if k.startswith("pkg_")},
    "lgbm": info.get("lightgbm_ok"),
    "lgbm_s": info.get("lightgbm_smoke_s"),
}, ensure_ascii=False, default=str)[:4000])
print("PROBE_DONE", flush=True)
