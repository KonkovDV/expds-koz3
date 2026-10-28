# ── Шаг 1 (TPU). --require-tpu (по умолчанию) — до pip ────────────────────────────────────────────────────────
# Роли: главный процесс (оркестратор, сам TPU не трогает при CNX_TPU_PROCS>1) | воркер одного чипа (env CNX_TPU_CHIP).
# Главный проверяет TPU в отдельном подпроцессе-пробе: импорт torch_xla в главном мог бы занять чипы/топологию до воркеров.
# @@TPU_KERNEL@@ (метка: по ней воркер находит исходник этого ядра)
TPU_CHIP = os.environ.get("CNX_TPU_CHIP")
IS_WORKER = TPU_CHIP is not None
XLA_DEV = os.environ.get("CNX_TPU_DEV", "xla")      # xla | cpu (cpu — только бокс: проверка логики без torch_xla-компиляции)
os.environ.setdefault("PJRT_DEVICE", "CPU" if ARGS.allow_cpu else "TPU")
if os.environ.pop("XLA_USE_BF16", None) is not None:   # устарел в torch_xla 2.6, удалён в 2.8 (релиз v2.6.0, «Deprecations»)
    print("XLA_USE_BF16 снят: bf16 — через torch.autocast('xla', dtype=torch.bfloat16)", flush=True)
TPU_PROBE_SRC = r'''
import json, os, time
t = time.time()
import torch, torch_xla, torch_xla.runtime as xr, torch_xla.core.xla_model as xm
d = torch_xla.device(); v = float(((torch.ones(8, device=d) * 2).sum()).cpu())
r = dict(torch=torch.__version__, torch_xla=torch_xla.__version__, pjrt=os.environ.get("PJRT_DEVICE"), device_type=xr.device_type(),
         n_dev=len(torch_xla.devices()), global_devices=xr.global_runtime_device_count(), ok=v == 16.0, t_s=round(time.time() - t, 1),
         accel_type=os.environ.get("TPU_ACCELERATOR_TYPE"), istpuvm=os.environ.get("ISTPUVM"))
try: r["hw"] = xm.xla_device_hw(d)
except Exception as e: r["hw"] = repr(e)
try:
    import torch_xla.tpu as T; r["tpu_version"] = T.version()
except Exception as e: r["tpu_version"] = repr(e)
for m in ("libtpu", "libtpu-nightly", "jax", "jaxlib"):
    try:
        from importlib.metadata import version; r[m] = version(m)
    except Exception: pass
print("TPU_PROBE " + json.dumps(r), flush=True)
'''
TPU_INFO = {}
def tpu_probe():
    import importlib.util
    if importlib.util.find_spec("torch_xla") is None:
        raise SystemExit("--require-tpu: в образе нет torch_xla — вероятно, при push закрепился CPU/GPU-образ "
                         "(github.com/Kaggle/kaggle-cli/issues/1197): убрать docker_image, docker_image_pinning_type=latest. Останов.")
    try: p = subprocess.run([sys.executable, "-c", TPU_PROBE_SRC], capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired: raise SystemExit("--require-tpu: проба TPU зависла > 20 мин. Останов.")
    line = [l for l in p.stdout.splitlines() if l.startswith("TPU_PROBE ")]
    if p.returncode or not line:
        print(p.stdout[-3000:], p.stderr[-3000:], flush=True)
        raise SystemExit(f"--require-tpu: проба torch_xla упала (rc={p.returncode}). Останов.")
    r = json.loads(line[-1][len("TPU_PROBE "):])
    if not r["ok"]: raise SystemExit(f"--require-tpu: неверный результат на устройстве: {r}")
    return r
if NEED_GPU and XLA_DEV == "xla" and not IS_WORKER:
    import glob as _g
    _acc = sorted(_g.glob("/dev/accel*") + _g.glob("/dev/vfio/[0-9]*"))
    if not ARGS.allow_cpu and not _acc and os.environ.get("ISTPUVM") != "1":
        raise SystemExit("--require-tpu: нет /dev/accel*, /dev/vfio/* и ISTPUVM≠1 — сессия получила не TPU. Останов.")
    TPU_INFO = tpu_probe(); TPU_INFO["dev_nodes"] = _acc
    if not ARGS.allow_cpu and TPU_INFO["device_type"] != "TPU":
        raise SystemExit(f"--require-tpu: PJRT видит {TPU_INFO['device_type']}, не TPU. Останов.")
    log("TPU:", json.dumps(TPU_INFO, ensure_ascii=False))
