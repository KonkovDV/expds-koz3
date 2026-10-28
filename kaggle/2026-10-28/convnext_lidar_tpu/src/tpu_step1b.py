# ── Шаг 1б (TPU). Устройства: процесс на чип (v5e: 1 чип = 1 TensorCore) ─────────────────────────────────────────
XLA_ON = XLA_DEV == "xla"
if XLA_DEV not in ("xla", "cpu") or (XLA_DEV == "cpu" and not ARGS.allow_cpu):
    raise SystemExit(f"CNX_TPU_DEV={XLA_DEV}: допустимо xla (TPU) или cpu (только с --allow-cpu на боксе)")
N_CHIPS = 1 if IS_WORKER else int(TPU_INFO.get("n_dev", 1) or 1)
NPROC = 1 if IS_WORKER else max(1, E("CNX_TPU_PROCS", CNX_TPU_PROCS_DEFAULT))
if XLA_ON and not ARGS.allow_cpu and not IS_WORKER: NPROC = min(NPROC, N_CHIPS)
def _mkdev():
    if not XLA_ON: return torch.device("cpu")
    import torch_xla, torch_xla.runtime as xr
    d = torch_xla.device()
    if not ARGS.allow_cpu and xr.device_type() != "TPU": raise SystemExit(f"--require-tpu: устройство {xr.device_type()}, не TPU. Останов.")
    nd = len(torch_xla.devices())
    if IS_WORKER and nd != 1: log(f"ВНИМАНИЕ: воркер чипа {TPU_CHIP} видит {nd} устройств (ожидалось 1: TPU_VISIBLE_CHIPS не применился?)")
    (torch.ones(8, device=d) * 2).sum().cpu()
    return d
DEV = _mkdev() if NEED_GPU and (IS_WORKER or NPROC == 1) else None
DEVS = [DEV] if DEV is not None else [f"tpu_chip{i}" for i in range(NPROC)]
log("TPU-процессы:", NPROC, "чипов (проба):", N_CHIPS, "роль:", f"воркер чипа {TPU_CHIP}" if IS_WORKER else "главный", "устройство:", DEV)
