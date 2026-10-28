# ── Ячейка 1. Конфиг, пути, защита от CPU ────────────────────────────────────────────────
import os, sys, json, time, glob, csv, math, shutil, tarfile, threading, subprocess, re
from pathlib import Path
import numpy as np

CFG = json.loads(r'''__CFG__''')

# Локальный сухой прогон (только вне Kaggle, только по явной переменной): CPU, 1 батч, без предобучения.
LOCAL_TEST = os.environ.get("KOZ3_BOT9_LOCAL_TEST") == "1" and not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
INPUT_DIR = Path(os.environ.get("KOZ3_BOT9_INPUT", CFG["input_dir"])) if LOCAL_TEST else Path(CFG["input_dir"])
CODE_DIR = Path(os.environ.get("KOZ3_BOT9_CODE", CFG["code_bundle_dir"])) if LOCAL_TEST else Path(CFG["code_bundle_dir"])
WORK = Path(os.environ.get("KOZ3_BOT9_WORK", "/kaggle/working")) if LOCAL_TEST else Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
if LOCAL_TEST:
    CFG.update(CFG["local_test_overrides"])
    print("ЛОКАЛЬНЫЙ СУХОЙ ПРОГОН: CPU, 1 батч, без предобучения — результаты не значимы", flush=True)

import torch

def gpu_guard(min_gpus):
    if os.environ.get("KOZ3_ALLOW_CPU_FALLBACK"):
        raise RuntimeError("KOZ3_ALLOW_CPU_FALLBACK задан — откат на CPU запрещён")
    if not torch.cuda.is_available():
        raise RuntimeError("--require-gpu: torch.cuda.is_available() == False. Ядро запущено без GPU "
                           "(Accelerator → GPU T4 x2; второе одновременное GPU-ядро аккаунта молча получает CPU). Остановка.")
    n = torch.cuda.device_count()
    if n < min_gpus:
        raise RuntimeError(f"--require-gpu: видно GPU {n}, нужно ≥ {min_gpus}. Остановка, без отката на CPU.")
    names = []
    for i in range(n):
        t = torch.ones(8, device=f"cuda:{i}") * 2
        if t.device.type != "cuda" or t.device.index != i or float(t.sum()) != 16.0:
            raise RuntimeError(f"проверочный тензор не на cuda:{i}: {t.device}")
        names.append(torch.cuda.get_device_name(i))
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if smi.returncode != 0 or smi.stdout.count("GPU ") < min_gpus:
        raise RuntimeError(f"nvidia-smi не подтвердил {min_gpus} GPU: {smi.stdout} {smi.stderr}")
    print("GPU OK:", names, flush=True)
    return [torch.device(f"cuda:{i}") for i in range(min_gpus)]

if CFG["require_gpu"] and not LOCAL_TEST:
    DEVICES = gpu_guard(CFG["min_gpus"])
else:
    DEVICES = [torch.device("cpu")] * CFG["min_gpus"]
torch.backends.cudnn.benchmark = True
