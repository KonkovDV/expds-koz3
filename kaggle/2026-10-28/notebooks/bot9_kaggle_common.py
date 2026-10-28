"""Общие функции ноутбуков bot-9 для Kaggle 2×T4. Ничего не запускает при импорте.

Защита от отката на CPU: gpu_guard() падает, если torch не видит CUDA или GPU меньше, чем нужно,
и проверяет, что тензор реально создаётся на каждом GPU. Каждая команда обучения получает
--require-gpu и EXPDS_REQUIRE_GPU=1; патченный expds.train сам проверяет модель и первый батч.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
PATCH_MARKERS = ("--center-pre-flip", "--ema-warmup", "--select-min-epoch", "--min-gpus")


def gpu_guard(min_gpus: int = 2) -> list[str]:
    import torch

    if os.environ.get("KOZ3_ALLOW_CPU_FALLBACK"):
        raise RuntimeError("KOZ3_ALLOW_CPU_FALLBACK задан — в этих ноутбуках запрещён")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() == False: ядро без GPU (Accelerator → GPU T4 x2). "
                           "Второе одновременное ядро аккаунта Kaggle молча получает CPU.")
    n = torch.cuda.device_count()
    if n < min_gpus:
        raise RuntimeError(f"Видно GPU: {n}, нужно ≥ {min_gpus}. Остановка, без отката на CPU.")
    names = []
    for i in range(n):
        t = torch.ones(8, device=f"cuda:{i}") * 2
        if not t.is_cuda or t.device.index != i or float(t.sum()) != 16.0:
            raise RuntimeError(f"Проверочный тензор не на cuda:{i}: {t.device}")
        names.append(torch.cuda.get_device_name(i))
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if out.returncode != 0 or out.stdout.count("GPU ") < min_gpus:
        raise RuntimeError(f"nvidia-smi не подтвердил {min_gpus} GPU: {out.stdout} {out.stderr}")
    os.environ["EXPDS_REQUIRE_GPU"] = "1"
    print("GPU OK:", names, flush=True)
    return names


def find_bundle() -> Path:
    root = Path("/kaggle/input")
    for p in [root, *root.rglob("*")]:
        if p.is_dir() and ((p / "expds_code.tar.gz").exists() or (p / "expds_code" / "expds").exists()):
            return p
    raise FileNotFoundError("в /kaggle/input нет expds_code(.tar.gz) — подключите датасет-бандл")


def setup_code(bundle: Path, work: Path = WORK) -> Path:
    if (bundle / "expds_code.tar.gz").exists():
        with tarfile.open(bundle / "expds_code.tar.gz", "r:gz") as tar:
            tar.extractall(path=work)
    else:
        shutil.copytree(bundle / "expds_code", work, dirs_exist_ok=True)
    src = (work / "expds" / "train.py").read_text(encoding="utf-8")
    miss = [m for m in PATCH_MARKERS if m not in src]
    if miss:
        raise RuntimeError(f"В бандле старый expds/train.py без патча bot-9 (нет {miss}). "
                           "Пересоберите expds_code.tar.gz из ветки bot9-kaggle-queue.")
    hf = bundle / "hf_hub_cache"
    if hf.is_dir():
        os.environ["HF_HUB_CACHE"] = str(hf)
        os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(work))
    return work


def setup_fold_caches(bundle: Path, folds_cfg: dict, work: Path = WORK) -> Path:
    """Как worker оркестратора, плюс проверка leave-family-out: тестовых семей фолда нет в его кэше."""
    src = bundle / "li_tin_59_t128" if (bundle / "li_tin_59_t128").exists() else bundle
    root = work / "cache"
    for name in ["full"] + [f"fold{k}" for k in range(5)]:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        for it in src.iterdir():
            if it.name == "meta.json" or it.suffix == ".npy" or (name == "full" and it.name == "index.csv"):
                if not (d / it.name).exists():
                    os.symlink(it.resolve(), d / it.name)
        if name != "full" and not (d / "index.csv").exists():
            shutil.copy2(src / f"index_{name}.csv", d / "index.csv")
    for f in folds_cfg["folds"]:
        k = f["fold"]
        with (root / f"fold{k}" / "index.csv").open(encoding="utf-8") as fh:
            fams = {r["family"] for r in csv.DictReader(fh)}
        leak = fams & set(f["test_families"])
        if leak:
            raise RuntimeError(f"fold{k}: тестовые семьи в обучении: {sorted(leak)}")
        if f["val_family"] not in fams:
            raise RuntimeError(f"fold{k}: семьи проверки {f['val_family']} нет в кэше")
    print("LFO OK: тестовые семьи отсутствуют в кэшах фолдов", flush=True)
    return root


def run_queue(cmds_by_gpu: dict[int, list[str]], log_dir: Path, work: Path = WORK) -> list[dict]:
    """Команды по очередям GPU параллельно; каждая — с --require-gpu и --device cuda:i."""
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EXPDS_REQUIRE_GPU="1", OMP_NUM_THREADS="2", PYTHONPATH=str(work))
    results: list[dict] = []
    lock = threading.Lock()

    for gpu, cmds in cmds_by_gpu.items():  # проверка до старта: исключение в потоке не всплывает
        for c in cmds:
            if "--require-gpu" not in c or f"--device cuda:{gpu}" not in c:
                raise RuntimeError(f"команда без --require-gpu/--device cuda:{gpu}: {c}")
    errors: list[str] = []

    def worker(gpu: int, cmds: list[str]):
        try:
            _work(gpu, cmds)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cuda:{gpu}: {exc!r}")

    def _work(gpu: int, cmds: list[str]):
        for j, c in enumerate(cmds):
            t0 = time.time()
            lf = log_dir / f"gpu{gpu}_step{j}.log"
            with lf.open("w", encoding="utf-8") as fh:
                rc = subprocess.run(c, shell=True, cwd=work, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            r = {"gpu": gpu, "step": j, "cmd": c, "rc": rc, "elapsed_sec": round(time.time() - t0, 1), "log": str(lf)}
            with lock:
                results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            if rc != 0:
                print(lf.read_text(encoding="utf-8")[-3000:], flush=True)

    th = [threading.Thread(target=worker, args=(g, c)) for g, c in cmds_by_gpu.items()]
    [t.start() for t in th]
    [t.join() for t in th]
    (log_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [r for r in results if r["rc"] != 0]
    if errors:
        raise RuntimeError("; ".join(errors))
    if bad:
        raise RuntimeError(f"упало команд: {len(bad)}")
    return results


def check_logs_gpu(log_dir: Path) -> None:
    """Пост-проверка: в каждом логе обучения устройство cuda, а не cpu."""
    for lf in sorted(log_dir.glob("gpu*_step*.log")):
        t = lf.read_text(encoding="utf-8", errors="replace")
        if '"device": "cpu"' in t or "устройство cpu" in t:
            raise RuntimeError(f"{lf}: обучение шло на CPU")
