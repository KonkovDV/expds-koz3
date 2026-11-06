#!/usr/bin/env python3
"""Автономный оркестратор КОЗ №3 для обучения на кэше 59 облаков LAS (li_tin_59_t128)
на облачной GPU-ферме Kaggle (4 узла × 2 NVIDIA Tesla T4 = 8 × T4, 128 ГБ VRAM).

Особенности:
- Безопасная работа с пулом токенов /home/box/.kaggle_tokens/pool.json (значения токенов
  никогда не выводятся в stdout/stderr/логи).
- Исправлена маршрутизация токенов для kagglesdk >= 1.8: передаётся путь к файлу токена
  через переменную окружения KAGGLE_API_TOKEN=/tmp/kaggle_node_<id>/access_token.
- Полная автономность контейнера Kaggle:
  * В expds_code.tar.gz упакован пакет segmentation_models_pytorch (не требует pip install в онлайне).
  * В hf_hub_cache.tar.gz упакованы предобученные веса ImageNet для ResNet-18 и ResNet-34.
  * Прямой вызов python -m expds.train по предразбитым кэшам фолдов (index_fold0..4.csv) без
    импорта laspy/rasterio, с записью всех весов .pt напрямую в /kaggle/working/.
- Параллельное обучение на 2 × T4 (--device cuda:0 и --device cuda:1) с автоматическим
  ожиданием готовности датасета перед kernels push --accelerator NvidiaTeslaT4.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

POOL_PATH = Path("/home/box/.kaggle_tokens/pool.json")
REPO_ROOT = Path("/workspace/expds-koz3")
CACHE_PATH = Path("/workspace/expds_work/cache/li_tin_59_t128")
RUNNER_DIR = Path("/tmp/kaggle_koz3_runner")
DATASET_SLUG = "koz3-li-tin-59-t128-bundle"
KERNEL_SLUG = "koz3-li-tin-59-t128-dual-t4"


def load_pool(pool_path: Path = POOL_PATH) -> list[dict]:
    if not pool_path.exists():
        raise FileNotFoundError(f"Пул токенов не найден: {pool_path}")
    raw = json.loads(pool_path.read_text(encoding="utf-8"))
    nodes = (raw.get("nodes") or raw.get("accounts")) if isinstance(raw, dict) else raw
    out = []
    for idx, item in enumerate(nodes or []):
        out.append({
            "node_id": idx,
            "username": item.get("username", f"node_{idx}"),
            "gpus": item.get("gpus", "2x NVIDIA Tesla T4 (32 GB VRAM)"),
            "has_token": bool(item.get("token")),
            "_token": item.get("token", ""),
        })
    return out


def prepare_node_env(node: dict) -> dict[str, str]:
    """Создаёт изолированный файл токена с правами 0600 и возвращает env для Kaggle CLI."""
    nid = node["node_id"]
    token_dir = Path(f"/tmp/kaggle_node_{nid}")
    token_dir.mkdir(parents=True, exist_ok=True)
    token_file = token_dir / "access_token"
    if node.get("_token"):
        token_file.write_text(node["_token"], encoding="utf-8")
        token_file.chmod(0o600)
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(token_dir)
    env["KAGGLE_API_TOKEN"] = str(token_file)
    return env


def kaggle_bin() -> str:
    cand = Path.home() / ".local" / "bin" / "kaggle"
    return str(cand) if cand.exists() else "kaggle"


def check_node_live(node: dict) -> dict:
    env = prepare_node_env(node)
    kbin = kaggle_bin()
    res = {
        "node_id": node["node_id"],
        "username": node["username"],
        "gpus": node["gpus"],
        "has_token": node["has_token"],
        "ok": False,
        "kernels": [],
        "datasets": [],
    }
    try:
        pk = subprocess.run([kbin, "kernels", "list", "--mine", "--page-size", "10"],
                            env=env, capture_output=True, text=True, timeout=25)
        pd = subprocess.run([kbin, "datasets", "list", "--mine"],
                            env=env, capture_output=True, text=True, timeout=25)
        res["ok"] = (pk.returncode == 0)
        for ln in pk.stdout.splitlines():
            s = ln.strip()
            if not s or s.startswith("ref ") or s.startswith("---") or s == "Not found":
                continue
            ref = s.split()[0]
            ps = subprocess.run([kbin, "kernels", "status", ref],
                                env=env, capture_output=True, text=True, timeout=15)
            st_line = ps.stdout.strip() or ps.stderr.strip()
            res["kernels"].append({"ref": ref, "status": st_line})
        for ln in pd.stdout.splitlines():
            s = ln.strip()
            if not s or s.startswith("ref ") or s.startswith("---") or s == "No datasets found":
                continue
            res["datasets"].append(s.split()[0])
    except Exception as exc:
        res["error"] = str(exc)
    return res


def family_folds(families: list[str], weights: list[int], k: int) -> dict[str, int]:
    tot: dict[str, int] = {}
    for f, w in zip(families, weights):
        tot[f] = tot.get(f, 0) + w
    load = [0] * k
    fold = {}
    for f in sorted(tot, key=lambda f: (-tot[f], f)):
        i = min(range(k), key=lambda idx: load[idx])
        fold[f] = i
        load[i] += tot[f]
    return fold


def compute_fold_splits() -> list[dict]:
    """Вычисляет точное 5-фолдовое разбиение по семьям (candidates.family_folds) для li_tin_59_t128."""
    with (CACHE_PATH / "index.csv").open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fields, rows = list(rd.fieldnames or []), list(rd)
    tiles = collections.Counter(r["family"] for r in rows)
    pos = collections.Counter(r["family"] for r in rows if r["positive"] == "1")
    fams = sorted(tiles)
    fold = family_folds(fams, [pos[f] for f in fams], 5)
    splits = []
    for k in range(5):
        test = {f for f in fams if fold[f] == k}
        train_fams = [f for f in fams if f not in test]
        n_train = sum(tiles[f] for f in train_fams)
        pool = [f for f in train_fams if pos[f] > 0] or train_fams
        val = min(pool, key=lambda f: (abs(tiles[f] - 0.1 * n_train), f))
        train_rows = [r for r in rows if r["family"] not in test]
        splits.append({
            "fold": k,
            "val": val,
            "test_families": sorted(test),
            "train_tiles": len(train_rows),
            "fields": fields,
            "train_rows": train_rows,
        })
    return splits


# Точные проверочные семьи для каждого из 5 фолдов li_tin_59_t128:
# Fold 0 -> auto, Fold 1 -> auto, Fold 2 -> auto, Fold 3 -> auto, Fold 4 -> auto
FOLD_VAL_FAMILIES = {}  # computed dynamically from fold splits


def build_dual_gpu_plan() -> dict:
    """Формирует двухуровневый план обучения на 4 узлах (8 × Tesla T4, по 2 GPU на узел)."""
    # в HEAD имя splits_val_map нигде не задано → NameError в plan/stage.
    splits_val_map = {sp["fold"]: sp["val"] for sp in compute_fold_splits()}
    wave1 = [
        {
            "node_id": 0,
            "username": "dmitrijkv",
            "recipe": "li_r18_hard2_59 (Unet + resnet18, target=mask, Folds 0 & 1)",
            "gpu0_cmds": [
                "python -m expds.train --cache cache/fold0 --out /kaggle/working/li_r18_hard2_59_fold0.pt "
                "--encoder resnet18 --arch Unet --target mask --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[0]} --device {{DEV0}}"
            ],
            "gpu1_cmds": [
                "python -m expds.train --cache cache/fold1 --out /kaggle/working/li_r18_hard2_59_fold1.pt "
                "--encoder resnet18 --arch Unet --target mask --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[1]} --device {{DEV1}}"
            ],
        },
        {
            "node_id": 1,
            "username": "usertest1234",
            "recipe": "li_r18_hard2_59 (Unet + resnet18, target=mask, Folds 2 & 3)",
            "gpu0_cmds": [
                "python -m expds.train --cache cache/fold2 --out /kaggle/working/li_r18_hard2_59_fold2.pt "
                "--encoder resnet18 --arch Unet --target mask --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[2]} --device {{DEV0}}"
            ],
            "gpu1_cmds": [
                "python -m expds.train --cache cache/fold3 --out /kaggle/working/li_r18_hard2_59_fold3.pt "
                "--encoder resnet18 --arch Unet --target mask --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[3]} --device {{DEV1}}"
            ],
        },
        {
            "node_id": 2,
            "username": "aaaa0000004",
            "recipe": "li_r18_hard2_59 (Fold 4 + финальные веса на всех 59 облаках LAS)",
            "gpu0_cmds": [
                "python -m expds.train --cache cache/fold4 --out /kaggle/working/li_r18_hard2_59_fold4.pt "
                "--encoder resnet18 --arch Unet --target mask --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[4]} --device {{DEV0}}"
            ],
            "gpu1_cmds": [
                "python -m expds.train --cache cache/full --out /kaggle/working/li_r18_hard2_59_final.pt "
                "--encoder resnet18 --arch Unet --target mask --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                "--all-data --device {DEV1}"
            ],
        },
        {
            "node_id": 3,
            "username": "aaaadg4gg4g4g",
            "recipe": "High-capacity: resnet34 + FPN + --center-edge-ignore (5 folds across 2×T4)",
            "gpu0_cmds": [
                "python -m expds.train --cache cache/fold0 --out /kaggle/working/li_r34_fpn_edge_59_fold0.pt "
                "--encoder resnet34 --arch FPN --target centers --center-loss focal "
                "--center-radius 4.0 --center-sigma 0.0 --center-edge-ignore --ema 0.999 "
                f"--epochs 30 --batch 64 --lr 1.4e-3 --require-gpu --val-families {splits_val_map[0]} --device {{DEV0}}",
                "python -m expds.train --cache cache/fold1 --out /kaggle/working/li_r34_fpn_edge_59_fold1.pt "
                "--encoder resnet34 --arch FPN --target centers --center-loss focal "
                "--center-radius 4.0 --center-sigma 0.0 --center-edge-ignore --ema 0.999 "
                f"--epochs 30 --batch 64 --lr 1.4e-3 --require-gpu --val-families {splits_val_map[1]} --device {{DEV0}}",
                "python -m expds.train --cache cache/fold2 --out /kaggle/working/li_r34_fpn_edge_59_fold2.pt "
                "--encoder resnet34 --arch FPN --target centers --center-loss focal "
                "--center-radius 4.0 --center-sigma 0.0 --center-edge-ignore --ema 0.999 "
                f"--epochs 30 --batch 64 --lr 1.4e-3 --require-gpu --val-families {splits_val_map[2]} --device {{DEV0}}",
            ],
            "gpu1_cmds": [
                "python -m expds.train --cache cache/fold3 --out /kaggle/working/li_r34_fpn_edge_59_fold3.pt "
                "--encoder resnet34 --arch FPN --target centers --center-loss focal "
                "--center-radius 4.0 --center-sigma 0.0 --center-edge-ignore --ema 0.999 "
                f"--epochs 30 --batch 64 --lr 1.4e-3 --require-gpu --val-families {splits_val_map[3]} --device {{DEV1}}",
                "python -m expds.train --cache cache/fold4 --out /kaggle/working/li_r34_fpn_edge_59_fold4.pt "
                "--encoder resnet34 --arch FPN --target centers --center-loss focal "
                "--center-radius 4.0 --center-sigma 0.0 --center-edge-ignore --ema 0.999 "
                f"--epochs 30 --batch 64 --lr 1.4e-3 --require-gpu --val-families {splits_val_map[4]} --device {{DEV1}}",
            ],
        },
    ]
    wave2 = [
        {
            "node_id": 0,
            "username": "dmitrijkv",
            "recipe": "High-capacity mask rescore: resnet34 + FPN (target=mask, Folds 0 & 1)",
            "gpu0_cmds": [
                "python -m expds.train --cache cache/fold0 --out /kaggle/working/li_r34_fpn_mask_59_fold0.pt "
                "--encoder resnet34 --arch FPN --target mask --ema 0.999 --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[0]} --device {{DEV0}}"
            ],
            "gpu1_cmds": [
                "python -m expds.train --cache cache/fold1 --out /kaggle/working/li_r34_fpn_mask_59_fold1.pt "
                "--encoder resnet34 --arch FPN --target mask --ema 0.999 --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[1]} --device {{DEV1}}"
            ],
        },
        {
            "node_id": 1,
            "username": "usertest1234",
            "recipe": "High-capacity mask rescore: resnet34 + FPN (target=mask, Folds 2 & 3)",
            "gpu0_cmds": [
                "python -m expds.train --cache cache/fold2 --out /kaggle/working/li_r34_fpn_mask_59_fold2.pt "
                "--encoder resnet34 --arch FPN --target mask --ema 0.999 --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[2]} --device {{DEV0}}"
            ],
            "gpu1_cmds": [
                "python -m expds.train --cache cache/fold3 --out /kaggle/working/li_r34_fpn_mask_59_fold3.pt "
                "--encoder resnet34 --arch FPN --target mask --ema 0.999 --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[3]} --device {{DEV1}}"
            ],
        },
        {
            "node_id": 2,
            "username": "aaaa0000004",
            "recipe": "High-capacity mask rescore: resnet34 + FPN (Fold 4 + all-data final)",
            "gpu0_cmds": [
                "python -m expds.train --cache cache/fold4 --out /kaggle/working/li_r34_fpn_mask_59_fold4.pt "
                "--encoder resnet34 --arch FPN --target mask --ema 0.999 --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                f"--val-families {splits_val_map[4]} --device {{DEV0}}"
            ],
            "gpu1_cmds": [
                "python -m expds.train --cache cache/full --out /kaggle/working/li_r34_fpn_mask_59_final.pt "
                "--encoder resnet34 --arch FPN --target mask --ema 0.999 --epochs 30 --batch 64 --lr 1.4e-3 --require-gpu "
                "--all-data --device {DEV1}"
            ],
        },
    ]
    return {"wave1_parallel_8xT4": wave1, "wave2_parallel_6xT4": wave2}


WORKER_TEMPLATE = '''#!/usr/bin/env python3
"""Автономный воркер Kaggle (2 × NVIDIA Tesla T4) для обучения КОЗ №3 на кэше li_tin_59_t128.
Узел: node_{node_id} ({username})
Запускает два независимых потока задач параллельно на cuda:0 и cuda:1 (с авто-адаптацией к числу GPU)
и сохраняет все обученные чекпойнты .pt напрямую в /kaggle/working/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

NODE_ID = {node_id}
USERNAME = {username!r}
WAVE1_GPU0 = {w1_gpu0!r}
WAVE1_GPU1 = {w1_gpu1!r}
WAVE2_GPU0 = {w2_gpu0!r}
WAVE2_GPU1 = {w2_gpu1!r}


def find_input_bundle() -> Path:
    for root in (Path("/kaggle/input"), Path("/tmp/kaggle_koz3_runner/dataset_bundle")):
        if not root.exists():
            continue
        for p in [root, *root.rglob("*")]:
            if not p.is_dir():
                continue
            if (p / "expds_code.tar.gz").exists() or (p / "expds_code" / "expds").exists():
                return p
    # Диагностика содержимого /kaggle/input при ошибке
    found = [str(x) for x in Path("/kaggle/input").rglob("*")][:40] if Path("/kaggle/input").exists() else []
    raise FileNotFoundError(f"Не найден входной датасет в /kaggle/input. Найдено: {found}")


def setup_environment(bundle: Path, work_root: Path) -> Path:
    # 1. Код expds + segmentation_models_pytorch (поддержка как .tar.gz, так и распакованной папки Kaggle)
    if (bundle / "expds_code.tar.gz").exists():
        with tarfile.open(bundle / "expds_code.tar.gz", "r:gz") as tar:
            tar.extractall(path=work_root)
    elif (bundle / "expds_code").is_dir():
        shutil.copytree(bundle / "expds_code", work_root, dirs_exist_ok=True)

    # 2. Кэш весов HuggingFace Hub (ResNet-18 и ResNet-34 ImageNet)
    hf_dst = work_root / "hf_hub_cache"
    hf_dst.mkdir(parents=True, exist_ok=True)
    if (bundle / "hf_hub_cache").is_dir():
        hf_src = bundle / "hf_hub_cache"
        specs = [
            (
                "models--smp-hub--resnet18.imagenet",
                "3f2325ff978283d47aa6a1d6878ca20565622683",
                hf_src / "models--smp-hub--resnet18.imagenet/blobs/879c59b3870194ae248b52920479fed460415e62",
                hf_src / "blobs/d5/d5da100346311aff6748ca74f737a7e9178dae9e48b071f61a8ccf9fa3ad88d7",
            ),
            (
                "models--smp-hub--resnet34.imagenet",
                "7a57b34f723329ff020b3f8bc41771163c519d0c",
                hf_src / "models--smp-hub--resnet34.imagenet/blobs/879c59b3870194ae248b52920479fed460415e62",
                hf_src / "blobs/bc/bcb6928a067cda2ccbcf44c2c810cf759f69ff578332b4ac57ff2a6eb3984b13",
            ),
        ]
        for repo_dir, snap_hash, cfg_src, weight_src in specs:
            if cfg_src.exists() and weight_src.exists() and weight_src.stat().st_size > 1000:
                snap_dir = hf_dst / repo_dir / "snapshots" / snap_hash
                refs_dir = hf_dst / repo_dir / "refs"
                snap_dir.mkdir(parents=True, exist_ok=True)
                refs_dir.mkdir(parents=True, exist_ok=True)
                (refs_dir / "main").write_text(snap_hash, encoding="utf-8")
                if not (snap_dir / "config.json").exists():
                    os.symlink(cfg_src.resolve(), snap_dir / "config.json")
                if not (snap_dir / "model.safetensors").exists():
                    os.symlink(weight_src.resolve(), snap_dir / "model.safetensors")
    elif (bundle / "hf_hub_cache.tar.gz").exists():
        with tarfile.open(bundle / "hf_hub_cache.tar.gz", "r:gz") as tar:
            tar.extractall(path=hf_dst)

    os.environ["HF_HUB_CACHE"] = str(hf_dst)
    os.environ["HF_HUB_OFFLINE"] = "1"

    cache_src = bundle / "li_tin_59_t128" if (bundle / "li_tin_59_t128").exists() else bundle
    cache_root = work_root / "cache"

    # Полный кэш (59 облаков)
    full_dir = cache_root / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    for item in cache_src.iterdir():
        if item.name in ("meta.json", "index.csv") or item.suffix == ".npy":
            dst = full_dir / item.name
            if not dst.exists():
                os.symlink(item.resolve(), dst)

    # Кэши 5 фолдов (index_fold0..4.csv)
    for k in range(5):
        fdir = cache_root / f"fold{k}"
        fdir.mkdir(parents=True, exist_ok=True)
        for item in cache_src.iterdir():
            if item.name == "meta.json" or item.suffix == ".npy":
                dst = fdir / item.name
                if not dst.exists():
                    os.symlink(item.resolve(), dst)
        fold_idx = cache_src / f"index_fold{k}.csv"
        dst_idx = fdir / "index.csv"
        if fold_idx.exists() and not dst_idx.exists():
            shutil.copy2(fold_idx, dst_idx)

    return cache_root


def run_cmd_sequence(cmds: list[str], dev: str, work_root: Path, out_dir: Path, wave_name: str, slot: int, results: dict) -> None:
    slot_res = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_root)
    env["HF_HUB_CACHE"] = str(work_root / "hf_hub_cache")
    env["HF_HUB_OFFLINE"] = "1"
    env["EXPDS_REQUIRE_GPU"] = "1"
    for idx, raw_cmd in enumerate(cmds):
        cmd_str = raw_cmd.format(DEV0=dev, DEV1=dev).replace("/kaggle/working", str(out_dir))
        cmd = [sys.executable if tok == "python" else tok for tok in cmd_str.split()]
        log_path = out_dir / f"{wave_name}_slot{slot}_step{idx}.log"
        t0 = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, cwd=str(work_root), env=env, stdout=lf, stderr=subprocess.STDOUT)
        slot_res.append({
            "step": idx,
            "cmd": " ".join(cmd),
            "rc": p.returncode,
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "log": log_path.name,
        })
    results[f"slot{slot}"] = slot_res


def run_wave(work_root: Path, out_dir: Path, cmds0: list[str], cmds1: list[str], wave_name: str, n_gpus: int) -> dict:
    dev0 = "cuda:0" if n_gpus >= 1 else "cpu"
    dev1 = "cuda:1" if n_gpus >= 2 else ("cuda:0" if n_gpus == 1 else "cpu")
    results: dict = {"dev0": dev0, "dev1": dev1}
    t0 = time.perf_counter()

    if n_gpus >= 1:
        threads = []
        if cmds0:
            t = threading.Thread(target=run_cmd_sequence, args=(cmds0, dev0, work_root, out_dir, wave_name, 0, results))
            t.start()
            threads.append(t)
        if cmds1:
            t = threading.Thread(target=run_cmd_sequence, args=(cmds1, dev1, work_root, out_dir, wave_name, 1, results))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    else:
        if cmds0:
            run_cmd_sequence(cmds0, dev0, work_root, out_dir, wave_name, 0, results)
        if cmds1:
            run_cmd_sequence(cmds1, dev1, work_root, out_dir, wave_name, 1, results)

    results["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return results


def main() -> None:
    import torch

    n_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    print(json.dumps({"node_id": NODE_ID, "username": USERNAME, "gpus": n_gpus, "gpu_names": gpu_names}), flush=True)
    if n_gpus < 1 and os.environ.get("KOZ3_ALLOW_CPU_FALLBACK", "0") != "1":
        raise RuntimeError(
            f"На узле node_{NODE_ID} ({USERNAME}) не выделен GPU (detected_gpus=0)! "
            "Отказ от тихого отката на CPU (правило 1 GPU-сессия на аккаунт)."
        )

    bundle = find_input_bundle()
    work_root = Path(f"/tmp/koz3_run_node_{NODE_ID}")
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else (work_root / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_environment(bundle, work_root)

    summary = {
        "node_id": NODE_ID,
        "username": USERNAME,
        "detected_gpus": n_gpus,
        "gpu_names": gpu_names,
        "wave1": run_wave(work_root, out_dir, WAVE1_GPU0, WAVE1_GPU1, "wave1", n_gpus),
    }
    out_summary = out_dir / f"summary_node_{NODE_ID}.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Запускаем Волну 2 только если доступны GPU и остаётся запас времени сессии
    if n_gpus >= 1 and os.environ.get("KOZ3_RUN_WAVE2", "1") == "1" and (WAVE2_GPU0 or WAVE2_GPU1):
        summary["wave2"] = run_wave(work_root, out_dir, WAVE2_GPU0, WAVE2_GPU1, "wave2", n_gpus)
        out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    saved_pts = sorted(p.name for p in out_dir.glob("*.pt"))
    summary["saved_checkpoints"] = saved_pts
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


def stage_bundles(nodes: list[dict]) -> dict:
    """Собирает dataset_bundle (expds_code.tar.gz + hf_hub_cache.tar.gz + кэш li_tin_59_t128 + фолды) и 4 папки ядер."""
    ds_dir = RUNNER_DIR / "dataset_bundle"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Архив кода expds + чистый Python-пакет segmentation_models_pytorch
    code_tar = ds_dir / "expds_code.tar.gz"
    smp_site = Path("/tmp/expds-venv/lib/python3.12/site-packages/segmentation_models_pytorch")
    with tarfile.open(code_tar, "w:gz") as tar:
        for rel in ("expds", "scripts", "configs", "pyproject.toml", "requirements.txt"):
            src = REPO_ROOT / rel
            if src.exists():
                tar.add(src, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name else ti)
        if smp_site.exists():
            tar.add(smp_site, arcname="segmentation_models_pytorch",
                    filter=lambda ti: None if "__pycache__" in ti.name else ti)

    # 2. Архив кэша весов ImageNet (ResNet-18 и ResNet-34) для мгновенного старта без сети
    hf_tar = ds_dir / "hf_hub_cache.tar.gz"
    hf_src = Path.home() / ".cache" / "huggingface" / "hub"
    with tarfile.open(hf_tar, "w:gz") as tar:
        for model_dir in ("models--smp-hub--resnet18.imagenet", "models--smp-hub--resnet34.imagenet", "blobs"):
            p = hf_src / model_dir
            if p.exists():
                tar.add(p, arcname=model_dir)

    # 3. Жёсткие ссылки на кэш li_tin_59_t128 + предразбитые index_fold0..4.csv
    cache_link_dir = ds_dir / "li_tin_59_t128"
    cache_link_dir.mkdir(parents=True, exist_ok=True)
    linked_files = []
    for f in sorted(CACHE_PATH.iterdir()):
        dst = cache_link_dir / f.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(f, dst)
        except OSError:
            os.symlink(f.resolve(), dst)
        linked_files.append(f.name)

    splits = compute_fold_splits()
    for sp in splits:
        fk = sp["fold"]
        f_csv = cache_link_dir / f"index_fold{fk}.csv"
        with f_csv.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=sp["fields"])
            wr.writeheader()
            wr.writerows(sp["train_rows"])
        linked_files.append(f_csv.name)

    # 4. Сборка 4 папок ядер Kaggle
    plan = build_dual_gpu_plan()
    w1_by_node = {x["node_id"]: x for x in plan["wave1_parallel_8xT4"]}
    w2_by_node = {x["node_id"]: x for x in plan["wave2_parallel_6xT4"]}

    default_owner = nodes[0]["username"] if nodes else "dmitrijkv"
    ds_meta = {
        "title": "KOZ3 Li TIN 59 T128 Cache and Code Bundle",
        "id": f"{default_owner}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    (ds_dir / "dataset-metadata.json").write_text(json.dumps(ds_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    kernels_root = RUNNER_DIR / "kernels"
    kernels_root.mkdir(parents=True, exist_ok=True)
    staged_kernels = []

    for node in nodes:
        nid = node["node_id"]
        uname = node["username"]
        kdir = kernels_root / f"node_{nid}_{uname}"
        kdir.mkdir(parents=True, exist_ok=True)

        w1 = w1_by_node.get(nid, {})
        w2 = w2_by_node.get(nid, {})
        worker_code = (
            WORKER_TEMPLATE
            .replace("{node_id}", str(nid))
            .replace("{username!r}", repr(uname))
            .replace("{w1_gpu0!r}", repr(w1.get("gpu0_cmds", [])))
            .replace("{w1_gpu1!r}", repr(w1.get("gpu1_cmds", [])))
            .replace("{w2_gpu0!r}", repr(w2.get("gpu0_cmds", [])))
            .replace("{w2_gpu1!r}", repr(w2.get("gpu1_cmds", [])))
        )
        worker_file = kdir / "koz3_dual_t4_worker.py"
        worker_file.write_text(worker_code, encoding="utf-8")
        worker_file.chmod(0o755)

        # Важно: title обязан совпадать со слагом KERNEL_SLUG, чтобы Kaggle не менял URL ядра!
        kmeta = {
            "id": f"{uname}/{KERNEL_SLUG}",
            "title": KERNEL_SLUG,
            "code_file": "koz3_dual_t4_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "machine_shape": "NvidiaTeslaT4",
            "enable_tpu": "false",
            "enable_internet": "false",
            "dataset_sources": [f"{uname}/{DATASET_SLUG}"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (kdir / "kernel-metadata.json").write_text(json.dumps(kmeta, ensure_ascii=False, indent=2), encoding="utf-8")

        node_ds_meta = {
            "title": f"KOZ3 Li TIN 59 T128 Bundle Node {nid}",
            "id": f"{uname}/{DATASET_SLUG}",
            "licenses": [{"name": "other"}],
        }
        (kdir / "dataset-metadata.node.json").write_text(
            json.dumps(node_ds_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staged_kernels.append({"node_id": nid, "username": uname, "kernel_dir": str(kdir)})

    return {
        "dataset_bundle_dir": str(ds_dir),
        "code_tar_bytes": code_tar.stat().st_size,
        "hf_tar_bytes": hf_tar.stat().st_size,
        "cache_files": linked_files,
        "kernels": staged_kernels,
    }


KNOWN_GPU_SLUGS = (
    KERNEL_SLUG,
    "koz3-gpu-s1",
    "koz3-gpu-s2",
    "koz3-gpu-s3",
)


def get_running_kernels_on_account(kbin: str, env: dict, uname: str, exclude_slug: str | None = None) -> list[str]:
    """Проверяет, нет ли уже запущенной GPU-сессии на аккаунте (Kaggle даёт только 1 GPU-сессию на аккаунт).

    При одновременном пуше второго ядра на тот же аккаунт Kaggle тихо сбрасывает второе ядро на CPU.
    """
    running = []
    for slug in KNOWN_GPU_SLUGS:
        if exclude_slug and slug == exclude_slug:
            continue
        ref = f"{uname}/{slug}"
        try:
            p = subprocess.run([kbin, "kernels", "status", ref],
                               env=env, capture_output=True, text=True, timeout=15)
            st = (p.stdout + " " + p.stderr).lower()
            if "running" in st or "queued" in st:
                running.append(ref)
        except Exception:
            continue
    return running


def wait_for_dataset_ready(kbin: str, env: dict, dataset_ref: str, max_wait_sec: int = 120) -> str:
    """Ожидает, пока созданный приватный датасет Kaggle перейдёт в статус ready."""
    t0 = time.time()
    last_st = ""
    while time.time() - t0 < max_wait_sec:
        p = subprocess.run([kbin, "datasets", "status", dataset_ref],
                           env=env, capture_output=True, text=True, timeout=20)
        st = (p.stdout.strip() or p.stderr.strip()).lower()
        last_st = st
        if "ready" in st:
            return st
        time.sleep(5)
    return last_st


def push_to_kaggle(nodes: list[dict], target_nodes: list[int] | None, push_ds: bool, push_kn: bool) -> list[dict]:
    kbin = kaggle_bin()
    ds_dir = RUNNER_DIR / "dataset_bundle"
    results = []
    for node in nodes:
        nid = node["node_id"]
        if target_nodes is not None and nid not in target_nodes:
            continue
        uname = node["username"]
        env = prepare_node_env(node)
        kdir = RUNNER_DIR / "kernels" / f"node_{nid}_{uname}"
        ds_ref = f"{uname}/{DATASET_SLUG}"
        node_res: dict = {"node_id": nid, "username": uname, "dataset_ref": ds_ref, "kernel_ref": f"{uname}/{KERNEL_SLUG}"}

        if push_ds:
            shutil.copy2(kdir / "dataset-metadata.node.json", ds_dir / "dataset-metadata.json")
            # Проверим, существует ли уже датасет на этом аккаунте
            p_st = subprocess.run([kbin, "datasets", "status", ds_ref],
                                  env=env, capture_output=True, text=True, timeout=20)
            if p_st.returncode == 0 and "ready" in (p_st.stdout + p_st.stderr).lower():
                node_res["dataset_push"] = {"rc": 0, "status": "already_exists_and_ready", "stdout": p_st.stdout.strip()}
            else:
                p = subprocess.run([kbin, "datasets", "create", "-p", str(ds_dir), "--dir-mode", "tar"],
                                   env=env, capture_output=True, text=True, timeout=600)
                node_res["dataset_push"] = {"rc": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}

        if push_kn:
            # Защита от одновременного запуска двух ядер на одном аккаунте (1 GPU-сессия на аккаунт)
            busy = get_running_kernels_on_account(kbin, env, uname, exclude_slug=KERNEL_SLUG)
            if busy and os.environ.get("KOZ3_FORCE_CONCURRENT_PUSH", "0") != "1":
                node_res["kernel_push"] = {
                    "rc": 1,
                    "skipped": True,
                    "reason": f"На аккаунте {uname} уже выполняется ядро {busy}; параллельный пуш второго ядра приведёт к откату на CPU.",
                }
                results.append(node_res)
                continue
            # Дождёмся готовности датасета и при необходимости повторим kernels push с ретраями
            ds_state = wait_for_dataset_ready(kbin, env, ds_ref, max_wait_sec=90)
            node_res["dataset_ready_status"] = ds_state
            for attempt in range(1, 7):
                p = subprocess.run(
                    [kbin, "kernels", "push", "-p", str(kdir), "--accelerator", "NvidiaTeslaT4"],
                    env=env, capture_output=True, text=True, timeout=120,
                )
                out_all = (p.stdout + "\n" + p.stderr).strip()
                node_res["kernel_push"] = {
                    "attempt": attempt,
                    "rc": p.returncode,
                    "stdout": p.stdout.strip(),
                    "stderr": p.stderr.strip(),
                }
                if p.returncode == 0 and "error" not in out_all.lower():
                    break
                time.sleep(15)
        results.append(node_res)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="проверить живой статус всех 4 узлов Kaggle без раскрытия токенов")
    sub.add_parser("plan", help="вывести план параллельного обучения (2×T4 на узел) для рецептов A и B")
    sub.add_parser("stage", help="собрать dataset_bundle и 4 директории ядер в /tmp/kaggle_koz3_runner/")

    ps = sub.add_parser("push", help="отправить датасет и/или ядра в Kaggle API")
    ps.add_argument("--nodes", nargs="*", type=int, default=None, help="номера узлов (по умолчанию все 0..3)")
    ps.add_argument("--datasets", action="store_true", help="создать приватный датасет с кэшем и кодом")
    ps.add_argument("--kernels", action="store_true", help="запустить ядра обучения на 2×T4")

    args = ap.parse_args()
    nodes = load_pool()

    if args.cmd == "status":
        res = [check_node_live(n) for n in nodes]
        print(json.dumps({"total_nodes": len(res), "total_t4_gpus": len(res) * 2, "nodes": res},
                         ensure_ascii=False, indent=2))
    elif args.cmd == "plan":
        print(json.dumps(build_dual_gpu_plan(), ensure_ascii=False, indent=2))
    elif args.cmd == "stage":
        print(json.dumps(stage_bundles(nodes), ensure_ascii=False, indent=2))
    elif args.cmd == "push":
        print(json.dumps(push_to_kaggle(nodes, args.nodes, args.datasets, args.kernels),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
