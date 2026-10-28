#!/usr/bin/env python3
"""Оркестратор облачной фермы GPU (4 аккаунта Kaggle = 8 × Tesla T4) и роя CPU-воркеров.

Читает пул токенов из ~/.kaggle_tokens/pool.json (или переменной окружения KAGGLE_POOL_JSON),
формирует шардированный план обучения 5 фолдов + --all-data по 4 независимым узлам Kaggle
(по 2 × Tesla T4 на каждый узел, 16 ГБ VRAM на карту) без участия домашнего ПК.

Примеры:
    python scripts/cloud_gpu_farm.py status
    python scripts/cloud_gpu_farm.py plan --cache work/cache/li_tin_59_t128 --markup li --epochs 25 --batch 64
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_POOL_PATH = Path.home() / ".kaggle_tokens" / "pool.json"


def load_pool(pool_path: Path = DEFAULT_POOL_PATH) -> list[dict]:
    """Возвращает метаданные узлов фермы без раскрытия секретных токенов в консоль."""
    if not pool_path.exists():
        return []
    raw = json.loads(pool_path.read_text(encoding="utf-8"))
    nodes = (raw.get("nodes") or raw.get("accounts")) if isinstance(raw, dict) else raw
    out = []
    for idx, item in enumerate(nodes or []):
        out.append({
            "node_id": idx,
            "username": item.get("username", f"node_{idx}"),
            "gpus": item.get("gpus", "2x Tesla T4 (32 GB VRAM)"),
            "has_token": bool(item.get("token")),
            "_token": item.get("token", ""),
        })
    return out


def build_fold_plan(cache: str, markup: str, work: str, out: str, epochs: int = 25,
                    batch: int = 64, encoder: str = "resnet18", target: str = "mask",
                    n_nodes: int = 4) -> list[dict]:
    """Распределяет 5 фолдов OOF + финальное обучение --all-data по 4 узлам фермы (2×T4 на узел)."""
    assignments = [
        {"node_id": 0, "role": "OOF folds 0 & 1 (GPU0: fold0, GPU1: fold1)", "folds": [0, 1], "all_data": False},
        {"node_id": 1, "role": "OOF folds 2 & 3 (GPU0: fold2, GPU1: fold3)", "folds": [2, 3], "all_data": False},
        {"node_id": 2, "role": "OOF fold 4 + seed2 ensemble", "folds": [4], "all_data": False},
        {"node_id": 3, "role": "Full-data submission weights (--all-data)", "folds": [], "all_data": True},
    ]
    plan = []
    for a in assignments[:max(1, n_nodes)]:
        if a["all_data"]:
            cmd = (f"python -m expds.train --cache {cache} --out {work}/{markup}_all.pt "
                   f"--encoder {encoder} --epochs {epochs} --batch {batch} --target {target} --all-data --device cuda:0 & "
                   f"python -m expds.train --cache {cache} --out {work}/{markup}_all_s1.pt "
                   f"--encoder {encoder} --epochs {epochs} --batch {batch} --target {target} --all-data --seed 1 --device cuda:1 & wait")
        else:
            f_str = " ".join(str(f) for f in a["folds"])
            p_flag = " --parallel-gpus" if len(a["folds"]) > 1 else " --device cuda:0"
            cmd = (f"python scripts/nn_oof.py --markup {markup} --root . --cache {cache} "
                   f"--work {work} --out {out} --fold-indices {f_str} --train-only{p_flag} "
                   f"--encoder {encoder} --epochs {epochs} --batch {batch} --target {target}")
        plan.append({
            "node_id": a["node_id"],
            "role": a["role"],
            "folds": a["folds"],
            "all_data": a["all_data"],
            "command": cmd,
        })
    return plan


def check_node(node: dict) -> dict:
    """Проверяет доступность Kaggle API для узла без печати токена."""
    env = os.environ.copy()
    tok = node.get("_token", "")
    if not tok:
        return {"node_id": node["node_id"], "username": node["username"], "ok": False, "error": "no_token"}
    token_dir = Path(f"/tmp/kaggle_node_{node['node_id']}")
    token_dir.mkdir(parents=True, exist_ok=True)
    token_file = token_dir / "access_token"
    token_file.write_text(tok, encoding="utf-8")
    token_file.chmod(0o600)
    env["KAGGLE_CONFIG_DIR"] = str(token_dir)
    env["KAGGLE_API_TOKEN"] = str(token_file)
    kaggle_bin = Path.home() / ".local" / "bin" / "kaggle"
    cmd = [str(kaggle_bin) if kaggle_bin.exists() else "kaggle", "kernels", "list", "--mine", "--page-size", "5"]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=15)
        ok = proc.returncode == 0
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        return {
            "node_id": node["node_id"],
            "username": node["username"],
            "gpus": node["gpus"],
            "ok": ok,
            "kernels_count": max(0, len(lines) - 2) if ok else 0,
        }
    except Exception as exc:
        return {"node_id": node["node_id"], "username": node["username"], "ok": False, "error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="проверить статус всех узлов облачной GPU-фермы")
    st.add_argument("--pool", default=str(DEFAULT_POOL_PATH))
    st.add_argument("--live", action="store_true", help="выполнить живой запрос списка ядер Kaggle по каждому узлу")

    pl = sub.add_parser("plan", help="сгенерировать план параллельного обучения 5 фолдов + all-data")
    pl.add_argument("--cache", required=True)
    pl.add_argument("--markup", default="li")
    pl.add_argument("--work", default="work/oof_run")
    pl.add_argument("--out", default="work/oof_run.geojson")
    pl.add_argument("--epochs", type=int, default=25)
    pl.add_argument("--batch", type=int, default=64)
    pl.add_argument("--encoder", default="resnet18")
    pl.add_argument("--target", choices=("mask", "centers"), default="mask")

    args = ap.parse_args()
    if args.cmd == "status":
        nodes = load_pool(Path(args.pool))
        if args.live:
            res = [check_node(n) for n in nodes]
        else:
            res = [{k: v for k, v in n.items() if not k.startswith("_")} for n in nodes]
        print(json.dumps({"total_nodes": len(res), "total_t4_gpus": len(res) * 2, "nodes": res},
                         ensure_ascii=False, indent=2))
    elif args.cmd == "plan":
        plan = build_fold_plan(args.cache, args.markup, args.work, args.out,
                               epochs=args.epochs, batch=args.batch,
                               encoder=args.encoder, target=args.target)
        print(json.dumps({"nodes": len(plan), "plan": plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
