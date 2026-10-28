#!/usr/bin/env python3
"""Фоновый демон-монитор и сборщик весов с 4 узлов Kaggle (8 × Tesla T4)
в /workspace/expds_work/models_59/.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/tmp/kaggle_koz3_runner")
import orchestrate_kaggle_koz3 as ok

OUT_ROOT = Path("/workspace/expds_work/models_59")
STATUS_FILE = OUT_ROOT / "collector_status.json"
LOG_FILE = OUT_ROOT / "collector_daemon.log"


def log_msg(msg: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def organize_checkpoints(raw_dir: Path) -> list[str]:
    copied = []
    for pt in sorted(raw_dir.glob("*.pt")):
        dst_flat = OUT_ROOT / pt.name
        shutil.copy2(pt, dst_flat)
        copied.append(str(dst_flat))
        # Также раскладываем по папкам fold{k}/li.pt для совместимости с nn_oof.py --resume
        m = re.match(r"^(li_r18_hard2_59|li_r34_fpn_edge_59|li_r34_fpn_mask_59)_fold(\d)\.pt$", pt.name)
        if m:
            exp_name, fk = m.group(1), m.group(2)
            fold_dir = OUT_ROOT / exp_name / f"fold{fk}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pt, fold_dir / "li.pt")
    for meta_f in sorted(list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.log"))):
        shutil.copy2(meta_f, OUT_ROOT / meta_f.name)
    return copied


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    nodes = ok.load_pool()
    kbin = ok.kaggle_bin()
    completed_nodes: set[int] = set()
    state: dict = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "target_dir": str(OUT_ROOT),
        "nodes": {},
        "downloaded_models": [],
    }
    log_msg(f"Старт демона-сборщика для {len(nodes)} узлов Kaggle -> {OUT_ROOT}")

    while len(completed_nodes) < len(nodes):
        for node in nodes:
            nid = node["node_id"]
            uname = node["username"]
            ref = f"{uname}/{ok.KERNEL_SLUG}"
            if nid in completed_nodes:
                continue
            env = ok.prepare_node_env(node)
            p = subprocess.run([kbin, "kernels", "status", ref],
                               env=env, capture_output=True, text=True, timeout=30)
            st_text = (p.stdout.strip() or p.stderr.strip())
            state["nodes"][str(nid)] = {
                "username": uname,
                "kernel_ref": ref,
                "status": st_text,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            }
            if "COMPLETE" in st_text:
                log_msg(f"Узел node_{nid} ({uname}) завершил работу ({st_text}). Скачиваю артефакты...")
                raw_dir = OUT_ROOT / f"raw_node_{nid}_{uname}"
                raw_dir.mkdir(parents=True, exist_ok=True)
                po = subprocess.run([kbin, "kernels", "output", ref, "-p", str(raw_dir)],
                                    env=env, capture_output=True, text=True, timeout=300)
                copied = organize_checkpoints(raw_dir)
                state["nodes"][str(nid)]["output_rc"] = po.returncode
                state["nodes"][str(nid)]["copied_files"] = copied
                state["downloaded_models"] = sorted(set(state["downloaded_models"]) | set(copied))
                completed_nodes.add(nid)
                log_msg(f"Узел node_{nid} ({uname}): скачано {len(copied)} весов .pt")
            elif "ERROR" in st_text or "CANCEL" in st_text:
                log_msg(f"ВНИМАНИЕ: узел node_{nid} ({uname}) в статусе {st_text}")
                raw_dir = OUT_ROOT / f"raw_node_{nid}_{uname}"
                raw_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run([kbin, "kernels", "output", ref, "-p", str(raw_dir)],
                               env=env, capture_output=True, text=True, timeout=120)
                organize_checkpoints(raw_dir)
                completed_nodes.add(nid)

        STATUS_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        if len(completed_nodes) < len(nodes):
            time.sleep(60)

    log_msg("Все 4 узла обработаны. Демон-сборщик завершает работу.")


if __name__ == "__main__":
    main()
