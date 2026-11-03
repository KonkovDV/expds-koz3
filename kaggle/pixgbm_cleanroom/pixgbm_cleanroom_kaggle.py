"""Kaggle entry: PIXGBM retrain on TPU-VM as big CPU host (bot-6 / BACKLOG TPU-2).

Rebuilds train tree from xfer shards s000–s009 + gt-train-v2, then trains
geo 200–217 only (original PIXGBM filter). Flags: --enable-retrain always
passed by this job script; repo CLI still defaults OFF.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(os.environ.get("PIXGBM_OUT", "/kaggle/working/pixgbm_retrain"))
LOG_DIR = OUT / "logs"
CKPT = OUT / "checkpoints"
STATUS = OUT / "STATUS.txt"
TREE = Path(os.environ.get("PIXGBM_TREE", "/kaggle/working/train-expds-3"))
REPO_CANDIDATES = [
    Path("/kaggle/working/expds-koz3"),
    Path("/kaggle/input/expds-koz3"),
    Path("."),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(state: str, note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(f"{state}\t{utc_now()}\t{note}\n", encoding="utf-8")


def find_repo() -> Path:
    for p in REPO_CANDIDATES:
        if (p / "scripts" / "pixgbm_retrain_cleanroom.py").is_file():
            return p.resolve()
    # Kernel directory may sit next to scripts/ when push packs the repo snapshot
    here = Path(__file__).resolve().parent
    for p in (here.parent.parent, here.parent, Path("/kaggle/working")):
        if (p / "scripts" / "pixgbm_retrain_cleanroom.py").is_file():
            return p.resolve()
    raise FileNotFoundError(
        "scripts/pixgbm_retrain_cleanroom.py not found; attach/copy repo snapshot"
    )


def ensure_geo_pixgbm(repo: Path) -> None:
    dest = repo / "expds" / "geo_pixgbm.py"
    if dest.is_file():
        return
    for base in Path("/kaggle/input").glob("*"):
        for cand in base.rglob("geo_pixgbm.py"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, dest)
            print("copied geo_pixgbm from", cand, flush=True)
            return
    raise FileNotFoundError(
        "expds/geo_pixgbm.py missing (gitignored on main). "
        "Attach build-time copy sha256 35e4331b… per docs/PIXGBM_DECISION.md"
    )


def load_rebuild():
    # Prefer sibling module in kernel dir; else repo path
    here = Path(__file__).resolve().parent
    mod_path = here / "rebuild_train_tree.py"
    if not mod_path.is_file():
        repo = find_repo()
        mod_path = repo / "kaggle" / "pixgbm_cleanroom" / "rebuild_train_tree.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("rebuild_train_tree", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def default_threads() -> int:
    """Historical models were trained with num_threads=2. Do not default to cpu_count()."""
    env = os.environ.get("PIXGBM_NUM_THREADS")
    if env:
        return max(1, int(env))
    return 2


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", "start")
    log_path = LOG_DIR / "train.log"
    try:
        models = OUT / "models" / "all"
        if STATUS.exists() and "DONE" in STATUS.read_text() and (models / "poly.txt").is_file():
            print("already DONE — skip", flush=True)
            return 0

        # 1) rebuild data tree from xfer shards
        write_status("RUNNING", "rebuild_tree")
        rebuild = load_rebuild()
        report = rebuild.rebuild_train_tree(dest=TREE)
        (CKPT / "rebuild_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("rebuild:", json.dumps({k: report[k] for k in ("root", "geo_region_dirs", "shards_found")}, ensure_ascii=False), flush=True)
        root = Path(report["root"])

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.7.0"])
        repo = find_repo()
        ensure_geo_pixgbm(repo)
        threads = default_threads()
        rounds = int(os.environ.get("PIXGBM_ROUNDS", "300"))
        hashes = repo / "models" / "pixgbm" / "all" / "MODELS_NOT_IN_GIT.md"
        cmd = [
            sys.executable,
            str(repo / "scripts" / "pixgbm_retrain_cleanroom.py"),
            "--enable-retrain",
            "--match-reference",
            "--root",
            str(root),
            "--out",
            str(OUT),
            "--rounds",
            str(rounds),
            "--num-threads",
            str(threads),
        ]
        if hashes.is_file():
            cmd += ["--expected-hashes", str(hashes)]
        write_status("RUNNING", f"train threads={threads}")
        t0 = time.time()
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"\n=== {utc_now()} cmd={' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, cwd=str(repo), stdout=logf, stderr=subprocess.STDOUT, text=True)
        (CKPT / f"after_train_{utc_now().replace(':','')}.txt").write_text(
            f"returncode={proc.returncode}\nwall_s={time.time() - t0:.1f}\nthreads={threads}\n",
            encoding="utf-8",
        )
        for name in ("poly.txt", "finds.txt"):
            p = models / name
            if p.is_file():
                import hashlib

                h = hashlib.sha256(p.read_bytes()).hexdigest()
                (models / f"{name}.sha256").write_text(h + "\n", encoding="utf-8")
                print(name, h, flush=True)
        if proc.returncode == 0:
            write_status("DONE", f"wall_s={time.time() - t0:.1f}")
            return 0
        write_status("FAILED", f"retrain_exit={proc.returncode}")
        return proc.returncode
    except Exception as e:
        write_status("FAILED", repr(e)[:200])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as logf:
            traceback.print_exc(file=logf)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
