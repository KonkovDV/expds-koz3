"""Kaggle entry: PIXGBM retrain on TPU-VM used as big CPU host (bot-6 / BACKLOG TPU-2).

Does not change submission inference. Requires contest train data + geo_pixgbm.py.
"""
from __future__ import annotations

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
    # Kernel may ship only this file — clone is forbidden in factory; expect dataset attach.
    raise FileNotFoundError(
        "scripts/pixgbm_retrain_cleanroom.py not found; attach repo snapshot or copy scripts/"
    )


def find_data_root() -> Path:
    env = os.environ.get("PIXGBM_DATA_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    for base in Path("/kaggle/input").glob("*"):
        for name in ("train-expds-3", "train_expds_3", "TRAIN"):
            cand = base / name
            if cand.is_dir():
                return cand
        # flat dataset that IS the train root (has region folders 200*)
        if any(base.glob("200*")):
            return base
    raise FileNotFoundError(
        "КОЗ №3 train root not found under /kaggle/input. "
        "OPEN: create/attach private dataset or sync via S3 (koz3-s3-cred). "
        "Set PIXGBM_DATA_ROOT."
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", "start")
    log_path = LOG_DIR / "train.log"
    try:
        # Idempotent: if DONE already and models exist, exit 0
        models = OUT / "models" / "all"
        if STATUS.exists() and "DONE" in STATUS.read_text() and (models / "poly.txt").is_file():
            print("already DONE — skip", flush=True)
            return 0

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.7.0"],
        )
        repo = find_repo()
        ensure_geo_pixgbm(repo)
        root = find_data_root()
        threads = int(os.environ.get("PIXGBM_NUM_THREADS", "8"))
        rounds = int(os.environ.get("PIXGBM_ROUNDS", "300"))
        tol = os.environ.get("PIXGBM_METRIC_TOL", "1e-4")
        hashes = repo / "models" / "pixgbm" / "all" / "MODELS_NOT_IN_GIT.md"
        cmd = [
            sys.executable,
            str(repo / "scripts" / "pixgbm_retrain_cleanroom.py"),
            "--enable-retrain",
            "--root",
            str(root),
            "--out",
            str(OUT),
            "--rounds",
            str(rounds),
            "--num-threads",
            str(threads),
            "--metric-tol",
            str(tol),
        ]
        if hashes.is_file():
            cmd += ["--expected-hashes", str(hashes)]
        write_status("RUNNING", f"train threads={threads}")
        t0 = time.time()
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"\n=== {utc_now()} cmd={' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, cwd=str(repo), stdout=logf, stderr=subprocess.STDOUT, text=True)
        # checkpoint marker after train attempt
        (CKPT / f"after_train_{utc_now().replace(':','')}.txt").write_text(
            f"returncode={proc.returncode}\nwall_s={time.time() - t0:.1f}\n", encoding="utf-8"
        )
        # sha256 sidecars
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
