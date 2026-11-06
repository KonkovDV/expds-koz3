#!/usr/bin/env python3
""". Deterministic, resumable training of gor_v3_head.npz + gor_dense_head.npz.

Обучение головы: признаки DINOv2, затем логистическая регрессия.
  1) extract_feats  — DINOv2-S vignettes 384+768 m → feats.npz (1536-d)
  2) fit_v3_head    — PCA-64 + L2-logreg C=1 + Landauer clean → gor_v3_head.npz
  3) dense_feats    — dense patch tokens GSDS 2+4 m → dense.npz (768-d)
  4) export_head    — variant 'all', centre_s via CP_GRID → gor_dense_head.npz
                     (expected centre_s=250; legacy md5 e35570e1… / acb95c52…)

PROTOCOL.md §4: STATUS.txt RUNNING/DONE/FAILED, stage checkpoints, idempotent restart.
Seed fixed (SEED=0). RULES: dinov2-small Apache-2.0; KOZ3 SpOr only; no Yandex/Google/Bing/Esri.

Env:
  DATA   — train-expds-3 root (SpOr tifs + UTM.json). PC: F:/2/train-expds-3
  MODEL  — dinov2-small snapshot dir (config.json + model.safetensors). PC: …/models/dinov2-small
  WORK   — work dir (default work/gor_heads)
  DEVICE — cpu|cuda (default cpu)
  THREADS — torch/OMP threads (default 2; PC currently has ~2 free)
  SEED   — default 0
  SKIP_EXTRACT=1 — reuse existing feats/dense npz
  REPO   — repo root (default parent of scripts/); needed so import_dfine finds inference/lib/vendor

Requires Linux/Docker for vendor transformers (manylinux .so). Host Windows .venv lacks
the pin — use pc/runners/run_bot5_gor_heads_docker.ps1. Preflight fails STATUS=FAILED if
MODEL files or transformers import are missing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURE_INPUTS = HERE / "fixtures" / "gor_dense_inputs.json"

DATA = Path(os.environ.get("DATA", "/workspace/data/train-expds-3"))
MODEL = Path(os.environ.get("MODEL", str(REPO / "models" / "dinov2-small")))
WORK = Path(os.environ.get("WORK", str(REPO / "work" / "gor_heads")))
DEVICE = os.environ.get("DEVICE", "cpu")
THREADS = os.environ.get("THREADS", "2")
SEED = os.environ.get("SEED", "0")
SKIP_EXTRACT = os.environ.get("SKIP_EXTRACT", "0") == "1"

V3_DIR = WORK / "v3"
DENSE_DIR = WORK / "dense"
HEADS = WORK / "heads"
STATUS = WORK / "STATUS.txt"
CKPT = WORK / "ckpt.json"
LOG = WORK / "train_heads.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(state: str, extra: str = "") -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        f"{state}\nstarted_or_updated={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n{extra}\n",
        encoding="utf-8",
    )


def load_ckpt() -> dict:
    if CKPT.exists():
        return json.loads(CKPT.read_text(encoding="utf-8"))
    return {"stages": {}}


def save_ckpt(ck: dict) -> None:
    CKPT.write_text(json.dumps(ck, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def run(cmd: list[str], env: dict) -> None:
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed rc={r.returncode}: {' '.join(cmd)}")


def stage_done(ck: dict, name: str, outputs: list[Path]) -> bool:
    st = ck["stages"].get(name)
    if not st or st.get("status") != "DONE":
        return False
    return all(p.exists() for p in outputs)


def mark(ck: dict, name: str, outputs: list[Path], extra: dict | None = None) -> None:
    info = {
        "status": "DONE",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "outputs": {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p), "md5": md5(p)} for p in outputs},
    }
    if extra:
        info.update(extra)
    ck["stages"][name] = info
    save_ckpt(ck)



def preflight() -> None:
    """Fail loud before long extract: MODEL files + pinned transformers (vendor/env)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.gor_heads.dinov2_loader import preflight_model_dir, probe_transformers

    reasons: list[str] = []
    reasons.extend(preflight_model_dir(MODEL))
    try:
        info = probe_transformers(REPO)
        log(f"preflight transformers ok source={info.get('source')} ver={info.get('version')} "
            f"vendor_present={info.get('vendor_present')}")
    except Exception as e:  # noqa: BLE001
        reasons.append(
            f"transformers import failed ({type(e).__name__}: {e}). "
            "Need inference/lib/vendor (manylinux) inside Docker pantum/n7q, "
            "or env transformers==5.21.0. Host Windows .venv cannot load vendor .so."
        )
    if reasons:
        msg = "preflight FAILED: " + " | ".join(reasons)
        write_status("FAILED", msg)
        log(msg)
        raise SystemExit(2)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    V3_DIR.mkdir(parents=True, exist_ok=True)
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    HEADS.mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", f"pid={os.getpid()}\nwork={WORK}\ndata={DATA}\nmodel={MODEL}")
    ck = load_ckpt()
    env_base = {
        **os.environ,
        "DATA": str(DATA),
        "MODEL": str(MODEL),
        "DEVICE": DEVICE,
        "THREADS": THREADS,
        "OMP_NUM_THREADS": THREADS,
        "SEED": SEED,
        "REPO": str(REPO),
        "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    try:
        preflight()
        # --- 1 extract_feats ---
        feats = V3_DIR / "feats.npz"
        meta = V3_DIR / "meta.json"
        if stage_done(ck, "extract_v3", [feats, meta]) or (SKIP_EXTRACT and feats.exists() and meta.exists()):
            log("skip extract_v3 (checkpoint)")
            if not stage_done(ck, "extract_v3", [feats, meta]):
                mark(ck, "extract_v3", [feats, meta], {"note": "SKIP_EXTRACT"})
        else:
            env = {**env_base, "OUT": str(V3_DIR)}
            run([sys.executable, str(REPO / "scripts/gor_patch/extract_feats.py")], env)
            mark(ck, "extract_v3", [feats, meta])

        # --- 2 fit_v3 ---
        v3_head = HEADS / "gor_v3_head.npz"
        if stage_done(ck, "fit_v3", [v3_head]):
            log("skip fit_v3")
        else:
            env = {
                **env_base,
                "OUT": str(V3_DIR),
                "HEAD": str(v3_head),
                "INPUTS": str(FIXTURE_INPUTS),
            }
            run([sys.executable, str(HERE / "fit_v3_head.py")], env)
            mark(ck, "fit_v3", [v3_head])

        # --- 3 dense_feats ---
        dense = DENSE_DIR / "dense.npz"
        files_j = DENSE_DIR / "files.json"
        if stage_done(ck, "extract_dense", [dense, files_j]) or (SKIP_EXTRACT and dense.exists() and files_j.exists()):
            log("skip extract_dense (checkpoint)")
            if not stage_done(ck, "extract_dense", [dense, files_j]):
                mark(ck, "extract_dense", [dense, files_j], {"note": "SKIP_EXTRACT"})
        else:
            env = {**env_base, "OUT": str(DENSE_DIR), "TAG": ""}
            run([sys.executable, str(REPO / "scripts/gor_dense/dense_feats.py")], env)
            mark(ck, "extract_dense", [dense, files_j])

        # --- 4 inputs for export_head ---
        inputs_dst = DENSE_DIR / "gor_dense_inputs.json"
        if not inputs_dst.exists() or inputs_dst.stat().st_size != FIXTURE_INPUTS.stat().st_size:
            shutil.copy2(FIXTURE_INPUTS, inputs_dst)
        if not stage_done(ck, "inputs_dense", [inputs_dst]):
            mark(ck, "inputs_dense", [inputs_dst])

        # --- 5 export_head ---
        dense_head = HEADS / "gor_dense_head.npz"
        if stage_done(ck, "export_dense", [dense_head]):
            log("skip export_dense")
        else:
            env = {
                **env_base,
                "DOUT": str(DENSE_DIR),
                "OUT": str(dense_head),
            }
            # unset FOLD for full-train export
            env.pop("FOLD", None)
            run([sys.executable, str(REPO / "scripts/gor_dense/export_head.py")], env)
            mark(ck, "export_dense", [dense_head])

        # --- 6 validate ---
        env = {
            **env_base,
            "V3_HEAD": str(v3_head),
            "DENSE_HEAD": str(dense_head),
        }
        run([sys.executable, str(HERE / "validate_heads.py")], env)
        mark(ck, "validate", [v3_head, dense_head])

        # install into models/ for local use (not committed)
        for src, dst in (
            (v3_head, REPO / "models/gor_gate/gor_v3_head.npz"),
            (dense_head, REPO / "models/gor_dense/gor_dense_head.npz"),
        ):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            log(f"installed {dst} sha256={sha256(dst)} md5={md5(dst)} bytes={dst.stat().st_size}")

        summary = {
            "gor_v3_head": ck["stages"]["fit_v3"]["outputs"][str(v3_head)],
            "gor_dense_head": ck["stages"]["export_dense"]["outputs"][str(dense_head)],
            "legacy_md5": {
                "gor_v3": "acb95c52956d712a16c3dd389237a691",
                "gor_dense": "e35570e10d17edac32bea8d26fa0f5f4",
            },
            "note": "Byte-identical md5 vs legacy only if feats match original extract; recipe+seed match.",
        }
        (WORK / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_status("DONE", json.dumps(summary))
        log("DONE " + json.dumps(summary))
        return 0
    except Exception as e:
        write_status("FAILED", repr(e))
        log(f"FAILED {e!r}")
        raise


if __name__ == "__main__":
    sys.exit(main())
