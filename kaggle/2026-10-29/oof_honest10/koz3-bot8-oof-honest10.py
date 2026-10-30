#!/usr/bin/env python3
"""KOZ3 · bot-8 / GPU-5: OOF-инференс моделей сдачи на трейне для калибровки порогов (honest-10).

Slug: dmitrijkv/koz3-bot8-oof-honest10
Ресурс: NvidiaTeslaT4, internet, dataset dmitrijkv/koz3-s3-cred.
Возобновляемость: /kaggle/working/oof_honest10/jobs/<id>/done.json + oof/*.geojson;
STATUS.txt RUNNING|DONE|FAILED; SHA256SUMS; пропуск завершённых при рестарте.

Режим OOF_MODE=weights (default): scripts/nn_oof.py --weights <pt> на train root.
OOF_MODE=fold: --predict-only --resume при наличии fold-весов в work/cache (иначе FAIL job).
OOF_SMOKE=1: CPU synthetic без S3/GPU (тот же контракт STATUS/sha256).

Код репо + models/*.pt: attach snapshot (см. README) или S3 (OOF_CODE_S3_KEY / models prefix).
Ключи S3 не печатаются и не пишутся в /kaggle/working.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path(os.environ.get("OOF_OUT", "/kaggle/working/oof_honest10"))
LOG = OUT / "logs"
T0 = time.time()
MODE = os.environ.get("OOF_MODE", "weights")  # weights | fold
SMOKE = os.environ.get("OOF_SMOKE", "").strip() in ("1", "true", "yes")
DEADLINE_H = float(os.environ.get("OOF_DEADLINE_H", "11.0"))
JOB_FILTER = [x for x in os.environ.get("OOF_JOBS", "").split(",") if x.strip()]
BUCKET = os.environ.get("OOF_S3_BUCKET", "train-expds-3")
ENDPOINT = os.environ.get("OOF_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
DOCKER_HINT = "gcr.io/kaggle-private-byod/python"


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def _load_jobs_lib(repo: Path | None = None):
    candidates = []
    if repo:
        candidates.append(repo / "scripts" / "oof_kaggle_jobs.py")
    here = Path(__file__).resolve().parent
    candidates += [
        here / "oof_kaggle_jobs.py",
        Path("/kaggle/working/expds-koz3/scripts/oof_kaggle_jobs.py"),
        Path("scripts/oof_kaggle_jobs.py"),
    ]
    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("oof_kaggle_jobs", p)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            sys.modules["oof_kaggle_jobs"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("scripts/oof_kaggle_jobs.py не найден (attach repo snapshot)")


def find_repo() -> Path:
    for p in (
        Path(os.environ["OOF_REPO"]) if os.environ.get("OOF_REPO") else None,
        Path("/kaggle/working/expds-koz3"),
        Path("/kaggle/input/expds-koz3"),
        Path("."),
    ):
        if p is None:
            continue
        if (p / "scripts" / "nn_oof.py").is_file() and (p / "scripts" / "oof_kaggle_jobs.py").is_file():
            return p.resolve()
    # dataset that contains the repo as a subfolder
    for base in Path("/kaggle/input").glob("*"):
        for cand in base.rglob("scripts/nn_oof.py"):
            root = cand.parent.parent
            if (root / "scripts" / "oof_kaggle_jobs.py").is_file():
                return root.resolve()
    raise FileNotFoundError(
        "репозиторий с scripts/nn_oof.py + oof_kaggle_jobs.py не найден; "
        "прикрепите private dataset snapshot или задайте OOF_REPO"
    )


def find_models(repo: Path) -> Path:
    env = os.environ.get("OOF_MODELS_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    if (repo / "models").is_dir() and any((repo / "models").glob("*.pt")):
        return repo / "models"
    for base in Path("/kaggle/input").glob("*"):
        for name in ("models", "koz3-models"):
            cand = base / name
            if cand.is_dir() and any(cand.glob("*.pt")):
                return cand
        if any(base.glob("*.pt")):
            return base
    raise FileNotFoundError("models/*.pt не найдены (OOF_MODELS_ROOT или dataset)")


def find_train_root() -> Path:
    env = os.environ.get("OOF_DATA_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    for base in Path("/kaggle/input").glob("*"):
        for name in ("train-expds-3", "train_expds_3", "TRAIN"):
            cand = base / name
            if cand.is_dir():
                return cand
    # after S3 sync
    syn = Path("/kaggle/working/train-expds-3")
    if syn.is_dir():
        return syn
    raise FileNotFoundError(
        "train-expds-3 не найден; sync через S3 (koz3-s3-cred) или attach dataset; OOF_DATA_ROOT"
    )


def _cred():
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient()
        kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception as e:
        log("Kaggle Secrets:", type(e).__name__)
    if not kid or not sec:
        kid, sec = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not kid or not sec:
        for p in sorted(glob.glob("/kaggle/input/**/cred.json", recursive=True)):
            try:
                c = json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception:
                continue
            kid = c.get("id") or c.get("aws_access_key_id")
            sec = c.get("key") or c.get("aws_secret_access_key")
            if kid and sec:
                log("S3 keys from private dataset")
                break
    if not kid or not sec:
        raise SystemExit(f"нет ключей S3: Secrets {SECRET_ID}/{SECRET_KEY} или dmitrijkv/koz3-s3-cred")
    return kid, sec


def s3_client():
    import boto3
    from botocore.config import Config
    kid, sec = _cred()
    return boto3.client(
        "s3", endpoint_url=ENDPOINT, aws_access_key_id=kid, aws_secret_access_key=sec,
        region_name="ru-7",
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=32),
    )


def maybe_sync_train(dst: Path, prefix: str = "") -> Path:
    """Опционально: OOF_S3_SYNC=1 скачивает префикс бакета в dst (резюмируемо)."""
    if os.environ.get("OOF_S3_SYNC", "").strip() not in ("1", "true", "yes"):
        return find_train_root()
    dst.mkdir(parents=True, exist_ok=True)
    cli = s3_client()
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(max_concurrency=4)
    token = None
    n = 0
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        resp = cli.list_objects_v2(**kw)
        for obj in resp.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            target = dst / key
            if target.is_file() and target.stat().st_size == obj["Size"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            cli.download_file(BUCKET, key, str(tmp), Config=cfg)
            tmp.rename(target)
            n += 1
            if n % 50 == 0:
                log("s3 synced", n, "files")
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    log("s3 sync done, files touched/new≈", n)
    return dst


def pip_deps():
    need = []
    for mod, pin in {
        "rasterio": "rasterio==1.4.3",
        "shapely": "shapely==2.1.1",
        "boto3": "boto3==1.40.61",
        "segmentation_models_pytorch": "segmentation-models-pytorch==0.3.4",
    }.items():
        try:
            importlib.import_module(mod)
        except Exception:
            need.append(pin)
    if need:
        log("pip install", need)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need])
        if r.returncode:
            raise SystemExit(f"pip failed: {need}")


def run_weights_job(repo: Path, root: Path, models: Path, work: Path, job: dict, lib) -> None:
    out = lib.job_out_path(work, job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    wpaths = []
    for w in job["weights"]:
        p = models / w
        if not p.is_file():
            raise FileNotFoundError(f"нет весов {p}")
        wpaths.append(str(p))
    cmd = [
        sys.executable, "-X", "utf8", str(repo / "scripts" / "nn_oof.py"),
        "--root", str(root),
        "--out", str(out),
        "--markup", job["markup"],
        "--config", str(repo / "configs" / "default.toml"),
        "--weights", *wpaths,
    ]
    if os.environ.get("OOF_TTA"):
        cmd += ["--tta", os.environ["OOF_TTA"]]
    log("run", job["id"], cmd[-6:])
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(repo)}
    logf = LOG / f"{job['id']}.log"
    with logf.open("w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=str(repo), env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"nn_oof rc={r.returncode} see {logf}")
    if not out.is_file():
        raise RuntimeError(f"нет выхода {out}")


def run_fold_job(repo: Path, root: Path, work: Path, job: dict, lib) -> None:
    """Ожидает кэш и fold-веса в work/fold_work/<id>/ (OOF_FOLD_WORK)."""
    cache = Path(os.environ.get("OOF_CACHE", work / "cache" / job["id"]))
    fold_work = Path(os.environ.get("OOF_FOLD_WORK", work / "fold_work" / job["id"]))
    out = lib.job_out_path(work, job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (cache / "index.csv").is_file():
        raise FileNotFoundError(f"fold mode: нет кэша {cache}/index.csv")
    cmd = [
        sys.executable, "-X", "utf8", str(repo / "scripts" / "nn_oof.py"),
        "--root", str(root),
        "--cache", str(cache),
        "--work", str(fold_work),
        "--out", str(out),
        "--markup", job["markup"],
        "--config", str(repo / "configs" / "default.toml"),
        "--folds", os.environ.get("OOF_FOLDS", "5"),
        "--predict-only", "--resume",
    ]
    log("run fold", job["id"])
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(repo)}
    logf = LOG / f"{job['id']}_fold.log"
    with logf.open("w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=str(repo), env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"nn_oof fold rc={r.returncode} see {logf}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    # smoke early — no repo/S3 required if lib is beside the kernel or in PYTHONPATH
    if SMOKE:
        # prefer lib next to this file (shipped) or from repo
        lib_path = Path(__file__).resolve().parent / "oof_kaggle_jobs.py"
        if not lib_path.is_file():
            # copy from cwd scripts if present
            src = Path("scripts/oof_kaggle_jobs.py")
            if src.is_file():
                shutil.copy2(src, lib_path)
        try:
            lib = _load_jobs_lib()
        except FileNotFoundError:
            # inline minimal: import from scripts if cwd is repo
            sys.path.insert(0, str(Path(".").resolve()))
            lib = _load_jobs_lib(Path("."))
        lib.write_status(OUT, "RUNNING", phase="smoke", mode="smoke")
        result = lib.run_smoke(OUT)
        log("smoke", result)
        return 0

    lib = None
    try:
        pip_deps()
        repo = find_repo()
        lib = _load_jobs_lib(repo)
        lib.write_status(OUT, "RUNNING", phase="boot", mode=MODE)
        log("repo", repo)
        models = find_models(repo)
        log("models", models)
        if os.environ.get("OOF_S3_SYNC", "").strip() in ("1", "true", "yes"):
            root = maybe_sync_train(Path("/kaggle/working/train-expds-3"))
        else:
            try:
                root = find_train_root()
            except FileNotFoundError:
                log("train root missing — trying S3 sync")
                root = maybe_sync_train(Path("/kaggle/working/train-expds-3"))
        log("train", root)

        jobs = lib.select_jobs(JOB_FILTER or None)
        deadline = T0 + DEADLINE_H * 3600
        done_n = fail_n = skip_n = 0
        for job in jobs:
            if time.time() > deadline:
                log("deadline — stop, resume later")
                lib.write_status(OUT, "RUNNING", phase="deadline", done=done_n, failed=fail_n, skipped=skip_n)
                return 0
            if lib.is_job_complete(OUT, job):
                log("skip done", job["id"])
                skip_n += 1
                continue
            t1 = time.perf_counter()
            try:
                if MODE == "fold":
                    run_fold_job(repo, root, OUT, job, lib)
                else:
                    run_weights_job(repo, root, models, OUT, job, lib)
                out = lib.job_out_path(OUT, job["out"])
                sha = lib.sha256_file(out)
                n = lib.count_features(out)
                lib.write_done(OUT, job, sha, n, time.perf_counter() - t1)
                done_n += 1
                log("done", job["id"], "feats", n, "sha", sha[:12])
            except Exception as e:
                fail_n += 1
                log("FAIL", job["id"], type(e).__name__, e)
                (LOG / f"{job['id']}_error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                # continue other jobs
        lib.write_sha256sums(OUT, jobs)
        state = "DONE" if fail_n == 0 else ("FAILED" if done_n == 0 else "DONE")
        # partial success still DONE with failed=; hard FAILED only if nothing new and failures
        if fail_n and done_n == 0 and skip_n == 0:
            state = "FAILED"
        elif fail_n:
            state = "DONE"  # partial; details in STATUS.json
        lib.write_status(
            OUT, state, phase="finish", mode=MODE,
            done=done_n, failed=fail_n, skipped=skip_n,
            elapsed_h=round((time.time() - T0) / 3600, 3),
        )
        log("finish", state, "done", done_n, "fail", fail_n, "skip", skip_n)
        return 0 if state != "FAILED" else 1
    except Exception:
        log("FATAL", traceback.format_exc())
        if lib:
            lib.write_status(OUT, "FAILED", phase="fatal", error="see logs")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "STATUS.txt").write_text(
                f"FAILED\nts={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\nphase=fatal\n",
                encoding="utf-8",
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
