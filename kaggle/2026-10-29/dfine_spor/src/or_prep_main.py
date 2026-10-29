# KOZ3 · koz3-dfine-or-prep (bot-9): CPU-ядро без GPU. Огромные ортофото Or из S3 → /kaggle/working/or_prep/*.tif
# (≈0,25 м/пикс, исходная CRS, deflate) + manifest.json. GPU-ядро koz3-dfine-or берёт их через kernel_sources и не тратит
# GPU-часы на скачивание ~65 ГБ. Файлы обрабатываются по одному; источник удаляется сразу после уменьшения.
# Ключи S3: Kaggle Secrets KOZ3_S3_* → env → /kaggle/input/**/cred.json (датасет dmitrijkv/koz3-s3-cred). Значения не печатаются.
import os, sys, subprocess, importlib
PINS = {"rasterio": "rasterio==1.4.3", "pyproj": "pyproj==3.7.1", "boto3": "boto3==1.40.61"}
need = []
for mod, pin in PINS.items():
    try: importlib.import_module(mod)
    except Exception: need.append(pin)
if need:
    print("pip install:", need, flush=True)
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", *need]).returncode:
        raise SystemExit(f"pip install не удался ({need}); нужен enable_internet=true")
    importlib.invalidate_caches()
import json, time, re, math, glob, shutil, hashlib, threading
import numpy as np
from pathlib import Path
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
OUT = Path(os.environ.get("DFINE_OUT", "/kaggle/working")) / "or_prep"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("DFINE_CACHE", "/tmp/s3cache"))
BUCKET, ENDPOINT = "train-expds-3", os.environ.get("DFINE_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
#@@S3@@
#@@PREP@@
CLI = s3_client(); CLI.list_objects_v2(Bucket=BUCKET, MaxKeys=1); log("S3 OK")
regions = os.environ.get("DFINE_REGIONS", "008_Герите,014_Лешки,033_Елизаветинское,056_Фалхан,093_Старый_Барах,185_Изборск").split(",")
man = {"prep_res_m": PREP_RES, "min_bytes": PREP_MIN_BYTES, "files": []}
for r in regions:
    tok, keys = None, []
    while True:
        kw = dict(Bucket=BUCKET, Prefix=r + "/")
        if tok: kw["ContinuationToken"] = tok
        o = CLI.list_objects_v2(**kw); keys += [(c["Key"], c["Size"]) for c in o.get("Contents", [])]
        if not o.get("IsTruncated"): break
        tok = o["NextContinuationToken"]
    for key, size in keys:
        n = key.rsplit("/", 1)[-1]
        if "_Or_" not in key or not n.lower().endswith((".tif", ".tiff")) or size < PREP_MIN_BYTES: continue
        dst = OUT / prep_name(key, size)
        if dst.exists(): log("уже есть", dst.name); continue
        free = shutil.disk_usage(CACHE.parent if CACHE.parent.exists() else "/").free
        log(f"{r}: {n} {size / 1e9:.2f} ГБ, свободно {free / 1e9:.1f} ГБ")
        if free < size * 1.02 + 2e9: man["files"].append({"key": key, "size": size, "error": "мало диска"}); log("ПРОПУСК: мало диска"); continue
        src = CACHE / key; src.parent.mkdir(parents=True, exist_ok=True); t = time.time()
        try:
            from boto3.s3.transfer import TransferConfig
            CLI.download_file(BUCKET, key, str(src), Config=TransferConfig(max_concurrency=8, multipart_chunksize=64 << 20))
            dl = time.time() - t; log(f"скачано за {dl:.0f} с ({size / 1e6 / max(dl, 1):.0f} МБ/с)")
            info = prep_tif(src, dst, log)
            h = hashlib.sha256(dst.read_bytes()).hexdigest()
            man["files"].append({"key": key, "size": size, "file": dst.name, "bytes": dst.stat().st_size, "sha256": h, "download_sec": round(dl), **info})
        except Exception as e:
            log("ОШИБКА", n, repr(e)[:300]); man["files"].append({"key": key, "size": size, "error": repr(e)[:300]})
        finally:
            src.unlink(missing_ok=True)
        json.dump(man, open(OUT / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
man["total_sec"] = round(time.time() - T0)
json.dump(man, open(OUT / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
errs = [f["key"] for f in man["files"] if "error" in f]
log("ГОТОВО:", len(man["files"]) - len(errs), "файлов; ошибки:", errs)
if errs: raise SystemExit("часть файлов Or не подготовлена")
