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
def _cred_from_inputs():
    # приватный датасет с cred.json (поля id/key или aws_access_key_id/aws_secret_access_key); значения не печатаются
    import glob, json as _j
    for _p in sorted(glob.glob("/kaggle/input/**/cred.json", recursive=True)):
        try:
            with open(_p, encoding="utf-8") as _fh: _c = _j.load(_fh)
        except Exception as _e:
            log("cred.json не читается:", type(_e).__name__); continue
        if not isinstance(_c, dict): continue
        _k = _c.get("id") or _c.get("aws_access_key_id"); _s = _c.get("key") or _c.get("aws_secret_access_key")
        if _k and _s:
            log("ключи S3 из приватного датасета"); return _k, _s
    return None, None
def s3_client():
    import boto3
    from botocore.config import Config
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient(); kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception as e:
        log("Kaggle Secrets недоступны:", type(e).__name__, "— пробую env / приватный датасет")
    if not kid or not sec:
        kid, sec = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not kid or not sec:
        kid, sec = _cred_from_inputs()
    if not kid or not sec:
        raise SystemExit(f"нет ключей S3: Kaggle Secrets {SECRET_ID}/{SECRET_KEY} или датасет dmitrijkv/koz3-s3-cred (cred.json)")
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=kid, aws_secret_access_key=sec, region_name="ru-7",
                        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=32))
# ── Or-подготовка (29.10): огромные ортофото Or (до 26,7 ГБ, полосы без сжатия, 1–2 см/пикс) → компактный GeoTIFF
# в исходной CRS с шагом ≈PREP_RES м (усреднение блоками строк, маска альфа/nodata → 0, первые ≤3 канала, dtype как в
# источнике). Дальше Scene, как для любого растра, приводит его к UTM 0,5 м (average). Источник удаляется сразу после
# подготовки (диск Kaggle). Используется и CPU-ядром koz3-dfine-or-prep, и GPU-ядром Or (как запасной путь).
PREP_RES = float(os.environ.get("DFINE_PREP_RES", 0.25)); PREP_MIN_BYTES = int(float(os.environ.get("DFINE_PREP_MIN_BYTES", 1.5e9)))
PREP_LOCK = threading.Lock()  # не больше одного огромного скачивания одновременно
def prep_name(key, size): return f"{Path(key).stem}__{size}__r{PREP_RES:g}.tif"
def prep_tif(src_path, dst_path, log=print):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from rasterio.warp import calculate_default_transform
    t = time.time()
    with rasterio.open(src_path) as s:
        try: nat = abs(calculate_default_transform(s.crs, "EPSG:3857", s.width, s.height, *s.bounds)[0].a)
        except Exception: nat = abs(s.transform.a)
        # 3857 растягивает метры на 1/cos(lat): поправка через UTM-подобную оценку ширины в метрах
        try:
            from pyproj import Geod, Transformer
            lon0, lat0 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform(*s.xy(s.height // 2, 0))
            lon1, lat1 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform(*s.xy(s.height // 2, s.width - 1))
            nat = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)[2] / max(1, s.width - 1)
        except Exception: pass
        f = max(1, int(PREP_RES / max(nat, 1e-6)))
        W, H = math.ceil(s.width / f), math.ceil(s.height / f); nb = min(3, s.count); R = max(1, int(64e6 // (s.width * f * 2 * s.count)))
        prof = dict(driver="GTiff", width=W, height=H, count=nb, dtype=s.dtypes[0], crs=s.crs, nodata=0, tiled=True,
                    blockxsize=256, blockysize=256, compress="deflate", predictor=2, BIGTIFF="IF_SAFER",
                    transform=s.transform * s.transform.scale(s.width / W, s.height / H))
        # 4-й канал uint8 = альфа, если так помечен или содержит только 0/255 (GDAL не всегда видит альфу без ExtraSamples)
        alpha = False
        if s.count >= 4 and s.dtypes[3] == "uint8":
            from rasterio.enums import ColorInterp
            v = np.unique(s.read(4, out_shape=(min(512, s.height), min(512, s.width)), resampling=Resampling.nearest))
            alpha = s.colorinterp[3] == ColorInterp.alpha or set(v.tolist()) <= {0, 255}
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True); tmp = str(dst_path) + ".part"
        with rasterio.open(tmp, "w", **prof) as d:
            for oy in range(0, H, R):
                r = min(R, H - oy); sy = oy * f; sh = min(s.height - sy, r * f)
                win = Window(0, sy, s.width, sh)
                a = s.read(list(range(1, nb + 1)), window=win, out_shape=(nb, r, W), resampling=Resampling.average)
                m = s.dataset_mask(window=win, out_shape=(r, W), resampling=Resampling.nearest)
                if alpha: m = np.minimum(m, s.read(4, window=win, out_shape=(r, W), resampling=Resampling.nearest))
                a[:, m == 0] = 0
                d.write(a, window=Window(0, oy, W, r))
        os.replace(tmp, dst_path)
    log(f"[prep] {Path(src_path).name}: нативно {nat:.4f} м/пикс, шаг ×{f} → {W}x{H}x{nb}, альфа {alpha}, "
        f"{Path(dst_path).stat().st_size / 1e6:.1f} МБ, {time.time() - t:.0f} с")
    return {"native_m": round(nat, 5), "factor": f, "alpha_band4": bool(alpha), "W": W, "H": H, "bands": nb, "sec": round(time.time() - t)}
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
