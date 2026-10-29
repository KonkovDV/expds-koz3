# KOZ3 №3 · convnext_lidar (bot-9): U-Net с энкодером ConvNeXt-T на каналах рельефа LiDAR (0,5 м/пикс).
# Рецепт L1: /workspace/koz3_shared/bot-9/DETECTORS_SOTA.md §4.4–4.5 (кроп 512, 12k ит., copy-paste, heatmap-пики).
# Фолды: честные семейные (configs/folds_li_tin_59_t128.json, семьи — index.csv li_tin_59_t128), 5 фолдов, LOFO.
# Режимы CNX_MODE: prep (CPU: S3 LAS → каналы рельефа *.npy) | smoke (10 ит.) | full (12k ит.).
# Варианты (флаг CNX_JOBS / встроенный список): c3|c5 (каналы lrm_s,lrm_l,slope [+svf_r,opp_r]) × in|sc (ImageNet|scratch).
# ВАЖНО: инициализация ImageNet (timm convnext_tiny.fb_in1k) — ТОЛЬКО для измерения, в сдачу — лишь после письма
# организаторов о весах. Сдаваемые кандидаты — варианты *-sc.
# Ключи S3: Kaggle Secrets KOZ3_S3_ACCESS_KEY_ID/KOZ3_S3_SECRET_ACCESS_KEY → env → /kaggle/input/**/cred.json.
# Значения ключей никогда не печатаются и не пишутся в /kaggle/working.
# @@EMBED@@
import os, sys, subprocess, importlib, argparse, time, json, shutil
AP = argparse.ArgumentParser()
AP.add_argument("--allow-cpu", action="store_true", help="только для бокса: разрешить запуск без GPU")
ARGS, _ = AP.parse_known_args()
MODE = os.environ.get("CNX_MODE", CNX_MODE_DEFAULT)
NEED_GPU = MODE != "prep"
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)

# ── Шаг 1. --require-gpu (по умолчанию) — до pip ─────────────────────────────────────────────────────────────
if NEED_GPU and not ARGS.allow_cpu:
    try: _smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
    except Exception: _smi = ""
    if "GPU" not in _smi:
        raise SystemExit("--require-gpu: GPU нет (сессия получила CPU — второй одновременный GPU-кернел?). Останов.")
    log("nvidia-smi:", _smi.strip().replace("\n", " | "))

# ── Шаг 2. Зависимости: пины + жёсткий импорт-чек; torch/torchvision — из образа ──────────────────────────────
PINS = {"timm": "timm==1.0.29", "segmentation_models_pytorch": "segmentation-models-pytorch==0.6.0",
        "rasterio": "rasterio==1.4.3", "pyproj": "pyproj==3.7.1", "shapely": "shapely==2.1.1", "boto3": "boto3==1.40.61",
        "scipy": "scipy==1.16.3", "laspy": "laspy==2.7.0"}
NODEPS = {"rvt": "rvt-py==2.2.1"}          # rvt-py тянет gdal/matplotlib — ставим --no-deps, нужен только rvt.vis
STRICT = {"timm", "segmentation_models_pytorch", "rvt"}  # версия влияет на модель/каналы — строго пин
def _ver(mod):
    try:
        from importlib.metadata import version
        return version({"segmentation_models_pytorch": "segmentation-models-pytorch", "rvt": "rvt-py"}.get(mod, mod))
    except Exception: return "?"
def _deps():
    import torch as _t
    cons = "/tmp/cnx_constraints.txt"   # не дать pip заменить torch/numpy образа
    with open(cons, "w") as fh:
        for p in ("torch", "torchvision", "numpy", "huggingface_hub", "safetensors"):
            v = _ver(p)
            if v != "?": fh.write(f"{p}=={v}\n")
    for group, extra in ((PINS, []), (NODEPS, ["--no-deps"])):
        need = []
        for mod, pin in group.items():
            try:
                importlib.import_module(mod)
                if mod in STRICT and _ver(mod) != pin.split("==")[1]: need.append(pin)
            except Exception: need.append(pin)
        if need:
            print("pip install:", need, extra, flush=True)
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", "-c", cons, *extra, *need])
            if r.returncode: raise SystemExit(f"pip install не удался ({need}); нужен enable_internet=true")
            importlib.invalidate_caches()
            for k in [k for k in sys.modules if k.split(".")[0] in ("timm", "segmentation_models_pytorch", "rvt")]: del sys.modules[k]
    bad = []
    for mod in ["torch", "torchvision", "numpy", *PINS, "rvt.vis"]:
        try: importlib.import_module(mod); print(f"  {mod} {_ver(mod.split('.')[0])}", flush=True)
        except Exception as e: bad.append(f"{mod}: {e!r}")
    for mod, pin in {**PINS, **NODEPS}.items():
        if mod in STRICT and _ver(mod) != pin.split("==")[1]: bad.append(f"{mod} {_ver(mod)} ≠ {pin}")
    if bad: raise SystemExit("импорт-чек не пройден: " + "; ".join(bad))
_deps()

# вендор: минимальный expds (Rust expds_core) + relief_ext бота 2 + compute_metrics (koz3-1.5.0)
import base64, io, tarfile, hashlib
VENDOR = "/tmp/cnx_vendor"
if os.path.isdir(VENDOR): shutil.rmtree(VENDOR)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(VENDOR_B64)), mode="r:gz") as _tf: _tf.extractall(VENDOR, filter="data")
VMAN = json.load(open(f"{VENDOR}/vendor_manifest.json", encoding="utf-8"))
for _n, _h in VMAN["sha256"].items():
    if hashlib.sha256(open(f"{VENDOR}/{_n}", "rb").read()).hexdigest() != _h: raise SystemExit(f"вендор повреждён: {_n}")
sys.path.insert(0, VENDOR)
os.environ.setdefault("RAYON_NUM_THREADS", str(max(1, min(4, os.cpu_count() or 1))))
from expds import accel as XA, lidar as XL
from expds.candidates import to_work_grid
import relief_ext as RX
import rvt
if XA._rs is None or not hasattr(XA._rs, "relief_channels"): raise SystemExit("expds_core (Rust) не загрузился — каналы не совпадут")
# sha только файлов, влияющих на каналы (смена головы/метрики не требует пересборки prep)
RELIEF_SHA = {k: v for k, v in VMAN["sha256"].items() if k.startswith("expds") or k == "relief_ext.py"}
log("вендор OK:", XA._rs.version(), "rvt", _ver("rvt"), "relief_ext", VMAN["sha256"]["relief_ext.py"][:12])

import re, math, glob, random, threading, collections, concurrent.futures as cf
from pathlib import Path
import numpy as np, torch

E = lambda k, d: type(d)(os.environ.get(k, d))
SMOKE = MODE == "smoke"
CFG = dict(mode=MODE, res=0.5, crop=E("CNX_CROP", 512), bs=E("CNX_BS", 8), iters=E("CNX_ITERS", 10 if SMOKE else 12000),
           lr=3e-4, wd=1e-4, warmup=E("CNX_WARMUP", 500), ema=0.999, workers=E("CNX_WORKERS", 2), amp=True,
           obj_frac=0.6, cls_w=[2.0, 1.0, 1.0, 2.0, 1.0],          # kt, kp, gor, fort, arch (§4.4: povr×1, tselye×2, fort×2)
           sigma_k=0.15, sigma_min_px=1.0, default_diam_m=8.0,
           p_height=0.5, height=(0.5, 2.2), p_scale=0.5, scale=(0.7, 1.45), p_noise=0.5, noise_m=(0.02, 0.05),
           p_paste=0.5, paste_max=3, paste_bank=E("CNX_PASTE_BANK", 1500),
           infer_stride=E("CNX_STRIDE", 384), tta=E("CNX_TTA", 8), infer_bs=E("CNX_INFER_BS", 8),
           peak_k=E("CNX_PEAK_K", 11), min_score=E("CNX_MIN_SCORE", 0.05), mask_thr=0.5, gor_min_px=600, fort_min_px=30,
           arch_min_px=30, max_det=E("CNX_MAX_DET", 20000), circle_r_m=E("CNX_CIRCLE_R", 1.0),
           deadline_h=E("CNX_DEADLINE_H", 11.3), min_iter_frac=E("CNX_MIN_ITER_FRAC", 0.25), infer_reserve_h=E("CNX_INFER_RESERVE_H", 0.6),
           jobs=os.environ.get("CNX_JOBS", CNX_JOBS_DEFAULT), regions=os.environ.get("CNX_REGIONS", CNX_REGIONS_DEFAULT),
           build_relief=E("CNX_BUILD_RELIEF", 1 if MODE in ("prep", "smoke") else 0), seed=E("CNX_SEED", 0),
           log_every=E("CNX_LOG_EVERY", 100),
           budget_warm_it=E("CNX_BUDGET_WARM_IT", 100), budget_it=E("CNX_BUDGET_IT", 150))   # бюджет: медиана с/ит ит. warm+1..budget (после прогрева)
OUT = Path(os.environ.get("CNX_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("CNX_CACHE", "/tmp/s3cache")); RELIEF_DIR = Path(os.environ.get("CNX_RELIEF_DIR", "/tmp/cnx_relief"))
BUCKET, ENDPOINT = "train-expds-3", os.environ.get("CNX_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
RU2CLS = {"курганы_целые": 0, "курганы_поврежденные": 1, "городища": 2, "фортификации": 3, "архитектуры": 4}
TAGS5 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r")   # порядок каналов = контракт модели (c3 = первые три)
DEADLINE = T0 + CFG["deadline_h"] * 3600
# сменная голова курганов (HEAD_INTERFACE.md): heat (встроенная) | centerhead (bot-2) | pkg.mod:factory
import cnx_head as HD
HEAD_NAME = os.environ.get("CNX_HEAD", globals().get("CNX_HEAD_DEFAULT", "heat"))
HEAD = HD.get_head(HEAD_NAME, CFG); HD.check_head(HEAD)
def _head_selftest():                    # ранний фейл: forward/цели/лосс/backward/untta на 64×64, scratch-энкодер
    m = HEAD.build(3, "tu-convnext_tiny", None); x = torch.randn(2, 3, 64, 64)
    o = m(x); assert set(HEAD.out_keys) <= set(k for k, v in o.items() if v is not None), (HEAD.out_keys, list(o))
    hs = 64 // HEAD.stride; shp = {k: tuple(v.shape) for k, v in o.items() if v is not None}
    assert o["mask"].shape == (2, 3, 64, 64) and all(sh[-2:] == (hs, hs) for k, sh in shp.items() if k != "mask") and o["hm"].shape[1] == 2, shp
    tg = HEAD.targets(np.array([[20.3, 30.7, 12.0, 0], [40.0, 10.0, 8.0, 1]]), 64); assert all(v.shape[-2:] == (hs, hs) for v in tg.values()), {k: v.shape for k, v in tg.items()}
    tb = {k: torch.from_numpy(np.stack([v, v])) for k, v in tg.items()}
    l, parts = HEAD.loss(o, tb, torch.ones(2, 64, 64)); l.backward(); assert math.isfinite(float(l)), parts
    u = HEAD.untta({k: v.detach() for k, v in o.items() if v is not None and k != "mask"}, 1, True); assert u["hm"].shape == o["hm"].shape
    return dict(head=HEAD.name, stride=HEAD.stride, out_shapes=shp, loss_parts=parts)
assert CFG["crop"] % HEAD.stride == 0 and CFG["infer_stride"] % HEAD.stride == 0, "crop/шаг тайлов не кратны stride головы"
CIRCLE_R = float(os.environ["CNX_CIRCLE_R"]) if "CNX_CIRCLE_R" in os.environ else (getattr(HEAD, "circle_r_m", None) or CFG["circle_r_m"])
CFG["circle_r_m"] = CIRCLE_R; SAVE_MAPS = E("CNX_SAVE_MAPS", 1)
log("CFG", json.dumps(CFG, ensure_ascii=False))   # после переопределения CIRCLE_R головой/env (было: до него → в логе 1.0 при факт. 0.5)
log("голова:", HEAD_NAME, json.dumps(_head_selftest(), ensure_ascii=False))

# ── Шаг 1б. GPU через torch ───────────────────────────────────────────────────────────────────────────────────
NG = torch.cuda.device_count() if torch.cuda.is_available() else 0
if NEED_GPU:
    if NG == 0 and not ARGS.allow_cpu: raise SystemExit("--require-gpu: torch не видит CUDA. Останов.")
    for i in range(NG): (torch.ones(8, device=f"cuda:{i}") * 2).sum().item()
    log("GPU:", [torch.cuda.get_device_name(i) for i in range(NG)])
DEVS = [torch.device(f"cuda:{i}") for i in range(NG)][:2] or [torch.device("cpu")]

# ── Шаг 2б. Задания и веса ImageNet (ранний фейл) ─────────────────────────────────────────────────────────────
def parse_jobs(s):
    out = []
    for tok in [t for t in s.split(",") if t.strip()]:
        v, f = tok.strip().split(":")
        ch, ini = v.split("-")
        assert ch in ("c3", "c5") and ini in ("in", "sc"), tok
        out.append(dict(variant=v, ch=int(ch[1]), init=ini, fold=int(f)))
    return out
JOBS = parse_jobs(CFG["jobs"]) if MODE != "prep" else []
TIMM_NAME = "convnext_tiny.fb_in1k"
WSHA = None
if any(j["init"] == "in" for j in JOBS):
    try:
        import timm
        _m = timm.create_model(TIMM_NAME, pretrained=True, num_classes=0)
        _sd = _m.state_dict(); _h = hashlib.sha256()
        for k in sorted(_sd): _h.update(k.encode()); _h.update(_sd[k].float().cpu().numpy().tobytes())
        WSHA = _h.hexdigest(); del _m, _sd
    except Exception as e: raise SystemExit(f"веса ImageNet {TIMM_NAME} не скачались: {e!r}")
    log("веса ImageNet OK", TIMM_NAME, "sha256(state_dict)", WSHA[:16], "— ТОЛЬКО измерение, не в сдачу")

# ── Шаг 3. Секреты → S3 (ранний фейл) ─────────────────────────────────────────────────────────────────────────
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
CLI = s3_client()
try: CLI.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
except Exception as e: raise SystemExit(f"S3 недоступен ({type(e).__name__}): проверьте секреты/интернет (детали скрыты).")
log("S3 OK")

# ── Шаг 4. Фолды по семьям: проверка утечки (две карты семей) ─────────────────────────────────────────────────
FAM, CLU = FOLDS["family"], FOLDS["cluster"]
FOLD_OF_FAM = {}
for f in FOLDS["folds"]:
    for fam in f["test_families"]:
        assert fam not in FOLD_OF_FAM, f"семья {fam} в двух фолдах"; FOLD_OF_FAM[fam] = f["fold"]
FOLD = {r: FOLD_OF_FAM[FAM[r]] for r in FAM}
assert set(FAM) == set(FOLD), "участок без фолда"
def split(fold, pool):
    test = [r for r in pool if FOLD[r] == fold]
    tclu = {CLU.get(r, FAM[r]) for r in test}; tfam = {FAM[r] for r in test}
    train = [r for r in pool if FOLD[r] != fold and FAM[r] not in tfam and CLU.get(r, FAM[r]) not in tclu]
    leak = [r for r in train if FAM[r] in tfam or CLU.get(r, FAM[r]) in tclu]
    if leak: raise SystemExit(f"утечка семей в фолде {fold}: {leak}")
    return train, test
for f in range(5): split(f, list(FAM))
# кластерная карта (folds_family_k5) не должна связывать участки разных фолдов
_cf = collections.defaultdict(set)
for r in FAM: _cf[CLU.get(r, FAM[r])].add(FOLD[r])
_bad = {k: sorted(v) for k, v in _cf.items() if len(v) > 1}
if _bad: raise SystemExit(f"утечка: кластер семей в нескольких фолдах {_bad}")
log("фолды OK:", len(FAM), "участков,", len(FOLD_OF_FAM), "семей; кластеры k5 не пересекают фолды")

def list_prefix(prefix, delim=None):
    out, tok = [], None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix)
        if delim: kw["Delimiter"] = delim
        if tok: kw["ContinuationToken"] = tok
        r = CLI.list_objects_v2(**kw)
        out += [p["Prefix"] for p in r.get("CommonPrefixes", [])] if delim else [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"): return out
        tok = r["NextContinuationToken"]

def fetch(key, size):
    from boto3.s3.transfer import TransferConfig
    dst = CACHE / key
    if dst.exists() and dst.stat().st_size == size: return dst
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_name(dst.name + ".part")
    for a in range(5):
        try: CLI.download_file(BUCKET, key, str(tmp), Config=TransferConfig(max_concurrency=4)); tmp.rename(dst); return dst
        except Exception as e: log("повтор", a + 1, key, type(e).__name__); time.sleep(5 * (a + 1))
    raise RuntimeError("не скачан " + key)

def region_inv(region):
    las, marks, utm = None, [], None
    for k, s in list_prefix(region + "/"):
        n = k.rsplit("/", 1)[-1]
        if n == "UTM.json": utm = (k, s)
        elif n.lower().endswith(".las") and "_La_" in n: las = (k, s)
        elif n.endswith(".geojson") and "разметка" in k:
            m = re.search(r"_Li_(.+)\.geojson$", n)
            if m and m.group(1) in RU2CLS: marks.append((k, s, RU2CLS[m.group(1)]))
    return {"region": region, "las": las, "marks": sorted(marks), "utm": utm}

POOL = sorted(FAM) if not CFG["regions"] else [r for r in CFG["regions"].split(",") if r]
for r in POOL:
    if r not in FAM: raise SystemExit(f"участок {r} вне фолдов")
with cf.ThreadPoolExecutor(16) as ex: INV = {i["region"]: i for i in ex.map(region_inv, POOL)}
_nolas = [r for r, i in INV.items() if not i["las"]]
if _nolas: raise SystemExit(f"нет LAS в S3: {_nolas}")
log(f"S3-инвентарь: {len(INV)} участков, LAS {sum(i['las'][1] for i in INV.values()) / 1e9:.2f} ГБ, "
    f"файлов разметки Li {sum(len(i['marks']) for i in INV.values())}")
for j in JOBS:
    tr, te = split(j["fold"], POOL)
    if not te or not tr: raise SystemExit(f"задание {j}: пустой train ({len(tr)}) или test ({len(te)}) в выбранных участках")

def region_crs(inv):
    if inv["utm"]:
        try: d = json.load(open(fetch(*inv["utm"]), encoding="utf-8"))
        except Exception: d = {}
        c = str(d.get("crs") or d.get("CRS") or "")
        m = re.search(r"EPSG:*\s*(\d+)", c, re.I)
        if m: return f"EPSG:{m.group(1)}"
        u = str(d.get("utm") or d.get("UTM") or "").strip().upper()
        dg = "".join(ch for ch in u if ch.isdigit())
        if dg and 1 <= int(dg) <= 60:
            tail = u.replace("UTM", "").replace("ZONE", "").strip()
            return f"EPSG:{327 if tail.endswith('S') and not tail.endswith('NS') else 326}{int(dg):02d}"
    return None   # build_dem возьмёт CRS из заголовка LAS (expds.crs.las_crs)

# ── Шаг 5. Каналы рельефа: prep-выход (kernel_sources) или сборка здесь ───────────────────────────────────────
def build_relief(region):
    inv = INV[region]; t = time.time()
    las = fetch(*inv["las"]); crs = region_crs(inv)
    d = XL.build_dem(las, res=0.1, fill="delaunay", crs=crs)
    base, tr = XL.relief_channels(d, CFG["res"], TAGS5[:3])
    z, step = to_work_grid(d.dem, d.res, CFG["res"])
    if z.shape != base.shape[1:]: raise RuntimeError(f"{region}: сетка z {z.shape} ≠ каналы {base.shape}")
    z = np.where(base[0] > 0, z, np.nan).astype(np.float32)
    ext, stats = RX.extra_channels(z, step, d.left, d.top, accel=None, tags=TAGS5[3:])
    arr = np.concatenate([base, ext]).astype(np.uint8)
    np.save(RELIEF_DIR / f"{region}.npy", arr)
    meta = dict(region=region, crs=str(d.crs), left=float(d.left), top=float(d.top), step=float(step), shape=list(arr.shape),
                tags=list(TAGS5), site_stats=stats, dem_res=float(d.res), dem_fill="delaunay", backend=XA.backend(),
                rvt=_ver("rvt"), vendor=VMAN["sha256"], valid_px=int((arr[0] > 0).sum()), t_s=round(time.time() - t, 1))
    json.dump(meta, open(RELIEF_DIR / f"{region}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if MODE == "prep" or os.environ.get("CNX_KEEP_LAS", "0") != "1":
        try: Path(las).unlink()
        except Exception: pass
    return meta

def find_prep():
    for p in sorted(glob.glob("/kaggle/input/**/cnx_relief/manifest.json", recursive=True)):
        return Path(p).parent
    p = os.environ.get("CNX_PREP_DIR")
    return Path(p) if p and (Path(p) / "manifest.json").exists() else None

RELIEF_DIR.mkdir(parents=True, exist_ok=True)
META = {}
if MODE == "prep":
    RELIEF_DIR = OUT / "cnx_relief"; RELIEF_DIR.mkdir(parents=True, exist_ok=True)
    order = sorted(POOL, key=lambda r: INV[r]["las"][1])
    with cf.ThreadPoolExecutor(1) as pre:       # скачивание следующего LAS параллельно со сборкой текущего
        nxt = pre.submit(fetch, *INV[order[0]]["las"])
        for i, r in enumerate(order):
            nxt.result()
            if i + 1 < len(order): nxt = pre.submit(fetch, *INV[order[i + 1]]["las"])
            if (RELIEF_DIR / f"{r}.json").exists(): META[r] = json.load(open(RELIEF_DIR / f"{r}.json", encoding="utf-8")); continue
            META[r] = build_relief(r)
            log(f"relief {i + 1}/{len(order)} {r}: {META[r]['shape']} за {META[r]['t_s']} с")
    json.dump(dict(regions=sorted(META), tags=list(TAGS5), vendor=RELIEF_SHA, backend=XA.backend(), rvt=_ver("rvt"),
                   res=CFG["res"], dem_res=0.1, dem_fill="delaunay"),
              open(RELIEF_DIR / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("prep готов:", len(META), "участков →", RELIEF_DIR); raise SystemExit(0)

PREP = find_prep()
need = sorted({r for j in JOBS for r in sum(split(j["fold"], POOL), [])})
if PREP is not None:
    man = json.load(open(PREP / "manifest.json", encoding="utf-8"))
    if man["vendor"] != RELIEF_SHA or man["tags"] != list(TAGS5):
        raise SystemExit("prep-выход собран другим кодом каналов (sha вендора/теги не совпали) — перезапустите prep")
    log("prep-выход:", PREP, len(man["regions"]), "участков")
for r in need:
    if PREP is not None and (PREP / f"{r}.npy").exists():
        META[r] = json.load(open(PREP / f"{r}.json", encoding="utf-8")); META[r]["npy"] = str(PREP / f"{r}.npy")
    elif CFG["build_relief"]:
        META[r] = build_relief(r); META[r]["npy"] = str(RELIEF_DIR / f"{r}.npy"); log("relief собран здесь:", r, META[r]["shape"], META[r]["t_s"], "с")
    else:
        raise SystemExit(f"нет каналов для {r}: подключите kernel_sources prep-ядра или CNX_BUILD_RELIEF=1")

# ── Шаг 6. Разметка Li → объекты (пиксели сетки) и маски ──────────────────────────────────────────────────────
from shapely.geometry import shape, mapping, Point
from shapely.ops import transform as stf
from pyproj import Transformer
def load_gt(region):
    m = META[region]; out = []
    for k, s, c in INV[region]["marks"]:
        d = json.load(open(fetch(k, s), encoding="utf-8"))
        n = ((d.get("crs") or {}).get("properties") or {}).get("name", ""); mm = re.search(r"EPSG::?(\d+)", n)
        tr = Transformer.from_crs(f"EPSG:{mm.group(1)}" if mm else "OGC:CRS84", m["crs"], always_xy=True).transform
        for f in d.get("features", []):
            g = f.get("geometry")
            if not g: continue
            try: g = stf(tr, shape(g))
            except Exception: continue
            if not g.is_valid: g = g.buffer(0)
            if not g.is_empty: out.append((g, c))
    return out

def region_targets(region):
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    m = META[region]; C, H, W = m["shape"]; st = m["step"]; tr = from_origin(m["left"], m["top"], st, st)
    gts = load_gt(region); objs, samp, shapes = [], [], {2: [], 3: [], 4: []}
    for g, c in gts:
        if c in (0, 1):
            p = g.centroid if g.geom_type != "Point" else g
            if g.geom_type in ("Polygon", "MultiPolygon") and not g.contains(p): p = g.representative_point()
            dm = 2 * math.sqrt(g.area / math.pi) if g.area > 0 else CFG["default_diam_m"]
            x, y = (p.x - m["left"]) / st, (m["top"] - p.y) / st
            objs.append((x, y, dm / st, c)); samp.append((x, y, c))
        else:
            gg = g if g.area > 0 else g.buffer(1.0)
            shapes[c].append(gg); p = gg.representative_point()
            samp.append(((p.x - m["left"]) / st, (m["top"] - p.y) / st, c))
    mask = np.zeros((H, W), np.uint8)
    for c, bit in ((2, 1), (3, 2), (4, 4)):
        if shapes[c]:
            r = rasterize([(s, 1) for s in shapes[c]], out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
            mask |= (r * bit).astype(np.uint8)
    return dict(objs=np.array(objs, np.float32).reshape(-1, 4), samp=np.array(samp, np.float32).reshape(-1, 3), mask=mask, gts=gts)
TGT = {}
with cf.ThreadPoolExecutor(4) as ex:
    for r, t in zip(need, ex.map(region_targets, need)): TGT[r] = t
log("разметка:", {c: int(sum((t["samp"][:, 2] == i).sum() for t in TGT.values())) for i, c in enumerate(CLASSES)})

# ── Шаг 7. Данные: окна, аугментации L1 ───────────────────────────────────────────────────────────────────────
import torch.nn.functional as F
U8_PER_M = {0: 127.0 / 0.75, 1: 127.0 / 2.0}   # lrm_s клип ±0,75 м, lrm_l ±2 м (expds.lidar.RELIEF) → уровни u8 на метр
SLOPE_MAX = 45.0
def _rel(r): return np.load(META[r]["npy"], mmap_mode="r")
def read_win(arr, x0, y0, n):
    C, H, W = arr.shape; out = np.zeros((C, n, n), arr.dtype)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + n), min(H, y0 + n)
    if xb > xa and yb > ya: out[:, ya - y0:yb - y0, xa - x0:xb - x0] = arr[:, ya:yb, xa:xb]
    return out
class WinDS(torch.utils.data.Dataset):
    def __init__(self, regions, nch, seed, length):
        self.reg, self.nch, self.seed, self.n = regions, nch, seed, length
        self.valid = np.array([META[r]["valid_px"] for r in regions], np.float64); self.valid /= self.valid.sum()
        sl = [(i, k) for i, r in enumerate(regions) for k in range(len(TGT[r]["samp"]))]
        self.sl = np.array(sl, np.int64).reshape(-1, 2)
        w = np.array([CFG["cls_w"][int(TGT[regions[i]]["samp"][k, 2])] for i, k in self.sl], np.float64)
        self.sw = w / w.sum() if len(w) else None
        self.bank = self._bank(np.random.default_rng(seed))
    def _bank(self, rng):
        bank, cand = [], [(r, o) for r in self.reg for o in TGT[r]["objs"]]
        for idx in rng.permutation(len(cand))[:CFG["paste_bank"]]:
            r, (x, y, d, c) = cand[idx]; rad = int(math.ceil(0.9 * d)) + 2
            if rad > 60: continue
            p = read_win(_rel(r), int(x) - rad, int(y) - rad, 2 * rad + 1)[:self.nch]
            if (p[0] > 0).mean() > 0.95: bank.append((p, d, int(c), x - (int(x) - rad), y - (int(y) - rad)))
        return bank
    def __len__(self): return self.n
    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        S = CFG["crop"]; s = rng.uniform(*CFG["scale"]) if rng.random() < CFG["p_scale"] else 1.0
        n = int(round(S / s))
        if self.sw is not None and rng.random() < CFG["obj_frac"]:
            ri, k = self.sl[rng.choice(len(self.sl), p=self.sw)]; x, y, _ = TGT[self.reg[ri]]["samp"][k]
            x0 = int(x - n * rng.uniform(0.2, 0.8)); y0 = int(y - n * rng.uniform(0.2, 0.8))
        else:
            ri = rng.choice(len(self.reg), p=self.valid); a = _rel(self.reg[ri]); _, H, W = a.shape
            for _ in range(20):
                x0, y0 = int(rng.integers(-n // 4, max(1, W - 3 * n // 4))), int(rng.integers(-n // 4, max(1, H - 3 * n // 4)))
                if a[0, min(H - 1, max(0, y0 + n // 2)), min(W - 1, max(0, x0 + n // 2))] > 0: break
        r = self.reg[ri]; img = read_win(_rel(r), x0, y0, n)[:self.nch].astype(np.float32)
        msk = read_win(TGT[r]["mask"][None], x0, y0, n)[0]
        o = TGT[r]["objs"]; o = o[(o[:, 0] > x0 - 50) & (o[:, 0] < x0 + n + 50) & (o[:, 1] > y0 - 50) & (o[:, 1] < y0 + n + 50)].copy()
        o[:, 0] -= x0; o[:, 1] -= y0
        if n != S:
            img = F.interpolate(torch.from_numpy(img)[None], size=(S, S), mode="bilinear", align_corners=False)[0].numpy()
            msk = F.interpolate(torch.from_numpy(msk)[None, None].float(), size=(S, S), mode="nearest")[0, 0].numpy().astype(np.uint8)
            o[:, :3] *= S / n
        valid = img[0] > 0.5
        img = np.where(valid[None], img, 0)
        objs = [tuple(v) for v in o]
        if self.bank and rng.random() < CFG["p_paste"]:          # copy-paste курганов (альфа-круг)
            for _ in range(int(rng.integers(1, CFG["paste_max"] + 1))):
                p, d, c, px, py = self.bank[int(rng.integers(len(self.bank)))]; R = p.shape[1] // 2
                cx, cy = int(rng.integers(R, S - R)), int(rng.integers(R, S - R))
                if not valid[cy, cx] or msk[cy - R:cy + R + 1, cx - R:cx + R + 1].any(): continue
                if any((ox - cx) ** 2 + (oy - cy) ** 2 < (0.6 * (od + d)) ** 2 for ox, oy, od, _ in objs): continue
                yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; dist = np.sqrt((xx + R - px) ** 2 + (yy + R - py) ** 2)
                al = np.clip((R - dist) / max(1.0, 0.3 * R), 0, 1)[None] * (p[0:1] > 0) * valid[None, cy - R:cy + R + 1, cx - R:cx + R + 1]
                sl = (slice(None), slice(cy - R, cy + R + 1), slice(cx - R, cx + R + 1))
                img[sl] = al * p + (1 - al) * img[sl]
                objs.append((cx - R + px, cy - R + py, d, c))
        if rng.random() < CFG["p_height"]:                       # масштаб высоты ×0,5–2,2 в пространстве u8
            h = rng.uniform(*CFG["height"])
            for c in [c for c in (0, 1, 3, 4) if c < self.nch]: img[c] = 128 + (img[c] - 128) * h
            sd = np.deg2rad((img[2] - 1) / 254 * SLOPE_MAX); img[2] = 1 + np.rad2deg(np.arctan(h * np.tan(sd))) / SLOPE_MAX * 254
        if rng.random() < CFG["p_noise"]:                        # шум 2–5 см на lrm
            sm = rng.uniform(*CFG["noise_m"])
            for c in (0, 1): img[c] += rng.normal(0, sm * U8_PER_M[c], img[c].shape).astype(np.float32)
        img = np.where(valid[None], np.clip(img, 1, 255), 0)
        k, fl = int(rng.integers(4)), bool(rng.integers(2))         # D8
        def d8(a): a = np.rot90(a, k, axes=(-2, -1)); return np.ascontiguousarray(a[..., ::-1] if fl else a)
        m3 = np.stack([(msk & b) > 0 for b in (1, 2, 4)]).astype(np.uint8)
        ob = np.array(objs, np.float64).reshape(-1, 4)              # цели головы — в итоговой (после D8) системе кропа
        if len(ob): ob[:, :2] = HD.d8_points(ob[:, :2], S, k, fl)
        tgt = {kk: torch.from_numpy(np.ascontiguousarray(v)) for kk, v in HEAD.targets(ob, S).items()}
        return torch.from_numpy(d8(img).astype(np.uint8)), tgt, torch.from_numpy(d8(m3)), torch.from_numpy(d8(valid.astype(np.uint8)))

# ── Шаг 8. Модель, лосс, обучение ─────────────────────────────────────────────────────────────────────────────
import segmentation_models_pytorch as smp
def build_model(ch, init):
    return HEAD.build(ch, f"tu-{TIMM_NAME}" if init == "in" else "tu-convnext_tiny", True if init == "in" else None)
def norm(x): return (x.float() - 128.0) / 127.0
def mask_loss(logit, gt, valid):
    v = valid[:, None].float(); g = gt.float(); lg = logit.float()
    bce = (F.binary_cross_entropy_with_logits(lg, g, reduction="none") * v).sum() / (v.sum() * lg.shape[1]).clamp(min=1)
    p = torch.sigmoid(lg) * v; inter = (p * g).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + (g * v).sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return bce + dice.mean()

def _vram(dev):
    """Пик памяти устройства (МБ) с последнего reset_peak_memory_stats; на CPU — None. Только для лога/отчёта."""
    if dev.type != "cuda": return dict(max_memory_allocated_mb=None, max_memory_reserved_mb=None)
    return dict(max_memory_allocated_mb=round(torch.cuda.max_memory_allocated(dev) / 2**20, 1),
                max_memory_reserved_mb=round(torch.cuda.max_memory_reserved(dev) / 2**20, 1))

def train_job(job, dev, lane_left_iters, log_j):
    torch.manual_seed(CFG["seed"] + job["fold"]); train, test = split(job["fold"], POOL)
    model = build_model(job["ch"], job["init"]).to(dev).to(memory_format=torch.channels_last)
    msd = model.state_dict(); ema = {k: v.detach().clone().float() for k, v in msd.items()}
    fk = [k for k, v in msd.items() if v.dtype.is_floating_point]; ok_ = [k for k in msd if k not in fk]
    e_l, m_l = [ema[k] for k in fk], [msd[k] for k in fk]
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    iters = CFG["iters"]; wu = min(CFG["warmup"], max(1, iters // 10))
    lr_at = lambda it, n: CFG["lr"] * (it + 1) / wu if it < wu else CFG["lr"] * 0.5 * (1 + math.cos(math.pi * (it - wu) / max(1, n - wu)))
    use_amp = CFG["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ds = WinDS(train, job["ch"], CFG["seed"] * 1000 + job["fold"] * 10 + job["ch"], 10 ** 8)
    nw = CFG["workers"] if dev.type == "cuda" else 0
    dl = torch.utils.data.DataLoader(ds, batch_size=CFG["bs"], shuffle=False, num_workers=nw, pin_memory=dev.type == "cuda",
                                     persistent_workers=nw > 0, prefetch_factor=4 if nw else None, drop_last=True,
                                     worker_init_fn=lambda _: torch.set_num_threads(1))
    log_j(f"train: {len(train)} участков, объектов {sum(len(TGT[r]['samp']) for r in train)}, банк copy-paste {len(ds.bank)}; test {test}")
    hist, t0, it, s_it = [], time.time(), 0, None; model.train()
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)   # только счётчик пика памяти (лог/отчёт)
    dw_all = dw_warm = 0.0; t_end = t_warm0 = t0                   # ожидание батча загрузчика: от конца прошлой ит. до получения батча (только лог/отчёт)
    b_warm, b_it = int(CFG["budget_warm_it"]), int(CFG["budget_it"]); dts, t_prev = [], t0   # с/ит для бюджета — после прогрева загрузчиков/cudnn
    for x, tgt, m3, val in dl:
        t_got = time.time(); dw_all += t_got - t_end
        if it > b_warm: dw_warm += t_got - t_end
        if it >= iters: break
        x = norm(x.to(dev, non_blocking=True)).contiguous(memory_format=torch.channels_last)
        tgt = {k: v.to(dev, non_blocking=True) for k, v in tgt.items()}
        m3, val = m3.to(dev, non_blocking=True), val.to(dev, non_blocking=True)
        for g in opt.param_groups: g["lr"] = lr_at(it, iters)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
        lh, parts = HEAD.loss(out, tgt, val); lm = mask_loss(out["mask"], m3, val); loss = lh + lm
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        with torch.no_grad():
            dcy = min(CFG["ema"], (1 + it) / (10 + it))
            torch._foreach_mul_(e_l, dcy); torch._foreach_add_(e_l, [v.float() for v in m_l], alpha=1 - dcy)
            for k in ok_: ema[k].copy_(msd[k])
        it += 1
        if not math.isfinite(loss.item()): raise RuntimeError(f"loss не конечен на ит. {it}")   # .item() синхронизирует → dt честный
        t_now = time.time()
        if it > b_warm: dts.append(t_now - t_prev)
        t_prev = t_now
        if it == b_it and iters > b_it:                    # бюджет полосы под дедлайн: медиана с/ит ит. b_warm+1..b_it (было: среднее 1..50 с прогревом)
            s_it = float(np.median(dts)); log_j(f"с/ит: медиана ит. {b_warm + 1}–{b_it} {s_it:.3f}, среднее 1–{it} {(t_now - t0) / it:.3f}")
            left = DEADLINE - time.time() - CFG["infer_reserve_h"] * 3600 * (1 + lane_left_iters // max(1, CFG["iters"]))
            need_s = s_it * ((iters - it) + lane_left_iters)
            if need_s > left:
                new = max(int(CFG["min_iter_frac"] * CFG["iters"]), int(iters * left / max(need_s, 1)))
                log_j(f"БЮДЖЕТ: {s_it:.3f} с/ит, нужно {need_s / 3600:.2f} ч > есть {left / 3600:.2f} ч → итераций {iters} → {new}")
                iters = new
            else: log_j(f"бюджет OK: {s_it:.3f} с/ит, прогноз обучения {s_it * iters / 3600:.2f} ч")
        if it % CFG["log_every"] == 0 or it == iters or it <= 2:
            hist.append(dict(it=it, loss=round(loss.item(), 4), head=round(lh.item(), 4), mask=round(lm.item(), 4), **{k: round(v, 4) for k, v in parts.items()},
                             lr=round(opt.param_groups[0]["lr"], 7), t=round(time.time() - t0, 1)))
            log_j(f"ит {it}/{iters} loss {loss.item():.4f} (head {lh.item():.4f} {parts} mask {lm.item():.4f})")
        if it == b_warm: t_warm0 = time.time()
        t_end = time.time()
    t_train = time.time() - t0
    mem = _vram(dev)
    dwi = dict(data_wait_s=round(dw_all, 1), data_wait_share=round(dw_all / max(t_train, 1e-9), 4),
               data_wait_share_after_warm=(round(dw_warm / max(t_end - t_warm0, 1e-9), 4) if it > b_warm else None))
    log_j(f"память/загрузчик (обучение): {json.dumps(dict(**mem, **dwi))}")
    if s_it is None: s_it = t_train / max(1, it)
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()}); model.eval()
    del dl
    return model, dict(iters_planned=CFG["iters"], iters_effective=it, s_per_it=round(s_it, 4), t_train_s=round(t_train, 1),
                       bank=len(ds.bank), hist=hist, train_regions=train, test_regions=test, vram_train=mem, **dwi)

# ── Шаг 9. Инференс (D8 TTA), декодирование пиков, OOF ────────────────────────────────────────────────────────
@torch.no_grad()
def predict_region(model, region, ch, dev):
    """→ {"mask": (3,H,W) вероятности на 0,5 м, <ключи головы>: (K,ceil(H/s),ceil(W/s)) — prob_keys sigmoid, прочие сырые}."""
    a = _rel(region); C, H, W = a.shape; S, st, s = CFG["crop"], CFG["infer_stride"], HEAD.stride
    h4, w4, n4 = -(-H // s), -(-W // s), S // s
    tiles = [t for t in HD.tile_origins(H, W, S, st, s) if (a[0, t[0]:t[0] + S:8, t[1]:t[1] + S:8] > 0).any()]
    acc, cnt, cnt4 = {}, np.zeros((H, W), np.float32), np.zeros((h4, w4), np.float32)
    use_amp = dev.type == "cuda"; tf = [(k, f) for f in (False, True) for k in range(4)][:max(1, CFG["tta"])]
    for b in range(0, len(tiles), CFG["infer_bs"]):
        bt = tiles[b:b + CFG["infer_bs"]]
        x = torch.from_numpy(np.stack([read_win(a, xx, yy, S)[:ch] for yy, xx in bt])).to(dev)
        x = norm(x); o = {}
        for k, fl in tf:                 # TTA: среднее логитов (hm, mask) / сырых wh, off — в исходной системе
            xi = torch.rot90(x, k, (2, 3)); xi = xi.flip(3) if fl else xi
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                yo = model(xi.contiguous(memory_format=torch.channels_last))
            yo = {kk: v.float() for kk, v in yo.items() if v is not None}
            ym = HD.untta_map(yo.pop("mask"), k, fl); yo = HEAD.untta(yo, k, fl); yo["mask"] = ym
            for kk, v in yo.items(): o[kk] = o[kk] + v if kk in o else v
        for kk, v in o.items():
            v = (v / len(tf)).cpu().numpy(); full = kk == "mask"; Hh, Ww, n, d = (H, W, S, 1) if full else (h4, w4, n4, s)
            if kk not in acc: acc[kk] = np.zeros((v.shape[1], Hh, Ww), np.float32)
            for (yy, xx), oi in zip(bt, v):
                y0, x0 = yy // d, xx // d; hh, ww = min(n, Hh - y0), min(n, Ww - x0); acc[kk][:, y0:y0 + hh, x0:x0 + ww] += oi[:, :hh, :ww]
        for yy, xx in bt:
            hh, ww = min(S, H - yy), min(S, W - xx); cnt[yy:yy + hh, xx:xx + ww] += 1
            y0, x0 = yy // s, xx // s; hh, ww = min(n4, h4 - y0), min(n4, w4 - x0); cnt4[y0:y0 + hh, x0:x0 + ww] += 1
    va = (a[0] > 0) & (cnt > 0); va4 = HD.block_any(va, s) & (cnt4 > 0)
    for kk in acc:
        full = kk == "mask"; c_, v_ = (cnt, va) if full else (cnt4, va4)
        acc[kk] /= np.maximum(c_, 1)[None]
        if full or kk in HEAD.prob_keys: acc[kk] = (1 / (1 + np.exp(-acc[kk]))) * v_[None]
    acc["_valid"], acc["_valid_h"] = va, va4
    return acc

def save_maps(maps, region, tag, fold, ch):
    """OOF-карты для реранкера bot-2 (без утечки: участок — тест фолда `fold`). mask → uint8 (p·255); hm (prob) → float32; wh/off (сырые) → float16."""
    m = META[region]; d = {}
    for kk, v in maps.items():
        if kk.startswith("_"): continue
        d[kk] = np.round(v * 255).astype(np.uint8) if kk == "mask" else v.astype(np.float32 if kk in HEAD.prob_keys else np.float16)   # hm — f32: u8/f16 дают плато → лишние пики
    meta = dict(region=region, fold=fold, family=FAM[region], tag=tag, head=HEAD.name, head_api=HD.HEAD_API, stride=HEAD.stride,
                left=m["left"], top=m["top"], step=m["step"], step_head=m["step"] * HEAD.stride, crs=m["crs"],
                H=int(maps["mask"].shape[1]), W=int(maps["mask"].shape[2]), tags=list(TAGS5[:ch]),
                enc={k: ("u8/255 prob" if k == "mask" else ("f32 prob" if k in HEAD.prob_keys else "f16 raw")) for k in d},
                keys={"hm": list(HD.HM_CLASSES), "mask": list(HD.MASK_CLASSES)})
    p = OUT / "maps" / f"maps_cnx_{tag}_{region}.npz"; p.parent.mkdir(exist_ok=True)
    np.savez_compressed(p, valid=maps["_valid"].astype(np.uint8), valid_h=maps["_valid_h"].astype(np.uint8),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)), **d)
    return p.stat().st_size

def decode(maps, region):
    from scipy import ndimage as ndi
    m = META[region]; st = m["step"]; dets = []; mk = maps["mask"]
    gor = mk[0] > CFG["mask_thr"]
    for c, mn in ((2, CFG["gor_min_px"]), (3, CFG["fort_min_px"]), (4, CFG["arch_min_px"])):
        pc = mk[c - 2]; lab, n = ndi.label(pc > CFG["mask_thr"])
        if not n: continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]; conf = ndi.mean(pc, lab, np.arange(1, n + 1))
        boxes = ndi.find_objects(lab)
        for i in np.where(sizes >= mn)[0]:
            sl = boxes[i]; comp = lab[sl] == i + 1
            dt = ndi.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]; yy, xx = np.unravel_index(np.argmax(dt), dt.shape)
            dets.append((sl[1].start + xx, sl[0].start + yy, c, float(conf[i])))   # полюс недоступности (на скелете)
    dets = [(m["left"] + (x + 0.5) * st, m["top"] - (y + 0.5) * st, c, s) for x, y, c, s in dets]
    sup = HD.block_mean(gor, HEAD.stride) > 0.5                                  # клетка головы внутри городища
    t = HEAD.decode({k: v for k, v in maps.items() if k != "mask" and not k.startswith("_")}, m["left"], m["top"],
                    st * HEAD.stride, maps["_valid_h"], sup)                         # кандидаты курганов, метры родной СК
    for X, Y, c, s in zip(t["X"], t["Y"], t["cls"], t["score"]): dets.append((float(X), float(Y), int(c), float(s)))
    to3857 = Transformer.from_crs(m["crs"], "EPSG:3857", always_xy=True).transform
    feats = []
    for ux, uy, c, s in dets:
        g = stf(to3857, Point(ux, uy).buffer(CIRCLE_R, quad_segs=4))
        ring = [[round(a, 2), round(b, 2)] for a, b in g.exterior.coords]
        feats.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[ring]), properties=dict(
            class_name=CLASSES[c], confidence=round(min(1.0, max(0.0, s)), 5), region_name=region, crs="EPSG:3857", markup_type="li",
            threshold_group="li_cnx", original_crs=m["crs"])))   # своя группа порогов в стеке (не li стека)
    return feats

def gt_features(regions):
    out = []
    for r in regions:
        tr = Transformer.from_crs(META[r]["crs"], "EPSG:3857", always_xy=True).transform
        for g, c in TGT[r]["gts"]:
            out.append(dict(type="Feature", geometry=mapping(stf(tr, g)), properties=dict(
                class_name=CLASSES[c], region_name=r, markup_type="li", crs="EPSG:3857")))
    return out

def write_fc(path, feats):
    for i, f in enumerate(feats): f["properties"] = dict(fid=i, **{k: v for k, v in f["properties"].items() if k != "fid"})
    fc = dict(type="FeatureCollection", name=path.stem, crs=dict(type="name", properties=dict(name="urn:ogc:def:crs:EPSG::3857")),
              features=feats)
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")

def score(pred, gt):
    try:
        import logging; logging.getLogger("compute_metrics").setLevel(logging.ERROR)
        import compute_metrics as CM
        r = CM.F1Calculator().calculate_f1(pred, gt)
        return dict(final_score=r.get("final_score"), **{k: r[k] for k in ("per_class",) if k in r})
    except Exception as e: return dict(error=repr(e))

SCORE_THR_GRID = [round(0.05 * k, 2) for k in range(1, 20)]   # 0.05…0.95
def score_thr(pred, gt):
    """Только отчёт: балл li-only при пороге 0.3 и лучший порог/F1 по каждому классу (сетка 0.05…0.95, in-sample, оптимистично)
    + итог при этих порогах. OOF и GT не меняются; ошибка не роняет задание."""
    try:
        import logging; logging.getLogger("compute_metrics").setLevel(logging.ERROR)
        import compute_metrics as CM
        feats = pred["features"]; cls_present = sorted({f["properties"]["class_name"] for f in feats})
        def at(thr):
            keep = [f for f in feats if float(f["properties"]["confidence"]) >= thr.get(f["properties"]["class_name"], 1e9)]
            return CM.F1Calculator().calculate_f1(dict(pred, features=keep), gt)
        best, t_s = {}, time.time()
        for t in SCORE_THR_GRID:
            r = at({c: t for c in cls_present})
            for c, v in r.get("per_class", {}).items():
                if c not in cls_present or not v.get("num_ground_truth"): continue
                if c not in best or v["f1"] > best[c]["f1"]:
                    best[c] = dict(thr=t, f1=v["f1"], tp=v["tp"], fp=v["fp"], fn=v["fn"], num_predictions=v["num_predictions"], num_ground_truth=v["num_ground_truth"])
            if t == 0.3: r03 = r
        rb = at({c: best[c]["thr"] for c in best}) if best else {}
        return dict(grid=[SCORE_THR_GRID[0], SCORE_THR_GRID[-1], 0.05], note="in-sample по GT теста фолда — оптимистично; без порога см. score_li_only",
                    final_score_thr0_3=r03.get("final_score"),
                    per_class_thr0_3={c: {k: v[k] for k in ("f1", "tp", "fp", "fn")} for c, v in r03.get("per_class", {}).items() if v.get("num_predictions") or v.get("num_ground_truth")},
                    best_per_class=best, final_score_best_thr=rb.get("final_score"), t_s=round(time.time() - t_s, 1))
    except Exception as e: return dict(error=repr(e))

# ── Шаг 9б. Веса в формате контракта bot-8 (bot-9/CNX_CONTRACT_FOR_BOT8.md) — ДОПОЛНИТЕЛЬНО к ckpt_cnx_*.pt ────────
CONTRACT_VERSION = 1
RELIEF_SHA_ALL = hashlib.sha256(json.dumps(RELIEF_SHA, sort_keys=True).encode()).hexdigest()   # = relief_sha TPU build.py
def _prim(x):
    """Только примитивы/тензоры (torch.load(weights_only=True) без пиклинга классов)."""
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, dict): return {str(k): _prim(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_prim(v) for v in x]
    if isinstance(x, (bool, int, float, str)) or x is None: return x
    if isinstance(x, np.generic): return x.item()
    raise TypeError(f"контракт: не примитив {type(x)}")
def contract_dict(model, sd, job):
    """→ dict контракта: state_dict fp16 + arch {library,name,kwargs} (сборка офлайн, без скачивания весов) + вход/выход/пороги."""
    ch = int(job["ch"]); enc = "tu-convnext_tiny"          # «in» = тот же граф, веса ImageNet только для инициализации
    lib = f"segmentation_models_pytorch=={_ver('segmentation_models_pytorch')}; timm=={_ver('timm')}"
    if HEAD.name == "centerhead":
        arch = dict(library=lib + "; bot-9 src/cnx_contract.py (= bot-2 centerhead.head.UnetCenterNet @" + VMAN.get("centerhead_commit", "?")[:12] + ")",
                    name="cnx_contract.CnxUnetCenter", kwargs=dict(encoder_name=enc, encoder_weights=None, in_channels=ch, classes=3,
                    stride=int(model.stride), feat_block=None if model._block is None else int(model._block), feat_ch=int(model.feat_ch),
                    n_cls=2, head_ch=int(model.center.hm[0].out_channels)))
        names = ["hm_kurgany_tselye", "hm_kurgany_povrezhdennye", "wh_log_r_px", "off_x", "off_y", "mask_gorodishcha", "mask_fortifikatsii", "mask_arkhitektury"]
        s_ = int(model.stride); strides = [s_] * 5 + [1] * 3; act = ["sigmoid", "sigmoid", "none", "none", "none", "sigmoid", "sigmoid", "sigmoid"]
        sdc = sd
    elif HEAD.name == "heat":                              # Dec2Dict(smp.Unet classes=5): ключи без префикса unet. = чистый smp.Unet
        arch = dict(library=lib, name="segmentation_models_pytorch.Unet", kwargs=dict(encoder_name=enc, encoder_weights=None, in_channels=ch, classes=5))
        names = ["hm_kurgany_tselye", "hm_kurgany_povrezhdennye", "mask_gorodishcha", "mask_fortifikatsii", "mask_arkhitektury"]
        strides = [1] * 5; act = ["sigmoid"] * 5
        sdc = {k[len("unet."):]: v for k, v in sd.items() if k.startswith("unet.")}
    else:
        raise TypeError(f"контракт: голова {HEAD.name} не описана")
    hc = getattr(HEAD, "cfg", {}) or {}
    thr = dict(min_score=CFG["min_score"], peak_k=CFG["peak_k"], mask_thr=CFG["mask_thr"], gor_min_px=CFG["gor_min_px"],
               fort_min_px=CFG["fort_min_px"], arch_min_px=CFG["arch_min_px"], max_det=CFG["max_det"], circle_r_m=CIRCLE_R,
               **{f"head_{k}": hc[k] for k in ("min_score", "peak_pool", "topk", "nms_min_m", "nms_frac", "nms_cross_class", "size_mode") if k in hc},
               note="пороги декодирования, которыми получены oof_cnx_*.geojson; НЕ подобраны по OOF (подбор — вне ядра)")
    return _prim(dict(
        format="koz3-weights-contract-bot8", contract_version=CONTRACT_VERSION,
        state_dict={k: v.detach().cpu().contiguous() for k, v in sdc.items()}, arch=arch,
        in_kind=f"cnx{ch}", in_channels=ch, in_tags=list(TAGS5[:ch]), in_norm="x=(u8-128)/127; u8 0 = вне съёмки (→ -1.0079); каналов valid нет",
        n_out=len(names), out_names=names, out_stride=strides, out_act=act, out_logits=True,
        out_format="forward(x)→dict hm(B,2,H/s,W/s) wh(B,1,..) off(B,2,..) mask(B,3,H,W); out_names = cat(hm,wh,off,mask) по каналам" if HEAD.name == "centerhead" else "forward(x)→(B,5,H,W) логиты",
        cell_m=float(CFG["res"]), prep_code_sha=RELIEF_SHA_ALL, prep_code_files=RELIEF_SHA,
        thresholds=thr, oof_thresholds=None,
        infer=dict(tile=CFG["crop"], stride=CFG["infer_stride"], tta=CFG["tta"], tta_avg="logits", autocast="float16", pad="u8 0 (вне растра)", hw_multiple=32),
        train=dict(variant=job["variant"], fold=int(job["fold"]), init=job["init"], head=HEAD.name, head_api=HD.HEAD_API,
                   timm_pretrained=TIMM_NAME if job["init"] == "in" else None, centerhead_commit=VMAN.get("centerhead_commit")),
        versions={m: _ver(m) for m in ("torch", "timm", "segmentation_models_pytorch")}))
def save_contract(model, sd, job, tag):
    p = OUT / f"weights_cnx_{tag}.pt"; torch.save(contract_dict(model, sd, job), p)
    c = torch.load(p, map_location="cpu", weights_only=True)          # проверка формата сразу после записи
    assert c["n_out"] == len(c["out_names"]) and len(c["state_dict"]) > 0
    return p.name

# ── Шаг 10. Полосы GPU: задания по очереди, отчёт ─────────────────────────────────────────────────────────────
REPORT = dict(cfg=CFG, jobs=[], vendor=VMAN, head=dict(name=HEAD.name, api=HD.HEAD_API, out_keys=list(HEAD.out_keys),
              cfg=getattr(HEAD, "cfg", None)), timm=TIMM_NAME, imagenet_sha=WSHA,
              note="ImageNet-варианты (*-in) — только измерение, не в сдачу до письма организаторов о весах",
              versions={m: _ver(m) for m in ("torch", "torchvision", "numpy", *PINS, "rvt")}, gpus=[str(d) for d in DEVS])
LOCK = threading.Lock(); ERR = []
def lane(li, dev, jobs):
    for n, job in enumerate(jobs):
        tag = f"{job['variant']}_f{job['fold']}"; lj = lambda *a: log(f"[{dev} {tag}]", *a)
        try:
            left = sum(CFG["iters"] for _ in jobs[n + 1:])
            model, info = train_job(job, dev, left, lj); t = time.time(); feats = []
            for r in info["test_regions"]:
                prob = predict_region(model, r, job["ch"], dev); fr = decode(prob, r); feats += fr
                mb = save_maps(prob, r, tag, job["fold"], job["ch"]) if SAVE_MAPS else 0; info.setdefault("maps_bytes", {})[r] = mb
                lj(f"инференс {r}: {len(fr)} детекций, карты {mb / 1e6:.1f} МБ"); del prob
            info["t_infer_s"] = round(time.time() - t, 1); info["n_det"] = len(feats)
            p = OUT / f"oof_cnx_{tag}.geojson"; write_fc(p, feats)
            gtf = gt_features(info["test_regions"]); write_fc(OUT / f"gt_li_cnx_{tag}.geojson", [dict(f) for f in gtf])
            info["score_li_only"] = score(json.loads(p.read_text(encoding="utf-8")), dict(type="FeatureCollection", features=gtf))
            info["score_li_only_thr"] = score_thr(json.loads(p.read_text(encoding="utf-8")), dict(type="FeatureCollection", features=gtf))
            try: info["vram_total"] = _vram(dev)   # пик с начала задания (обучение + инференс)
            except Exception as e: info["vram_total"] = dict(error=repr(e))
            sd = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in model.state_dict().items()}
            torch.save(dict(state_dict=sd, head=HEAD.name, head_api=HD.HEAD_API, job=job, tags=list(TAGS5[:job["ch"]]), timm=TIMM_NAME, cfg=CFG), OUT / f"ckpt_cnx_{tag}.pt")
            try: info["weights_contract"] = save_contract(model, sd, job, tag)
            except Exception as e: info["weights_contract_error"] = repr(e); lj("контракт весов bot-8: ОШИБКА (задание продолжается)", repr(e))
            lj("готово:", json.dumps({k: info[k] for k in ("iters_effective", "s_per_it", "t_train_s", "t_infer_s", "n_det")}),
               "score(Li-only, compute_metrics)", info["score_li_only"].get("final_score"))
            try:   # только лог; .get — у TPU-сборки (tpu_train) нет vram_train/data_wait_*
                st = info.get("score_li_only_thr") or {}
                lj("score(Li-only) при пороге 0.3:", st.get("final_score_thr0_3"), "| лучший порог по классам:",
                   json.dumps({c: [v["thr"], round(v["f1"], 4)] for c, v in (st.get("best_per_class") or {}).items()}),
                   "| итог при лучших:", st.get("final_score_best_thr"), "| ошибка:" if "error" in st else "", st.get("error", ""))
                lj("VRAM:", json.dumps(dict(train=info.get("vram_train"), total=info.get("vram_total"))), "| ожидание данных:",
                   info.get("data_wait_share"), "после прогрева", info.get("data_wait_share_after_warm"))
            except Exception as e: lj("лог порогов/VRAM: ошибка (задание продолжается)", repr(e))
            with LOCK: REPORT["jobs"].append(dict(job=job, **info))
            del model
            if dev.type == "cuda": torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc(); lj("ОШИБКА", repr(e))
            with LOCK: ERR.append((tag, repr(e))); REPORT["jobs"].append(dict(job=job, error=repr(e)))
(OUT / "region_fold.json").write_text(json.dumps(dict(
    note="фолд CNN, где участок был тестом (семейные фолды folds_family_k5, LOFO); для OOFReranker(cnn_fold_of=...)",
    region_fold={r: FOLD[r] for r in sorted(FAM)}, region_family={r: FAM[r] for r in sorted(FAM)},
    jobs=[dict(variant=j["variant"], fold=j["fold"]) for j in JOBS]), ensure_ascii=False, indent=1), encoding="utf-8")
lanes = [[] for _ in DEVS]
for i, j in enumerate(JOBS): lanes[i % len(DEVS)].append(j)
log("полосы:", [[f"{j['variant']}:{j['fold']}" for j in l] for l in lanes])
ths = [threading.Thread(target=lane, args=(i, d, l)) for i, (d, l) in enumerate(zip(DEVS, lanes)) if l]
for t in ths: t.start()
for t in ths: t.join()
# объединённые OOF по вариантам (внутри ядра; между ядрами — src/merge_oof.py на боксе)
for v in sorted({j["variant"] for j in JOBS}):
    feats = []
    for j in JOBS:
        p = OUT / f"oof_cnx_{v}_f{j['fold']}.geojson"
        if j["variant"] == v and p.exists(): feats += json.loads(p.read_text(encoding="utf-8"))["features"]
    write_fc(OUT / f"oof_cnx_{v}.geojson", feats)
REPORT["t_total_s"] = round(time.time() - T0, 1); REPORT["errors"] = ERR
(OUT / f"report_cnx_{MODE}.json").write_text(json.dumps(REPORT, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
shutil.rmtree(RELIEF_DIR, ignore_errors=True) if MODE == "full" else None
log("ИТОГ:", "ошибки " + str(ERR) if ERR else "все задания OK", f"за {REPORT['t_total_s']} с")

# ── Шаг 11 (только smoke, опционально). GPU-бенч смешивания bot-8 (чистая ветка bot8-cpu-a-clean, bench/acc_gpu_T4.md) ──
# host vs device_exact vs device: время, byte_equal device_exact, max|Δ| device, пик VRAM → OUT/bench_acc_gpu_t4.json.
# Любая ошибка/таймаут бенча — только запись в JSON и лог; код выхода smoke определяется заданиями выше.
BENCH_RUNS = [("stub_30M_1", ["--px", "5480", "5480", "--net", "stub", "--tta", "1"]),
              ("unet_30M_1", ["--px", "5480", "5480", "--net", "unet", "--tta", "1", "--reps", "1"]),
              ("stub_30M_d8", ["--px", "5480", "5480", "--net", "stub", "--tta", "d8"]),
              ("stub_75M_1", ["--px", "8660", "8660", "--net", "stub", "--tta", "1"])]
def bench_acc_gpu(out_json, cap_s):
    import gc, tempfile
    t0 = time.time(); res = dict(note="bot-8 bench_acc_gpu.py, чистая ветка (bench/acc_gpu_T4.md); встроен в koz3-cnx-li-smoke", runs={})
    def dump():
        res["t_total_s"] = round(time.time() - t0, 1); Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    try:
        d = Path(tempfile.mkdtemp(prefix="b8bench_"))
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BENCH_B64)), mode="r:gz") as tf:
            try: tf.extractall(d, filter="data")
            except TypeError: tf.extractall(d)          # Python без tarfile filter
        man = json.load(open(d / "bench_manifest.json", encoding="utf-8")); res["source"] = man
        for n, h in man["sha256"].items():
            if hashlib.sha256((d / n).read_bytes()).hexdigest() != h: raise RuntimeError(f"бенч: sha256 {n} не совпал")
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if res["gpu"] is None: res["skipped"] = "нет CUDA"; dump(); return res
        clk = None
        try: clk = subprocess.Popen(["nvidia-smi", "--query-gpu=index,name,clocks.sm,power.draw,memory.used", "--format=csv", "-lms", "500"],
                                    stdout=open(Path(out_json).with_name("bench_acc_gpu_t4_clocks.csv"), "w"), stderr=subprocess.DEVNULL)
        except Exception as e: res["clocks_error"] = repr(e)
        try:
            for name, args in BENCH_RUNS:
                left = min(cap_s - (time.time() - t0), DEADLINE - time.time() - 300)
                if left < 60: res["runs"][name] = dict(skipped=f"нет времени ({left:.0f} с)"); continue
                oj = d / f"{name}.json"; cmd = [sys.executable, str(d / "bench/bench_acc_gpu.py"), "--repo", str(d / "repo"), "--dev", "cuda", *args, "--out", str(oj)]
                r = dict(args=args, timeout_s=round(left)); t1 = time.time()
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True, timeout=left, env={**os.environ, "PYTHONHASHSEED": "0"})
                    r.update(rc=p.returncode, stdout_tail=p.stdout[-3000:], stderr_tail=p.stderr[-3000:])
                except subprocess.TimeoutExpired as e: r.update(rc=None, error="timeout")
                r["wall_s"] = round(time.time() - t1, 1)
                if oj.exists():
                    j = json.loads(oj.read_text(encoding="utf-8")); r["result"] = j; md = j.get("modes", {})
                    r["summary"] = {m: {k: md[m].get(k) for k in ("wall_med", "cpu_med", "peak_vram_mb", "byte_equal", "max_abs", "px_diff", "flips", "x_wall", "peaks_only_new", "peaks_only_ref")} for m in md}
                res["runs"][name] = r; log("бенч bot-8", name, "rc", r.get("rc"), json.dumps(r.get("summary"), default=str)[:600]); dump()
        finally:
            if clk is not None: clk.terminate()
    except Exception as e:
        import traceback; res["error"] = repr(e); res["traceback"] = traceback.format_exc()[-3000:]
    dump(); return res
if MODE == "smoke" and globals().get("BENCH_B64") and os.environ.get("CNX_BENCH", "1") == "1":
    try:
        _b = bench_acc_gpu(OUT / "bench_acc_gpu_t4.json", float(os.environ.get("CNX_BENCH_MAX_S", 1800)))
        log("бенч bot-8: записан", OUT / "bench_acc_gpu_t4.json", "ошибка" if _b.get("error") else "OK", f"{_b.get('t_total_s')} с")
    except BaseException as e: log("бенч bot-8: пропущен (не влияет на код выхода)", repr(e))
if ERR: raise SystemExit(1)
