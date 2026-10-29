# KOZ3 · dfine_spor (bot-9): детектор рамок D-FINE-L на спутнике SpOr, 0,5 м/пикс, окна 640 px, 5 семейных фолдов.
# Рецепт: /workspace/koz3_shared/bot-5/NOLIDAR.md §6.1 (гиперпараметры — факты из их таблицы; код конкурентов не копирован).
# Веса: ustc-community/dfine-large-coco (HF, Apache-2.0), ревизия закреплена. ТОЛЬКО обучить и измерить — в сдачу
# лишь после письма организаторов о весах. Режимы: DFINE_MODE = smoke (10 ит., мини-данные) | full.
# Ключи S3: Kaggle Secrets KOZ3_S3_* → env → /kaggle/input/**/cred.json (датасет dmitrijkv/koz3-s3-cred, поля id/key).
# Значения ключей никогда не печатаются и не пишутся в /kaggle/working.
# ТОЧКА ПОДКЛЮЧЕНИЯ ТРЕНЕРА БОТА 5: если найден файл dfine_train_bot5.py (рядом со скриптом или в /kaggle/input/**),
# из него берётся функция train(model, batches, cfg, device, log) -> model вместо встроенной train_loop (тот же контракт).
# DFINE_TRAINER = bot5 (по умолчанию: тренер бота 5 @0256251, встроенная копия dfine_train_bot5.py) | builtin (train_loop ниже).
# DFINE_AUG = ours (по умолчанию, §6.1 с QAUG) | bot5 (jitter() бота 5 вместо наших аугментаций; сдвиг окна ±1/3 — наш).
# Фикс 28.10 (Kaggle smoke v1: «Cannot copy out of meta tensor»): from_pretrained в transformers 5 подменяет
# PreTrainedModel.tie_weights глобально (init.no_tie_weights) — два потока gpu_worker грузили модель одновременно, привязка
# class_embed/bbox_embed ↔ model.decoder.* пропускалась и 48 тензоров оставались на meta. Теперь загрузка под замком,
# с проверкой meta/формы головы/привязки до .to(), и пробная загрузка + forward на CPU до построения чипов.
import os, sys, subprocess, importlib
PINS = {"transformers": "transformers==5.21.0", "rasterio": "rasterio==1.4.3", "pyproj": "pyproj==3.7.1",
        "shapely": "shapely==2.1.1", "boto3": "boto3==1.40.61", "cv2": "opencv-python-headless==4.11.0.86", "scipy": "scipy==1.16.3"}
HF_REPO, HF_REV = "ustc-community/dfine-large-coco", "aa8f8211545e4cff5adf52ea512f9ac74a9824ec"
def _deps():
    need = []
    for mod, pin in PINS.items():
        try:
            m = importlib.import_module(mod)
            if mod == "transformers" and m.__version__ != pin.split("==")[1]: need.append(pin)  # D-FINE API — строго пин
        except Exception: need.append(pin)
    if need:
        print("pip install:", need, flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", *need])
        if r.returncode: raise SystemExit(f"pip install не удался ({need}); нужен enable_internet=true")
        for k in [k for k in sys.modules if k.split(".")[0] in ("transformers", "huggingface_hub", "tokenizers")]: del sys.modules[k]
        importlib.invalidate_caches()
    bad = []
    for mod in ["torch", "torchvision", "huggingface_hub", "safetensors", *PINS]:
        try: m = importlib.import_module(mod); print(f"  {mod} {getattr(m, '__version__', '?')}", flush=True)
        except Exception as e: bad.append(f"{mod}: {e!r}")
    try: from transformers import DFineForObjectDetection  # noqa
    except Exception as e: bad.append(f"DFineForObjectDetection: {e!r}")
    try:
        import transformers as _t
        if _t.__version__ != PINS["transformers"].split("==")[1]: bad.append(f"transformers {_t.__version__} != пин {PINS['transformers']}")
    except Exception as e: bad.append(f"transformers: {e!r}")
    if bad: raise SystemExit("импорт-чек не пройден: " + "; ".join(bad))
_deps()
import json, time, re, math, glob, random, threading, argparse, collections, concurrent.futures as cf
from pathlib import Path
import numpy as np, torch

MODE = os.environ.get("DFINE_MODE", DFINE_MODE_DEFAULT)
AP = argparse.ArgumentParser()
AP.add_argument("--allow-cpu", action="store_true")
ARGS, _ = AP.parse_known_args()
E = lambda k, d: type(d)(os.environ.get(k, d))
CFG = dict(mode=MODE, res=0.5, tile=E("DFINE_TILE", 640), overlap=E("DFINE_OVERLAP", 160), max_windows=E("DFINE_MAX_WINDOWS", 0), bs=E("DFINE_BS", 8),
           iters=E("DFINE_ITERS", 10 if MODE == "smoke" else 8000), lr=1e-4, bb_mult=0.1, wd=1e-4, pct=0.05, clip=0.1,
           num_dn=E("DFINE_NUM_DN", -1), obj_frac=0.6, cls_cap=4.0, shift=1 / 3, qaug=0.5, workers=E("DFINE_WORKERS", 2),
           max_obj_chips_region=400, min_score=0.05, nms_iou=0.5, max_det=100,
           smoke_train_regions=E("DFINE_SMOKE_TRAIN_REGIONS", 8), smoke_test_regions=E("DFINE_SMOKE_TEST_REGIONS", 1),
           deadline_h=E("DFINE_DEADLINE_H", 11.3), folds=os.environ.get("DFINE_FOLDS", "0,1,2,3,4"),
           regions=os.environ.get("DFINE_REGIONS", ""), trainer=os.environ.get("DFINE_TRAINER", "bot5"),
           aug=os.environ.get("DFINE_AUG", "ours"))
assert CFG["trainer"] in ("bot5", "builtin") and CFG["aug"] in ("ours", "bot5"), "DFINE_TRAINER=bot5|builtin, DFINE_AUG=ours|bot5"
OUT = Path(os.environ.get("DFINE_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("DFINE_CACHE", "/tmp/s3cache")); CHIPS = Path(os.environ.get("DFINE_CHIPS", "/tmp/chips"))
BUCKET, ENDPOINT = "train-expds-3", os.environ.get("DFINE_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
RU2CLS = {"курганы_целые": 0, "курганы_поврежденные": 1, "городища": 2, "фортификации": 3, "архитектуры": 4}
T0 = time.time(); DEADLINE = T0 + CFG["deadline_h"] * 3600
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
log("CFG", json.dumps(CFG, ensure_ascii=False))

# ── Шаг 1. --require-gpu ─────────────────────────────────────────────────────────────────────────────────────
NG = torch.cuda.device_count() if torch.cuda.is_available() else 0
log("GPU:", [torch.cuda.get_device_name(i) for i in range(NG)])
if NG == 0 and not ARGS.allow_cpu:
    raise SystemExit("--require-gpu: GPU нет (сессия получила CPU — второй одновременный GPU-кернел?). Останов.")
DEVS = [torch.device(f"cuda:{i}") for i in range(NG)][:2] or [torch.device("cpu")]

# ── Шаг 2. Секреты → S3 (ранний фейл), веса HF (ранний фейл) ───────────────────────────────────────────────────
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
try:
    from huggingface_hub import snapshot_download
    CKPT = snapshot_download(HF_REPO, revision=HF_REV, allow_patterns=["*.json", "*.safetensors"])
except Exception as e:
    raise SystemExit(f"веса {HF_REPO}@{HF_REV[:8]} не скачались: {e!r}")
log("веса HF OK", HF_REPO, HF_REV[:8], "(Apache-2.0)")

# >>> MODEL (блок тестируется отдельно: kaggle/dfine_spor/fix/test_model_block.py)
from transformers import PreTrainedModel as _PTM
_PTM_ORIG = {n: _PTM.__dict__[n] for n in ("tie_weights", "init_weights") if n in _PTM.__dict__}  # снято в главном потоке до загрузок
MODEL_LOCK = threading.Lock()
def _meta_names(m):
    return [n for n, p in m.named_parameters(remove_duplicate=False) if p.is_meta] + \
           [n for n, b in m.named_buffers(remove_duplicate=False) if b.is_meta]
def _repair_meta(m, meta):
    # 1) повторная привязка весов исходной tie_weights (class_embed/bbox_embed → model.decoder.*)
    log(f"модель: на meta {len(meta)} тензоров ({', '.join(meta[:3])}…) — tie_weights()")
    _PTM_ORIG.get("tie_weights", type(m).tie_weights)(m)
    meta = _meta_names(m)
    if meta:  # 2) материализация оставшихся: to_empty + _init_weights модуля, затем снова привязка
        log(f"модель: после tie_weights на meta ещё {len(meta)} — to_empty + _init_weights")
        mods = {n.rsplit(".", 1)[0] if "." in n else "" for n in meta}
        for mn in sorted(mods):
            mod = m.get_submodule(mn); mod.to_empty(device="cpu", recurse=False); m._init_weights(mod)
        _PTM_ORIG.get("tie_weights", type(m).tie_weights)(m)
        meta = _meta_names(m)
    return meta
def check_model(m):
    """Жёсткие проверки ДО .to(device): нет meta, голова на 5 классов, привязка верхних голов к декодеру."""
    meta = _meta_names(m)
    if meta: raise RuntimeError(f"D-FINE: {len(meta)} параметров/буферов на meta ({', '.join(meta[:4])}…): "
                                "веса не материализованы (гонка from_pretrained/tie_weights?) — .to(device) упадёт")
    assert not any(p.is_meta for p in m.parameters()) and not any(b.is_meta for b in m.buffers())
    K, d = len(CLASSES), m.config.d_model; dec = m.model.decoder
    assert m.config.num_labels == K, f"num_labels {m.config.num_labels} != {K}"
    for i, l in enumerate(dec.class_embed):
        assert tuple(l.weight.shape) == (K, d) and tuple(l.bias.shape) == (K,), f"decoder.class_embed[{i}] {tuple(l.weight.shape)}"
    assert tuple(m.model.enc_score_head.weight.shape) == (K, d), f"enc_score_head {tuple(m.model.enc_score_head.weight.shape)}"
    if m.config.num_denoising > 0:
        assert m.model.denoising_class_embed.num_embeddings == K + 1, f"denoising_class_embed {m.model.denoising_class_embed.num_embeddings}"
    assert len(m.class_embed) == len(dec.class_embed) and len(m.bbox_embed) == len(dec.bbox_embed)
    for a, b in [(m.class_embed, dec.class_embed), (m.bbox_embed, dec.bbox_embed)]:
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert pa.data_ptr() == pb.data_ptr(), "class_embed/bbox_embed не привязаны к model.decoder.*"
    return {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "num_labels": K,
            "heads": [len(dec.class_embed), K, d]}
def make_model(seed=None):
    from transformers import DFineForObjectDetection, PreTrainedModel
    kw = dict(num_labels=len(CLASSES), id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)},
              ignore_mismatched_sizes=True, device_map=None)
    if CFG["num_dn"] >= 0: kw["num_denoising"] = CFG["num_dn"]
    # transformers 5: from_pretrained временно подменяет PreTrainedModel.tie_weights/init_weights на уровне класса —
    # это глобальное состояние процесса, поэтому загрузка строго по одной (замок) и с восстановлением оригиналов.
    with MODEL_LOCK:
        for n, f in _PTM_ORIG.items():
            if PreTrainedModel.__dict__.get(n) is not f:
                log(f"ВНИМАНИЕ: PreTrainedModel.{n} был подменён — восстанавливаю"); setattr(PreTrainedModel, n, f)
        if seed is not None: torch.manual_seed(seed)
        m = DFineForObjectDetection.from_pretrained(CKPT, **kw)
        meta = _meta_names(m)
        if meta:
            meta = _repair_meta(m, meta)
        info = check_model(m)
    return m, info
# <<< MODEL

# ── Шаг 2б. Ранний фейл модели: загрузка на CPU, проверки meta/головы/привязки, один forward — ДО чипов ───────────
_t = time.time()
try:
    _m, _info = make_model(seed=0); _m.eval()
    with torch.no_grad(): _o = _m(pixel_values=torch.rand(1, 3, 128, 128))
    assert tuple(_o.logits.shape[-1:]) == (len(CLASSES),) and _o.pred_boxes.shape[-1] == 4, f"logits {tuple(_o.logits.shape)}"
    _info["logits"] = list(_o.logits.shape); del _m, _o
except Exception as e:
    raise SystemExit(f"модель D-FINE не прошла ранний чек: {e!r}")
log("модель OK (CPU):", json.dumps(_info), f"{time.time() - _t:.0f} с")

# ── Шаг 3. Фолды по семьям (проверка) и инвентарь участков ─────────────────────────────────────────────────────
FAM = FOLDS["family_of"]; FOLD = {r: f["fold"] for f in FOLDS["folds"] for r in f["test_regions"]}
ff = collections.defaultdict(set)
for r, f in FOLD.items(): ff[FAM[r]].add(f)
assert not {k: v for k, v in ff.items() if len(v) > 1}, "семья в двух фолдах"
log("фолды OK:", len(FOLD), "участков,", len(ff), "семей")

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
    keys = list_prefix(region + "/"); tifs, marks, utm = [], [], None
    for k, s in keys:
        n = k.rsplit("/", 1)[-1]
        if n == "UTM.json": utm = (k, s)
        elif "_SpOr_" in k and n.lower().endswith((".tif", ".tiff")): tifs.append((k, s))
        elif n.endswith(".geojson") and "разметка" in k:
            m = re.search(r"_SpOr_(.+)\.geojson$", n)
            if m and m.group(1) in RU2CLS: marks.append((k, s, RU2CLS[m.group(1)]))
    return {"region": region, "tifs": sorted(tifs), "marks": marks, "utm": utm}

regions = [p.rstrip("/") for p in list_prefix("", "/")]
if CFG["regions"]: regions = [r for r in regions if r in CFG["regions"].split(",")]
with cf.ThreadPoolExecutor(16) as ex: INV = [i for i in ex.map(region_inv, regions) if i["tifs"]]
INV = [i for i in INV if i["region"] in FOLD] + [i for i in INV if i["region"] not in FOLD]
skipped = [i["region"] for i in INV if i["region"] not in FOLD]
INV = [i for i in INV if i["region"] in FOLD]
log(f"участков со SpOr в фолдах: {len(INV)}; вне фолдов (пропуск): {len(skipped)}")

def target_crs(inv, ds):
    if inv["utm"]:
        d = json.load(open(fetch(*inv["utm"]), encoding="utf-8")); m = re.search(r"EPSG::?(\d+)", d.get("crs", ""))
        if m: return f"EPSG:{m.group(1)}"
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(*ds.xy(ds.height // 2, ds.width // 2))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"

def load_gt(inv, crs):
    from shapely.geometry import shape
    from shapely.ops import transform as st
    from pyproj import Transformer
    out = []
    for k, s, c in inv["marks"]:
        d = json.load(open(fetch(k, s), encoding="utf-8"))
        n = ((d.get("crs") or {}).get("properties") or {}).get("name", ""); m = re.search(r"EPSG::?(\d+)", n)
        tr = Transformer.from_crs(f"EPSG:{m.group(1)}" if m else "OGC:CRS84", crs, always_xy=True).transform
        for f in d.get("features", []):
            g = f.get("geometry")
            if not g or g.get("type") not in ("Polygon", "MultiPolygon"): continue
            g = st(tr, shape(g)); g = g if g.is_valid else g.buffer(0)
            if not g.is_empty: out.append((g.bounds, c))
    return out

class Scene:
    """Растр SpOr, приведённый на лету к UTM 0,5 м (WarpedVRT; при уменьшении — average, иначе bilinear), RGB uint8."""
    def __init__(self, inv, key, size):
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform
        self.src = rasterio.open(fetch(key, size)); self.crs = target_crs(inv, self.src)
        tr, w, h = calculate_default_transform(self.src.crs, self.crs, self.src.width, self.src.height, *self.src.bounds,
                                               resolution=CFG["res"])
        try: src_m = abs(calculate_default_transform(self.src.crs, self.crs, self.src.width, self.src.height, *self.src.bounds)[0].a)
        except Exception: src_m = CFG["res"]
        rs = Resampling.average if src_m < CFG["res"] else Resampling.bilinear
        self.vrt = WarpedVRT(self.src, crs=self.crs, transform=tr, width=w, height=h, resampling=rs, nodata=0)
        self.W, self.H, self.T = w, h, tr; self.bands = list(range(1, min(3, self.src.count) + 1))
        self.stretch = None
        if self.src.dtypes[0] != "uint8":
            ov = self.src.read(1, out_shape=(min(1024, self.src.height), min(1024, self.src.width)))
            v = ov[ov > 0]; self.stretch = tuple(np.percentile(v, [1, 99])) if v.size else (0, 1)
    def read(self, x0, y0, n):
        from rasterio.windows import Window
        # WarpedVRT не умеет boundless — читаем пересечение и дополняем нулями вручную
        a = np.zeros((len(self.bands), n, n), dtype=self.src.dtypes[0])
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(self.W, x0 + n), min(self.H, y0 + n)
        if cx1 > cx0 and cy1 > cy0:
            a[:, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = self.vrt.read(self.bands, window=Window(cx0, cy0, cx1 - cx0, cy1 - cy0))
        if a.shape[0] < 3: a = np.repeat(a[:1], 3, 0)
        if self.stretch:
            lo, hi = self.stretch; a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return np.ascontiguousarray(a.astype(np.uint8).transpose(1, 2, 0))
    def to_px(self, b):
        inv = ~self.T; (x0, y1), (x1, y0) = inv * (b[0], b[1]), inv * (b[2], b[3])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    def close(self): self.vrt.close(); self.src.close()

# ── Шаг 4. Банк обучающих чипов (центр на объекте ±1/3 окна хранится как запас поля; фон — случайные окна) ──────────
BIG = int(round(CFG["tile"] * (1 + 2 * CFG["shift"])))
def build_chips(inv):
    import cv2
    rng = random.Random(hash(inv["region"]) & 0xFFFF); rec = []
    for key, size in inv["tifs"]:
        try:
            sc = Scene(inv, key, size); gt = load_gt(inv, sc.crs)
            boxes = [(sc.to_px(b), c) for b, c in gt]
            boxes = [(b, c) for b, c in boxes if b[2] > 0 and b[3] > 0 and b[0] < sc.W and b[1] < sc.H]
            objs = rng.sample(boxes, min(len(boxes), CFG["max_obj_chips_region"]))
            nb = max(2, int(len(objs) * (1 - CFG["obj_frac"]) / CFG["obj_frac"]))
            cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, c) for b, c in objs] + \
                    [(rng.uniform(0, sc.W), rng.uniform(0, sc.H), -1) for _ in range(nb)]
            for j, (cx, cy, c) in enumerate(cents):
                x0, y0 = int(cx - BIG / 2), int(cy - BIG / 2); img = sc.read(x0, y0, BIG)
                if (img.max(2) == 0).mean() > 0.5: continue
                bx = [[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0, cc] for b, cc in boxes
                      if b[2] > x0 and b[3] > y0 and b[0] < x0 + BIG and b[1] < y0 + BIG]
                p = CHIPS / f"{inv['region']}__{Path(key).stem}__{j}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                rec.append({"path": str(p), "region": inv["region"], "fold": FOLD[inv["region"]], "cls": c, "boxes": bx})
            sc.close()
        except Exception as e:
            log("чипы: ошибка", inv["region"], key.rsplit("/", 1)[-1], repr(e)[:200])
    return rec

# ── Шаг 5. Датасет с аугментациями (по рецепту §6.1) ─────────────────────────────────────────────────────────────
class ChipSet(torch.utils.data.IterableDataset):
    def __init__(self, recs, seed):
        self.recs, self.seed = recs, seed
        n = collections.Counter(r["cls"] for r in recs if r["cls"] >= 0); nmax = max(n.values()) if n else 1
        self.obj = [r for r in recs if r["cls"] >= 0]; self.bg = [r for r in recs if r["cls"] < 0]
        self.w = np.array([min(CFG["cls_cap"], math.sqrt(nmax / n[r["cls"]])) for r in self.obj], float)
        self.w = self.w / self.w.sum() if len(self.w) else self.w
    def __iter__(self):
        import cv2
        wi = torch.utils.data.get_worker_info(); rs = np.random.RandomState(self.seed + (wi.id if wi else 0))
        prng = random.Random(self.seed + (wi.id if wi else 0))
        T = CFG["tile"]
        while True:
            use_obj = self.obj and (not self.bg or rs.rand() < CFG["obj_frac"])
            r = self.obj[rs.choice(len(self.obj), p=self.w)] if use_obj else self.bg[rs.randint(len(self.bg))]
            img = cv2.imread(r["path"])[:, :, ::-1]; S = img.shape[0]
            ox, oy = rs.randint(0, S - T + 1), rs.randint(0, S - T + 1); img = img[oy:oy + T, ox:ox + T].astype(np.float32)
            bx = []
            for x0, y0, x1, y1, c in r["boxes"]:
                a = max(1e-6, (x1 - x0) * (y1 - y0)); cx0, cy0, cx1, cy1 = max(x0 - ox, 0), max(y0 - oy, 0), min(x1 - ox, T), min(y1 - oy, T)
                if cx1 > cx0 and cy1 > cy0 and (cx1 - cx0) * (cy1 - cy0) >= 0.5 * a: bx.append([cx0, cy0, cx1, cy1, c])
            bx = np.array(bx, np.float32).reshape(-1, 5)
            if CFG["aug"] == "bot5":  # jitter() бота 5 (xywh px), без --strong-aug, как в его dfine_folds.sh
                ji, jb = BOT5.jitter(img.astype(np.uint8), np.stack([bx[:, 0], bx[:, 1], bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]], 1), prng, False)
                img = ji.astype(np.float32); bx = np.concatenate([jb[:, :2], jb[:, :2] + jb[:, 2:4], bx[:, 4:5]], 1).astype(np.float32)
            else:
                img, bx = self._ours(img, bx, rs, T)
            img = np.ascontiguousarray(np.clip(img, 0, 255).transpose(2, 0, 1), dtype=np.float32) / 255.0
            cxcywh = np.stack([(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2, bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]], 1) / T
            yield torch.from_numpy(img), {"class_labels": torch.from_numpy(bx[:, 4].astype(np.int64)),
                                          "boxes": torch.from_numpy(np.clip(cxcywh, 0, 1).astype(np.float32))}
    @staticmethod
    def _ours(img, bx, rs, T):
            import cv2
            if rs.rand() < 0.3: img = cv2.GaussianBlur(img, (0, 0), rs.uniform(0.5, 1.5))
            img = img * rs.uniform(0.75, 1.3) * rs.uniform(0.85, 1.15, 3) + rs.uniform(-20, 20)
            img = np.clip(img, 0, 255)
            if rs.rand() < 0.3: img = 255 * (img / 255) ** rs.uniform(0.7, 1.4)
            if rs.rand() < 0.15: img = np.repeat(img.mean(2, keepdims=True), 3, 2)
            if rs.rand() < CFG["qaug"]:
                f = rs.uniform(1.5, 3.2); s = max(8, int(T / f))
                img = cv2.resize(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA), (T, T), interpolation=cv2.INTER_LINEAR)
                ok, enc = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rs.uniform(15, 60))])
                img = cv2.imdecode(enc, 1).astype(np.float32) + rs.normal(0, rs.uniform(0, 6), (T, T, 3))
            if rs.rand() < 0.5: img = img[:, ::-1]; bx[:, [0, 2]] = T - bx[:, [2, 0]]
            k = rs.randint(4)
            for _ in range(k):  # rot90 против часовой: (x, y) -> (y, T - x)
                img = np.rot90(img); bx = bx[:, [1, 2, 3, 0, 4]].copy(); bx[:, [1, 3]] = T - bx[:, [1, 3]]
                bx[:, [1, 3]] = np.sort(bx[:, [1, 3]], 1)
            return img, bx

def batches(recs, seed, dev):
    dl = torch.utils.data.DataLoader(ChipSet(recs, seed), batch_size=CFG["bs"], num_workers=CFG["workers"],
                                     collate_fn=lambda b: (torch.stack([x for x, _ in b]), [y for _, y in b]),
                                     persistent_workers=CFG["workers"] > 0, prefetch_factor=4 if CFG["workers"] else None)
    for x, y in dl: yield x.to(dev, non_blocking=True), [{k: v.to(dev) for k, v in t.items()} for t in y]

# ── Шаг 6. Модель и встроенный тренер ────────────────────────────────────────────────────────────────────────────
def train_loop(model, it, cfg, dev, log, budget_s=None, tag=""):
    bb = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg["lr"] * cfg["bb_mult"]}, {"params": rest, "lr": cfg["lr"]}], weight_decay=cfg["wd"])
    def sched(total, done=0):
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg["lr"] * cfg["bb_mult"], cfg["lr"]], total_steps=max(2, total - done),
                                                   pct_start=cfg["pct"] if done == 0 else 0.01)
    total = cfg["iters"]; sch = sched(total)
    amp = dev.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp); model.train()
    t0 = time.time(); nan_run = 0; hist = []; step = 0; tm = []
    while step < total:
        x, y = next(it); ts = time.time()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            out = model(pixel_values=x, labels=y)
        loss = out.loss.float()
        if not torch.isfinite(loss):
            nan_run += 1; opt.zero_grad(set_to_none=True)
            if nan_run > 50: raise RuntimeError("loss NaN 50 раз подряд — перезапустить с DFINE_NUM_DN=0")
            continue
        nan_run = 0; opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]); scaler.step(opt); scaler.update()
        if step < sch.total_steps - 1: sch.step()
        step += 1
        if dev.type == "cuda": torch.cuda.synchronize(dev)
        tm.append(time.time() - ts)
        if step == 50 and budget_s:  # адаптация числа итераций под бюджет времени
            spi = float(np.median(tm[10:])); fit = int(budget_s / spi)
            if fit < total:
                log(f"{tag} бюджет: {spi:.3f} с/ит → итераций {total} → {max(step + 10, fit)}"); total = max(step + 10, fit); sch = sched(total, step)
        if step % 100 == 0 or step == total or cfg["mode"] == "smoke":
            hist.append({"it": step, "loss": round(loss.item(), 4), "s_it": round(float(np.median(tm[-50:])), 4)})
            log(tag, json.dumps(hist[-1]))
        if time.time() > DEADLINE - 1800: log(tag, "СТОП по времени на ит.", step); break
    model.train_report = {"iters": step, "planned": cfg["iters"], "s_it_median": float(np.median(tm[min(5, len(tm) - 1):])) if tm else None,
                          "hist": hist, "sec": round(time.time() - t0)}
    return model

TRAIN, TRAIN_SRC = train_loop, "builtin train_loop"
if CFG["trainer"] == "bot5" or CFG["aug"] == "bot5":
    import types, hashlib
    BOT5 = None
    for p in [Path(__file__).with_name("dfine_train_bot5.py") if "__file__" in globals() else None,
              *map(Path, glob.glob("/kaggle/input/**/dfine_train_bot5.py", recursive=True))]:
        if p and p.exists():
            spec = importlib.util.spec_from_file_location("dfine_train_bot5", p); BOT5 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(BOT5); TRAIN_SRC = f"{p} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}"; break
    if BOT5 is None and "BOT5_TRAINER_SRC" in globals():  # встроенная копия (build.py), Kaggle загружает только файл кода
        BOT5 = types.ModuleType("dfine_train_bot5"); exec(compile(BOT5_TRAINER_SRC, "<dfine_train_bot5.py>", "exec"), BOT5.__dict__)
        TRAIN_SRC = f"встроенная копия sha256 {hashlib.sha256(BOT5_TRAINER_SRC.encode('utf-8')).hexdigest()[:16]}"
    if BOT5 is None: raise SystemExit("DFINE_TRAINER/AUG=bot5, но dfine_train_bot5.py не найден (ни встроенного, ни файла)")
    if CFG["trainer"] == "bot5":
        TRAIN = lambda model, it, cfg, dev, log, budget_s=None, tag="": BOT5.train(
            model, it, dict(cfg, budget_s=budget_s, stop_ts=DEADLINE - 1800, tag=tag), dev, log)
    log("тренер:", CFG["trainer"], "| аугментации:", CFG["aug"], "| бот 5:", TRAIN_SRC)
else:
    log("тренер: builtin train_loop | аугментации:", CFG["aug"])

# ── Шаг 7. Инференс OOF на полном растре (окна 640, перекрытие 160, flip-TTA, NMS) → полигоны EPSG:3857 ───────────
@torch.no_grad()
def infer_region(model, inv, dev):
    from torchvision.ops import batched_nms
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as st
    from pyproj import Transformer
    model.eval(); T, S = CFG["tile"], CFG["tile"] - CFG["overlap"]; feats = []
    for key, size in inv["tifs"]:
        sc = Scene(inv, key, size); B, L, Cc = [], [], []
        xs = list(range(0, max(1, sc.W - CFG["overlap"]), S)) or [0]; ys = list(range(0, max(1, sc.H - CFG["overlap"]), S)) or [0]
        wins = [(x, y) for y in ys for x in xs]
        if CFG["max_windows"]: wins = wins[:CFG["max_windows"]]  # только для сухого прогона на CPU
        t_r = time.time(); log(f"[infer] {inv['region']} {key.rsplit('/', 1)[-1]}: {sc.W}x{sc.H} px, окон {len(wins)}")
        for b in range(0, len(wins), CFG["bs"]):
            chunk, imgs = [], []
            for x, y in wins[b:b + CFG["bs"]]:
                im = sc.read(x, y, T)
                if (im.max(2) == 0).mean() > 0.95: continue
                chunk.append((x, y)); imgs.append(im)
            if not imgs: continue
            px = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
            for flip in (False, True):
                inp = px.flip(3) if flip else px
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    o = model(pixel_values=inp)
                prob = o.logits.float().sigmoid(); bx = o.pred_boxes.float()
                if flip: bx = bx.clone(); bx[..., 0] = 1 - bx[..., 0]
                Q, C = prob.shape[1:]
                sc_, idx = prob.flatten(1).topk(min(CFG["max_det"], Q * C), 1)
                for i, (x, y) in enumerate(chunk):
                    keep = sc_[i] >= CFG["min_score"]; q = idx[i][keep] // C; c = idx[i][keep] % C; bb = bx[i][q] * T
                    xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2 + x, bb[:, 1] - bb[:, 3] / 2 + y, bb[:, 0] + bb[:, 2] / 2 + x, bb[:, 1] + bb[:, 3] / 2 + y], 1)
                    B.append(xyxy.cpu()); L.append(sc_[i][keep].cpu()); Cc.append(c.cpu())
            if (b // CFG["bs"]) % 200 == 199: log(f"[infer] {inv['region']}: {b + CFG['bs']}/{len(wins)} окон, {time.time() - t_r:.0f} с")
        log(f"[infer] {inv['region']}: готово {len(wins)} окон за {time.time() - t_r:.0f} с")
        if B:
            B, L, Cc = torch.cat(B), torch.cat(L), torch.cat(Cc); k = batched_nms(B, L, Cc, CFG["nms_iou"]); B, L, Cc = B[k], L[k], Cc[k]
            to3857 = Transformer.from_crs(sc.crs, "EPSG:3857", always_xy=True).transform; t = np.linspace(0, 2 * np.pi, 33)[:-1]
            for (x0, y0, x1, y1), s, c in zip(B.numpy(), L.numpy(), Cc.numpy()):
                cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, .5), max((y1 - y0) / 2, .5)
                ring = [sc.T * (cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in t]
                feats.append({"type": "Feature", "geometry": mapping(st(to3857, Polygon(ring))),
                              "properties": {"class_name": CLASSES[int(c)], "confidence": round(float(s), 5), "region_name": inv["region"],
                                             "crs": "EPSG:3857", "markup_type": "sp_or", "original_crs": sc.crs, "group": "sp_dfine",
                                             "fold": FOLD[inv["region"]], "tif": key.rsplit("/", 1)[-1]}})
        sc.close()
    return feats

# ── Шаг 8. Оркестрация: чипы (CPU, параллельно) → фолды по 2 GPU в потоках → OOF-файлы и отчёт ────────────────────
folds_run = [int(f) for f in CFG["folds"].split(",")]
inv_train = [i for i in INV if i["marks"]]
if MODE == "smoke":
    random.Random(0).shuffle(inv_train)
    inv_train = sorted(inv_train, key=lambda i: sum(s for _, s in i["tifs"]))[:CFG["smoke_train_regions"]]
    folds_run = sorted({FOLD[i["region"]] for i in inv_train})[:len(DEVS)] or folds_run[:1]
t = time.time()
with cf.ThreadPoolExecutor(8) as ex: RECS = [r for rs in ex.map(build_chips, inv_train) for r in rs]
log(f"чипов: {len(RECS)} (объектных {sum(r['cls'] >= 0 for r in RECS)}), участков {len(inv_train)}, {time.time() - t:.0f} с;",
    "классы:", dict(collections.Counter(CLASSES[r['cls']] for r in RECS if r['cls'] >= 0)))
if not RECS: raise SystemExit("пустой банк чипов — проверьте разметку/растры")

REPORT, lock = {"cfg": CFG, "hf": f"{HF_REPO}@{HF_REV}", "folds": {}, "skipped_regions_no_fold": skipped}, threading.Lock()
queue = list(folds_run)
def gpu_worker(gi, dev):
    while True:
        with lock:
            if not queue: return
            k = queue.pop(0); left = len(queue)
        tag = f"[gpu{gi} f{k}]"
        try:
            tr = [r for r in RECS if r["fold"] != k]
            assert not {FAM[r["region"]] for r in tr} & {FAM[i["region"]] for i in INV if FOLD[i["region"]] == k}, "утечка семьи"
            test = [i for i in INV if FOLD[i["region"]] == k]
            if MODE == "smoke": test = sorted(test, key=lambda i: sum(s for _, s in i["tifs"]))[:CFG["smoke_test_regions"]]
            per_gpu_left = math.ceil((left + 1) / len(DEVS))
            budget = (DEADLINE - time.time() - 1800) / per_gpu_left - 1200  # минус ~20 мин на инференс фолда
            model, minfo = make_model(seed=100 + k); model = model.to(dev)  # check_model уже прошёл (нет meta, 5 классов)
            log(tag, f"train чипов {len(tr)}, test участков {len(test)}, бюджет обучения {budget / 3600:.2f} ч")
            model = TRAIN(model, batches(tr, 1000 + k, dev), CFG, dev, log, budget_s=budget, tag=tag)
            rep = dict(getattr(model, "train_report", {})); rep["model_check"] = minfo; rep["trainer"] = CFG["trainer"]
            if MODE == "full": model.save_pretrained(OUT / f"dfine_f{k}", safe_serialization=True)
            ti = time.time(); feats = []
            for inv in test:
                try: feats += infer_region(model, inv, dev)
                except Exception as e: log(tag, "инференс: ошибка", inv["region"], repr(e)[:200])
            json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
                       "features": feats}, open(OUT / f"oof_dfine_f{k}.geojson", "w", encoding="utf-8"), ensure_ascii=False)
            rep.update({"n_train_chips": len(tr), "test_regions": [i["region"] for i in test], "n_det": len(feats),
                        "infer_sec": round(time.time() - ti)})
            log(tag, "готово", json.dumps({a: b for a, b in rep.items() if a != "hist"}, ensure_ascii=False))
            del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
        except Exception as e:
            import traceback; rep = {"error": repr(e), "tb": traceback.format_exc()[-2000:]}; log(tag, "ОШИБКА", repr(e))
        with lock:
            REPORT["folds"][k] = rep
            json.dump(REPORT, open(OUT / "dfine_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ths = [threading.Thread(target=gpu_worker, args=(gi, d)) for gi, d in enumerate(DEVS)]
for th in ths: th.start()
for th in ths: th.join()
allf = []
for k in folds_run:
    p = OUT / f"oof_dfine_f{k}.geojson"
    if p.exists(): allf += json.load(open(p, encoding="utf-8"))["features"]
for i, f in enumerate(allf): f["properties"]["fid"] = i
json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": allf},
          open(OUT / "oof_dfine_spor.geojson", "w", encoding="utf-8"), ensure_ascii=False)
s_its = [v.get("s_it_median") for v in REPORT["folds"].values() if v.get("s_it_median")]
if s_its:
    REPORT["projection_full_h"] = round(float(np.mean(s_its)) * 8000 * 5 / 2 / 3600, 2)
    log("ПРОГНОЗ full: с/ит", round(float(np.mean(s_its)), 3), "→ 8000 ит × 5 фолдов / 2 GPU ≈", REPORT["projection_full_h"], "ч (+инференс)")
REPORT["total_sec"] = round(time.time() - T0)
json.dump(REPORT, open(OUT / "dfine_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
errs = [k for k, v in REPORT["folds"].items() if "error" in v]
log("ГОТОВО", len(allf), "детекций; ошибки фолдов:", errs)
if errs: raise SystemExit(f"фолды с ошибкой: {errs}")
