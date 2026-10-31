# KOZ3 · koz3-w1-dfine-or-final (Or-вариант, 3-way merge ae-final ⊕ koz3-dfine-or P_OR_FIX, coord 29.10) (bot-W1-aeor). Основа — dmitrijkv/koz3-dfine-ae (bot-9/bot-5, рецепт Ae: obj2coco,
# v2 (29.10 23:40 MSK, координатор): чекпойнты/выводы в /kaggle/working после каждого фолда, пофолдовые OOF не удаляются, возобновление фолда по готовым весам+OOF из /kaggle/input.
# 0,5 м/пикс, окна 640/160, 4000 ит, тренер bot5@0256251). Правки W1 (29.10.2026):
#  1) задача 'final': обучение на ВСЕХ участках Ae (4000 ит) → /kaggle/working/dfine_ae/{model.safetensors,config.json}
#     (это веса для [dfine_aeor] weights.ae); идёт первой в очереди, чтобы веса появились даже при сбое OOF;
#  2) OOF по тем же 5 семейным фолдам (FOLDS ниже — копия исходника) с сохранением весов dfine_f{k};
#  3) инференс как в ветке dfine_aeor: окна с прижатием к краю, NMS 0.5 внутри вида, затем ТРИ варианта:
#     noflip (V1: один вид), flipnms (исходный рецепт koz3-dfine-ae), flipwbf (V5: flip-TTA + WBF 0.55, как в деплое);
#     файлы oof_dfine_ae_{noflip,flipnms,flipwbf}.geojson в формате исходного OOF (свойства те же), conf ≥ OOF_SAVE_MIN;
#  4) если в /kaggle/input уже есть dfine_f{k}/model.safetensors или dfine_ae/model.safetensors — обучение пропускается.
# Дедлайн 11,3 ч (лимит Kaggle 12 ч). Ключи S3 не печатаются.
# KOZ3 · dfine_spor (bot-9): детектор рамок D-FINE-L на спутнике SpOr, 0,5 м/пикс, окна 640 px, 5 семейных фолдов.
# Рецепт: /workspace/koz3_shared/bot-5/NOLIDAR.md §6.1 (гиперпараметры — факты из их таблицы; код конкурентов не копирован).
# Веса: ustc-community/dfine-large-coco (HF, Apache-2.0), ревизия закреплена. ТОЛЬКО обучить и измерить — в сдачу
# лишь после письма организаторов о весах. Режимы: DFINE_MODE = smoke (10 ит., мини-данные) | full.
# 29.10 вариант Or/Ae (DFINE_MOD): та же схема на модальности Or/Ae, база obj2coco, threshold_group or_dfine/ae_dfine.
# 29.10 P_OR_FIX (Or): CRS из UTM.json только проекционная (Герите 1×1 px), фолды Кавказ-ядро | Фалхан+Елизаветинское | Изборск,
# негативные чипы, сэмплер obj/neg/bg 0,5/0,3/0,2, cls_cap 6. См. bot-9/DFINE_FINAL_ORAE.md §11.
# Ключи S3: Kaggle Secrets KOZ3_S3_* → env → /kaggle/input/**/cred.json (датасет elenakonkova/koz3-s3-cred, поля id/key).
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
HF_BASES = {"coco": ("ustc-community/dfine-large-coco", "aa8f8211545e4cff5adf52ea512f9ac74a9824ec"),  # как фолды full/full_rest
            "obj2coco": ("ustc-community/dfine-large-obj2coco-e25", "c89de47153e338c07cddc81fc21a776b97cbf75e")}  # Objects365→COCO, Apache-2.0
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
DFINE_MODE_DEFAULT = 'full'
FOLDS_DEFAULT = 'final,0,1,2'
DEADLINE_H_DEFAULT = 11.3
FULL_ITERS_DEFAULT = 2400
DFINE_BASE_DEFAULT = 'obj2coco'
MOD_DEFAULT = 'Or'
FOLDS = __import__('json').loads(r'''{"family_of": {"008_Герите": "Кавказ-ядро", "014_Лешки": "Кавказ-ядро", "093_Старый_Барах": "Кавказ-ядро", "056_Фалхан": "Фалхан", "033_Елизаветинское": "Елизаветинское", "185_Изборск": "Изборск"}, "folds": [{"fold": 0, "test_families": ["Кавказ-ядро"], "test_regions": ["008_Герите", "014_Лешки", "093_Старый_Барах"]}, {"fold": 1, "test_families": ["Фалхан", "Елизаветинское"], "test_regions": ["056_Фалхан", "033_Елизаветинское"]}, {"fold": 2, "test_families": ["Изборск"], "test_regions": ["185_Изборск"]}]}''')
BOT5_TRAINER_SRC = r'''# Адаптер тренера бота 5 под хук dfine_main.py: train(model, batches, cfg, device, log) -> model.
# Вендорено из /workspace/koz3_shared/bot-5/repo, ветка bot5-dfine-kaggle, коммит 0256251,
#   scripts/det_train_hf.py (git blob 45a67faee1823acde38f930e5c6f3a7b6e3363b3, sha256 78cab4ac41081c2d…):
#   jitter() — дословно (стр. 49–80); цикл обучения main() (AdamW 2 группы, OneCycle, GradScaler, autocast, clip) — дословно
#   по логике, переписан как функция. Отличия адаптера (документированы в bot-9/DFINE.md):
#   * батчи приходят из нашего конвейера (pixel_values 0..1 без нормализации = RTDetrImageProcessor do_rescale, do_normalize=false;
#     labels class_labels + boxes cxcywh норм. — тот же формат, что даёт процессор), поэтому AutoImageProcessor и COCO-датасет
#     бота 5 не нужны;
#   * добавлены: стоп по бюджету времени/дедлайну (cfg budget_s/stop_ts), пропуск нефинитного loss (≤50 подряд), отчёт
#     model.train_report в формате встроенного тренера; fp16 включается на cuda (как --fp16 бота 5).
import math, random, time
import numpy as np
from PIL import Image, ImageFilter

def jitter(img: np.ndarray, boxes: np.ndarray, rng: random.Random, strong: bool):
    """[бот 5, дословно] img HxWx3 uint8, boxes Nx4 xywh в пикселях."""
    h, w = img.shape[:2]
    x = img.astype(np.float32)
    b = boxes.astype(np.float32).copy()
    if rng.random() < 0.5:  # отражение по горизонтали
        x = x[:, ::-1]
        if len(b):
            b[:, 0] = w - b[:, 0] - b[:, 2]
    k = rng.randrange(4)
    for _ in range(k):  # поворот на 90° против часовой: (x, y) -> (y, W - x)
        hh, ww = x.shape[:2]
        x = np.rot90(x)
        if len(b):
            nx, ny = b[:, 1].copy(), ww - b[:, 0] - b[:, 2]
            b = np.stack([nx, ny, b[:, 3], b[:, 2]], 1)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.3:
        x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.1:
        x = np.repeat(x.mean(2, keepdims=True), 3, 2)
    out = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    if rng.random() < 0.2:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if strong and rng.random() < 0.3:
        f = rng.uniform(0.4, 0.8)
        sz = out.size
        out = out.resize((max(8, int(sz[0] * f)), max(8, int(sz[1] * f))), Image.BILINEAR).resize(sz, Image.BILINEAR)
    return np.asarray(out), b

def train(model, batches, cfg, device, log):
    import torch
    dev = device; tag = cfg.get("tag", "")
    iters = int(cfg["iters"]); lr, mult, wd = cfg["lr"], cfg["bb_mult"], cfg["wd"]
    fp16 = dev.type == "cuda"
    # --- бот 5: оптимизатор и расписание ---
    bb = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    lrs = [lr * mult, lr]
    opt = torch.optim.AdamW([{"params": bb, "lr": lrs[0]}, {"params": rest, "lr": lrs[1]}], weight_decay=wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lrs, total_steps=max(2, iters), pct_start=cfg["pct"])
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    model.train()
    t = time.time(); hist = []; tm = []; rep = []; it = 0; bad = 0
    log_every = 1 if cfg.get("mode") == "smoke" else 100
    budget = cfg.get("budget_s"); stop_ts = cfg.get("stop_ts")
    while it < iters:
        pv, lab = next(batches); ts = time.time()
        with torch.autocast(dev.type, enabled=fp16):
            o = model(pixel_values=pv.to(dev), labels=[{k: v.to(dev) for k, v in l.items()} for l in lab])
        if not torch.isfinite(o.loss.float()):  # [адаптер] защита, у бота 5 её нет
            bad += 1; opt.zero_grad(set_to_none=True)
            if bad > 50: raise RuntimeError("loss NaN 50 раз подряд — перезапустить с DFINE_NUM_DN=0")
            continue
        bad = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(o.loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        scaler.step(opt); scaler.update()
        if it < sch.total_steps - 1: sch.step()
        if fp16: torch.cuda.synchronize(dev)
        hist.append(float(o.loss.detach())); tm.append(time.time() - ts); it += 1
        if it % log_every == 0 or it == iters:
            rep.append({"it": it, "loss": round(float(np.mean(hist[-log_every:])), 4), "s_it": round(float(np.median(tm[-50:])), 4)})
            log(tag, "[bot5]", rep[-1])
        now = time.time()
        if (budget and now - t > budget) or (stop_ts and now > stop_ts):  # [адаптер] стоп по времени
            log(tag, "[bot5] СТОП по времени на ит.", it); break
    model.train_report = {"trainer": "bot5@0256251", "iters": it, "planned": iters,
                          "s_it_median": float(np.median(tm[min(5, len(tm) - 1):])) if tm else None,
                          "hist": rep, "sec": round(time.time() - t), "loss_first": hist[0] if hist else None,
                          "loss_last": hist[-1] if hist else None}
    return model
'''
import json, time, re, math, glob, random, threading, argparse, collections, concurrent.futures as cf
from pathlib import Path
import numpy as np, torch

MODE = os.environ.get("DFINE_MODE", DFINE_MODE_DEFAULT)
DFINE_BASE = os.environ.get("DFINE_BASE", globals().get("DFINE_BASE_DEFAULT", "coco")); assert DFINE_BASE in HF_BASES, "DFINE_BASE=coco|obj2coco"
HF_REPO, HF_REV = HF_BASES[DFINE_BASE]; SP_GROUP = "sp_dfine" if DFINE_BASE == "coco" else "sp_dfine_o2c"
MOD = os.environ.get("DFINE_MOD", globals().get("MOD_DEFAULT", "SpOr")); assert MOD in ("SpOr", "Or", "Ae"), "DFINE_MOD=SpOr|Or|Ae"
MARKUP_TYPE = {"SpOr": "sp_or", "Or": "or", "Ae": "ae"}[MOD]
if MOD != "SpOr": SP_GROUP = MOD.lower() + "_dfine" + ("" if DFINE_BASE == "obj2coco" else "_coco")  # or_dfine / ae_dfine
AP = argparse.ArgumentParser()
AP.add_argument("--allow-cpu", action="store_true")
ARGS, _ = AP.parse_known_args()
E = lambda k, d: type(d)(os.environ.get(k, d))
CFG = dict(mode=MODE, res=0.5, tile=E("DFINE_TILE", 1024), overlap=E("DFINE_OVERLAP", 288), max_windows=E("DFINE_MAX_WINDOWS", 0), bs=E("DFINE_BS", 4),
           iters=E("DFINE_ITERS", 10 if MODE == "smoke" else globals().get("FULL_ITERS_DEFAULT", 8000)), lr=1e-4, bb_mult=0.1, wd=1e-4, pct=0.05, clip=0.1,
           num_dn=E("DFINE_NUM_DN", -1), obj_frac=E("DFINE_OBJ_FRAC", 0.5), cls_cap=E("DFINE_CLS_CAP", 6.0), neg_share=E("DFINE_NEG_SHARE", 0.6),
           neg_ratio=E("DFINE_NEG_RATIO", 1.0), neg_min=E("DFINE_NEG_MIN", 8), neg_cap=E("DFINE_NEG_CAP", 60), neg_margin=E("DFINE_NEG_MARGIN", 32),
           neg_valid=E("DFINE_NEG_VALID", 0.25), neg_masked=E("DFINE_NEG_MASKED", 1), neg_mask_max=E("DFINE_NEG_MASK_MAX", 0.4), shift=1 / 3, qaug=0.5, workers=E("DFINE_WORKERS", 2),
           max_obj_chips_region=400, min_score=0.05, nms_iou=0.5, max_det=100,
           smoke_train_regions=E("DFINE_SMOKE_TRAIN_REGIONS", 8), smoke_test_regions=E("DFINE_SMOKE_TEST_REGIONS", 1),
           deadline_h=E("DFINE_DEADLINE_H", float(globals().get("DEADLINE_H_DEFAULT", 11.3))), folds=os.environ.get("DFINE_FOLDS", globals().get("FOLDS_DEFAULT", "0,1,2,3,4")),
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
        raise SystemExit(f"нет ключей S3: Kaggle Secrets {SECRET_ID}/{SECRET_KEY} или датасет elenakonkova/koz3-s3-cred (cred.json)")
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
        elif f"_{MOD}_" in k and n.lower().endswith((".tif", ".tiff")): tifs.append((k, s))
        elif n.endswith(".geojson") and "разметка" in k:
            m = re.search(rf"_{MOD}_(.+)\.geojson$", n)
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
        if m:
            from pyproj import CRS as _CRS
            if _CRS.from_user_input(f"EPSG:{m.group(1)}").is_projected: return f"EPSG:{m.group(1)}"
            log(f"[crs] {inv['region']}: UTM.json задаёт EPSG:{m.group(1)} (географическая, градусы) — беру UTM по центру растра")
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
PREP_DIR = os.environ.get("DFINE_PREP_DIR", "/kaggle/input")
def raster_path(key, size):
    if MOD != "Or" or size < PREP_MIN_BYTES: return fetch(key, size)
    import shutil; nm = prep_name(key, size)
    for p in sorted(glob.glob(f"{PREP_DIR}/**/or_prep/{nm}", recursive=True)): return Path(p)  # из ядра koz3-dfine-or-prep
    dst = CACHE / "or_prep" / nm
    with PREP_LOCK:
        if dst.exists(): return dst
        free = shutil.disk_usage(CACHE.parent if CACHE.parent.exists() else "/").free
        log(f"[prep] нет {nm} из koz3-dfine-or-prep — скачиваю {size / 1e9:.1f} ГБ в ядре (свободно {free / 1e9:.0f} ГБ)")
        if free < size * 1.02 + 2e9: raise RuntimeError("мало диска для " + nm)
        t = time.time(); src = fetch(key, size); log(f"[prep] скачано за {time.time() - t:.0f} с ({size / 1e6 / max(1, time.time() - t):.0f} МБ/с)")
        try: prep_tif(src, dst, log)
        finally: src.unlink(missing_ok=True)
    return dst

class Scene:
    """Растр SpOr, приведённый на лету к UTM 0,5 м (WarpedVRT; при уменьшении — average, иначе bilinear), RGB uint8."""
    def __init__(self, inv, key, size):
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform
        self.src = rasterio.open(raster_path(key, size)); self.crs = target_crs(inv, self.src)
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
# Фикс 28.10 (smoke v2): раньше окно BIG всегда центрировалось на объекте и чип выбрасывался при >50 % nodata —
# на растрах меньше BIG (≈1067 px при tile 640) и у краёв терялись ВСЕ объекты (8 smoke-участков: 38 из 89 GT).
# Теперь окно сдвигается внутрь растра (объект остаётся внутри), центр объекта в чипе хранится в rec["c"],
# объектный чип отбрасывается только если в самой рамке >50 % nodata, фоновый — если валидных пикселей < T²/2.
def _clamp0(v, n):  # левый край окна BIG внутри растра длины n (если растр меньше окна — растр по центру окна)
    return int(min(max(v, 0), n - BIG)) if n >= BIG else int((n - BIG) // 2)
COVER, NEG = [], []
def build_chips(inv):
    import cv2, zlib
    rng = random.Random(zlib.crc32(inv["region"].encode())); rec = []; drop = collections.Counter()  # детерминированно (hash() солёный)
    for key, size in inv["tifs"]:
        try:
            sc = Scene(inv, key, size); gt = load_gt(inv, sc.crs); _org = []
            boxes = [(sc.to_px(b), c) for b, c in gt]
            boxes = [(b, c) for b, c in boxes if b[2] > 0 and b[3] > 0 and b[0] < sc.W and b[1] < sc.H]
            objs = rng.sample(boxes, min(len(boxes), CFG["max_obj_chips_region"]))
            nb = max(2, int(len(objs) * (1 - CFG["obj_frac"]) / CFG["obj_frac"]))
            cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, c, b) for b, c in objs] + \
                    [(rng.uniform(0, sc.W), rng.uniform(0, sc.H), -1, None) for _ in range(nb)]
            for j, (cx, cy, c, ob) in enumerate(cents):
                x0, y0 = _clamp0(cx - BIG / 2, sc.W), _clamp0(cy - BIG / 2, sc.H); img = sc.read(x0, y0, BIG)
                valid = img.max(2) > 0
                if ob is None:
                    if valid.sum() < 0.5 * CFG["tile"] ** 2: drop["bg_nodata"] += 1; continue
                else:
                    a0, b0 = max(0, int(ob[0] - x0)), max(0, int(ob[1] - y0)); a1, b1 = min(BIG, int(math.ceil(ob[2] - x0))), min(BIG, int(math.ceil(ob[3] - y0)))
                    if a1 <= a0 or b1 <= b0 or valid[b0:b1, a0:a1].mean() < 0.5: drop["obj_nodata"] += 1; continue
                bx = [[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0, cc] for b, cc in boxes
                      if b[2] > x0 and b[3] > y0 and b[0] < x0 + BIG and b[1] < y0 + BIG]
                p = CHIPS / f"{inv['region']}__{Path(key).stem}__{j}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                _org.append((x0, y0)); rec.append({"path": str(p), "region": inv["region"], "fold": FOLD[inv["region"]], "cls": c, "boxes": bx,
                            "c": (cx - x0, cy - y0) if ob is not None else None})
            # P_OR_FIX: явные негативы — окна T×T, не задевающие ни одну рамку GT (+поле neg_margin px); окно может выходить за
            # край растра (как окна инференса), но валидных пикселей ≥ neg_valid·T²; окна сдвинуты друг от друга ≥ T/3 по x или y.
            # Сколько: clip(round(neg_ratio·объектов участка), neg_min, neg_cap). Обучение на них: пустые цели (проверено: HF D-FINE
            # даёт конечный loss только из VFL, бот-9/or_fix/test_empty_targets.log).
            T, M = CFG["tile"], CFG["neg_margin"]
            nneg = 0 if CFG["neg_ratio"] <= 0 else min(CFG["neg_cap"], max(CFG["neg_min"], int(round(CFG["neg_ratio"] * len(objs)))))
            got, tries, negw = 0, 0, []
            while got < nneg and tries < 60 * max(1, nneg):
                tries += 1; x0, y0 = int(rng.uniform(-T / 2, sc.W - T / 2)), int(rng.uniform(-T / 2, sc.H - T / 2))
                if any(b[2] + M > x0 and b[3] + M > y0 and b[0] - M < x0 + T and b[1] - M < y0 + T for b, _c in boxes): continue
                if any(abs(x0 - u) < T / 3 and abs(y0 - v) < T / 3 for u, v in negw): continue
                img = sc.read(x0, y0, T)
                if (img.max(2) > 0).mean() < CFG["neg_valid"]: drop["neg_nodata"] += 1; continue
                negw.append((x0, y0)); got += 1
                p = CHIPS / f"{inv['region']}__{Path(key).stem}__neg{got}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                rec.append({"path": str(p), "region": inv["region"], "fold": FOLD[inv["region"]], "cls": -2, "boxes": [], "c": None})
            # запас: если чистых окон не хватило (малые растры Изборск/Елизаветинское: окно 1024 не помещается вне GT) — «маскированные»
            # негативы: окно с GT, рамки GT (+поле) залиты 0 = nodata (как пустые края растра на инференсе); замазано ≤ neg_mask_max окна.
            gotm, tries2 = 0, 0
            while CFG["neg_masked"] and got + gotm < nneg and tries2 < 60 * max(1, nneg):
                tries2 += 1; x0, y0 = int(rng.uniform(-T / 2, sc.W - T / 2)), int(rng.uniform(-T / 2, sc.H - T / 2))
                if any(abs(x0 - u) < T / 3 and abs(y0 - v) < T / 3 for u, v in negw): continue
                img = sc.read(x0, y0, T); mk = np.zeros((T, T), bool)
                for b, _c in boxes:
                    a0, b0 = max(0, int(b[0] - M - x0)), max(0, int(b[1] - M - y0)); a1, b1 = min(T, int(math.ceil(b[2] + M - x0))), min(T, int(math.ceil(b[3] + M - y0)))
                    if a1 > a0 and b1 > b0: mk[b0:b1, a0:a1] = True
                if not mk.any() or mk.mean() > CFG["neg_mask_max"]: continue
                img[mk] = 0
                if (img.max(2) > 0).mean() < CFG["neg_valid"]: drop["negm_nodata"] += 1; continue
                negw.append((x0, y0)); gotm += 1
                p = CHIPS / f"{inv['region']}__{Path(key).stem}__negm{gotm}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                rec.append({"path": str(p), "region": inv["region"], "fold": FOLD[inv["region"]], "cls": -2, "boxes": [], "c": None, "masked": True})
            NEG.append({"region": inv["region"], "tif": key.rsplit("/", 1)[-1], "W": sc.W, "H": sc.H, "want": nneg, "got_clean": got,
                        "got_masked": gotm, "tries": tries + tries2})
            log("негативы:", json.dumps(NEG[-1], ensure_ascii=False))
            # проверка покрытия GT: рамка целиком внутри хотя бы одного сохранённого чипа (BIG×BIG до сдвига ±1/3)
            _cov = sum(any(b[0] >= x and b[1] >= y and b[2] <= x + BIG and b[3] <= y + BIG for x, y in _org) for b, c in boxes)
            COVER.append({"region": inv["region"], "tif": key.rsplit("/", 1)[-1], "W": sc.W, "H": sc.H, "gt_total": len(gt),
                          "gt_in_raster": len(boxes), "gt_covered_by_chip": _cov, "chips": len(_org)})
            log("покрытие GT:", json.dumps(COVER[-1], ensure_ascii=False))
            sc.close()
        except Exception as e:
            log("чипы: ошибка", inv["region"], key.rsplit("/", 1)[-1], repr(e)[:200])
    if drop: log("чипы:", inv["region"], "отброшено", dict(drop))
    return rec

# ── Шаг 5. Датасет с аугментациями (по рецепту §6.1) ─────────────────────────────────────────────────────────────
class ChipSet(torch.utils.data.IterableDataset):
    def __init__(self, recs, seed):
        self.recs, self.seed = recs, seed
        n = collections.Counter(r["cls"] for r in recs if r["cls"] >= 0); nmax = max(n.values()) if n else 1
        self.obj = [r for r in recs if r["cls"] >= 0]; self.bg = [r for r in recs if r["cls"] == -1]; self.neg = [r for r in recs if r["cls"] == -2]
        self.w = np.array([min(CFG["cls_cap"], math.sqrt(nmax / n[r["cls"]])) for r in self.obj], float)
        self.w = self.w / self.w.sum() if len(self.w) else self.w
    def __iter__(self):
        import cv2
        wi = torch.utils.data.get_worker_info(); rs = np.random.RandomState(self.seed + (wi.id if wi else 0))
        prng = random.Random(self.seed + (wi.id if wi else 0))
        T = CFG["tile"]
        while True:
            use_obj = self.obj and (not (self.bg or self.neg) or rs.rand() < CFG["obj_frac"])
            if use_obj: r = self.obj[rs.choice(len(self.obj), p=self.w)]
            else:  # P_OR_FIX: не-объектный чип — негатив с вероятностью neg_share, иначе случайный фон
                pool = self.neg if self.neg and (not self.bg or rs.rand() < CFG["neg_share"]) else self.bg
                r = pool[rs.randint(len(pool))]
            img = cv2.imread(r["path"])[:, :, ::-1]; S = img.shape[0]
            if r.get("c") is not None:  # объект в центре кадра ±shift·T (как раньше при центрированном чипе), в пределах чипа
                sh = CFG["shift"] * T
                ox, oy = [int(min(max(v - T / 2 + rs.uniform(-sh, sh), 0), S - T)) for v in r["c"]]
            else:
                ox, oy = rs.randint(0, S - T + 1), rs.randint(0, S - T + 1)
            img = img[oy:oy + T, ox:ox + T].astype(np.float32)
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
def _valid(g): return g if g.is_valid else g.buffer(0)
OOF_SAVE_MIN = float(os.environ.get("OOF_SAVE_MIN", 0.2))
VARIANTS = ("noflip", "flipnms", "flipwbf")
def _wins(W, H, tile, overlap):  # как dfine_aeor.windows: последний ряд/столбец прижат к краю
    step = tile - overlap
    def axis(n):
        if n <= tile: return [0]
        xs = list(range(0, n - tile + 1, step))
        if xs[-1] + tile < n: xs.append(n - tile)
        return xs
    return [(x, y) for y in axis(int(H)) for x in axis(int(W))]
def wbf_fast(B, L, C, n_views=2, thr=0.55, cell=128.0):
    """Тот же алгоритм, что dfine_aeor.wbf (по классу, по убыванию уверенности, слитый бокс — среднее с весами,
    уверенность = сумма / max(n_views, членов)), но кандидаты ищутся через сеточный индекс (миллионы боксов)."""
    B = np.asarray(B, np.float64).reshape(-1, 4); L = np.asarray(L, np.float64).reshape(-1); C = np.asarray(C, int).reshape(-1)
    oB, oS, oL = [], [], []
    for c in np.unique(C):
        idx = np.where(C == c)[0]; idx = idx[np.argsort(-L[idx], kind="stable")]
        fused, wsum, ssum, cnt, grid = [], [], [], [], collections.defaultdict(list)
        def cells(b):
            return [(i, j) for i in range(int(b[0] // cell), int(b[2] // cell) + 1) for j in range(int(b[1] // cell), int(b[3] // cell) + 1)]
        for i in idx:
            b = B[i]; best, bj = -1.0, -1
            cand = {j for ce in cells(b) for j in grid.get(ce, ())}
            for j in cand:
                f = fused[j]; w = max(0.0, min(b[2], f[2]) - max(b[0], f[0])); h = max(0.0, min(b[3], f[3]) - max(b[1], f[1]))
                inter = w * h; ua = (b[2] - b[0]) * (b[3] - b[1]) + (f[2] - f[0]) * (f[3] - f[1]) - inter
                iou = inter / ua if ua > 0 else 0.0
                if iou > best: best, bj = iou, j
            if bj >= 0 and best >= thr:
                wsum[bj] = wsum[bj] + b * L[i]; ssum[bj] += L[i]; cnt[bj] += 1
                fused[bj] = wsum[bj] / max(ssum[bj], 1e-12)
                for ce in cells(fused[bj]):
                    if bj not in grid[ce]: grid[ce].append(bj)
            else:
                j = len(fused); fused.append(b.copy()); wsum.append(b * L[i]); ssum.append(float(L[i])); cnt.append(1)
                for ce in cells(b): grid[ce].append(j)
        for f, s, n in zip(fused, ssum, cnt):
            oB.append(f); oS.append(min(1.0, s / max(n_views, n))); oL.append(int(c))
    if not oB: return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return np.asarray(oB), np.asarray(oS), np.asarray(oL, int)
@torch.no_grad()
def infer_region(model, inv, dev):
    from torchvision.ops import batched_nms
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as st
    from pyproj import Transformer
    model.eval(); T = CFG["tile"]; out = {v: [] for v in VARIANTS}
    def nms_t(B, L, C):
        if not len(B): return B, L, C
        k = batched_nms(B, L, C, CFG["nms_iou"]); return B[k], L[k], C[k]
    for key, size in inv["tifs"]:
        sc = Scene(inv, key, size); views = [([], [], []), ([], [], [])]
        wins = _wins(sc.W, sc.H, T, CFG["overlap"])
        if CFG["max_windows"]: wins = wins[:CFG["max_windows"]]
        t_r = time.time(); log(f"[infer] {inv['region']} {key.rsplit('/', 1)[-1]}: {sc.W}x{sc.H} px, окон {len(wins)}")
        for b in range(0, len(wins), CFG["bs"]):
            chunk, imgs = [], []
            for x, y in wins[b:b + CFG["bs"]]:
                im = sc.read(x, y, T)
                if (im.max(2) == 0).mean() > 0.95: continue
                chunk.append((x, y)); imgs.append(im)
            if not imgs: continue
            px = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
            for vi, flip in enumerate((False, True)):
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
                    views[vi][0].append(xyxy.cpu()); views[vi][1].append(sc_[i][keep].cpu()); views[vi][2].append(c.cpu())
            if (b // CFG["bs"]) % 200 == 199: log(f"[infer] {inv['region']}: {b + CFG['bs']}/{len(wins)} окон, {time.time() - t_r:.0f} с")
        cat = lambda v: (torch.cat(v[0]), torch.cat(v[1]), torch.cat(v[2])) if v[0] else (torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long))
        v0, v1 = cat(views[0]), cat(views[1])
        n0, n1 = nms_t(*v0), nms_t(*v1)                       # NMS внутри вида (перекрытие окон), как dfine_aeor.detect_scene
        res = {"noflip": n0,
               "flipnms": nms_t(*(torch.cat([a, b]) for a, b in zip(v0, v1))),   # исходный рецепт koz3-dfine-ae
               "flipwbf": wbf_fast(torch.cat([n0[0], n1[0]]).numpy(), torch.cat([n0[1], n1[1]]).numpy(), torch.cat([n0[2], n1[2]]).numpy(), 2, 0.55)}
        log(f"[infer] {inv['region']}: готово {len(wins)} окон за {time.time() - t_r:.0f} с;", {k: len(v[1]) for k, v in res.items()})
        to3857 = Transformer.from_crs(sc.crs, "EPSG:3857", always_xy=True).transform; t = np.linspace(0, 2 * np.pi, 33)[:-1]
        for vn, (B, L, Cc) in res.items():
            B, L, Cc = (np.asarray(B), np.asarray(L), np.asarray(Cc))
            for (x0, y0, x1, y1), s, c in zip(B, L, Cc):
                if s < OOF_SAVE_MIN: continue
                cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, .5), max((y1 - y0) / 2, .5)
                ring = [sc.T * (cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in t]
                out[vn].append({"type": "Feature", "geometry": mapping(_valid(st(to3857, Polygon(ring)))),
                                "properties": {"class_name": CLASSES[int(c)], "confidence": round(float(s), 5), "region_name": inv["region"],
                                               "crs": "EPSG:3857", "markup_type": MARKUP_TYPE, "original_crs": sc.crs, "group": SP_GROUP, "threshold_group": SP_GROUP,
                                               "fold": FOLD[inv["region"]], "tif": key.rsplit("/", 1)[-1], "tta": vn}})
        sc.close()
    return out

# ── Шаг 8. Оркестрация: чипы (CPU, параллельно) → фолды по 2 GPU в потоках → OOF-файлы и отчёт ────────────────────
folds_run = [f if f == "final" else int(f) for f in CFG["folds"].split(",")]
inv_train = [i for i in INV if i["marks"]]
if MODE == "smoke":
    random.Random(0).shuffle(inv_train)
    inv_train = sorted(inv_train, key=lambda i: sum(s for _, s in i["tifs"]))[:CFG["smoke_train_regions"]]
    folds_run = ["final"] + (sorted({FOLD[i["region"]] for i in inv_train})[:1] or [0])
t = time.time()
with cf.ThreadPoolExecutor(8) as ex: RECS = [r for rs in ex.map(build_chips, inv_train) for r in rs]
log(f"чипов: {len(RECS)} (объектных {sum(r['cls'] >= 0 for r in RECS)}), участков {len(inv_train)}, {time.time() - t:.0f} с;",
    "классы:", dict(collections.Counter(CLASSES[r['cls']] for r in RECS if r['cls'] >= 0)))
if not RECS: raise SystemExit("пустой банк чипов — проверьте разметку/растры")
FOLD_CHIPS = {}  # P_OR_FIX: состав train каждого фолда (классы, фон, негативы)
for k in folds_run:
    _tr = [r for r in RECS if r["fold"] != k]
    FOLD_CHIPS[k] = {"test": sorted(i["region"] for i in INV if FOLD[i["region"]] == k), "train_chips": len(_tr),
                     "obj": dict(collections.Counter(CLASSES[r["cls"]] for r in _tr if r["cls"] >= 0)),
                     "bg": sum(r["cls"] == -1 for r in _tr), "neg": sum(r["cls"] == -2 for r in _tr),
                     "neg_masked": sum(bool(r.get("masked")) for r in _tr),
                     "neg_by_region": dict(collections.Counter(r["region"] for r in _tr if r["cls"] == -2))}
    log(f"[фолд {k}] train:", json.dumps(FOLD_CHIPS[k], ensure_ascii=False))
    _miss = [c for c in ("gorodishcha", "fortifikatsii", "arkhitektury") if not FOLD_CHIPS[k]["obj"].get(c)]
    if _miss: log(f"[фолд {k}] ВНИМАНИЕ: в train нет классов {_miss}")
if os.environ.get("DFINE_CHIPS_ONLY") == "1":
    json.dump({"cfg": CFG, "gt_coverage": COVER, "negatives": NEG, "fold_chips": FOLD_CHIPS, "n_chips": len(RECS)},
              open(OUT / "chips_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("DFINE_CHIPS_ONLY=1: собран только банк чипов, обучение пропущено"); raise SystemExit(0)

REPORT, lock = {"cfg": CFG, "hf": f"{HF_REPO}@{HF_REV}", "base": DFINE_BASE, "mod": MOD, "markup_type": MARKUP_TYPE, "gt_coverage": COVER, "negatives": NEG, "fold_chips": FOLD_CHIPS, "folds": {}, "skipped_regions_no_fold": skipped}, threading.Lock()
queue = list(folds_run)
def _ready(name):
    for p in [OUT / name, *map(Path, glob.glob(f"/kaggle/input/**/{name}", recursive=True))]:
        if (p / "model.safetensors").exists() and (p / "config.json").exists(): return p
    return None
def load_ready(p):
    from transformers import DFineForObjectDetection
    with MODEL_LOCK:
        for n, f in _PTM_ORIG.items(): setattr(_PTM, n, f)
        m = DFineForObjectDetection.from_pretrained(str(p), device_map=None)
        info = check_model(m)
    return m, info
def _ckpt(event, **kw):
    """v2: журнал чекпойнтов в /kaggle/working (пишется сразу, под общим замком)."""
    p = OUT / "checkpoints.json"
    try: cur = json.load(open(p, encoding="utf-8")) if p.exists() else []
    except Exception: cur = []
    cur.append(dict(event=event, t=round(time.time() - T0), **kw))
    json.dump(cur, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1); log("CHECKPOINT", event, json.dumps(kw, ensure_ascii=False))
def _find_out(name):
    for p in [OUT / name, *map(Path, glob.glob(f"/kaggle/input/**/{name}", recursive=True))]:
        if p.exists(): return p
    return None
def _merge_oof(final_pass=False):
    """v2: сводный OOF по готовым фолдам (после каждого фолда и в конце)."""
    for v in VARIANTS:
        allf = []
        for k in folds_run:
            if k == "final": continue
            p = OUT / f"oof_dfine_f{k}_{v}.geojson"
            if p.exists(): allf += json.load(open(p, encoding="utf-8"))["features"]
        for i, f in enumerate(allf): f["properties"]["fid"] = i
        json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": allf},
                  open(OUT / f"oof_dfine_{MOD.lower()}_{v}.geojson", "w", encoding="utf-8"), ensure_ascii=False)
        log("OOF" + (" (итог)" if final_pass else " (промежуточный)"), v, len(allf)); del allf
def gpu_worker(gi, dev):
    while True:
        with lock:
            if not queue: return
            k = queue.pop(0); left = len(queue)
        tag = f"[gpu{gi} {k if k == 'final' else 'f' + str(k)}]"
        try:
            final = k == "final"; wname = "dfine_or" if final else f"dfine_f{k}"
            tr = RECS if final else [r for r in RECS if r["fold"] != k]
            if not final:
                assert not {FAM[r["region"]] for r in tr} & {FAM[i["region"]] for i in INV if FOLD[i["region"]] == k}, "утечка семьи"
            test = [] if final else [i for i in INV if FOLD[i["region"]] == k]
            if MODE == "smoke": test = sorted(test, key=lambda i: sum(s for _, s in i["tifs"]))[:CFG["smoke_test_regions"]]
            per_gpu_left = math.ceil((left + 1) / len(DEVS))
            budget = (DEADLINE - time.time() - 1800) / per_gpu_left - 1200
            rp = _ready(wname)
            if not final and rp is not None:
                got = {v: _find_out(f"oof_dfine_f{k}_{v}.geojson") for v in VARIANTS}
                if all(got.values()):
                    import shutil
                    for v, p in got.items():
                        if p != OUT / p.name: shutil.copy(p, OUT / p.name)
                    rep = {"resumed": str(rp), "oof_from": {v: str(p) for v, p in got.items()}}
                    log(tag, "фолд уже готов (веса + OOF) — пропуск:", rp)
                    with lock:
                        REPORT["folds"][str(k)] = rep
                        json.dump(REPORT, open(OUT / "dfine_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                        _ckpt("fold_resumed", fold=k); _merge_oof()
                    continue
            if rp is not None:
                model, minfo = load_ready(rp); model = model.to(dev); rep = {"reused": str(rp)}
                log(tag, "веса уже есть — обучение пропущено:", rp)
            else:
                model, minfo = make_model(seed=(199 if final else 100 + k)); model = model.to(dev)
                log(tag, f"train чипов {len(tr)}, test участков {len(test)}, бюджет обучения {budget / 3600:.2f} ч")
                model = TRAIN(model, batches(tr, (1999 if final else 1000 + k), dev), CFG, dev, log, budget_s=budget, tag=tag)
                rep = dict(getattr(model, "train_report", {}))
                if MODE == "full" or final:
                    model.save_pretrained(OUT / wname, safe_serialization=True)
                    log(tag, "веса сохранены:", str(OUT / wname), sorted(x.name for x in (OUT / wname).iterdir()))
                    with lock: _ckpt("weights", fold=k, dir=str(OUT / wname))
            rep["model_check"] = minfo; rep["trainer"] = CFG["trainer"]
            if final:
                rep.update({"n_train_chips": len(tr), "dir": str(OUT / wname), "train_regions": sorted({r["region"] for r in tr})})
            else:
                ti = time.time(); feats = {v: [] for v in VARIANTS}
                for inv in test:
                    if time.time() > DEADLINE: log(tag, "инференс: потолок времени, пропуск", inv["region"]); rep.setdefault("infer_skipped", []).append(inv["region"]); continue
                    try:
                        r_ = infer_region(model, inv, dev)
                        for v in VARIANTS: feats[v] += r_[v]
                    except Exception as e: log(tag, "инференс: ошибка", inv["region"], repr(e)[:200]); rep.setdefault("infer_errors", []).append(inv["region"])
                for v in VARIANTS:
                    json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
                               "features": feats[v]}, open(OUT / f"oof_dfine_f{k}_{v}.geojson", "w", encoding="utf-8"), ensure_ascii=False)
                rep.update({"n_train_chips": len(tr), "test_regions": [i["region"] for i in test], "n_det": {v: len(feats[v]) for v in VARIANTS},
                            "infer_sec": round(time.time() - ti)})
                del feats
            if dev.type == "cuda": rep["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
            log(tag, "готово", json.dumps({a: b for a, b in rep.items() if a not in ("hist", "train_regions")}, ensure_ascii=False))
            del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
        except Exception as e:
            import traceback; rep = {"error": repr(e), "tb": traceback.format_exc()[-2000:]}; log(tag, "ОШИБКА", repr(e))
        with lock:
            REPORT["folds"][str(k)] = rep
            json.dump(REPORT, open(OUT / "dfine_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            _ckpt("fold_done", fold=k, error=("error" in rep))
            if k != "final" and "error" not in rep:
                try: _merge_oof()
                except Exception as e: log(tag, "промежуточный сводный OOF: ошибка", repr(e)[:200])
ths = [threading.Thread(target=gpu_worker, args=(gi, d)) for gi, d in enumerate(DEVS)]
for th in ths: th.start()
for th in ths: th.join()
_merge_oof(final_pass=True)
# v2: пофолдовые OOF остаются в /kaggle/working (чекпойнт для возобновления)
allf = []
s_its = [v.get("s_it_median") for v in REPORT["folds"].values() if v.get("s_it_median")]
if s_its:
    REPORT["projection_full_h"] = round(float(np.mean(s_its)) * CFG["iters"] * 6 / 2 / 3600, 2)
    log("ПРОГНОЗ full: с/ит", round(float(np.mean(s_its)), 3), "→ 8000 ит × 5 фолдов / 2 GPU ≈", REPORT["projection_full_h"], "ч (+инференс)")
REPORT["total_sec"] = round(time.time() - T0)
json.dump(REPORT, open(OUT / "dfine_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
errs = [k for k, v in REPORT["folds"].items() if "error" in v]
log("ГОТОВО; ошибки задач:", errs, "| веса final:", _ready("dfine_or"))
if errs: raise SystemExit(f"задачи с ошибкой: {errs}")
if _ready("dfine_or") is None: raise SystemExit("нет весов dfine_ae/model.safetensors")
