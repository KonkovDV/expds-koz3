# ── main: признаки виньеток extract_feats.py (bot-5 @COMMIT) DINOv2-reg B и L на GPU — часть A FOR_BOT9_DINOV2REG ─────
PINS = {"transformers": "transformers==5.21.0", "rasterio": "rasterio==1.5.2", "pyproj": "pyproj==3.8.0",
        "shapely": "shapely==2.2.0", "boto3": "boto3==1.40.61"}   # первые 4 — kaggle/requirements_gor.txt bot-5 (его ноутбук ставит их же)
VERS = deps(PINS)
import numpy as np
MODE = os.environ.get("B9_MODE", MODE_DEFAULT); assert MODE in ("smoke", "full"), MODE
DEADLINE = T0 + float(os.environ.get("B9_DEADLINE_H", DEADLINE_H_DEFAULT)) * 3600
WORK = Path(os.environ.get("B9_WORK", "/kaggle/working")); GP = WORK / "gp"; GP.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("B9_DRY_DATA") or "/tmp/data")
SUMMARY = dict(kernel=KERNEL_ID, commit=COMMIT, mode=MODE, versions=VERS, hf=MODELS, steps={})
def dump_summary(): json.dump(SUMMARY, open(WORK / "run_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
def remaining(): return DEADLINE - time.time()
log("MODE", MODE, "WORK", WORK, "DATA", DATA)

import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if DEV == "cuda" else "нет")
if DEV != "cuda" and not os.environ.get("B9_ALLOW_CPU"): raise SystemExit("GPU нет (второй одновременный GPU-кернел?). Останов.")
unpack_sources()
EXTRACT = SRC_DIR / "scripts/gor_patch/extract_feats.py"

# веса HF (ранний фейл, до данных); dry: B9_MODEL_DIR — один крошечный каталог для обоих
CK, HID = {}, {}
for t, (rid, rev) in MODELS.items():
    if os.environ.get("B9_MODEL_DIR"): CK[t] = os.environ["B9_MODEL_DIR"]
    else:
        from huggingface_hub import snapshot_download
        try: CK[t] = snapshot_download(rid, revision=rev, allow_patterns=["*.json", "*.safetensors"])
        except Exception as e: raise SystemExit(f"веса {rid}@{rev[:8]} не скачались: {e!r}")
    cfg = json.load(open(Path(CK[t]) / "config.json")); HID[t] = cfg["hidden_size"]
    log(f"веса {t}: {rid}@{rev[:8]} (Apache-2.0) hidden {HID[t]} регистров {cfg.get('num_register_tokens')} → {CK[t]}")

# данные: все SpOr *.tif + UTM.json этих участков (extract_feats глобит {DATA}/*/*SpOr*/*.tif*)
if not os.environ.get("B9_DRY_DATA"):
    import concurrent.futures as cf
    CLI = s3_client()
    try: CLI.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
    except Exception as e: raise SystemExit(f"S3 недоступен ({type(e).__name__}): проверьте секреты/интернет (детали скрыты).")
    regs = [p.rstrip("/") for p in list_prefix(CLI, "", "/")]
    def inv(reg):
        tif, utm = [], []
        for k, s in list_prefix(CLI, reg + "/"):
            parts = k.split("/")
            if parts[-1] == "UTM.json" and len(parts) == 2: utm.append((k, s))
            elif len(parts) == 3 and "SpOr" in parts[1] and parts[-1].lower().endswith((".tif", ".tiff")): tif.append((k, s))
        return tif, (utm if tif else [])
    with cf.ThreadPoolExecutor(16) as ex: INV = list(ex.map(inv, regs))
    TIF = sorted(x for t, _ in INV for x in t); UTM = {k.split("/")[0]: (k, s) for _, u in INV for k, s in u}
    log(f"S3: участков {len(regs)}, SpOr tif {len(TIF)}, UTM.json {len(UTM)}, {sum(s for _, s in TIF) / 1e9:.2f} ГБ")
    if MODE == "full": assert len(TIF) == 175 and len(UTM) == 132, f"ожидалось 175 tif / 132 UTM (bot-9/s3/listing.txt), есть {len(TIF)}/{len(UTM)}"
    if MODE == "smoke": TIF = TIF[:2]   # как LIMIT_FILES=2 у bot-5: первые 2 файла по сортировке путей
    KEYS = TIF + sorted({UTM[k.split("/")[0]] for k, _ in TIF})
    t = time.time()
    with cf.ThreadPoolExecutor(8) as ex: list(ex.map(lambda ks: fetch(CLI, ks[0], ks[1], DATA), KEYS))
    SUMMARY["download"] = dict(tif=len(TIF), gb=round(sum(s for _, s in KEYS) / 1e9, 3), s=round(time.time() - t)); dump_summary()
    log("данные скачаны", SUMMARY["download"])

BASE_ENV = dict(DATA=DATA, OUT=GP, DEVICE=DEV, THREADS=os.environ.get("B9_THREADS", "4"), REVISION="", HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)
def check(t, test):
    sfx = "_test" if test else ""
    F = np.load(GP / f"feats_reg{t}{sfx}.npz")["F"]; meta = json.load(open(GP / f"meta_reg{t}{sfx}.json", encoding="utf-8"))
    info = dict(shape=list(F.shape), finite=bool(np.isfinite(F.astype(np.float32)).all()), meta=len(meta), regions=len({m["reg"] for m in meta}))
    log(f"проверка feats_reg{t}{sfx}:", json.dumps(info))
    assert info["shape"][1] == 4 * HID[t] and info["finite"] and info["meta"] == info["shape"][0], \
        f"reg{t}: ожидается 4×{HID[t]} и всё конечное — СТОП (assert ноутбука bot-5)"
    return info

# smoke bot-5 (первым): LIMIT_FILES=2 для B и L, размерность 4×C
for t in MODELS:
    rc, dt = run_step(f"smoke_{t}", [EXTRACT], dict(BASE_ENV, MODEL=CK[t], TAG=f"_reg{t}", LIMIT_FILES=2), WORK / f"smoke_reg{t}.log", min(remaining(), 1800))
    SUMMARY["steps"][f"smoke_{t}"] = dict(rc=rc, s=round(dt)); dump_summary()
    if rc != 0: raise SystemExit(f"smoke reg{t} упал — СТОП (хвост: smoke_reg{t}.log)")
    SUMMARY["steps"][f"smoke_{t}"]["check"] = check(t, True); dump_summary()
for p in GP.glob("*_test.*"): p.unlink()
if MODE == "smoke": log("SMOKE OK"); sys.exit(0)

# полный: B, затем L (порядок ноутбука bot-5); extract_feats сам сохраняет каждые 20 файлов
for t in MODELS:
    if remaining() < 600: SUMMARY["steps"][t] = dict(rc="skipped_deadline"); dump_summary(); continue
    rc, dt = run_step(f"full_{t}", [EXTRACT], dict(BASE_ENV, MODEL=CK[t], TAG=f"_reg{t}", LIMIT_FILES=0), WORK / f"extract_reg{t}.log", remaining() - 300)
    SUMMARY["steps"][t] = dict(rc=rc, s=round(dt)); dump_summary()
    if rc == 0:
        SUMMARY["steps"][t]["check"] = c = check(t, False)
        if not os.environ.get("B9_DRY_DATA") and c["meta"] != 3982:
            log("ВНИМАНИЕ: виньеток", c["meta"], "≠ 3982 у DINOv2-S (bot-5/analysis/gor_patch/extract_feats.log) — сообщить bot-5")
        dump_summary()
if not os.environ.get("B9_DRY_DATA"): shutil.rmtree(DATA, ignore_errors=True)
import tarfile
with tarfile.open(WORK / "gor_patch_reg.tgz", "w:gz") as tf:
    for p in sorted(GP.glob("feats_reg*.npz")) + sorted(GP.glob("meta_reg*.json")): tf.add(p, arcname=p.name)
ok = all(SUMMARY["steps"].get(t, {}).get("rc") == 0 for t in MODELS)
log("ГОТОВО" if ok else "ЧАСТИЧНО", json.dumps({k: v.get("rc") for k, v in SUMMARY["steps"].items()}), "| выход: /kaggle/working/gor_patch_reg.tgz")
if not ok: sys.exit(1)
