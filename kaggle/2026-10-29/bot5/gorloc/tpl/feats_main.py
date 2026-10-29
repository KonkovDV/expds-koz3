# ── main: признаки dense_feats.py (bot-5 @COMMIT) на GPU ────────────────────────────────────────────────────────
PINS = {"transformers": "transformers==5.21.0", "rasterio": "rasterio==1.5.2", "pyproj": "pyproj==3.8.0",
        "shapely": "shapely==2.2.0", "boto3": "boto3==1.40.61"}   # первые 4 — kaggle/requirements_gor.txt bot-5; boto3 — как в кернелах bot-9
VERS = deps(PINS)
import numpy as np
EXPECT_DIM = int(os.environ.get("B9_EXPECT_DIM", EXPECT_DIM))   # только для сухого прогона с крошечной моделью
MODE = os.environ.get("B9_MODE", MODE_DEFAULT); assert MODE in ("smoke", "full"), MODE
DEADLINE = T0 + float(os.environ.get("B9_DEADLINE_H", DEADLINE_H_DEFAULT)) * 3600
WORK = Path(os.environ.get("B9_WORK", "/kaggle/working")); GD = WORK / "gd"; GD.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("B9_DRY_DATA") or "/tmp/data")
HF_REPO, HF_REV = "facebook/dinov2-with-registers-large", "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"   # Apache-2.0, пин bot-5
SMOKE_REG = "185_Изборск"
SUMMARY = dict(kernel=KERNEL_ID, commit=COMMIT, mode=MODE, versions=VERS, hf=[HF_REPO, HF_REV], steps={})
def dump_summary(): json.dump(SUMMARY, open(WORK / "run_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("MODE", MODE, "WORK", WORK, "DATA", DATA)

# 1. GPU (ранний фейл: второй одновременный GPU-кернел даёт CPU-сессию)
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if DEV == "cuda" else "нет")
if DEV != "cuda" and not os.environ.get("B9_ALLOW_CPU"): raise SystemExit("GPU нет (второй одновременный GPU-кернел?). Останов.")

# 2. исходники bot-5
unpack_sources()
INPUTS = json.loads((SRC_DIR / "kaggle/gor_dense_inputs.json").read_text(encoding="utf-8"))
REGS = sorted(INPUTS["scn"]) if MODE == "full" else [SMOKE_REG]
if os.environ.get("B9_ONLY_REGS"): REGS = [r for r in os.environ["B9_ONLY_REGS"].split(",") if r]
shutil.copy(SRC_DIR / "kaggle/gor_dense_inputs.json", GD / "gor_dense_inputs.json")

# 3. веса HF (ранний фейл, до скачивания данных)
if os.environ.get("B9_MODEL_DIR"): CKPT = os.environ["B9_MODEL_DIR"]; log("модель (dry):", CKPT)
else:
    from huggingface_hub import snapshot_download
    try: CKPT = snapshot_download(HF_REPO, revision=HF_REV, allow_patterns=["*.json", "*.safetensors"])
    except Exception as e: raise SystemExit(f"веса {HF_REPO}@{HF_REV[:8]} не скачались: {e!r}")
    log("веса HF OK", HF_REPO, HF_REV[:8], "(Apache-2.0)", CKPT)

# 4. данные: только SpOr *.tif и UTM.json нужных участков (структура ключей = структура папок train-expds-3)
if not os.environ.get("B9_DRY_DATA"):
    import concurrent.futures as cf
    CLI = s3_client()
    try: CLI.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
    except Exception as e: raise SystemExit(f"S3 недоступен ({type(e).__name__}): проверьте секреты/интернет (детали скрыты).")
    log("S3 OK")
    def inv(reg):
        ks = list_prefix(CLI, reg + "/"); out = []
        for k, s in ks:
            parts = k.split("/"); n = parts[-1]
            if n == "UTM.json" and len(parts) == 2: out.append((k, s))
            elif len(parts) == 3 and "SpOr" in parts[1] and n.lower().endswith((".tif", ".tiff")): out.append((k, s))
        return out
    with cf.ThreadPoolExecutor(16) as ex: KEYS = [x for l in ex.map(inv, REGS) for x in l]
    ntif = sum(k.lower().endswith((".tif", ".tiff")) for k, _ in KEYS); nutm = sum(k.endswith("UTM.json") for k, _ in KEYS)
    log(f"к скачиванию: {ntif} tif + {nutm} UTM.json, {sum(s for _, s in KEYS) / 1e9:.2f} ГБ, участков {len(REGS)}")
    if MODE == "full": assert ntif == 175 and nutm == 132, f"ожидалось 175 tif / 132 UTM (листинг bot-9/s3/listing.txt), есть {ntif}/{nutm}"
    t = time.time()
    with cf.ThreadPoolExecutor(8) as ex: list(ex.map(lambda ks: fetch(CLI, ks[0], ks[1], DATA), KEYS))
    SUMMARY["download"] = dict(tif=ntif, utm=nutm, gb=round(sum(s for _, s in KEYS) / 1e9, 3), s=round(time.time() - t))
    log("данные скачаны", SUMMARY["download"]); dump_summary()

FEATS = SRC_DIR / "scripts/gor_dense/dense_feats.py"
BASE_ENV = dict(DATA=DATA, OUT=GD, MODEL=CKPT, REVISION="", DEVICE=DEV, THREADS=os.environ.get("B9_THREADS", "4"), REPO=SRC_DIR,
                HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)   # веса уже в кэше; ONLY_REGS ниже
def check_npz(tag, dim=EXPECT_DIM):
    z = np.load(GD / f"dense{tag}.npz"); F = z["F"]; n = len(F)
    fin = all(np.isfinite(F[a:a + 200000].astype(np.float32)).all() for a in range(0, n, 200000))
    info = dict(shape=list(F.shape), dtype=str(F.dtype), finite=bool(fin), files=len(json.load(open(GD / f"files{tag}.json", encoding="utf-8"))["files"]),
                regions=len(set(f["reg"] for f in json.load(open(GD / f"files{tag}.json", encoding="utf-8"))["files"])),
                npz_bytes=(GD / f"dense{tag}.npz").stat().st_size)
    log(f"проверка dense{tag}.npz:", json.dumps(info)); del F, z
    assert info["shape"][1] == dim and info["finite"], f"dense{tag}: ожидается dim {dim} и всё конечное — СТОП, прислать хвост лога bot-5"
    return info
def remaining(): return DEADLINE - time.time()

# 5. smoke bot-5 (обязателен первым): один участок, GSDS=1,2 → F.shape[1]==2048, всё конечное
smoke_reg = os.environ.get("B9_ONLY_REGS", "").split(",")[0] or SMOKE_REG
rc, dt = run_step("smoke", [FEATS], dict(BASE_ENV, ONLY_REGS=smoke_reg, GSDS="1,2", TAG="_smoke"), WORK / "feats_smoke.log", min(remaining(), 1800))
SUMMARY["steps"]["smoke"] = dict(rc=rc, s=round(dt)); dump_summary()
if rc != 0: raise SystemExit("smoke dense_feats упал — СТОП (хвост: feats_smoke.log)")
SUMMARY["steps"]["smoke"]["check"] = check_npz("_smoke"); (GD / "dense_smoke.npz").unlink(); (GD / "files_smoke.json").unlink(); dump_summary()
if MODE == "smoke": log("SMOKE OK — признаки 1/2 м/пкс дают", EXPECT_DIM, "каналов; полный прогон: B9_MODE/MODE_DEFAULT=full"); sys.exit(0)

# 6. полный: _L24 (2/4 м/пкс, сравнимо с CPU-прототипом DINOv2-S), затем _L12 (1/2 м/пкс, основной)
for tag, gsds in (("_L24", "2,4"), ("_L12", "1,2")):
    if remaining() < 600: SUMMARY["steps"][tag] = dict(rc="skipped_deadline"); dump_summary(); log("пропуск", tag, "— дедлайн"); continue
    rc, dt = run_step(tag, [FEATS], dict(BASE_ENV, ONLY_REGS=os.environ.get("B9_ONLY_REGS", ""), GSDS=gsds, TAG=tag), WORK / f"feats{tag}.log", remaining() - 300)
    SUMMARY["steps"][tag] = dict(rc=rc, s=round(dt)); dump_summary()
    if rc != 0: log("ОШИБКА", tag, rc); continue
    SUMMARY["steps"][tag]["check"] = check_npz(tag)
    SUMMARY["steps"][tag]["files_sha256"] = sha256_file(GD / f"files{tag}.json"); dump_summary()
if not os.environ.get("B9_DRY_DATA"): shutil.rmtree(DATA, ignore_errors=True)
ok = all(SUMMARY["steps"].get(t, {}).get("rc") == 0 for t in ("_L24", "_L12"))
log("ГОТОВО" if ok else "ЧАСТИЧНО", json.dumps({k: v.get("rc") for k, v in SUMMARY["steps"].items()}),
    "| выход: /kaggle/working/gd/{dense,files}_L24/_L12 + gor_dense_inputs.json → вход кернелов koz3-b5-gorloc-probe-*")
if not ok: sys.exit(1)
