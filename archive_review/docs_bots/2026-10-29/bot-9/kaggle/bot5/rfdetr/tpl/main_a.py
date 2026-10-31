# KOZ3 · bot-9 для bot-5: RF-DETR-M (COCO) на SpOr, 5 семейных фолдов.
# FOR_BOT9_RFDETR.md; код wt_rf @c43cb08. ТОЛЬКО честный замер. «Сканирование» запрещено.
# det05 внутри ядра из S3 (реальные каталоги) + GT dmitrijkv/koz3-gt-train-v2.
KERNEL_ID = "dmitrijkv/koz3-b5-rfdetr"
COMMIT = "c43cb08bfa18c926b17ac69c8d0e8864f33caf5b"
SRC_TAR_SHA256 = "@@TAR_SHA@@"
GT_SHA256 = "d4bc1a3b01350c9b3785155dc4d7c13403fdfe7a098664bcb259f96bc168ff86"
MODE_DEFAULT = "full"
DEADLINE_H_DEFAULT = 11.0
ITERS_DEFAULT = 8000
MODEL_DEFAULT = "rfdetr-m"
SIZE_DEFAULT = 640
INIT_DEFAULT = "coco"
import os, sys, subprocess, importlib, json, time, hashlib, base64, gzip, glob, shutil, re, tarfile, io
from pathlib import Path
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
WORK = Path(os.environ.get("B9_WORK", "/kaggle/working")); WORK.mkdir(parents=True, exist_ok=True)
TMP = Path(os.environ.get("B9_TMP", "/kaggle/tmp")); TMP.mkdir(parents=True, exist_ok=True)
MODE = os.environ.get("B9_MODE", MODE_DEFAULT)
ITERS = int(os.environ.get("B9_ITERS", str(ITERS_DEFAULT if MODE == "full" else 50)))
MODEL = os.environ.get("B9_MODEL", MODEL_DEFAULT)
SIZE = int(os.environ.get("B9_SIZE", SIZE_DEFAULT))
INIT = os.environ.get("B9_INIT", INIT_DEFAULT)
SUMMARY = dict(kernel=KERNEL_ID, commit=COMMIT, mode=MODE, iters=ITERS, model=MODEL, size=SIZE, init=INIT, steps={})
def dump():
    json.dump(SUMMARY, open(WORK / "run_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
def pip_install(args):
    if os.environ.get("B9_DRY"): return
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])
    if r.returncode: raise SystemExit(f"pip failed: {args}")
log("MODE", MODE, "ITERS", ITERS, "MODEL", MODEL)
pip_install(["--only-binary", ":all:", "rasterio==1.4.3", "pyproj==3.7.1", "shapely==2.2.0", "boto3==1.40.61", "Pillow==11.3.0"])
pip_install(["--no-deps", "rfdetr==1.11.2"])
SRC_B64 = "@@SRC_B64@@"
SRC_DIR = TMP / "src"
if SRC_DIR.exists(): shutil.rmtree(SRC_DIR)
SRC_DIR.mkdir(parents=True)
_raw = gzip.decompress(base64.b64decode(SRC_B64))
assert hashlib.sha256(_raw).hexdigest() == SRC_TAR_SHA256, "src tar sha mismatch"
tarfile.open(fileobj=io.BytesIO(_raw), mode="r:").extractall(SRC_DIR)
bad = [str(p) for p in SRC_DIR.rglob("*") if p.is_file() and (b"satelite-scan" in p.read_bytes() or b"ackrome" in p.read_bytes())]
if bad: raise SystemExit(f"forbidden ref in {bad[:3]}")
log("src OK", COMMIT[:8])
req = SRC_DIR / "kaggle/requirements_rfdetr.txt"
if req.exists(): pip_install(["-r", str(req)])
sys.path.insert(0, str(SRC_DIR)); sys.path.insert(0, str(SRC_DIR / "scripts"))
