# W4: check_submission архива v3_on (b505656, sha ebc8e4ba) на Kaggle CPU, офлайн.
# Kaggle = python 3.13, площадка pantum/n7q = 3.12: вендор архива (regex cp312) на 3.13 не грузится.
# Прогон A: как check4 (laspy/lazrs/smp) — вендорный путь; прогон B: + офлайн transformers 5.21.0 стек cp313
# в окружение → fallback "env" ровно TRANSFORMERS_PIN (проверка логики архива, не образа площадки).
import glob, hashlib, os, shutil, subprocess, sys, time, json
SHA = "ebc8e4ba4c583403fd87485e6c394fc98eb1aee24fbb09d4a7d25cbe3e2d9401"
src = glob.glob("/kaggle/input/**/v3on.zip.bin", recursive=True)[0]
chk = glob.glob("/kaggle/input/**/check_submission.py", recursive=True)[0]
os.makedirs("/kaggle/temp", exist_ok=True)
z = "/kaggle/temp/koz3_v3on_b505656.zip"; shutil.copy(src, z)
h = hashlib.sha256(open(z, "rb").read()).hexdigest()
out = open("/kaggle/working/check_submission.log", "w")
def p(*a):
    s = " ".join(map(str, a)); print(s, flush=True); out.write(s + "\n"); out.flush()
p("sha256", h, "OK" if h == SHA else "MISMATCH", "size_MiB", round(os.path.getsize(z) / 2**20, 1))
p("python", sys.version.split()[0], "nproc", os.cpu_count(), "disk_free_GiB", round(shutil.disk_usage("/kaggle/temp").free / 2**30, 1))
def vers(tag):
    for m in ("torch", "transformers", "tokenizers", "regex", "huggingface_hub", "safetensors", "tqdm", "laspy", "segmentation_models_pytorch", "rasterio", "pyproj", "shapely", "lightgbm", "timm", "numpy"):
        try:
            mod = __import__(m); p(tag, "dep", m, getattr(mod, "__version__", "?"))
        except Exception as e:
            p(tag, "dep", m, "FAIL", repr(e)[:200])
def pip(wh, pk, deps=False):
    a = [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wh] + ([] if deps else ["--no-deps"]) + pk
    r = subprocess.run(a, capture_output=True, text=True); p("pip", r.returncode, r.stdout[-1500:], r.stderr[-2500:])
WH = os.path.dirname(glob.glob("/kaggle/input/**/laspy-2.7.0-py3-none-any.whl", recursive=True)[0])
pip(WH, ["laspy==2.7.0", "lazrs==0.8.2", "segmentation_models_pytorch==0.5.0"])
# v3: опорный участок D-FINE (check_submission DFINE_KNOWN 103): датасет koz3-w4-dfine-ref103 (ASCII-имена) → структура участка
REF_SHA = "93e9bd90f6b6315a3220f58c3759b70f13f286bad1e9acc0230eb12430b74179"
R = "103_Гнёздово_Ольшанская"; REG = f"/kaggle/temp/ref/{R}"
tdir = f"{REG}/04_Гнёздово_Ольшанская_SpOr_спутник"; os.makedirs(tdir, exist_ok=True)
rt = glob.glob("/kaggle/input/**/ref103_SpOr.tif", recursive=True)[0]
shutil.copy(rt, f"{tdir}/01_Гнёздово_Ольшанская_SpOr_спутник.tif")
shutil.copy(glob.glob("/kaggle/input/**/ref103_UTM.json", recursive=True)[0], f"{REG}/UTM.json")
rh = hashlib.sha256(open(rt, "rb").read()).hexdigest(); p("ref103 sha256", rh, "OK" if rh == REF_SHA else "MISMATCH")
def run(tag):
    vers(tag)
    t = time.time()
    wd = f"/kaggle/temp/wd_{tag}"; os.makedirs(wd, exist_ok=True)
    r = subprocess.run([sys.executable, chk, z, "--workdir", wd, "--timeout", "1500", "--python", sys.executable, "--dfine-region", REG], capture_output=True, text=True)
    p(f"===== RUN {tag} =====")
    p(r.stdout); p("STDERR:", r.stderr[-6000:]); p(tag, "RC", r.returncode, "wall_s", round(time.time() - t, 1))
    last = [l for l in r.stdout.splitlines() if "ИТОГ" in l]
    return {"rc": r.returncode, "wall_s": round(time.time() - t), "itog": last[-1] if last else None}
res = {"sha_ok": h == SHA, "python": sys.version.split()[0]}
res["A_vendor"] = run("A")
WH3 = os.path.dirname(glob.glob("/kaggle/input/**/transformers-5.21.0-*.whl", recursive=True)[0])
pip(WH3, ["transformers==5.21.0", "tokenizers==0.23.3", "regex==2026.9.29", "huggingface-hub", "safetensors", "typer==0.27.3", "rich==15.0.0"], deps=True)
res["B_env_pin"] = run("B")
p(json.dumps(res, ensure_ascii=False))
open("/kaggle/working/STATUS", "w").write(json.dumps(res, ensure_ascii=False) + "\n")
