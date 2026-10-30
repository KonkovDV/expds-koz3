# W4 (повтор с зависимостями образа): check_submission (полный прогон со смоуком) архива v2-clean 26ad1ba на Kaggle CPU (torch есть в образе).
import glob, hashlib, os, shutil, subprocess, sys, time
SHA = "545ef32a70b4994f4d694828ae5eb28f27c9161d7f19a96f5a9d3833b87e093f"
src = glob.glob("/kaggle/input/**/v2clean.zip.bin", recursive=True)[0]
chk = glob.glob("/kaggle/input/**/check_submission.py", recursive=True)[0]
os.makedirs("/kaggle/temp", exist_ok=True)
z = "/kaggle/temp/koz3_v2clean_70d335a.zip"; shutil.copy(src, z)
h = hashlib.sha256(open(z, "rb").read()).hexdigest()
out = open("/kaggle/working/check_submission.log", "w")
def p(*a):
    s = " ".join(map(str, a)); print(s, flush=True); out.write(s + "\n"); out.flush()
p("sha256", h, "OK" if h == SHA else "MISMATCH"); p("python", sys.version.split()[0], "nproc", os.cpu_count())
try:
    import torch; p("torch", torch.__version__)
except Exception as e: p("torch import FAIL", e)
DEPS = ["laspy[lazrs]==2.7.0", "segmentation_models_pytorch==0.5.0"]  # версии из requirements.lock (образ pantum/n7q)
WH = os.path.dirname(glob.glob("/kaggle/input/**/laspy-2.7.0-py3-none-any.whl", recursive=True)[0])  # офлайн-колёса (аккаунт без интернета)
p("wheels", WH, sorted(os.listdir(WH)))
r0 = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH, "--no-deps",
                     "laspy==2.7.0", "lazrs==0.8.2", "segmentation_models_pytorch==0.5.0"], capture_output=True, text=True)
p("pip", r0.returncode, r0.stdout[-1500:], r0.stderr[-1500:])
for m in ("laspy", "segmentation_models_pytorch", "rasterio", "pyproj", "shapely", "lightgbm", "timm", "scipy", "numpy"):
    try:
        mod = __import__(m); p("dep", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        p("dep", m, "FAIL", e)
t = time.time()
r = subprocess.run([sys.executable, chk, z, "--workdir", "/kaggle/temp", "--timeout", "1500", "--python", sys.executable], capture_output=True, text=True)
p(r.stdout); p("STDERR:", r.stderr[-6000:]); p("RC", r.returncode, "wall_s", round(time.time() - t, 1))
open("/kaggle/working/STATUS", "w").write(f"RC={r.returncode}\tsha_ok={h == SHA}\twall_s={time.time() - t:.0f}\n")
