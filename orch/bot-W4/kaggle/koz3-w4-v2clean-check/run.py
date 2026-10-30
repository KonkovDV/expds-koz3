# W4: check_submission (полный прогон со смоуком) архива v2-clean 26ad1ba на Kaggle CPU (torch есть в образе).
import glob, hashlib, os, shutil, subprocess, sys, time
SHA = "15d59e9f7c120b56079f4cc7bc8cb83c14b698db9a57228e75e286dddd9842c9"
src = glob.glob("/kaggle/input/**/v2clean.zip.bin", recursive=True)[0]
chk = glob.glob("/kaggle/input/**/check_submission.py", recursive=True)[0]
os.makedirs("/kaggle/temp", exist_ok=True)
z = "/kaggle/temp/koz3_v2clean_26ad1ba.zip"; shutil.copy(src, z)
h = hashlib.sha256(open(z, "rb").read()).hexdigest()
out = open("/kaggle/working/check_submission.log", "w")
def p(*a):
    s = " ".join(map(str, a)); print(s, flush=True); out.write(s + "\n"); out.flush()
p("sha256", h, "OK" if h == SHA else "MISMATCH"); p("python", sys.version.split()[0], "nproc", os.cpu_count())
try:
    import torch; p("torch", torch.__version__)
except Exception as e: p("torch import FAIL", e)
t = time.time()
r = subprocess.run([sys.executable, chk, z, "--workdir", "/kaggle/temp", "--timeout", "1500"], capture_output=True, text=True)
p(r.stdout); p("STDERR:", r.stderr[-6000:]); p("RC", r.returncode, "wall_s", round(time.time() - t, 1))
open("/kaggle/working/STATUS", "w").write(f"RC={r.returncode}\tsha_ok={h == SHA}\twall_s={time.time() - t:.0f}\n")
