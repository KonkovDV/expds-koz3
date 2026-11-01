# sat2 candidate compliance: check_submission full smoke on Kaggle CPU (no internet)
import glob, hashlib, json, os, shutil, subprocess, sys, time, zipfile, tomllib
SHA = "573eb87251fc967eb4dffd5d610d23b0aa1226dfa92520c8635be8da0bae88e9"
src = glob.glob("/kaggle/input/**/sat2.zip.bin", recursive=True)[0]
chk = glob.glob("/kaggle/input/**/check_submission.py", recursive=True)[0]
os.makedirs("/kaggle/temp", exist_ok=True)
z = "/kaggle/temp/submission_sat2_on_eb268d6.zip"
shutil.copy(src, z)
h = hashlib.sha256(open(z, "rb").read()).hexdigest()
out = open("/kaggle/working/check_submission.log", "w")
def p(*a):
    s = " ".join(map(str, a)); print(s, flush=True); out.write(s + "\n"); out.flush()
p("sha256", h, "OK" if h == SHA else "MISMATCH")
p("python", sys.version.split()[0], "nproc", os.cpu_count())
whs = glob.glob("/kaggle/input/**/laspy-2.7.0-py3-none-any.whl", recursive=True)
if whs:
    WH = os.path.dirname(whs[0])
    p("wheels", WH, sorted(os.listdir(WH)))
    r0 = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH, "--no-deps",
                         "laspy==2.7.0", "lazrs==0.8.2", "segmentation_models_pytorch==0.5.0"], capture_output=True, text=True)
    p("pip", r0.returncode, (r0.stderr or "")[-800:])
else:
    p("wheels MISSING")
for m in ("laspy", "segmentation_models_pytorch", "rasterio", "pyproj", "shapely", "lightgbm", "torch", "scipy", "numpy", "timm"):
    try:
        mod = __import__(m); p("dep", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        p("dep", m, "FAIL", e)
with zipfile.ZipFile(z) as zf:
    cfg = tomllib.loads(zf.read("configs/default.toml").decode())
    sd = cfg.get("sp_det2") or {}
    p("sp_det2", {k: sd.get(k) for k in ("enabled", "weights", "drop_prefixes")})
    assert sd.get("enabled") is True, "sp_det2 must be ON in candidate"
    assert "expds/sp_det2.py" in zf.namelist()
    assert "centerhead/__init__.py" in zf.namelist()
    assert "[post.frame_prior]" not in zf.read("configs/default.toml").decode()
    for i in range(5):
        assert f"models/sp_det2/sp_det2_cnxt_f{i}.pt" in zf.namelist()
t = time.time()
r = subprocess.run([sys.executable, chk, z, "--workdir", "/kaggle/temp", "--timeout", "1500", "--python", sys.executable],
                   capture_output=True, text=True)
p(r.stdout)
p("STDERR:", (r.stderr or "")[-6000:])
p("RC", r.returncode, "wall_s", round(time.time() - t, 1))
open("/kaggle/working/STATUS", "w").write(
    f"RC={r.returncode}\tsha_ok={h == SHA}\twall_s={time.time() - t:.0f}\tsp_det2={sd.get('enabled')}\n")
json.dump({"rc": r.returncode, "sha_ok": h == SHA, "sha": h, "sp_det2": sd, "wall_s": time.time() - t},
          open("/kaggle/working/result.json", "w"), ensure_ascii=False, indent=2)
