# W4 01.11: v3_on с изолированным вендором regex (cp311–cp314) — check_submission + побайтовый паритет выдач
# в окружениях Kaggle: C (по умолчанию, regex 2025.11.3), A (regex 2025.11.3 предзагружен .pth), B (regex удалён),
# NEG (D-FINE сломан нарочно), REF (старый ebc8e4ba в «рабочем» окружении: стек transformers 5.21 в env).
import glob, hashlib, json, os, shutil, site, subprocess, sys, time, zipfile
NEW_SHA = "1406c01a9aaafc996862a14f12cde14f6bf05b313ad0bbce14428df1d10ef41d"
OLD_SHA = "ebc8e4ba4c583403fd87485e6c394fc98eb1aee24fbb09d4a7d25cbe3e2d9401"
T = "/kaggle/temp"; os.makedirs(T, exist_ok=True)
out = open("/kaggle/working/check_submission.log", "w")
def p(*a):
    s = " ".join(map(str, a)); print(s, flush=True); out.write(s + "\n"); out.flush()
def sha(f): return hashlib.sha256(open(f, "rb").read()).hexdigest()
NEW = f"{T}/v3on_new.zip"; shutil.copy(glob.glob("/kaggle/input/**/v3onfix.zip.bin", recursive=True)[0], NEW)
OLD = f"{T}/v3on_old.zip"; shutil.copy(glob.glob("/kaggle/input/**/v3on.zip.bin", recursive=True)[0], OLD)
CHK = [f for f in glob.glob("/kaggle/input/**/check_submission.py", recursive=True) if "v3onfix" in f][0]
p("new", sha(NEW) == NEW_SHA, "old", sha(OLD) == OLD_SHA, "chk", CHK, "python", sys.version.split()[0])
def vers(tag):
    r = subprocess.run([sys.executable, "-c", "import importlib\nfor m in ('regex','transformers','tokenizers'):\n"
                        "  try:\n    x=importlib.import_module(m);print(m,x.__version__,x.__file__)\n"
                        "  except Exception as e:print(m,'FAIL',repr(e)[:160])"], capture_output=True, text=True)
    p(tag, "env:", r.stdout.replace("\n", " | "), r.stderr[-300:])
def pip(a):
    r = subprocess.run([sys.executable, "-m", "pip"] + a, capture_output=True, text=True); p("pip", a[:3], r.returncode, r.stderr[-800:])
WH = os.path.dirname(glob.glob("/kaggle/input/**/laspy-2.7.0-py3-none-any.whl", recursive=True)[0])
pip(["install", "-q", "--no-index", "--find-links", WH, "--no-deps", "laspy==2.7.0", "lazrs==0.8.2", "segmentation_models_pytorch==0.5.0"])
R = "103_Гнёздово_Ольшанская"; REF = f"{T}/ref"; REG = f"{REF}/{R}"
tdir = f"{REG}/04_Гнёздово_Ольшанская_SpOr_спутник"; os.makedirs(tdir, exist_ok=True)
shutil.copy(glob.glob("/kaggle/input/**/ref103_SpOr.tif", recursive=True)[0], f"{tdir}/01_Гнёздово_Ольшанская_SpOr_спутник.tif")
shutil.copy(glob.glob("/kaggle/input/**/ref103_UTM.json", recursive=True)[0], f"{REG}/UTM.json")
ENV = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "EXPDS_"))}
ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", EXPDS_DEVICE="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
res = {}
def run(tag, z):
    vers(tag); t = time.time(); wd = f"{T}/wd_{tag}"; os.makedirs(wd, exist_ok=True)
    r = subprocess.run([sys.executable, CHK, z, "--workdir", wd, "--timeout", "1500", "--python", sys.executable,
                        "--dfine-region", REG, "--keep"], capture_output=True, text=True)
    p(f"===== CHECK {tag} =====\n" + r.stdout); p("STDERR:", r.stderr[-3000:])
    itog = [l for l in r.stdout.splitlines() if "ИТОГ" in l]
    d = {"rc": r.returncode, "itog": itog[-1] if itog else None, "wall_s": round(time.time() - t),
         "v3_lines": [l for l in r.stdout.splitlines() if "] v3 " in l or "] dfine " in l][:12]}
    sm = glob.glob(f"{wd}/b8smoke_*/out")
    if sm:
        d["smoke_files"] = {f.name: sha(f) for f in sorted(__import__("pathlib").Path(sm[0]).iterdir()) if f.is_file()}
    # прямой прогон main.py на опорном участке 103
    code = f"{T}/code_{tag}"; shutil.rmtree(code, ignore_errors=True); zipfile.ZipFile(z).extractall(code)
    od = f"{T}/out_{tag}"; shutil.rmtree(od, ignore_errors=True); os.makedirs(od)
    t = time.time()
    m = subprocess.run([sys.executable, "inference/main.py", REF, f"{od}/result.geojson"], cwd=code, env=ENV,
                       capture_output=True, text=True, timeout=3000)
    log = m.stdout + m.stderr
    d["ref103"] = {"rc": m.returncode, "wall_s": round(time.time() - t),
                   "files": {f: sha(f"{od}/{f}") for f in sorted(os.listdir(od))},
                   "warn": [l[-300:] for l in log.splitlines() if "V3_DEGRADED" in l or "UNAVAILABLE" in l][:6],
                   "ndet": [l[-200:] for l in log.splitlines() if "n_det" in l][:4]}
    if os.path.exists(f"{od}/V3_DEGRADED.json"):
        d["ref103"]["marker"] = json.load(open(f"{od}/V3_DEGRADED.json"))
    p(tag, json.dumps(d, ensure_ascii=False)); res[tag] = d
    shutil.rmtree(code, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True)
run("C_default", NEW)
sp = site.getsitepackages()[0]; pth = f"{sp}/zz_w4_preload_regex.pth"
open(pth, "w").write("import regex\n")
run("A_regex2025_preloaded", NEW)
os.remove(pth)
# NEG: вендорный transformers сломан нарочно
NEG = f"{T}/v3on_neg.zip"
with zipfile.ZipFile(NEW) as a, zipfile.ZipFile(NEG, "w", zipfile.ZIP_DEFLATED) as b:
    for i in a.infolist():
        data = a.read(i)
        if i.filename == "inference/lib/vendor/transformers/__init__.py":
            data = b"raise ImportError('W4 negative test: D-FINE broken on purpose')\n"
        b.writestr(i, data)
run("NEG_dfine_broken", NEG); os.remove(NEG)
pip(["uninstall", "-y", "regex"])
run("B_no_regex", NEW)
WH3 = os.path.dirname(glob.glob("/kaggle/input/**/transformers-5.21.0-*.whl", recursive=True)[0])
pip(["install", "-q", "--no-index", "--find-links", WH3, "transformers==5.21.0", "tokenizers==0.23.3", "regex==2026.9.29",
     "huggingface-hub", "safetensors", "typer==0.27.3", "rich==15.0.0"])
run("REF_old_ebc8e4ba_env521", OLD)
run("D_new_env521", NEW)
ref = res["REF_old_ebc8e4ba_env521"]
summ = {}
for k, d in res.items():
    summ[k] = {"rc": d["rc"], "itog": d["itog"], "marker": "V3_DEGRADED.json" in d["ref103"]["files"],
               "smoke_eq_ref": d.get("smoke_files", {}).get("result.geojson") == ref.get("smoke_files", {}).get("result.geojson"),
               "ref103_eq_ref": d["ref103"]["files"].get("result.geojson") == ref["ref103"]["files"].get("result.geojson")}
p("SUMMARY", json.dumps(summ, ensure_ascii=False))
open("/kaggle/working/STATUS", "w").write(json.dumps({"summary": summ, "res": res}, ensure_ascii=False) + "\n")
