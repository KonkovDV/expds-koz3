# W4 01.11: паритет «надёжного» v2clean 545ef32a (без пересборки) в окружениях C/A/B/D прогона v3onfix2.
# REF_check4 = точный повтор check4 (Kaggle по умолчанию, check_submission из koz3-w4-v2clean-check) + --keep → эталон sha.
# C/A/B/D — новый check_submission (с проверкой V3_DEGRADED) из koz3-w4-v3onfix. Плюс прямой main.py на участке 103.
import glob, hashlib, json, os, pathlib, shutil, site, subprocess, sys, time, zipfile
SHA = "545ef32a70b4994f4d694828ae5eb28f27c9161d7f19a96f5a9d3833b87e093f"
T = "/kaggle/temp"; os.makedirs(T, exist_ok=True)
out = open("/kaggle/working/check_submission.log", "w")
def p(*a):
    s = " ".join(map(str, a)); print(s, flush=True); out.write(s + "\n"); out.flush()
def sha(f): return hashlib.sha256(open(f, "rb").read()).hexdigest()
Z = f"{T}/koz3_v2clean_70d335a.zip"; shutil.copy(glob.glob("/kaggle/input/**/v2clean.zip.bin", recursive=True)[0], Z)
chks = glob.glob("/kaggle/input/**/check_submission.py", recursive=True)
CHK_OLD = [f for f in chks if "v2clean" in f][0]; CHK_NEW = [f for f in chks if "v3onfix" in f][0]
p("zip_sha_ok", sha(Z) == SHA, "chk_old", CHK_OLD, sha(CHK_OLD)[:12], "chk_new", CHK_NEW, sha(CHK_NEW)[:12], "python", sys.version.split()[0])
def vers(tag):
    r = subprocess.run([sys.executable, "-c", "import importlib\nfor m in ('regex','transformers','tokenizers','huggingface_hub'):\n"
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
def run(tag, chk):
    vers(tag); t = time.time(); wd = f"{T}/wd_{tag}"; os.makedirs(wd, exist_ok=True)
    r = subprocess.run([sys.executable, chk, Z, "--workdir", wd, "--timeout", "1500", "--python", sys.executable, "--keep"],
                       capture_output=True, text=True)
    p(f"===== CHECK {tag} =====\n" + r.stdout); p("STDERR:", r.stderr[-3000:])
    lines = r.stdout.splitlines(); itog = [l for l in lines if "ИТОГ" in l]
    d = {"rc": r.returncode, "itog": itog[-1] if itog else None, "wall_s": round(time.time() - t),
         "fmt": [l for l in lines if "] format" in l or "записано" in l][:3],
         "v3_lines": [l for l in lines if "] v3 " in l or "] dfine " in l or "V3_DEGRADED" in l][:8]}
    sm = glob.glob(f"{wd}/b8smoke_*/out") or glob.glob(f"{wd}/**/out/result.geojson", recursive=True)
    if sm:
        o = pathlib.Path(sm[0]); o = o.parent if o.is_file() else o
        d["smoke_files"] = {f.name: [sha(f), f.stat().st_size] for f in sorted(o.iterdir()) if f.is_file()}
    code = f"{T}/code_{tag}"; shutil.rmtree(code, ignore_errors=True); zipfile.ZipFile(Z).extractall(code)
    od = f"{T}/out_{tag}"; shutil.rmtree(od, ignore_errors=True); os.makedirs(od)
    t = time.time()
    m = subprocess.run([sys.executable, "inference/main.py", REF, f"{od}/result.geojson"], cwd=code, env=ENV,
                       capture_output=True, text=True, timeout=3000)
    log = m.stdout + m.stderr
    d["ref103"] = {"rc": m.returncode, "wall_s": round(time.time() - t),
                   "files": {f: [sha(f"{od}/{f}"), os.path.getsize(f"{od}/{f}")] for f in sorted(os.listdir(od))},
                   "warn": [l[-300:] for l in log.splitlines() if "V3_DEGRADED" in l or "UNAVAILABLE" in l or "Traceback" in l][:6]}
    p(tag, json.dumps(d, ensure_ascii=False)); res[tag] = d
    shutil.rmtree(code, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True)
run("REF_check4", CHK_OLD)
run("C_default", CHK_NEW)
sp = site.getsitepackages()[0]; pth = f"{sp}/zz_w4_preload_regex.pth"
open(pth, "w").write("import regex\n")
run("A_regex2025_preloaded", CHK_NEW)
os.remove(pth)
pip(["uninstall", "-y", "regex"])
run("B_no_regex", CHK_NEW)
WH3 = os.path.dirname(glob.glob("/kaggle/input/**/transformers-5.21.0-*.whl", recursive=True)[0])
pip(["install", "-q", "--no-index", "--find-links", WH3, "transformers==5.21.0", "tokenizers==0.23.3", "regex==2026.9.29",
     "huggingface-hub", "safetensors", "typer==0.27.3", "rich==15.0.0"])
run("D_new_env521", CHK_NEW)
ref = res["REF_check4"]
def g(d, k): return (d.get(k, {}) if k == "smoke_files" else d["ref103"]["files"]).get("result.geojson", [None])[0]
summ = {}
for k, d in res.items():
    summ[k] = {"rc": d["rc"], "itog": d["itog"],
               "marker": "V3_DEGRADED.json" in d.get("smoke_files", {}) or "V3_DEGRADED.json" in d["ref103"]["files"],
               "smoke_sha": g(d, "smoke_files"), "smoke_eq_ref": g(d, "smoke_files") == g(ref, "smoke_files"),
               "smoke_bytes": d.get("smoke_files", {}).get("result.geojson", [0, 0])[1],
               "ref103_sha": g(d, "ref103"), "ref103_eq_ref": g(d, "ref103") == g(ref, "ref103"), "fmt": d["fmt"][:1]}
p("SUMMARY", json.dumps(summ, ensure_ascii=False))
open("/kaggle/working/STATUS", "w").write(json.dumps({"summary": summ, "res": res}, ensure_ascii=False) + "\n")
