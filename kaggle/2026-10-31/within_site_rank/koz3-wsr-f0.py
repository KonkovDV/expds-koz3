#!/usr/bin/env python3
"""KOZ3 within-site rank pilot (docs/plans/PROMPT_WITHIN_SITE_RANK_20261031.md @ 5fd33ce, Elena t211u 31.10).
Finetune the SAME map models/spor_focal_r18.pt (ResNet-18 U-Net centres, kurgany_povrezhdennye/tselye, sp_or):
loss = expds.centers.focal_center_loss + expds.centers.site_rank_loss (coeff 1, margin 1; GT centres vs <=8 loudest
false peaks per train site, re-mined every round). Pilot family held out of tiles and pairs; used only for the stop.
Stop metrics: scripts/wsr_peaks.py (old/new map, pilot family) + scripts/wsr_eval.py. One GPU, no credentials, no LB."""
import os, sys, subprocess, time, json, tarfile, shutil, importlib, importlib.metadata
from pathlib import Path
T0 = time.time()
DEADLINE_H = float(os.environ.get("WSR_DEADLINE_H", "5.5"))
PILOT = os.environ.get("WSR_PILOT", "Курганы")
ROUNDS = int(os.environ.get("WSR_ROUNDS", "6")); STEPS = int(os.environ.get("WSR_STEPS", "200"))
OUT = Path("/kaggle/working"); W = Path("/tmp/wsr"); W.mkdir(parents=True, exist_ok=True)
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
def status(s): (OUT / "STATUS.txt").write_text(s + "\n"); log("STATUS", s)
status("RUNNING")
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
if "GPU" not in smi: status("FAILED no GPU"); raise SystemExit("no GPU")
log("gpu", smi.strip())
import torch
cons = "/tmp/cons.txt"
Path(cons).write_text("".join(f"{p}=={importlib.metadata.version(p)}\n" for p in ("torch", "torchvision", "numpy")))
need = []
for mod, pin in {"segmentation_models_pytorch": "segmentation-models-pytorch==0.6.0", "timm": "timm==1.0.29",
                 "rasterio": "rasterio==1.4.3", "pyproj": "pyproj==3.7.1", "shapely": "shapely==2.1.1",
                 "scipy": "scipy==1.16.3", "laspy": "laspy==2.7.0"}.items():
    try: importlib.import_module(mod)
    except Exception: need.append(pin)
if need:
    log("pip", need)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", "-c", cons, *need])
    if r.returncode: status("FAILED pip"); raise SystemExit("pip failed")
INP = Path("/kaggle/input")
def find(name, is_dir=False):
    for p in INP.rglob(name):
        if p.is_dir() == is_dir: return p
    return None
# code + init weights + GT + UTM.json (koz3-wsr-data)
t = find("wsr_code.tar")
if t:
    with tarfile.open(t) as tf: tf.extractall(W)
else:
    shutil.copytree(find("wsr_code", True), W, dirs_exist_ok=True)
INIT = find("spor_focal_r18.pt"); GT = find("gt_kurgan_sp_or.geojson"); UTM = find("utm_by_region.json")
# tile cache (koz3-sp-large-f0-data cache.tar or auto-extracted dir)
ct = find("cache.tar")
if ct:
    with tarfile.open(ct) as tf: tf.extractall(W)
    CACHE = W / "cache_spor_r1_t128_s25"
else:
    CACHE = next(p.parent for p in INP.rglob("meta.json") if (p.parent / "index.csv").exists() and "spor" in str(p.parent))
# raw SpOr rasters (koz3-dino-gor-raw raw/<region>/...) -> /tmp/wsr/root with UTM.json
RAW = max((p for p in INP.rglob("raw") if p.is_dir()), key=lambda p: sum(1 for q in p.iterdir() if q.is_dir() and any("SpOr" in x.name for x in q.iterdir())))
ROOT = W / "root"; ROOT.mkdir(exist_ok=True)
utm = json.loads(UTM.read_text(encoding="utf-8"))
for reg in RAW.iterdir():
    if not reg.is_dir(): continue
    d = ROOT / reg.name; d.mkdir(exist_ok=True)
    for f in reg.rglob("*"):   # real dirs + file symlinks (catalog scan does not follow dir symlinks)
        if f.is_file():
            q = d / f.relative_to(reg); q.parent.mkdir(parents=True, exist_ok=True)
            if not q.exists(): os.symlink(f, q)
    if reg.name in utm: (d / "UTM.json").write_text(json.dumps(utm[reg.name], ensure_ascii=False), encoding="utf-8")
log("inputs", INIT, GT, CACHE, RAW, len(list(ROOT.iterdir())), "regions")
sys.path.insert(0, str(W))
import logging; logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s", stream=sys.stdout)
env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(W)}
def run(cmd, logf, budget):
    with open(logf, "w") as fh:
        try: return subprocess.run(cmd, cwd=W, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=budget).returncode
        except subprocess.TimeoutExpired: return "timeout"
# 1) old map on pilot family (reference for the stop; also the 598-like set)
rc = run([sys.executable, "-X", "utf8", "scripts/wsr_peaks.py", "--root", str(ROOT), "--weights", str(INIT),
          "--out", str(OUT / "pilot_old.geojson"), "--families", PILOT], OUT / "peaks_old.log", 3600)
log("old pilot peaks rc", rc)
if rc != 0: status(f"FAILED old peaks rc={rc}"); raise SystemExit(1)
# 2) finetune (pilot family out)
from expds.site_rank import finetune
budget_s = DEADLINE_H * 3600 - (time.time() - T0) - 1800   # 30 min reserve for pilot peaks + eval
log("finetune budget s", round(budget_s))
try:
    res = finetune(str(INIT), str(CACHE), str(ROOT), str(GT), str(OUT / "spor_focal_r18_wsr.pt"), [PILOT],
                   rounds=ROUNDS, steps_per_round=STEPS, deadline_s=budget_s, report_path=str(OUT / "train_history.json"))
except Exception as e:
    import traceback; traceback.print_exc(); status(f"FAILED finetune {type(e).__name__}: {e}"); raise SystemExit(1)
log("finetune", json.dumps({k: v for k, v in res.items() if k != "history"}))
torch.cuda.empty_cache()
# 3) new map on pilot family, 4) stop metrics
rc = run([sys.executable, "-X", "utf8", "scripts/wsr_peaks.py", "--root", str(ROOT), "--weights", str(OUT / "spor_focal_r18_wsr.pt"),
          "--out", str(OUT / "pilot_new.geojson"), "--families", PILOT], OUT / "peaks_new.log", 3600)
log("new pilot peaks rc", rc)
if rc != 0: status(f"FAILED new peaks rc={rc}"); raise SystemExit(1)
rc = run([sys.executable, "-X", "utf8", "scripts/wsr_eval.py", "--gt", str(GT), "--ref", str(OUT / "pilot_old.geojson"),
          "--new", str(OUT / "pilot_new.geojson"), "--family", PILOT, "--metric", str(W / "materials" / "compute_metrics.py"),
          "--out", str(OUT / "pilot_eval.json")], OUT / "eval.log", 1800)
log("eval rc", rc)
if rc != 0: status(f"FAILED eval rc={rc}"); raise SystemExit(1)
ev = json.loads((OUT / "pilot_eval.json").read_text(encoding="utf-8"))
cmp_ = ev["compare"]; d0 = ev["f1_ref"]["kurgany_povrezhdennye"]["f1_at_sub_thr"]; d1 = ev["f1_new"]["kurgany_povrezhdennye"]["f1_at_sub_thr"]
gate_median = cmp_["median_fp_louder_became"] is not None and cmp_["median_fp_louder_became"] <= 2
gate_f1 = d1 >= d0
report = {"prompt": "PROMPT_WITHIN_SITE_RANK_20261031 @ 5fd33ce", "pilot_family": PILOT, "wall_s": round(time.time() - T0),
          "median_fp_louder_was": cmp_["median_fp_louder_was"], "median_fp_louder_became": cmp_["median_fp_louder_became"],
          "median_fp_louder_became_with_inpoly_peak": cmp_["median_fp_louder_became_with_inpoly_peak"],
          "damaged_f1_old": d0, "damaged_f1_new": d1, "gate_median_le_2": gate_median, "gate_damaged_f1_not_worse": gate_f1,
          "verdict": "PASS" if (gate_median and gate_f1) else "STOP",
          "extras": {"n_quiet_ref": cmp_["n_quiet_ref"], "quiet_now_no_inpoly_peak": cmp_["quiet_ref_now_no_inpoly_peak"],
                     "quiet_now_kept": cmp_["quiet_ref_now_kept"], "gap_to_site_max_was": cmp_["median_gap_to_site_max_was"],
                     "gap_to_site_max_became": cmp_["median_gap_to_site_max_became"], "rank_was": cmp_["median_rank_was"],
                     "rank_became": cmp_["median_rank_became"]},
          "f1_ref": ev["f1_ref"], "f1_new": ev["f1_new"], "ref_summary": ev["ref"], "new_summary": ev.get("new"),
          "train": {k: v for k, v in res.items()}}
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=1))
status(f"DONE {report['verdict']} {time.time()-T0:.0f}s")
