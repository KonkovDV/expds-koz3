C_CLASSES = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
CACHE = TMP / "cache"; DET = TMP / "det"; DET.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "PYTHONPATH": str(SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
       "OMP_NUM_THREADS": os.environ.get("B9_THREADS", "2")}

def run(cmd, logname):
    log("RUN", " ".join(map(str, cmd))[:220])
    lp = WORK / logname
    with open(lp, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    SUMMARY["steps"][logname] = dict(rc=p.returncode); dump()
    if p.returncode:
        print(lp.read_text(encoding="utf-8")[-2000:])
        raise SystemExit(f"rc={p.returncode} {logname}")
    return p

run([sys.executable, "-m", "expds.tiles", "--root", str(DATA), "--gt", str(GT), "--markup", "sp_or",
     "--out", str(CACHE / "spor_det05"), "--res", "0.5", "--tile", "640", "--classes", *C_CLASSES],
    "tiles_windows.log")
run([sys.executable, "-m", "expds.tiles", "--root", str(DATA), "--gt", str(GT), "--markup", "sp_or",
     "--out", str(CACHE / "spor_det05_grid"), "--res", "0.5", "--tile", "640", "--grid-overlap", "0.25",
     "--classes", *C_CLASSES], "tiles_grid.log")
df_tmp()

folds = [0] if MODE == "smoke" else list(range(5))
for k in folds:
    run([sys.executable, str(SRC_DIR / "scripts/coco_from_cache.py"),
         "--cache", str(CACHE / "spor_det05"), "--out", str(DET / f"f{k}"),
         "--layout", "rfdetr", "--images", "--folds", "5", "--fold", str(k)], f"coco_f{k}.log")
    run([sys.executable, str(SRC_DIR / "scripts/coco_from_cache.py"),
         "--cache", str(CACHE / "spor_det05_grid"), "--out", str(DET / f"grid_f{k}"),
         "--layout", "rfdetr", "--images", "--folds", "5", "--fold", str(k)], f"coco_grid_f{k}.log")
    tr = DET / f"grid_f{k}" / "train"
    if tr.exists():
        for p in tr.glob("*.png"): p.unlink(missing_ok=True)
    df_tmp()

shutil.rmtree(CACHE, ignore_errors=True); df_tmp()
OUT = WORK / "rfdetr"; OUT.mkdir(parents=True, exist_ok=True)
COCO_W = os.environ.get("B9_COCO_W", "")
if not COCO_W:
    hits = sorted(glob.glob("/kaggle/input/**/rf-detr-medium.pth", recursive=True))
    if hits: COCO_W = hits[0]

if MODE == "smoke":
    cmd = [sys.executable, str(SRC_DIR / "scripts/rfdetr_train.py"),
           "--data", str(DET / "f0"), "--out", str(OUT / "smoke"),
           "--model", MODEL, "--size", str(SIZE), "--fp16", "--init", INIT, "--smoke"]
    if COCO_W: cmd += ["--coco-weights", COCO_W]
    run(cmd, "rf_smoke.log")
else:
    env2 = {**env, "INIT": INIT, "SIZE": str(SIZE)}
    if COCO_W: env2["COCO_W"] = COCO_W
    sh = SRC_DIR / "kaggle/rfdetr_folds.sh"
    log("folds.sh", sh, "ITERS", ITERS)
    p = subprocess.run(["bash", str(sh), str(DET), str(OUT), MODEL, str(ITERS)], env=env2, cwd=str(SRC_DIR))
    SUMMARY["steps"]["rfdetr_folds"] = dict(rc=p.returncode); dump()
    if p.returncode: raise SystemExit(f"folds rc={p.returncode}")

with tarfile.open(WORK / "rfdetr_preds.tgz", "w:gz") as t:
    for p in OUT.rglob("*"):
        if p.is_file(): t.add(p, arcname=str(p.relative_to(WORK)))
    dump(); t.add(WORK / "run_summary.json", arcname="run_summary.json")
SUMMARY["done"] = True; SUMMARY["t_total_s"] = round(time.time() - T0, 1); dump()
log("ГОТОВО", SUMMARY["t_total_s"], "с")
