# ── main: проба dense_probe.py (bot-5 @COMMIT) на CPU по выходу кернела признаков ─────────────────────────────────
PINS = {"shapely": "shapely==2.2.0"}   # из kaggle/requirements_gor.txt bot-5 (проба: numpy/scipy/shapely)
VERS = deps(PINS)
import collections
DEADLINE = T0 + float(os.environ.get("B9_DEADLINE_H", DEADLINE_H_DEFAULT)) * 3600
WORK = Path(os.environ.get("B9_WORK", "/kaggle/working")); WORK.mkdir(parents=True, exist_ok=True); PARTS = WORK / "parts"; PARTS.mkdir(exist_ok=True)
TAG = os.environ.get("B9_TAG", TAG_DEFAULT); LAYOUTS = [l for l in os.environ.get("B9_LAYOUTS", ",".join(LAYOUTS_DEFAULT)).split(",") if l]
THREADS = os.environ.get("B9_THREADS", "4")
FOLDS = os.environ.get("B9_FOLDS", FOLDS_DEFAULT if "FOLDS_DEFAULT" in dir() else "")  # только с одним LAYOUTS (bot-5 4a39427)
RESUME = os.environ.get("B9_RESUME", RESUME_DEFAULT if "RESUME_DEFAULT" in dir() else "1")
CKPT_DIR = os.environ.get("B9_CKPT_DIR", "")  # пусто → dense_probe: DOUT/ckpt{TAG}
if FOLDS and len(LAYOUTS) != 1:
    raise SystemExit(f"B9_FOLDS={FOLDS!r} требует ровно один LAYOUTS, сейчас {LAYOUTS}")
SUMMARY = dict(kernel=KERNEL_ID, commit=COMMIT, tag=TAG, layouts=LAYOUTS, folds=FOLDS or None, resume=RESUME, versions=VERS, steps={})
def dump_summary(): json.dump(SUMMARY, open(WORK / "run_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("TAG", TAG, "LAYOUTS", LAYOUTS, "FOLDS", FOLDS or "(все)", "RESUME", RESUME, "THREADS", THREADS)
unpack_sources()
# вход: выход кернела признаков (kernel_sources) — ищем gd/dense{TAG}.npz
GDIN = os.environ.get("B9_GD_IN")
if not GDIN:
    hits = sorted(glob.glob(f"/kaggle/input/**/gd/dense{TAG}.npz", recursive=True))
    if not hits: raise SystemExit(f"нет */gd/dense{TAG}.npz в /kaggle/input — сначала кернел признаков {FEATS_KERNEL} (kernel_sources)")
    GDIN = str(Path(hits[0]).parent)
GDIN = Path(GDIN); log("вход признаков:", GDIN, sorted(p.name for p in GDIN.iterdir()))
DOUT = Path(os.environ.get("B9_DOUT", "/tmp/gd")); DOUT.mkdir(parents=True, exist_ok=True)
for n in (f"dense{TAG}.npz", f"files{TAG}.json"):
    (DOUT / n).unlink(missing_ok=True); (DOUT / n).symlink_to(GDIN / n)
shutil.copy(SRC_DIR / "kaggle/gor_dense_inputs.json", DOUT / "gor_dense_inputs.json")   # метки/фолды — из коммита bot-5
if (GDIN / "gor_dense_inputs.json").exists():
    assert sha256_file(GDIN / "gor_dense_inputs.json") == sha256_file(DOUT / "gor_dense_inputs.json"), "gor_dense_inputs.json кернела признаков ≠ коммиту"
SUMMARY["files_sha256"] = sha256_file(DOUT / f"files{TAG}.json"); SUMMARY["inputs_sha256"] = sha256_file(DOUT / "gor_dense_inputs.json")
PROBE = SRC_DIR / "scripts/gor_dense/dense_probe.py"
# по одному раскрою на процесс (результат тот же: сиды фиксированы внутри fit/set_proj; кэш фолдов — внутри раскроя);
# так готовые раскрои сохраняются, даже если на последнем упрёмся в 12-часовой лимит.
for nm in LAYOUTS:
    rem = DEADLINE - time.time()
    if rem < 900: SUMMARY["steps"][nm] = dict(rc="skipped_deadline"); dump_summary(); log("пропуск", nm, "— дедлайн"); continue
    (DOUT / f"peaks{TAG}.json").unlink(missing_ok=True)
    env = dict(DOUT=DOUT, TAG=TAG, LAYOUTS=nm, REPO=SRC_DIR, OMP_NUM_THREADS=THREADS,
               OPENBLAS_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, RESUME=RESUME)
    if FOLDS: env["FOLDS"] = FOLDS
    if CKPT_DIR: env["CKPT_DIR"] = CKPT_DIR
    else: env["CKPT_DIR"] = str(WORK / f"ckpt{TAG}_{nm}")  # персистим чекпоинты фолдов в /kaggle/working
    Path(env["CKPT_DIR"]).mkdir(parents=True, exist_ok=True)
    rc, dt = run_step(f"probe{TAG}:{nm}", [PROBE], env, WORK / f"probe{TAG}_{nm}.log", rem - 300)
    SUMMARY["steps"][nm] = dict(rc=rc, s=round(dt))
    if rc == 0 and (DOUT / f"peaks{TAG}.json").exists():
        shutil.copy(DOUT / f"peaks{TAG}.json", PARTS / f"peaks{TAG}__{nm}.json")
        SUMMARY["steps"][nm]["stats"] = json.load(open(PARTS / f"peaks{TAG}__{nm}.json", encoding="utf-8"))["stats"].get(nm)
    dump_summary()

def merge(paths):
    """Слить peaks по раскроям в один peaks{TAG}.json того же формата, что пишет dense_probe.py."""
    R = None
    for p in paths:
        P = json.load(open(p, encoding="utf-8"))
        if R is None:
            R = {k: v for k, v in P.items() if k not in ("layouts", "stats", "hn_log", "cp_s_chosen")}
            R.update(layouts={}, stats={}, hn_log=collections.Counter(), cp_s_chosen={})
        for k in ("model", "gsds", "tag", "n_tokens", "token_labels", "centre_hit"):
            assert P.get(k) == R.get(k), f"несовпадение {k} в {p}"
        R["layouts"].update(P["layouts"]); R["stats"].update(P["stats"]); R["hn_log"].update(P.get("hn_log", {}))
        for v, c in P.get("cp_s_chosen", {}).items(): R["cp_s_chosen"].setdefault(v, collections.Counter()).update(c)
    return R
parts = [PARTS / f"peaks{TAG}__{nm}.json" for nm in LAYOUTS if (PARTS / f"peaks{TAG}__{nm}.json").exists()]
if parts:
    R = merge(parts); json.dump(R, open(WORK / f"peaks{TAG}.json", "w", encoding="utf-8"), ensure_ascii=False)
    shutil.copy(DOUT / f"files{TAG}.json", WORK / f"files{TAG}.json")
    import tarfile
    with tarfile.open(WORK / f"gor_dense{TAG}_{OUT_SUFFIX}.tgz", "w:gz") as tf:
        for n in (f"peaks{TAG}.json", f"files{TAG}.json", "run_summary.json"):
            if (WORK / n).exists(): tf.add(WORK / n, arcname=n)
        for p in parts: tf.add(p, arcname=f"parts/{p.name}")
    log("выход:", f"gor_dense{TAG}_{OUT_SUFFIX}.tgz", "раскрои:", list(R["layouts"]))
# чекпоинты фолдов (f###.npz/json) — в WORK/ckpt…; слияние кусков LOFO на боксе: python peaks_ckpt.py CK1,CK2 OUT.json
SUMMARY["peaks_ckpt"] = "scripts/gor_dense/peaks_ckpt.py"
SUMMARY["ckpt_dirs"] = {nm: str(WORK / f"ckpt{TAG}_{nm}") for nm in LAYOUTS if (WORK / f"ckpt{TAG}_{nm}").exists()}
SUMMARY["done_layouts"] = [p.name.split("__")[1][:-5] for p in parts]; dump_summary()
ok = len(parts) == len(LAYOUTS)
log("ГОТОВО" if ok else "ЧАСТИЧНО", json.dumps({k: v.get("rc") for k, v in SUMMARY["steps"].items()}))
if not ok: sys.exit(1)
