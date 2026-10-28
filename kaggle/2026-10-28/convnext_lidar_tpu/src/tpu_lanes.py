# ── Шаг 10 (TPU). Полосы: процесс на чип (независимые задания, без обмена градиентами) | 1 процесс на 1 устройстве ──
def _self_src():
    """Исходник этого ядра для воркеров: __file__ → argv[0] → /kaggle/src/*.py (по метке @@TPU_KERNEL@@)."""
    cand = [globals().get("__file__"), sys.argv[0] if sys.argv else None] + sorted(glob.glob("/kaggle/src/*.py"))
    for c in cand:
        try:
            if c and os.path.isfile(c) and "@@TPU_KERNEL@@" in open(c, encoding="utf-8").read(): return c
        except Exception: pass
    return None
def _chip_env(i, jobs, prep_dir):
    p0, p1 = 8476 + i, 8576 + i                   # свои порты на процесс (несвязанные процессы, 1 чип на процесс)
    env = dict(os.environ, CNX_TPU_CHIP=str(i), CNX_JOBS=",".join(f"{j['variant']}:{j['fold']}" for j in jobs),
               CNX_DEADLINE_ABS=repr(DEADLINE), CNX_REGIONS=CFG["regions"], CNX_VENDOR_DIR=f"/tmp/cnx_vendor_chip{i}",
               CNX_RELIEF_DIR=f"/tmp/cnx_relief_chip{i}", CNX_BUILD_RELIEF="0", PYTHONUNBUFFERED="1")   # каналы — только из prep/главного
    if prep_dir: env["CNX_PREP_DIR"] = str(prep_dir)
    if XLA_ON and not ARGS.allow_cpu:
        env.update(PJRT_DEVICE="TPU", TPU_VISIBLE_CHIPS=str(i), TPU_VISIBLE_DEVICES=str(i), TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1",
                   TPU_PROCESS_BOUNDS="1,1,1", TPU_PROCESS_PORT=str(p0), TPU_PROCESS_ADDRESSES=f"localhost:{p0}",
                   TPU_MESH_CONTROLLER_ADDRESS=f"localhost:{p1}", TPU_MESH_CONTROLLER_PORT=str(p1), CLOUD_TPU_TASK_ID="0")
    return env
def _run_chips(lanes):
    src = _self_src()
    if src is None: raise SystemExit("не найден исходник ядра для воркеров (__file__/argv[0]/kaggle/src) — задайте CNX_TPU_PROCS=1")
    prep_dir = PREP
    if prep_dir is None:          # каналы собраны здесь (smoke) → манифест, чтобы воркеры взяли их, а не собирали заново
        json.dump(dict(regions=sorted(META), tags=list(TAGS5), vendor=RELIEF_SHA, backend=XA.backend(), rvt=_ver("rvt"), res=CFG["res"],
                       dem_res=0.1, dem_fill="delaunay"), open(RELIEF_DIR / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        prep_dir = RELIEF_DIR
    (OUT / "logs_tpu").mkdir(exist_ok=True); procs = []
    for i, jobs in enumerate(lanes):
        if not jobs: continue
        fh = open(OUT / "logs_tpu" / f"chip{i}.log", "w", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-u", src, *sys.argv[1:]], env=_chip_env(i, jobs, prep_dir), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        def pump(p=p, fh=fh, i=i):
            for line in p.stdout:
                fh.write(line); fh.flush()
                if any(s in line for s in ("ит ", "ОШИБКА", "БЮДЖЕТ", "бюджет", "готово", "инференс", "Traceback", "Error", "TPU", "ВНИМАНИЕ", "ИТОГ")):
                    print(f"[chip{i}] {line.rstrip()}", flush=True)
            fh.close()
        th = threading.Thread(target=pump, daemon=True); th.start(); procs.append((i, p, th, jobs))
        names = ",".join(f"{j['variant']}:{j['fold']}" for j in jobs); log(f"chip{i}: pid {p.pid}, задания {names}")
    for i, p, th, jobs in procs:
        rc = p.wait(); th.join(timeout=60); rp = OUT / f"report_cnx_{MODE}_chip{i}.json"
        if rp.exists():
            r = json.loads(rp.read_text(encoding="utf-8")); REPORT["jobs"] += r["jobs"]; ERR.extend(tuple(e) for e in r["errors"])
            REPORT.setdefault("chips", {})[i] = {k: r.get(k) for k in ("xla", "t_total_s", "devices")}
        else:
            ERR.append((f"chip{i}", f"воркер rc={rc} без отчёта (см. logs_tpu/chip{i}.log)"))
            REPORT["jobs"] += [dict(job=j, error=f"chip{i} rc={rc}") for j in jobs]
        log(f"chip{i}: rc={rc}")
REPORT.update(tpu=TPU_INFO, nproc=NPROC, xla_dev=XLA_DEV)
REPORT["versions"].update({m: _ver(m) for m in ("torch_xla", "libtpu", "jax", "jaxlib")})
if IS_WORKER:
    lane(0, DEV, JOBS)
    (OUT / f"report_cnx_{MODE}_chip{TPU_CHIP}.json").write_text(json.dumps(dict(
        chip=int(TPU_CHIP), jobs=REPORT["jobs"], errors=[list(e) for e in ERR], xla=_xla_metrics(), devices=str(DEV),
        t_total_s=round(time.time() - T0, 1)), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("ИТОГ воркера:", "ошибки " + str(ERR) if ERR else "OK"); raise SystemExit(1 if ERR else 0)
NL = min(NPROC, len(JOBS)); lanes = [[] for _ in range(NL)]
for i, j in enumerate(JOBS): lanes[i % NL].append(j)
log("полосы:", [[f"{j['variant']}:{j['fold']}" for j in l] for l in lanes])
if NL <= 1 and DEV is not None: lane(0, DEV, JOBS)
else: _run_chips(lanes)
REPORT["jobs"].sort(key=lambda r: (r["job"]["variant"], r["job"]["fold"]))
# эталон T4 — bs 8 (CONVNEXT.md:86). TPU против T4 (критерий: TPU > 1,5× медленнее T4 на сэмпл → закрываем TPU-ветку)
_t4 = [float(v) for v in os.environ.get("CNX_T4_S_PER_IT", T4_S_PER_IT_DEFAULT).split(",")]
_sp = [r["s_per_sample_steady"] for r in REPORT["jobs"] if "s_per_sample_steady" in r]
if _sp:
    t4s = [v / 8 for v in _t4]; tpu_s = sorted(_sp)[len(_sp) // 2]; n_par = len([l for l in lanes if l])
    ratio = [tpu_s / t for t in t4s]
    REPORT["tpu_vs_t4"] = dict(tpu_s_per_sample_per_chip_median=tpu_s, per_job=_sp, t4_s_per_it=_t4, t4_s_per_sample=t4s,
                               t4_source=T4_SOURCE, ratio_per_device=[round(x, 3) for x in ratio], chips_parallel=n_par,
                               session_samples_per_s_tpu=round(n_par / tpu_s, 2), session_samples_per_s_2xT4=[round(2 / t, 2) for t in t4s],
                               compile_first_it_s=[r.get("t_first_it_s") for r in REPORT["jobs"]],
                               recompiles_after_warm=[r.get("recompiles_after_warm") for r in REPORT["jobs"]],
                               data_wait_frac=[r.get("data_wait_frac") for r in REPORT["jobs"]],
                               verdict="ЗАКРЫТЬ (TPU > 1,5× медленнее T4 на сэмпл)" if min(ratio) > 1.5 else
                                       ("ПОГРАНИЧНО" if max(ratio) > 1.5 else "ПРОДОЛЖАТЬ (TPU ≤ 1,5× T4 на сэмпл)"))
    log("TPU vs T4:", json.dumps(REPORT["tpu_vs_t4"], ensure_ascii=False))
