# KOZ3 №3 · koz3-b5-gordense-timing (bot-5): ЗАМЕР ВРЕМЕНИ на T4, без обучения. Пайплайн v4 + gor_dense ВКЛ. (блок
# включения из configs/default.toml ветки bot5-gor-dense-v4 @86a60c5), ORGATE ВЫКЛ., sp_dfine f0 (вариант R, make_on.py
# bot-7), если смонтирован kernel_source koz3-dfine-spor-full-rest. Вход = все 132 участка трейна с SpOr (175 tif, 5,40 ГБ;
# прокси тестового объёма: публичный тест SpOr 31 файл / 30 участков / 4,68 ГБ, bot-8/BUDGET.md:112 → прокси ≥ теста по ГБ
# (×1,15) и по файлам (×5,6), т.е. верхняя оценка). Данные: S3 train-expds-3 (как кернелы bot-9, ключи из Kaggle Secrets
# KOZ3_S3_* → env → датасет koz3-s3-cred). «Сканирование» и данные КОЗ №2 НЕ используются.
# Шаги: GPU-чек (жёстко) → пины → код+веса из приватного датасета (md5) → S3 → проверка CUDA у DINOv2 → прогон ON
# (main.py, замер) → при остатке времени прогон OFF (та же выборка, флаги v4 по умолчанию) для дельты.
# Жёсткий таймаут всего кернела HARD_S (55 мин); итог /kaggle/working/run_summary.json + logs/.
import os, sys, subprocess, json, time, hashlib, glob, re, shutil, tarfile, threading
from pathlib import Path
T0 = time.time(); HARD_S = float(os.environ.get("B5_HARD_S", 3300))
W = Path(os.environ.get("B5_WORK", "/kaggle/working")); LOGS = W / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("B5_DATA", "/tmp/data")); ROOTS = Path("/tmp/roots")
COMMIT = "86a60c5"; KERNEL = "dmitrijkv/koz3-b5-gordense-timing"
TARGETS = {"working_target_s": 960, "bot10_criterion_s": 900, "platform_limit_s": 1800}
MD5 = {"repo_86a60c5.tar": "5f481befaa1b0df1a43d4de321aa2a66",
       "models/dinov2-small/config.json": "689053885e41f529baca946d85d9b8dd",
       "models/dinov2-small/model.safetensors": "ea251be6e151ba8609ad0bc9b2e13a78",
       "models/dinov2-small/preprocessor_config.json": "e40d527806779cbf0c7fd6fca552eddb",
       "models/gor_gate/gor_v3_head.npz": "acb95c52956d712a16c3dd389237a691",
       "models/gor_dense/gor_dense_head.npz": "e35570e10d17edac32bea8d26fa0f5f4"}
F0_MD5 = {"config.json": "def4fb2d30fdad17ed43a02fb1ebfc2e", "model.safetensors": "3fe57318c9a17ccc0351fc00aae47f17"}
HF_DINO = ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056")   # Apache-2.0; запасной путь, если нет в датасете
SUMMARY = dict(kernel=KERNEL, commit=COMMIT, targets=TARGETS, steps={}, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
def log(*a): print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)
def left(): return HARD_S - (time.time() - T0)
def save():
    SUMMARY["elapsed_s"] = round(time.time() - T0, 1)
    json.dump(SUMMARY, open(W / "run_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
def die(msg): SUMMARY["fatal"] = msg; save(); log("FATAL:", msg); raise SystemExit(msg)
def md5(p, bs=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()
threading.Timer(HARD_S, lambda: (log("HARD TIMEOUT", HARD_S), SUMMARY.update(fatal="hard timeout"), save(), os._exit(3))).start()

# 1. GPU: только CUDA, без CPU-фолбэка
r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
log("nvidia-smi -L:", r.stdout.strip() or r.stderr.strip())
if r.returncode or "GPU 0" not in r.stdout: die("нет GPU (второй одновременный GPU-кернел?) — останов без CPU-фолбэка")
SUMMARY["gpu"] = r.stdout.strip()

# 2. пины (transformers 5.21.0 — как вендор inference/lib/vendor; остальные — requirements_gor bot-5 / requirements.txt репо)
PINS = ["transformers==5.21.0", "rasterio==1.5.2", "pyproj==3.8.0", "shapely==2.2.0", "boto3==1.40.61"]
OPT = ["lightgbm", "laspy[lazrs]", "segmentation-models-pytorch", "timm"]   # нужны импортам пайплайна; ставятся, только если нет
def have(mod):
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0
need = [p for p in PINS if subprocess.run([sys.executable, "-c", f"import {p.split('==')[0]} as m; assert m.__version__=='{p.split('==')[1]}'"],
                                         capture_output=True).returncode]
need += [p for p, m in zip(OPT, ["lightgbm", "laspy", "segmentation_models_pytorch", "timm"]) if not have(m)]
if need:
    log("pip install:", need); t = time.time()
    rr = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", *need])
    if rr.returncode: die(f"pip install не удался: {need} (нужен enable_internet)")
    SUMMARY["steps"]["pip"] = dict(pkgs=need, s=round(time.time() - t, 1))
vers = subprocess.run([sys.executable, "-c", "import json,torch,transformers,rasterio,numpy;print(json.dumps(dict(torch=torch.__version__,"
                      "cuda=torch.version.cuda,transformers=transformers.__version__,rasterio=rasterio.__version__,numpy=numpy.__version__)))"],
                      capture_output=True, text=True)
log("версии:", vers.stdout.strip() or vers.stderr[-400:]); SUMMARY["versions"] = vers.stdout.strip()

# 3. код @86a60c5 и веса из приватного датасета koz3-b5-gordense-timing (md5 каждого файла)
def find(rel):
    for base in sorted(glob.glob("/kaggle/input/*")):
        for p in (Path(base) / rel, Path(base) / Path(rel).name):
            if p.is_file(): return p
    hits = sorted(glob.glob(f"/kaggle/input/**/{Path(rel).name}", recursive=True))
    hits = [h for h in hits if Path(rel).parent.name in ("", Path(h).parent.name) or Path(rel).parent.as_posix() == "."]
    return Path(hits[0]) if hits else None
SRC = ROOTS / "src"; SRC.mkdir(parents=True, exist_ok=True); t = time.time()
tarp = find("repo_86a60c5.tar")
if tarp is not None:
    m = md5(tarp)
    if m != MD5["repo_86a60c5.tar"]: die(f"repo_86a60c5.tar md5 {m} != {MD5['repo_86a60c5.tar']}")
    with tarfile.open(tarp) as tf: tf.extractall(SRC)
else:   # Kaggle мог распаковать архив при загрузке датасета
    hits = sorted(glob.glob("/kaggle/input/**/inference/main.py", recursive=True))
    if not hits: die("в /kaggle/input нет repo_86a60c5.tar и распакованного репо (inference/main.py)")
    shutil.copytree(Path(hits[0]).parents[1], SRC, dirs_exist_ok=True, symlinks=False)
    log("репо: распакованный каталог", Path(hits[0]).parents[1], "(md5 tar не проверить; сверяю gor_dense.py)")
if not (SRC / "expds/gor_dense.py").is_file(): die("в репо нет expds/gor_dense.py — не та ветка (нужна bot5-gor-dense-v4)")
got = {}
for rel, want in MD5.items():
    if rel.endswith(".tar"): continue
    p = find(rel)
    if p is None and rel.startswith("models/dinov2-small/"):
        from huggingface_hub import hf_hub_download
        p = Path(hf_hub_download(HF_DINO[0], Path(rel).name, revision=HF_DINO[1])); log("из HF", HF_DINO, Path(rel).name)
    if p is None: die(f"нет файла датасета: {rel}")
    m = md5(p); got[rel] = m
    if m != want: die(f"{rel}: md5 {m} != {want}")
    dst = SRC / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(p, dst)
F0 = None
for c in sorted(glob.glob("/kaggle/input/**/dfine_f0/model.safetensors", recursive=True)):
    d = Path(c).parent
    if all((d / f).is_file() and md5(d / f) == h for f, h in F0_MD5.items()): F0 = d; break
if F0 is not None:
    (SRC / "models/dfine_f0").mkdir(parents=True, exist_ok=True)
    for f in F0_MD5: shutil.copy(F0 / f, SRC / "models/dfine_f0" / f)
log("sp_dfine f0:", F0 or "НЕ смонтирован — sp_dfine ВЫКЛ. (замер без D-FINE SpOr!)")
SUMMARY["steps"]["code_weights"] = dict(md5=got, dfine_f0=str(F0) if F0 else None, s=round(time.time() - t, 1)); save()

# 4. конфиги: OFF = v4 как есть; ON = блок включения gor_dense (+ sp_dfine R, если есть f0). Для ЗАМЕРА сняты
#    ограничители времени (иначе сторож/потолок урежут работу и время занизится): budget_s 3000, watchdog/окна/мягкий потолок выкл.
base = (SRC / "configs/default.toml").read_text(encoding="utf-8")
def sub1(pat, rep, s):
    s2, n = re.subn(pat, rep, s, flags=re.M); assert n == 1, pat; return s2
def timing_cfg(c):
    c = sub1(r"^budget_s = 1500\b", "budget_s = 3000", c); c = sub1(r"^watchdog = true\b", "watchdog = false", c)
    c = sub1(r"^window_deadline = true\b", "window_deadline = false", c); c = sub1(r"^soft_ceiling_s = 1200\b", "soft_ceiling_s = 0", c)
    assert re.search(r"(?m)^\[orgate\]\nenabled = false$", c), "ORGATE должен быть ВЫКЛ."
    return c
THR_F0 = {"kurgany_povrezhdennye": 0.555265, "kurgany_tselye": 0.670285, "gorodishcha": 1.01,   # gorodishcha 1.01 = noD (bot-10)
          "arkhitektury": 0.373015, "fortifikatsii": 0.869051}   # остальные — bot-7 make_on.py (bot-9 sweep.log R_max, f0)
on = sub1(r"^\[gor_dense\]\nenabled = false\n", '''[gor_dense]
enabled = true
backbone = "models/dinov2-small"
gate_head = "models/gor_gate/gor_v3_head.npz"
head = "models/gor_dense/gor_dense_head.npz"
head_md5 = "e35570e10d17edac32bea8d26fa0f5f4"
centre_s = 250.0

[thresholds_by_markup.gor_dense]
gorodishcha = 0.824066

[thresholds_by_markup.dfine_ae]
gorodishcha = 1.01
''', base)
if F0 is not None:
    on = sub1(r"^\[sp_dfine\]\nenabled = false\n", '[sp_dfine]\nenabled = true\nweights = ["models/dfine_f0"]\nflip_tta = true\nfp16 = true\n'
              'convert = "circle"\ndrop_groups = ["sp_or", "sp_focal", "sp_arch", "sp_focal2"]\n\n[thresholds_by_markup.sp_dfine]\n'
              + "\n".join(f"{k} = {v}" for k, v in THR_F0.items()) + "\n", on)
    on = sub1(r"^(\[thresholds_by_markup\.or\]\narkhitektury = )0\.065565", r"\g<1>0.483891", on)
else:   # без f0 таблицы sp_dfine нет — ставим её с одним ключом (noD конфигом, как в блоке включения)
    on = sub1(r"^(\[thresholds_by_markup\.gor_dense\]\ngorodishcha = 0\.824066\n)", r"\g<1>\n[thresholds_by_markup.sp_dfine]\ngorodishcha = 1.01\n", on)
import tomllib
for nm, c in (("on", timing_cfg(on)), ("off", timing_cfg(base))):
    d = ROOTS / nm
    if not d.exists(): shutil.copytree(SRC, d, symlinks=True)
    (d / "configs/default.toml").write_text(c, encoding="utf-8"); tomllib.loads(c)
    (LOGS / f"default_{nm}.toml").write_text(c, encoding="utf-8")
log("конфиги on/off записаны (logs/default_*.toml)")

# 5. данные S3: только SpOr *.tif и UTM.json (структура ключей = структура папок train-expds-3)
BUCKET, ENDPOINT = "train-expds-3", os.environ.get("B9_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
def s3_client():
    import boto3
    from botocore.config import Config
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient(); kid, sec = u.get_secret("KOZ3_S3_ACCESS_KEY_ID"), u.get_secret("KOZ3_S3_SECRET_ACCESS_KEY")
    except Exception as e: log("Kaggle Secrets недоступны:", type(e).__name__)
    if not kid or not sec: kid, sec = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not kid or not sec:
        for p in sorted(glob.glob("/kaggle/input/**/cred.json", recursive=True)):
            try: c = json.load(open(p, encoding="utf-8"))
            except Exception: continue
            kid, sec = c.get("id") or c.get("aws_access_key_id"), c.get("key") or c.get("aws_secret_access_key")
            if kid and sec: log("ключи S3 из приватного датасета"); break
    if not kid or not sec: die("нет ключей S3 (Secrets KOZ3_S3_* или датасет koz3-s3-cred)")
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=kid, aws_secret_access_key=sec, region_name="ru-7",
                        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=32))
def list_prefix(cli, prefix, delim=None):
    out, tok = [], None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix, **({"Delimiter": delim} if delim else {}), **({"ContinuationToken": tok} if tok else {}))
        r = cli.list_objects_v2(**kw)
        out += [p["Prefix"] for p in r.get("CommonPrefixes", [])] if delim else [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"): return out
        tok = r["NextContinuationToken"]
def fetch(cli, key, size):
    from boto3.s3.transfer import TransferConfig
    dst = DATA / key
    if dst.exists() and dst.stat().st_size == size: return
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_name(dst.name + ".part")
    for a in range(5):
        try:
            cli.download_file(BUCKET, key, str(tmp), Config=TransferConfig(max_concurrency=4)); tmp.rename(dst)
            assert dst.stat().st_size == size; return
        except Exception as e: log("повтор", a + 1, key, type(e).__name__); time.sleep(5 * (a + 1))
    raise RuntimeError("не скачан " + key)
import concurrent.futures as cf
t = time.time(); CLI = s3_client()
REGS = [p.rstrip("/") for p in list_prefix(CLI, "", "/")]
def inv(reg):
    out = []
    for k, s in list_prefix(CLI, reg + "/"):
        parts = k.split("/"); n = parts[-1]
        if n == "UTM.json" and len(parts) == 2: out.append((k, s))
        elif len(parts) == 3 and "SpOr" in parts[1] and n.lower().endswith((".tif", ".tiff")): out.append((k, s))
    return out if any(k.lower().endswith((".tif", ".tiff")) for k, _ in out) else []
with cf.ThreadPoolExecutor(16) as ex: KEYS = [x for l in ex.map(inv, REGS) for x in l]
ntif = sum(k.lower().endswith((".tif", ".tiff")) for k, _ in KEYS); nutm = sum(k.endswith("UTM.json") for k, _ in KEYS)
gb = sum(s for k, s in KEYS) / 1e9; nreg = len({k.split("/")[0] for k, _ in KEYS})
log(f"S3: {ntif} tif + {nutm} UTM.json, {gb:.2f} ГБ, участков {nreg}")
if not os.environ.get("B5_ANY_DATA") and (ntif, nutm) != (175, 132): die(f"ожидалось 175 tif / 132 UTM.json (bot-9 s3/listing.txt), есть {ntif}/{nutm}")
with cf.ThreadPoolExecutor(8) as ex: list(ex.map(lambda ks: fetch(CLI, *ks), KEYS))
SUMMARY["steps"]["download"] = dict(regions=nreg, tif=ntif, utm=nutm, gb=round(gb, 3), s=round(time.time() - t, 1)); save()
log("данные скачаны", SUMMARY["steps"]["download"])

# 6. CUDA у DINOv2 тем же кодом, что фаза gor_dense (gor_gate._device + load_backbone, fp16 на cuda)
chk = r'''
import sys, json, time, tomllib, torch
from pathlib import Path
R = Path(sys.argv[1]); sys.path.insert(0, str(R / "inference" / "lib")); sys.path.insert(0, str(R))
from expds import gor_dense as GD, gor_gate
st = GD.settings(tomllib.load(open(R / "configs/default.toml", "rb"))); dev = gor_gate._device(st)
m = gor_gate.load_backbone(st, R, dev); m = m.half() if dev.type == "cuda" else m
x = torch.zeros(1, 3, 518, 518, device=dev, dtype=next(m.parameters()).dtype); t = time.time()
with torch.no_grad(): m(pixel_values=x)
torch.cuda.synchronize() if dev.type == "cuda" else None
print("CUDA_CHECK", json.dumps(dict(device=str(dev), param_device=str(next(m.parameters()).device), dtype=str(next(m.parameters()).dtype),
      cuda=torch.cuda.is_available(), name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, fwd_s=round(time.time() - t, 3))))
'''
rr = subprocess.run([sys.executable, "-c", chk, str(ROOTS / "on")], capture_output=True, text=True, timeout=max(60, min(600, left() - 60)))
line = next((l for l in rr.stdout.splitlines() if l.startswith("CUDA_CHECK")), None)
log(line or ("CUDA_CHECK FAILED " + rr.stderr[-1500:]))
cc = json.loads(line.split(" ", 1)[1]) if line else None; SUMMARY["cuda_check"] = cc or rr.stderr[-1500:]; save()
if not cc or not cc["param_device"].startswith("cuda"): die("DINOv2 НЕ на cuda — замер недействителен")

# 7. прогоны main.py; nvidia-smi раз в 10 с (доказательство работы GPU: пик памяти/загрузки)
def run_main(nm):
    out = W / f"out_{nm}"; out.mkdir(exist_ok=True); lf = LOGS / f"main_{nm}.log"; smi = []; stop = threading.Event()
    def poll():
        while not stop.wait(10):
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            try: u, mem = map(int, q.stdout.strip().splitlines()[0].split(",")); smi.append((u, mem))
            except Exception: pass
    th = threading.Thread(target=poll, daemon=True); th.start()
    env = dict(os.environ, EXPDS_SOFT_CEILING_S="0", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
    tmo = left() - 90; log(f"▶ main.py {nm}: таймаут {tmo:.0f} с")
    t = time.time()
    with open(lf, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([sys.executable, str(ROOTS / nm / "inference/main.py"), str(DATA), str(out / "result.geojson")],
                             env=env, stdout=fh, stderr=subprocess.STDOUT)
        try: rc = p.wait(timeout=max(30, tmo))
        except subprocess.TimeoutExpired: p.kill(); rc = "timeout"
    wall = time.time() - t; stop.set(); txt = lf.read_text(encoding="utf-8", errors="replace")
    fz = re.findall(r"фазы: (.*)", txt); ready = re.findall(r"готово: (\{.*\})", txt)
    phases = json.loads(ready[-1]).get("phases") if ready else None
    gdl = re.findall(r"gor_dense: (\d+)/(\d+) участков, (\d+) точек, noD удалил (\d+) объектов sp_dfine, (\d+) с", txt)
    res = dict(rc=rc, wall_s=round(wall, 1), phases=phases, phases_line=fz[-1] if fz else None, gor_dense_summary=gdl[-1] if gdl else None,
               markers={k: txt.count(k) for k in ("GOR_DENSE_UNAVAILABLE", "SP_DFINE_UNAVAILABLE", "UNAVAILABLE", "мягкий потолок", "Traceback")},
               gpu_util_max=max((u for u, _ in smi), default=None), gpu_mem_max_mib=max((m for _, m in smi), default=None), smi_samples=len(smi),
               verdict={k: (wall <= v) for k, v in TARGETS.items()})
    rg = out / "result.geojson"
    if rg.is_file(): res.update(result_bytes=rg.stat().st_size, result_md5=md5(rg))
    log(f"■ main.py {nm}:", json.dumps(res, ensure_ascii=False)); SUMMARY["steps"][f"main_{nm}"] = res; save(); return res
on_r = run_main("on")
if left() > on_r["wall_s"] + 150: run_main("off")
else: log("OFF пропущен: не хватит времени до HARD_S"); SUMMARY["steps"]["main_off"] = "skipped (time)"
off_r = SUMMARY["steps"].get("main_off")
if isinstance(off_r, dict): SUMMARY["delta_on_minus_off_s"] = round(on_r["wall_s"] - off_r["wall_s"], 1)
SUMMARY["note"] = ("Время = верхняя оценка для теста: прокси 132 уч./175 SpOr/5,40 ГБ против публичного SpOr 30 уч./31 файл/4,68 ГБ; "
                   "T4 медленнее 4090 платформы; ограничители времени сняты только для замера.")
save(); log("ИТОГ:", json.dumps({k: SUMMARY.get(k) for k in ("cuda_check", "delta_on_minus_off_s")}, ensure_ascii=False),
            "ON wall", on_r["wall_s"], "s; цели", TARGETS); os._exit(0)
