# Сборка TPU-ядер convnext_lidar_tpu (bot-9). Исходник — GPU-ядро ../convnext_lidar/src/convnext_main.py (только чтение)
# + точечные TPU-патчи (каждый — точное совпадение ровно 1 раз, иначе сборка падает) + блоки src/tpu_*.py.
# Вендор (expds/Rust/relief_ext/centerhead bot-2 @ 788202a/compute_metrics) и фолды берутся из СОБРАННОГО prep-ядра
# ../convnext_lidar/koz3-cnx-li-prep/koz3-cnx-li-prep.py — тот же sha каналов (RELIEF_SHA), что у prep-выхода в kernel_sources.
import base64, hashlib, io, json, os, pathlib, re, tarfile
H = pathlib.Path(__file__).resolve().parent
G = H.parent / "convnext_lidar"
OWNER = "dmitrijkv"; PREP_SLUG = "koz3-cnx-li-prep"
SMOKE_REGIONS = "191_Осечки_1.3км,016_Хога"          # как GPU-smoke (convnext_lidar/build.py): фолды 3 и 4, малые LAS
T4_S_PER_IT = "0.37,0.45"                             # оценка T4, bs 8, кроп 512: bot-9/CONVNEXT.md:86 (из bench_t4/validity.json)
T4_SOURCE = "bot-9/CONVNEXT.md:86 (оценка по bench_t4/validity.json); заменить env CNX_T4_S_PER_IT=<s_per_it GPU-smoke>"
FULL = {                                              # slug-хвост → задания; только scratch (*-sc)
    "sc": ",".join([f"c3-sc:{f}" for f in range(5)] + [f"c5-sc:{f}" for f in range(5)]),   # основной: 10 заданий, 8 чипов
    "c3sc": ",".join(f"c3-sc:{f}" for f in range(5)),                                        # запасной сплит: 5 заданий / 5 чипов
    "c5sc": ",".join(f"c5-sc:{f}" for f in range(5)),
}

src = (G / "src/convnext_main.py").read_text(encoding="utf-8")
SRC_SHA = hashlib.sha256(src.encode()).hexdigest()
def block(n): return (H / "src" / n).read_text(encoding="utf-8")
def rep(s, old, new):
    assert s.count(old) == 1, ("патч не применим (исходник GPU-ядра изменился?)", old[:80], s.count(old)); return s.replace(old, new)
def cut(s, start, end, new):
    a = s.index(start); b = s.index(end, a); assert s.count(start) == 1 and s.count(end) == 1, (start, end); return s[:a] + new + s[b:]

t = src
t = rep(t, "# KOZ3 №3 · convnext_lidar (bot-9): U-Net", "# KOZ3 №3 · convnext_lidar_tpu (bot-9, TPU v5e-8, torch_xla) — сборка из GPU-ядра convnext_lidar.\n# KOZ3 №3 · convnext_lidar (bot-9): U-Net")
t = rep(t, 'help="только для бокса: разрешить запуск без GPU"', 'help="только для бокса: разрешить запуск без TPU (PJRT_DEVICE=CPU)"')
t = cut(t, "# ── Шаг 1. --require-gpu", "# ── Шаг 2. Зависимости", block("tpu_step1.py") + "\n")
t = rep(t, 'for p in ("torch", "torchvision", "numpy", "huggingface_hub", "safetensors"):',
           'for p in ("torch", "torchvision", "numpy", "huggingface_hub", "safetensors", "torch_xla", "libtpu", "jax", "jaxlib"):   # TPU-стек образа не трогать')
t = rep(t, 'for mod in ["torch", "torchvision", "numpy", *PINS, "rvt.vis"]:', 'for mod in ["torch", "torchvision", "numpy", *(["torch_xla"] if IS_WORKER and XLA_DEV == "xla" else []), *PINS, "rvt.vis"]:')
t = rep(t, 'VENDOR = "/tmp/cnx_vendor"', 'VENDOR = os.environ.get("CNX_VENDOR_DIR", "/tmp/cnx_vendor")   # свой каталог на процесс-чип')
t = rep(t, 'DEADLINE = T0 + CFG["deadline_h"] * 3600', 'DEADLINE = float(os.environ.get("CNX_DEADLINE_ABS", T0 + CFG["deadline_h"] * 3600))   # воркеры — дедлайн главного')
t = cut(t, "# ── Шаг 1б. GPU через torch", "# ── Шаг 2б. Задания", "")
# устройство XLA — ДО самопроверки головы: первый backward создаёт очереди autograd по числу устройств; если он прошёл на CPU
# до инициализации torch_xla, backward на xla:0 падает (engine.cpp:1539 device_ready_queues_, logs/dry_xla.log)
_st = 'log("голова:", HEAD_NAME, json.dumps(_head_selftest(), ensure_ascii=False))'
t = rep(t, _st, block("tpu_step1b.py") + _st)
t = cut(t, "def train_job(job, dev, lane_left_iters, log_j):", "# ── Шаг 9.", block("tpu_train.py"))
t = rep(t, 'use_amp = dev.type == "cuda"; tf =', 'use_amp = dev.type in ("cuda", "xla"); tf =')
t = rep(t, "with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):",
           "with torch.autocast(device_type=dev.type, dtype=torch.bfloat16 if dev.type == \"xla\" else torch.float16, enabled=use_amp):")
t = rep(t, "x = torch.from_numpy(np.stack([read_win(a, xx, yy, S)[:ch] for yy, xx in bt])).to(dev)",
           "x = torch.from_numpy(_pad_bs(np.stack([read_win(a, xx, yy, S)[:ch] for yy, xx in bt]), CFG[\"infer_bs\"])).to(dev)   # статическая форма")
t = rep(t, "yo = model(xi.contiguous(memory_format=torch.channels_last))",
           "yo = model(xi.contiguous() if dev.type == \"xla\" else xi.contiguous(memory_format=torch.channels_last))")
t = rep(t, "sd = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in model.state_dict().items()}",
           "sd = {k: (v.half() if v.dtype == torch.float32 else v).cpu() for k, v in model.state_dict().items()}   # XLA → CPU до torch.save")
t = rep(t, '(OUT / "region_fold.json").write_text(json.dumps(dict(', 'None if IS_WORKER else (OUT / "region_fold.json").write_text(json.dumps(dict(')
t = cut(t, "lanes = [[] for _ in DEVS]", "# объединённые OOF по вариантам", block("tpu_lanes.py"))
for bad in ("torch.cuda.amp", "GradScaler", "nvidia-smi -L", "channels_last).contiguous"):
    assert bad not in t, bad
assert "dtype=torch.float16, enabled=use_amp" not in t.split("def predict_region")[0], "в обучении остался fp16"

# вендор + фолды из собранного prep-ядра (то же, что произвело prep-выход)
P = (G / PREP_SLUG / f"{PREP_SLUG}.py").read_text(encoding="utf-8")
m_f = re.search(r"^FOLDS = .*$", P, re.M); m_v = re.search(r"^VENDOR_B64 = '([A-Za-z0-9+/=]+)'$", P, re.M)
assert m_f and m_v, "в prep-ядре нет FOLDS/VENDOR_B64"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(m_v.group(1))), mode="r:gz") as tf:
    VMAN = json.load(tf.extractfile("vendor_manifest.json"))
    for n, h in VMAN["sha256"].items(): assert hashlib.sha256(tf.extractfile(n).read()).hexdigest() == h, n
RELIEF = {k: v for k, v in VMAN["sha256"].items() if k.startswith("expds") or k == "relief_ext.py"}
_pre, _post = src.split("# @@EMBED@@")              # prep-ядро собрано из того же convnext_main.py?
assert P.startswith(_pre) and P.endswith(_post) and P.count(_post) == 1, "prep-ядро собрано из другой версии convnext_main.py — пересоберите GPU build.py"

def emit(name, mode, jobs, regions, ksrc, envd):
    for tok in jobs.split(","): assert tok.split(":")[0].endswith("-sc"), ("TPU — только scratch", tok)
    env_lines = "".join(f"__import__('os').environ.setdefault({k!r}, {v!r})\n" for k, v in envd.items())
    emb = (f"{env_lines}CNX_MODE_DEFAULT = {mode!r}\nCNX_HEAD_DEFAULT = 'centerhead'\nCNX_JOBS_DEFAULT = {jobs!r}\nCNX_REGIONS_DEFAULT = {regions!r}\n"
           f"CNX_TPU_PROCS_DEFAULT = 8\nT4_S_PER_IT_DEFAULT = {T4_S_PER_IT!r}\nT4_SOURCE = {T4_SOURCE!r}\n"
           f"CNX_BUILD_INFO = {json.dumps(dict(gpu_src_sha256=SRC_SHA, vendor_from=PREP_SLUG), ensure_ascii=False)!r}\n"
           f"{m_f.group(0)}\nVENDOR_B64 = {m_v.group(1)!r}")
    d = H / name; d.mkdir(exist_ok=True)
    code = t.replace("# @@EMBED@@", emb); (d / f"{name}.py").write_text(code, encoding="utf-8")
    md = {"id": f"{OWNER}/{name}", "title": name, "code_file": f"{name}.py", "language": "python", "kernel_type": "script",
          "is_private": True, "enable_gpu": False, "enable_tpu": True, "enable_internet": True, "machine_shape": "TpuV5E8",
          "docker_image_pinning_type": "latest",          # БЕЗ docker_image: закреплённый CPU/GPU-образ отменяет TPU (kaggle-cli #1197)
          "dataset_sources": [f"{OWNER}/koz3-s3-cred"], "competition_sources": [], "kernel_sources": ksrc, "model_sources": []}
    json.dump(md, open(d / "kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return d, len(code.encode())
COMMON = {"CNX_WORKERS": "6", "CNX_DEADLINE_H": "8.3", "CNX_INFER_RESERVE_H": "0.6"}   # 9 ч на TPU-сессию (kaggle.com/docs/tpu)
out = [emit("koz3-cnx-li-tpu-smoke", "smoke", "c3-sc:3,c5-sc:4", SMOKE_REGIONS, [],
            {**COMMON, "CNX_ITERS": "20", "CNX_LOG_EVERY": "5", "CNX_TPU_PROCS": "2", "CNX_DEADLINE_H": "2.0"})]
for tail, jobs in FULL.items(): out.append(emit(f"koz3-cnx-li-tpu-{tail}", "full", jobs, "", [f"{OWNER}/{PREP_SLUG}"], COMMON))
a = FULL["sc"].split(","); b = FULL["c3sc"].split(",") + FULL["c5sc"].split(",")
assert len(a) == len(set(a)) == 10 and sorted(a) == sorted(b)
for d, n in out: print(d.name, n, "байт")
print("gpu_src_sha256", SRC_SHA[:16], "relief_sha", hashlib.sha256(json.dumps(RELIEF, sort_keys=True).encode()).hexdigest()[:16],
      "centerhead", VMAN.get("centerhead_commit"))
