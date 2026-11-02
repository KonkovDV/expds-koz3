# ── Ячейка 1. Конфиг, пути, защита от CPU ────────────────────────────────────────────────
import os, sys, json, time, glob, csv, math, shutil, tarfile, threading, subprocess, re
from pathlib import Path
import numpy as np

CFG = json.loads(r'''{
 "version": "01_kurgan_subclass v2 (bot-9, 2026-10-28)",
 "input_dir": "/kaggle/input/koz3-kurgan-subclass-crops",
 "npz_name": "kurgan_subclass_crops.npz",
 "code_bundle_dir": "/kaggle/input/koz3-li-tin-59-t128-bundle",
 "schema": {
  "required": [
   "X",
   "y",
   "region"
  ],
  "channels": 3,
  "min_labeled": 50,
  "optional_len_n": [
   "family",
   "kind",
   "stack_idx",
   "cand_class"
  ]
 },
 "require_gpu": true,
 "min_gpus": 2,
 "pretrained": true,
 "models": [
  {
   "name": "r18_imnet",
   "encoder": "resnet18",
   "gpu": 0,
   "seed": 1
  },
  {
   "name": "r34_imnet",
   "encoder": "resnet34",
   "gpu": 1,
   "seed": 2
  }
 ],
 "lfo_folds": 10,
 "epochs": 20,
 "batch": 128,
 "lr": 0.0005,
 "weight_decay": 0.0001,
 "warmup_frac": 0.15,
 "dropout": 0.2,
 "aug": {
  "scale": 0.1,
  "shift_px": 4,
  "gain": 0.1,
  "bias": 0.1,
  "noise": 0.03
 },
 "tta": 8,
 "bootstrap": 1000,
 "apply_thr": 0.4,
 "why_thr": "0.4 = 20/(30+20): ошибка на povr дороже: веса классов метрики (povr 30, tselye 20; materials/compute_metrics.py:84–85) — порог Байеса при откалиброванной BCE без весов; задан заранее, не подбирается по OOF",
 "final_models": true,
 "max_batches": 0,
 "local_test_overrides": {"require_gpu": false}
}''')

# Локальный сухой прогон (только вне Kaggle, только по явной переменной): CPU, 1 батч, без предобучения.
LOCAL_TEST = os.environ.get("KOZ3_BOT9_LOCAL_TEST") == "1" and not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
INPUT_DIR = Path(os.environ.get("KOZ3_BOT9_INPUT", CFG["input_dir"])) if LOCAL_TEST else Path(CFG["input_dir"])
CODE_DIR = Path(os.environ.get("KOZ3_BOT9_CODE", CFG["code_bundle_dir"])) if LOCAL_TEST else Path(CFG["code_bundle_dir"])
WORK = Path(os.environ.get("KOZ3_BOT9_WORK", "/kaggle/working")) if LOCAL_TEST else Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
if LOCAL_TEST:
    CFG.update(CFG["local_test_overrides"])
    print("ЛОКАЛЬНЫЙ СУХОЙ ПРОГОН: CPU, 1 батч, без предобучения — результаты не значимы", flush=True)

import torch
torch.set_num_threads(int(os.environ.get('KTHREADS','4')))

def gpu_guard(min_gpus):
    if os.environ.get("KOZ3_ALLOW_CPU_FALLBACK"):
        raise RuntimeError("KOZ3_ALLOW_CPU_FALLBACK задан — откат на CPU запрещён")
    if not torch.cuda.is_available():
        raise RuntimeError("--require-gpu: torch.cuda.is_available() == False. Ядро запущено без GPU "
                           "(Accelerator → GPU T4 x2; второе одновременное GPU-ядро аккаунта молча получает CPU). Остановка.")
    n = torch.cuda.device_count()
    if n < min_gpus:
        raise RuntimeError(f"--require-gpu: видно GPU {n}, нужно ≥ {min_gpus}. Остановка, без отката на CPU.")
    names = []
    for i in range(n):
        t = torch.ones(8, device=f"cuda:{i}") * 2
        if t.device.type != "cuda" or t.device.index != i or float(t.sum()) != 16.0:
            raise RuntimeError(f"проверочный тензор не на cuda:{i}: {t.device}")
        names.append(torch.cuda.get_device_name(i))
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if smi.returncode != 0 or smi.stdout.count("GPU ") < min_gpus:
        raise RuntimeError(f"nvidia-smi не подтвердил {min_gpus} GPU: {smi.stdout} {smi.stderr}")
    print("GPU OK:", names, flush=True)
    return [torch.device(f"cuda:{i}") for i in range(min_gpus)]

if CFG["require_gpu"] and not LOCAL_TEST:
    DEVICES = gpu_guard(CFG["min_gpus"])
else:
    DEVICES = [torch.device("cpu")] * CFG["min_gpus"]
torch.backends.cudnn.benchmark = True


# ── Ячейка 2. Вход: наличие и схема npz ─────────────────────────────────────────────────
def _ls(root, n=60):
    root = Path(root)
    return [str(p) for p in sorted(root.rglob("*"))[:n]] if root.exists() else []

def find_npz():
    global INPUT_DIR
    if not INPUT_DIR.exists() and Path("/kaggle/input").exists():  # новый формат монтирования: /kaggle/input/datasets/<owner>/<slug>
        alt = [p for p in Path("/kaggle/input").rglob(INPUT_DIR.name) if p.is_dir()]
        if alt:
            print(f"{INPUT_DIR} нет, датасет найден в {alt[0]}", flush=True)
            INPUT_DIR = alt[0]
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Нет входного датасета {INPUT_DIR}. Подключите dmitrijkv/koz3-kurgan-subclass-crops "
            f"(kernel-metadata.json → dataset_sources). Содержимое /kaggle/input: {_ls('/kaggle/input')}")
    named = sorted(INPUT_DIR.rglob(CFG["npz_name"]))
    allz = sorted(INPUT_DIR.rglob("*.npz"))
    if named:
        return named[0]
    if len(allz) == 1:
        print(f"файл {CFG['npz_name']} не найден, беру единственный npz: {allz[0]}", flush=True)
        return allz[0]
    raise FileNotFoundError(f"В {INPUT_DIR} нет {CFG['npz_name']} (npz найдено: {[str(p) for p in allz]}). "
                            f"Содержимое: {_ls(INPUT_DIR)}")

def load_and_check(path):
    D = np.load(path, allow_pickle=False)
    keys = set(D.files)
    errs = []
    miss = [k for k in CFG["schema"]["required"] if k not in keys]
    if miss:
        raise ValueError(f"{path}: нет обязательных полей {miss}; есть {sorted(keys)}. "
                         "Ожидается вывод scripts/bot9_kurgan_subclass.py crops: "
                         "X uint8 (N,3,H,W), y int (−1 нет метки / 0 целый / 1 повреждённый), region str (N,)")
    X, y, region = D["X"], D["y"], D["region"]
    N = len(y)
    if X.ndim != 4:
        errs.append(f"X.ndim={X.ndim}, нужно 4 (N,C,H,W)")
    else:
        if X.shape[1] not in (1, 2, 3, 4) and X.shape[-1] in (1, 2, 3, 4):
            errs.append(f"X.shape={X.shape}: похоже на NHWC, нужно NCHW")
        if X.shape[1] != CFG["schema"]["channels"]:
            errs.append(f"каналов {X.shape[1]}, нужно {CFG['schema']['channels']} (lrm_s, lrm_l, slope)")
        if X.shape[2] != X.shape[3] or X.shape[2] < 32:
            errs.append(f"окно {X.shape[2]}×{X.shape[3]}: нужно квадратное ≥ 32")
        if len(X) != N:
            errs.append(f"len(X)={len(X)} ≠ len(y)={N}")
    if X.dtype != np.uint8:
        errs.append(f"X.dtype={X.dtype}, нужно uint8 (нормировка (x/255−0.5)/0.25 как у кэша обучения)")
    if not np.issubdtype(y.dtype, np.integer):
        errs.append(f"y.dtype={y.dtype}, нужно целое")
    else:
        bad = sorted(set(np.unique(y).tolist()) - {-1, 0, 1})
        if bad:
            errs.append(f"y содержит {bad}, допустимо только −1/0/1")
    if region.dtype.kind not in "UO" or len(region) != N:
        errs.append(f"region: dtype {region.dtype}, len {len(region)} — нужен str (N,)")
    for k in CFG["schema"]["optional_len_n"]:
        if k in keys and len(D[k]) != N:
            errs.append(f"{k}: len {len(D[k])} ≠ {N}")
    if errs:
        raise ValueError(f"{path}: схема не совпала:\n  - " + "\n  - ".join(errs))
    lab = y >= 0
    if lab.sum() < CFG["schema"]["min_labeled"]:
        raise ValueError(f"размеченных окон {int(lab.sum())} < {CFG['schema']['min_labeled']}")
    if len(set(y[lab].tolist())) < 2:
        raise ValueError("в разметке один класс — AUC не определён")
    return D

NPZ = find_npz()
D = load_and_check(NPZ)
X = np.ascontiguousarray(D["X"])
y = D["y"].astype(np.int64)
region = D["region"].astype(str)
kind = D["kind"].astype(str) if "kind" in D.files else np.array(["gt"] * len(y))
stack_idx = D["stack_idx"].astype(np.int64) if "stack_idx" in D.files else np.full(len(y), -1)
cand_class = D["cand_class"].astype(str) if "cand_class" in D.files else np.array([""] * len(y))
lab = y >= 0
print(json.dumps({"npz": str(NPZ), "N": int(len(y)), "shape": list(X.shape), "labeled": int(lab.sum()),
                  "povr_share": round(float((y[lab] == 1).mean()), 4), "kinds": {k: int((kind == k).sum()) for k in sorted(set(kind))},
                  "regions": int(len(set(region)))}, ensure_ascii=False), flush=True)


# ── Ячейка 3. smp и веса ImageNet из бандла (интернет не нужен) ────────────────────────────
def resolve_code_dir():
    global CODE_DIR
    ok = lambda d: (d / "expds_code.tar.gz").exists() or (d / "expds_code").is_dir()
    if not ok(CODE_DIR) and Path("/kaggle/input").exists():  # как find_bundle в koz3-gpu-s4.py:29–39
        hits = sorted(p.parent for p in Path("/kaggle/input").rglob("expds_code*") if ok(p.parent))
        if hits:
            CODE_DIR = hits[0]
    return CODE_DIR

resolve_code_dir()

def _pip_smp():
    """orch: последний шанс — pip install. Без интернета просто вернёт False."""
    import subprocess, importlib
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"],
                       capture_output=True, text=True)
    print("pip smp:", r.returncode, r.stderr[-400:])
    importlib.invalidate_caches()
    try:
        import segmentation_models_pytorch  # noqa: F401
        return True
    except ImportError:
        return False

def setup_smp():
    try:
        import segmentation_models_pytorch  # noqa: F401
        return "preinstalled"
    except ImportError:
        pass
    if _pip_smp():  # orch: в образе Kaggle (py3.13) smp нет, в бандле его нет → pip (enable_internet=true)
        return "pip"
    tars = sorted(CODE_DIR.rglob("expds_code.tar.gz")) if CODE_DIR.exists() else []
    dirs = sorted(p.parent for p in CODE_DIR.rglob("segmentation_models_pytorch/__init__.py")) if CODE_DIR.exists() else []
    dst = WORK / "code"
    if tars:
        with tarfile.open(tars[0], "r:gz") as t:
            t.extractall(dst)
    elif dirs:
        shutil.copytree(dirs[0], dst, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"нет segmentation_models_pytorch: подключите dmitrijkv/koz3-li-tin-59-t128-bundle "
                                f"(expds_code.tar.gz). Содержимое {CODE_DIR}: {_ls(CODE_DIR, 30)}")
    sys.path.insert(0, str(dst))
    import segmentation_models_pytorch  # noqa: F401
    return str(tars[0] if tars else dirs[0])

HF_SPECS = {  # пути внутри бандла dmitrijkv/koz3-li-tin-59-t128-bundle — как в /workspace/kpull/s4/koz3-gpu-s4.py:55–68
    "resnet18": ("models--smp-hub--resnet18.imagenet", "3f2325ff978283d47aa6a1d6878ca20565622683",
                 "models--smp-hub--resnet18.imagenet/blobs/879c59b3870194ae248b52920479fed460415e62",
                 "blobs/d5/d5da100346311aff6748ca74f737a7e9178dae9e48b071f61a8ccf9fa3ad88d7"),
    "resnet34": ("models--smp-hub--resnet34.imagenet", "7a57b34f723329ff020b3f8bc41771163c519d0c",
                 "models--smp-hub--resnet34.imagenet/blobs/879c59b3870194ae248b52920479fed460415e62",
                 "blobs/bc/bcb6928a067cda2ccbcf44c2c810cf759f69ff578332b4ac57ff2a6eb3984b13"),
}

def setup_hf_cache():
    """Kaggle-датасет теряет симлинки кэша HF: собираем snapshots/<hash>/{config.json,model.safetensors} заново."""
    src = next(iter(sorted(p for p in CODE_DIR.rglob("hf_hub_cache") if p.is_dir())), None) if CODE_DIR.exists() else None
    dst = WORK / "hf_hub_cache"
    if src is None and (CODE_DIR / "hf_hub_cache.tar.gz").exists():
        with tarfile.open(CODE_DIR / "hf_hub_cache.tar.gz", "r:gz") as t:  # архив хранит симлинки — кэш готов как есть
            t.extractall(dst)
        os.environ["HF_HUB_CACHE"] = str(dst); os.environ["HF_HUB_OFFLINE"] = "1"
        return {"from": "hf_hub_cache.tar.gz"}
    found = {}
    for enc, (repo, snap, cfg_rel, w_rel) in HF_SPECS.items():
        if src is None:
            break
        cfg, w = src / cfg_rel, src / w_rel
        if not (cfg.is_file() and w.is_file() and w.stat().st_size > 1_000_000):
            continue
        sd = dst / repo / "snapshots" / snap
        sd.mkdir(parents=True, exist_ok=True)
        (dst / repo / "refs").mkdir(parents=True, exist_ok=True)
        (dst / repo / "refs" / "main").write_text(snap)
        for name, p in (("config.json", cfg), ("model.safetensors", w)):
            if not (sd / name).exists():
                os.symlink(p.resolve(), sd / name)
        found[enc] = {"weights": str(w), "bytes": w.stat().st_size}
    need = sorted({m["encoder"] for m in CFG["models"]})
    if CFG["pretrained"] and not all(e in found for e in need):
        raise FileNotFoundError(f"нет весов ImageNet для {[e for e in need if e not in found]} в {src}. "
                                "Подключите dmitrijkv/koz3-li-tin-59-t128-bundle (hf_hub_cache).")
    os.environ["HF_HUB_CACHE"] = str(dst)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return found

SMP_FROM = setup_smp()
HF = setup_hf_cache() if CFG["pretrained"] else {}
print("smp:", SMP_FROM, "| imagenet в кэше:", json.dumps(HF, ensure_ascii=False), flush=True)
import segmentation_models_pytorch as smp

def make_model(encoder):
    weights = "imagenet" if CFG["pretrained"] else None
    try:
        enc = smp.encoders.get_encoder(encoder, in_channels=3, depth=5, weights=weights)
    except Exception as e:  # без молчаливого отката на случайную инициализацию
        raise RuntimeError(f"не загрузились веса {encoder}/{weights}: {e!r}. Проверьте бандл hf_hub_cache "
                           "или задайте CFG['pretrained']=false осознанно.") from e
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.drop = torch.nn.Dropout(CFG["dropout"])
            self.head = torch.nn.Linear(enc.out_channels[-1], 1)
        def forward(self, x):
            return self.head(self.drop(self.enc(x)[-1].mean(dim=(2, 3)))).squeeze(1)
    return Net()

for _enc in sorted({m["encoder"] for m in CFG["models"]}):
    make_model(_enc)  # падает здесь, до обучения, если весов нет
print("модели строятся:", sorted({m["encoder"] for m in CFG["models"]}), "pretrained", CFG["pretrained"], flush=True)


# ── Ячейка 4. Честные фолды по семьям и обучение ───────────────────────────────────────
# site_family / family_folds — копия expds/naming.py:108 и expds/metric.py:538 (как scoreboard --honest 10).
def site_family(dirname):
    m = re.match(r"^(\d{3})_(.+)$", dirname)
    name = m.group(2) if m else dirname
    return re.split(r"[_\s]", name, maxsplit=1)[0]

def family_folds(regions, family_of, k=None):
    fams = {}
    for r in regions:
        fams.setdefault(family_of(r), set()).add(r)
    names = sorted(fams)
    if not k or k >= len(names):
        return [fams[f] for f in names]
    bins, sizes = [set() for _ in range(k)], [0] * k
    for f in sorted(names, key=lambda f: (-len(fams[f]), f)):
        i = int(np.argmin(sizes)); bins[i] |= fams[f]; sizes[i] += len(fams[f])
    return [b for b in bins if b]

family = np.array([site_family(r) for r in region])
if "family" in D.files and not np.array_equal(family, D["family"].astype(str)):
    print(f"ВНИМАНИЕ: поле family в npz расходится с site_family(region) у {int((family != D['family'].astype(str)).sum())} окон; беру site_family", flush=True)
FOLDS = family_folds(sorted(set(region)), site_family, CFG["lfo_folds"])
fold_of = {r: k for k, regs in enumerate(FOLDS) for r in regs}
fold = np.array([fold_of[r] for r in region])
for k in range(len(FOLDS)):
    te_f = set(family[fold == k]); tr_f = set(family[fold != k])
    if te_f & tr_f:
        raise RuntimeError(f"утечка: семьи {sorted(te_f & tr_f)} в обучении и проверке фолда {k}")
print(f"фолдов {len(FOLDS)}, семей {len(set(family))}; ни одна семья не делится между фолдами", flush=True)

def norm(xb):
    return (xb.float() / 255.0 - 0.5) / 0.25  # как crop_verify._tensor и кэш обучения

def augment(x, g):
    """x: (B,C,H,W) норм. Поворот на любой угол + отражение + сдвиг/масштаб (reflect) + усиление/сдвиг каналов."""
    B = x.shape[0]; A = CFG["aug"]; dev = x.device
    ang = torch.rand(B, device=dev, generator=g) * 2 * math.pi
    sc = 1 + (torch.rand(B, device=dev, generator=g) * 2 - 1) * A["scale"]
    flip = torch.where(torch.rand(B, device=dev, generator=g) < 0.5, -1.0, 1.0)
    sh = (torch.rand(B, 2, device=dev, generator=g) * 2 - 1) * (2 * A["shift_px"] / x.shape[-1])
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, sh[:, 0]], 1), torch.stack([sin * flip, cos, sh[:, 1]], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    gain = 1 + (torch.rand(B, x.shape[1], 1, 1, device=dev, generator=g) * 2 - 1) * A["gain"]
    bias = (torch.rand(B, x.shape[1], 1, 1, device=dev, generator=g) * 2 - 1) * A["bias"]
    return x * gain + bias + torch.randn(x.shape, device=dev, generator=g) * A["noise"]

def tta_views(x):
    v = [torch.rot90(x, k, dims=(2, 3)) for k in range(4)]
    v += [t.flip(-1) for t in v]
    return v[:CFG["tta"]]

def fit(dev, Xd, tr, encoder, seed):
    torch.manual_seed(seed); g = torch.Generator(device=dev); g.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = make_model(encoder).to(dev)
    if CFG["require_gpu"] and not LOCAL_TEST:
        assert all(p.device.type == "cuda" for p in m.parameters()), "--require-gpu: модель не на GPU"
    opt = torch.optim.AdamW(m.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    steps_ep = max(1, len(tr) // CFG["batch"])
    if CFG.get("max_batches"):
        steps_ep = min(steps_ep, CFG["max_batches"])
    total = CFG["epochs"] * steps_ep
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=total, pct_start=CFG["warmup_frac"])
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ytr = torch.from_numpy(y).to(dev).float()
    trg = torch.from_numpy(tr).to(dev)
    for ep in range(CFG["epochs"]):
        m.train()
        order = trg[torch.from_numpy(rng.permutation(len(tr))).to(dev)]
        for s in range(steps_ep):
            b = order[s * CFG["batch"]:(s + 1) * CFG["batch"]]
            if len(b) < 2:
                continue
            x = augment(norm(Xd[b]), g)
            if s == 0 and ep == 0 and CFG["require_gpu"] and not LOCAL_TEST:
                assert x.is_cuda, "--require-gpu: батч не на GPU"
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                lo = torch.nn.functional.binary_cross_entropy_with_logits(m(x).float(), ytr[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(lo).backward(); scaler.step(opt); scaler.update(); sched.step()
    return m.eval()

@torch.no_grad()
def predict(m, dev, Xd, idx):
    out = np.zeros(len(idx)); use_amp = dev.type == "cuda"
    ig = torch.from_numpy(idx).to(dev)
    for s in range(0, len(idx), 512):
        x = norm(Xd[ig[s:s + 512]])
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            p = sum(torch.sigmoid(m(v).float()) for v in tta_views(x)) / len(tta_views(x))
        out[s:s + 512] = p.cpu().numpy()
    return out

def auc(yy, pp):
    yy = np.asarray(yy); pp = np.asarray(pp)
    n1 = int((yy == 1).sum()); n0 = int((yy == 0).sum())
    if not n1 or not n0:
        return None
    o = np.argsort(pp, kind="mergesort"); r = np.empty(len(pp)); r[o] = np.arange(1, len(pp) + 1)
    for v in np.unique(pp):  # средние ранги при равенствах
        t = pp == v
        if t.sum() > 1: r[t] = r[t].mean()
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ── Ячейка 5. 10 фолдов × модели, по одной модели на GPU параллельно ───────────────────
OOF = {}; LOG = []; ERR = []
def run_model(mi, spec):
    dev = DEVICES[spec["gpu"]]
    Xd = torch.from_numpy(X).to(dev)  # uint8 на GPU
    p = np.full(len(y), np.nan); t_all = time.time()
    for k in range(len(FOLDS)):
        te = np.flatnonzero(fold == k); tr = np.flatnonzero((fold != k) & lab)
        if len(te) == 0 or len(set(y[tr].tolist())) < 2:
            continue
        t0 = time.time()
        fp = WORK / "folds" / f"{spec['name']}_f{k}.npy"; fp.parent.mkdir(exist_ok=True)
        if fp.exists():
            p[te] = np.load(fp); print(f"{spec['name']} f{k}: из кэша {fp.name}", flush=True)
            LOG.append({"model": spec["name"], "fold": k, "cached": True}); continue
        m = fit(dev, Xd, tr, spec["encoder"], spec["seed"] * 100 + k)
        p[te] = predict(m, dev, Xd, te)
        np.save(fp.with_suffix(".tmp.npy"), p[te]); os.replace(fp.with_suffix(".tmp.npy"), fp)
        a = auc(y[te][lab[te]], p[te][lab[te]])
        rec = {"model": spec["name"], "fold": k, "train": int(len(tr)), "test": int(len(te)),
               "test_labeled": int(lab[te].sum()), "auc": a, "sec": round(time.time() - t0, 1), "device": str(dev)}
        LOG.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
        del m
        if dev.type == "cuda": torch.cuda.empty_cache()
    OOF[spec["name"]] = p
    print(f"{spec['name']}: все фолды за {time.time() - t_all:.0f} с", flush=True)

def _safe(mi, spec):
    try:
        run_model(mi, spec)
    except Exception as e:  # noqa: BLE001 — исключение потока иначе теряется
        import traceback; ERR.append(f"{spec['name']}: {e!r}\n{traceback.format_exc()}")

T0 = time.time()
th = [threading.Thread(target=_safe, args=(i, s)) for i, s in enumerate(CFG["models"])]
[t.start() for t in th]; [t.join() for t in th]
if ERR:
    raise RuntimeError("обучение упало:\n" + "\n".join(ERR))
print(f"OOF готово за {time.time() - T0:.0f} с", flush=True)


# ── Ячейка 6. AUC: общий, по фолдам, по семьям, бутстреп по семьям; OOF в /kaggle/working ───────
names = [s["name"] for s in CFG["models"]]
P = {n: OOF[n] for n in names}
P["ensemble"] = np.nanmean(np.stack([OOF[n] for n in names]), axis=0)
ok = lab & np.isfinite(P["ensemble"])
cand = kind == "cand"

def fam_table(p, mask):
    rows = []
    for f in sorted(set(family[mask])):
        mm = mask & (family == f)
        rows.append({"family": f, "fold": int(fold[mm][0]), "n": int(mm.sum()), "n_povr": int((y[mm] == 1).sum()),
                     "n_tsel": int((y[mm] == 0).sum()), "auc": auc(y[mm], p[mm])})
    return rows

def boot_auc(p, mask, n=CFG["bootstrap"], seed=0):
    fams = np.array(sorted(set(family[mask]))); rng = np.random.default_rng(seed); vals = []
    by = {f: np.flatnonzero(mask & (family == f)) for f in fams}
    for _ in range(n):
        ii = np.concatenate([by[f] for f in rng.choice(fams, len(fams), replace=True)])
        a = auc(y[ii], p[ii])
        if a is not None: vals.append(a)
    return {"p5": float(np.percentile(vals, 5)), "p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95)),
            "P_auc_ge_0.9": float(np.mean(np.array(vals) >= 0.9)), "n": len(vals)} if vals else None

REP = {"npz": str(NPZ), "cfg": CFG, "local_test": LOCAL_TEST, "folds": [sorted(f) for f in FOLDS],
       "per_fold_log": LOG, "models": {}}
for n, p in P.items():
    ft = fam_table(p, ok)
    fa = [r["auc"] for r in ft if r["auc"] is not None]
    REP["models"][n] = {
        "auc_oof_all": auc(y[ok], p[ok]),
        "auc_oof_cand": auc(y[ok & cand], p[ok & cand]),
        "auc_oof_gt": auc(y[ok & ~cand], p[ok & ~cand]),
        "auc_by_fold": {int(k): auc(y[ok & (fold == k)], p[ok & (fold == k)]) for k in sorted(set(fold[ok]))},
        "auc_family_macro": float(np.mean(fa)) if fa else None,
        "families_with_both_classes": len(fa),
        "by_family": ft,
    }
E = P["ensemble"]
REP["models"]["ensemble"]["bootstrap_by_family"] = boot_auc(E, ok)
thr = CFG["apply_thr"]
pred = (E >= thr).astype(int)
cur = np.array([1 if c == "kurgany_povrezhdennye" else 0 for c in cand_class])
mc = ok & cand & (cand_class != "")
REP["decision_at_thr"] = {
    "thr": thr, "why_thr": CFG["why_thr"],
    "acc_all_labeled": float((pred[ok] == y[ok]).mean()),
    "cand_labeled": int(mc.sum()),
    "cand_wrong_subclass_now": int((cur[mc] != y[mc]).sum()),
    "fixed_by_clf": int(((cur != y) & (pred == y))[mc].sum()),
    "broken_by_clf": int(((cur == y) & (pred != y))[mc].sum()),
}
a_ens = REP["models"]["ensemble"]["auc_oof_all"]
REP["target_auc_0.9_met"] = bool(a_ens is not None and a_ens >= 0.9)
REP["elapsed_sec_oof"] = round(time.time() - T0, 1)

(WORK / "kurgan_subclass_report.json").write_text(json.dumps(REP, ensure_ascii=False, indent=1), encoding="utf-8")
with open(WORK / "kurgan_subclass_auc_by_family.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["model", "family", "fold", "n", "n_povr", "n_tsel", "auc"])
    for n in P:
        for r in REP["models"][n]["by_family"]:
            w.writerow([n, r["family"], r["fold"], r["n"], r["n_povr"], r["n_tsel"], "" if r["auc"] is None else round(r["auc"], 4)])
with open(WORK / "kurgan_subclass_oof.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["stack_idx", "region", "family", "fold", "kind", "cand_class", "y", "p_povr", *[f"p_{n}" for n in names]])
    for i in range(len(y)):
        f6 = lambda v: "" if not np.isfinite(v) else round(float(v), 6)
        w.writerow([int(stack_idx[i]), region[i], family[i], int(fold[i]), kind[i], cand_class[i], int(y[i]), f6(E[i]), *[f6(P[n][i]) for n in names]])
summ = {n: {k: REP["models"][n][k] for k in ("auc_oof_all", "auc_oof_cand", "auc_family_macro", "families_with_both_classes")} for n in P}
print(json.dumps({"summary": summ, "bootstrap_ens": REP["models"]["ensemble"]["bootstrap_by_family"],
                  "decision": REP["decision_at_thr"], "AUC>=0.9": REP["target_auc_0.9_met"]}, ensure_ascii=False, indent=1))
print("→ /kaggle/working/kurgan_subclass_oof.csv, kurgan_subclass_report.json, kurgan_subclass_auc_by_family.csv", flush=True)


# ── Ячейка 7. Финальные модели на всех размеченных (для теста), веса в /kaggle/working/weights ───────
if CFG["final_models"] and not os.environ.get("KOZ3_SKIP_FINAL"):
    (WORK / "weights").mkdir(exist_ok=True)
    FERR = []
    def _final(spec):
        try:
            dev = DEVICES[spec["gpu"]]; Xd = torch.from_numpy(X).to(dev); t0 = time.time()
            m = fit(dev, Xd, np.flatnonzero(lab), spec["encoder"], spec["seed"] * 100 + 99)
            blob = {"state_dict": {k: (v.half() if v.is_floating_point() else v).cpu() for k, v in m.state_dict().items()},
                    "encoder": spec["encoder"], "dropout": CFG["dropout"], "norm": "(x/255-0.5)/0.25", "tile": int(X.shape[-1]),
                    "tta": CFG["tta"], "labeled": int(lab.sum()), "povr_share": float((y[lab] == 1).mean()),
                    "apply_thr": CFG["apply_thr"], "npz": str(NPZ), "label": "1 = kurgany_povrezhdennye, 0 = kurgany_tselye"}
            torch.save(blob, WORK / "weights" / f"{spec['name']}_all.pt")
            print(f"{spec['name']}: финальная модель {time.time() - t0:.0f} с", flush=True)
        except Exception as e:  # noqa: BLE001
            FERR.append(f"{spec['name']}: {e!r}")
    th = [threading.Thread(target=_final, args=(s,)) for s in CFG["models"]]
    [t.start() for t in th]; [t.join() for t in th]
    if FERR:
        raise RuntimeError("; ".join(FERR))
print("ГОТОВО за", round(time.time() - T0), "с", flush=True)
