# KOZ3 lidar kurgan retrain: relief height/zoom aug + site-balanced sampling + seeds.
# Идеи из обзора соперников (plans/COMPETITORS_GT05_20261101.md), реализация — наша (expds.train,
# ветка bots/a-lidar-aug). Чужого кода и весов нет. Ключей в тексте нет.
# Рецепт базы = li_r18_hard2 (Unet r18, mask, 30 эп, batch 64, lr 1.4e-3) на кэше li_tin_59_t128.
# Варианты: base | aug_hz (высота 0.5–2.2, масштаб 0.7–1.45, p 0.8) | aug_hzb (+ баланс участков alpha 0.5).
# Оценка (прокси, тайлы тестовых семей фолда): объекты = компоненты маски, TP = пик компоненты внутри эталона.
# Стоп-гейт пилота (фолд 0, Гнёздово не в обучении): F1 повреждённых aug ≥ base И медиана FP/GT по участкам < base.
import os, sys, json, csv, time, shutil, subprocess, threading, statistics
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = [int(s) for s in os.environ.get("LI_SEEDS", "__SEEDS__").split(",")]
VARIANTS = "__VARIANTS__".split(",")
FOLDS = [0, 1, 2, 3, 4]          # 0 = пилот, идёт первым
FINAL_VARIANT = "aug_hzb"        # all-data веса для сдачи-кандидата (эпохи = медиана лучших эпох фолдов)
CAP_S = 8 * 3600
T0 = time.time()
WORK = Path("/kaggle/working")
INP = next((p for p in Path("/kaggle/input").glob("*") if (p / "li_tin_59_t128" / "meta.json").exists()), None)
if INP is None:
    INP = next(p.parent.parent for p in Path("/kaggle/input").rglob("li_tin_59_t128/meta.json"))
print("input:", INP, "seeds", SEEDS, "variants", VARIANTS, flush=True)
CODE = WORK / "code"
if not CODE.exists():
    shutil.copytree(INP / "code", CODE)
os.environ["HF_HUB_CACHE"] = str(INP / "hf"); os.environ["HF_HUB_OFFLINE"] = "1"
def _pip(*pk):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pk], capture_output=True, text=True)
    print("pip", pk, r.returncode, r.stderr[-300:], flush=True)
try:
    import segmentation_models_pytorch  # noqa
except Exception:
    _pip("segmentation-models-pytorch==0.5.0")
try:
    import rasterio  # noqa
except Exception:
    _pip("rasterio")
PY_ENV = dict(os.environ, PYTHONPATH=f"{CODE}:{INP / 'pylib'}:" + os.environ.get("PYTHONPATH", ""))
sys.path[:0] = [str(CODE), str(INP / "pylib")]
import numpy as np
import torch
print("torch", torch.__version__, "gpus", torch.cuda.device_count(), flush=True)
if torch.cuda.device_count() < 1:
    raise SystemExit("нет GPU — стоп")
NG = torch.cuda.device_count()

SRC = INP / "li_tin_59_t128"
META = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))
with (SRC / "index.csv").open(encoding="utf-8") as fh:
    rd = csv.DictReader(fh); FIELDS = list(rd.fieldnames); ROWS = list(rd)
for i, r in enumerate(ROWS):
    r["_i"] = i
TEST = {0: ["Гнёздово"], 1: ["Тимерево"],
        2: ["Вышенское", "Демидовка", "Золотаревка", "КЧР", "Лихуша", "Московиты", "Никола", "Новотитаровская", "Усть", "Эсбю"],
        3: ["Бервенец", "Боровичи", "Верхмарево", "Вщиж", "Елизаветинское", "Макеево", "Миловидово", "Новенькое", "Нора", "Осечки", "Примаки", "Руна", "Семибратнее", "Тешилово", "Шетнево"],
        4: ["Армиево", "Вышегжа", "Граборг", "Копанское", "Красносельская", "Курган", "Лубно", "Муравлево", "Селяне", "Стеклянуха", "Стрекаловка", "Судьбищи", "Хальвасборген", "Хога", "Шумгора", "Эрентуна", "Юшково"]}
tiles = Counter(r["family"] for r in ROWS); pos = Counter(r["family"] for r in ROWS if r["positive"] == "1")

def val_family(k):
    test = set(TEST[k]); tr = [f for f in sorted(tiles) if f not in test]
    n = sum(tiles[f] for f in tr); pool = [f for f in tr if pos[f] > 0] or tr
    return min(pool, key=lambda f: (abs(tiles[f] - 0.1 * n), f))

def fold_cache(k):
    d = WORK / "cache" / f"fold{k}"
    if d.exists():
        return d
    d.mkdir(parents=True)
    for p in SRC.glob("*.npy"):
        os.symlink(p, d / p.name)
    shutil.copy(SRC / "meta.json", d / "meta.json")
    test = set(TEST[k])
    with (d / "index.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for r in ROWS:
            if r["family"] not in test:
                w.writerow({f: r[f] for f in FIELDS})
    return d

def full_cache():
    d = WORK / "cache" / "full"
    if not d.exists():
        d.mkdir(parents=True)
        for p in SRC.glob("*"):
            os.symlink(p, d / p.name)
    return d

VFLAGS = {"base": [],
          "aug_hz": ["--relief-height", "0.5", "2.2", "--relief-zoom", "0.7", "1.45", "--relief-p", "0.8"],
          "aug_hzb": ["--relief-height", "0.5", "2.2", "--relief-zoom", "0.7", "1.45", "--relief-p", "0.8",
                      "--site-balance", "0.5"]}
RECIPE = ["--encoder", "resnet18", "--arch", "Unet", "--target", "mask", "--batch", "64", "--lr", "1.4e-3",
          "--require-gpu", "--workers", "0"]
(WORK / "models").mkdir(exist_ok=True); (WORK / "logs").mkdir(exist_ok=True); (WORK / "probs").mkdir(exist_ok=True)
RESULTS = {}
LOCK = threading.Lock()

def run_train(name, cache, extra, gpu):
    out = WORK / "models" / f"{name}.pt"
    log = WORK / "logs" / f"{name}.log"
    if out.exists() and (WORK / "logs" / f"{name}.done").exists():
        return out
    cmd = [sys.executable, "-m", "expds.train", "--cache", str(cache), "--out", str(out), *RECIPE, *extra,
           "--device", f"cuda:{gpu}"]
    t = time.time()
    with log.open("w") as fh:
        r = subprocess.run(cmd, cwd=str(CODE), env=PY_ENV, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[{name}] gpu{gpu} rc={r.returncode} {time.time() - t:.0f}s", flush=True)
    if r.returncode != 0:
        print(log.read_text()[-2000:], flush=True)
        return None
    (WORK / "logs" / f"{name}.done").write_text("ok")
    return out

# ── оценка ───────────────────────────────────────────────────────────────────────
from scipy import ndimage

def load_tiles(idx_rows):
    by = defaultdict(list)
    for j, r in enumerate(idx_rows):
        by[int(r["shard"])].append((j, int(r["idx"])))
    x = np.zeros((len(idx_rows), META["channels"], META["tile"], META["tile"]), np.uint8)
    m = None
    for s, lst in by.items():
        im = np.load(SRC / f"images_{s:04d}.npy", mmap_mode="r"); mk = np.load(SRC / f"masks_{s:04d}.npy", mmap_mode="r")
        if m is None:
            m = np.zeros((len(idx_rows),) + mk.shape[1:], mk.dtype)
        jj = [a for a, _ in lst]; ii = [b for _, b in lst]
        x[jj] = im[ii]; m[jj] = mk[ii]
    return x, m

def predict(model_path, x, gpu):
    import segmentation_models_pytorch as smp
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    net = getattr(smp, ck.get("arch", "Unet"))(encoder_name=ck["encoder"], encoder_weights=None,
                                               in_channels=ck["in_channels"], classes=len(ck["classes"]))
    net.load_state_dict(ck["state_dict"]); dev = torch.device(f"cuda:{gpu}"); net.to(dev).eval()
    out = np.zeros((len(x), 2, x.shape[2], x.shape[3]), np.uint8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(x), 128):
            t = torch.from_numpy(x[i:i + 128]).to(dev).float().div_(255).sub_(0.5).div_(0.25)
            p = torch.sigmoid(net(t)[:, :2].float())
            out[i:i + 128] = (p * 255).round().byte().cpu().numpy()
    return out, int(ck.get("epoch", 0))

def obj_stats(prob, gtbits, thr):
    """По тайлам: TP/FP/FN для класса 0 (повреждённые), 1 (целые) и курганов вместе."""
    st = np.zeros((len(prob), 3, 3), np.int64)  # tile × {povr,tsel,any} × {tp,fp,fn}
    thr8 = int(round(thr * 255))
    for t in range(len(prob)):
        for c, sel in ((0, [0]), (1, [1]), (2, [0, 1])):
            g = np.zeros(prob.shape[2:], bool)
            for b in sel:
                g |= ((gtbits[t] >> b) & 1).astype(bool)
            p = prob[t, sel].max(axis=0)
            gl, ng = ndimage.label(g)
            pl, npr = ndimage.label(p >= thr8)
            if npr:
                pk = ndimage.maximum_position(p, pl, range(1, npr + 1))
                vals = [p[a] for a in pk]
                hit = set(); tp = 0
                for o in np.argsort(vals)[::-1]:
                    gi = gl[pk[o]]
                    if gi and gi not in hit:
                        hit.add(gi); tp += 1
                st[t, c] = (tp, npr - tp, ng - len(hit))
            else:
                st[t, c] = (0, 0, ng)
    return st

def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)

THRS = [0.3, 0.4, 0.5, 0.6, 0.7]

def evaluate(name, model_path, k, gpu):
    test = set(TEST[k]); vf = val_family(k)
    te = [r for r in ROWS if r["family"] in test]; va = [r for r in ROWS if r["family"] == vf]
    xv, mv = load_tiles(va); pv, ep = predict(model_path, xv, gpu)
    thr = max(THRS, key=lambda t: (f1(*obj_stats(pv, mv, t)[:, 0].sum(0)), -abs(t - 0.5)))  # порог по val-семье
    xt, mt = load_tiles(te); pt, _ = predict(model_path, xt, gpu)
    np.savez_compressed(WORK / "probs" / f"{name}.npz", prob=pt, rows=np.array([r["_i"] for r in te]))
    res = {"fold": k, "val_family": vf, "thr_val": thr, "epoch": ep, "by_thr": {}}
    for t in sorted(set(THRS + [thr])):
        st = obj_stats(pt, mt, t)
        reg = defaultdict(lambda: np.zeros(3, np.int64))
        for j, r in enumerate(te):
            reg[r["region"]] += st[j, 2]
        fpgt = [v[1] / max(v[0] + v[2], 1) for v in reg.values() if v[0] + v[2] > 0]
        s = st.sum(0)
        res["by_thr"][str(t)] = {"f1_povr": round(f1(*s[0]), 4), "f1_tsel": round(f1(*s[1]), 4),
                                 "f1_kurg": round(f1(*s[2]), 4), "tp_fp_fn_povr": s[0].tolist(),
                                 "median_fp_per_gt": round(statistics.median(fpgt), 4) if fpgt else None,
                                 "regions": len(fpgt)}
    return res

def job(name, k, v, seed, gpu):
    if time.time() - T0 > CAP_S:
        print("cap hit, skip", name, flush=True); return
    mp = run_train(name, fold_cache(k), ["--val-families", val_family(k), "--epochs", "30", "--seed", str(seed),
                                         *VFLAGS[v]], gpu)
    if mp is None:
        return
    r = evaluate(name, mp, k, gpu); r.update({"variant": v, "seed": seed})
    with LOCK:
        RESULTS[name] = r
        (WORK / "results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1))
    t = str(r["thr_val"]); b = r["by_thr"][t]
    print(f"[{name}] thr {t} F1povr {b['f1_povr']} F1kurg {b['f1_kurg']} medFP/GT {b['median_fp_per_gt']}", flush=True)

def worker(q, gpu):
    while q:
        with LOCK:
            if not q:
                return
            a = q.pop(0)
        try:
            job(*a, gpu=gpu)
        except Exception as e:
            print("job fail", a, repr(e), flush=True)

def run_queue(items):
    q = list(items); th = [threading.Thread(target=worker, args=(q, g)) for g in range(NG)]
    [t.start() for t in th]; [t.join() for t in th]

def gate(k=0):
    out = {}
    for seed in SEEDS:
        b = RESULTS.get(f"f{k}_base_s{seed}")
        for v in VARIANTS:
            if v == "base" or b is None or f"f{k}_{v}_s{seed}" not in RESULTS:
                continue
            a = RESULTS[f"f{k}_{v}_s{seed}"]
            ba, aa = b["by_thr"][str(b["thr_val"])], a["by_thr"][str(a["thr_val"])]
            ok = aa["f1_povr"] >= ba["f1_povr"] and (aa["median_fp_per_gt"] or 9) < (ba["median_fp_per_gt"] or 9)
            out[f"{v}_s{seed}"] = {"base": ba, "aug": aa, "pass": bool(ok)}
    return out

# 1) пилот: фолд 0, все варианты, все сиды
run_queue([(f"f0_{v}_s{s}", 0, v, s) for s in SEEDS for v in VARIANTS])
G = gate(0); (WORK / "gate_pilot.json").write_text(json.dumps(G, ensure_ascii=False, indent=1))
print("PILOT GATE", json.dumps(G, ensure_ascii=False), flush=True)
if not any(g["pass"] for g in G.values()):
    print("PILOT STOP: ни один aug-вариант не прошёл гейт фолда 0. Остальные фолды только для base/диагностики не учим.", flush=True)
    (WORK / "STOP").write_text("pilot gate failed")
else:
    # 2) остальные фолды
    run_queue([(f"f{k}_{v}_s{s}", k, v, s) for k in FOLDS[1:] for s in SEEDS for v in VARIANTS])
    # 3) финальные all-data веса для FINAL_VARIANT (эпохи — медиана лучших эпох фолдов)
    eps = [r["epoch"] for n, r in RESULTS.items() if r["variant"] == FINAL_VARIANT and r["epoch"]]
    ep = int(statistics.median(eps)) if eps else 30
    fq = [(s, g % NG) for g, s in enumerate(SEEDS)]
    def fin(seed, gpu):
        run_train(f"final_{FINAL_VARIANT}_s{seed}", full_cache(), ["--all-data", "--epochs", str(ep), "--seed", str(seed),
                                                                  *VFLAGS[FINAL_VARIANT]], gpu)
    th = [threading.Thread(target=fin, args=a) for a in fq]; [t.start() for t in th]; [t.join() for t in th]
# сводка по всем фолдам
summ = defaultdict(lambda: np.zeros(3, np.int64)); medfp = defaultdict(list)
for n, r in RESULTS.items():
    b = r["by_thr"][str(r["thr_val"])]; summ[(r["variant"], r["seed"])] += np.array(b["tp_fp_fn_povr"])
    if b["median_fp_per_gt"] is not None:
        medfp[(r["variant"], r["seed"])].append(b["median_fp_per_gt"])
S = {f"{v}_s{s}": {"oof_f1_povr": round(f1(*x), 4), "tp_fp_fn": x.tolist(), "folds": len(medfp[(v, s)]),
                   "mean_fold_median_fp_gt": round(float(np.mean(medfp[(v, s)])), 4) if medfp[(v, s)] else None}
     for (v, s), x in summ.items()}
(WORK / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1))
print("SUMMARY", json.dumps(S, ensure_ascii=False), f"wall {time.time() - T0:.0f}s", flush=True)
shutil.rmtree(WORK / "cache", ignore_errors=True); shutil.rmtree(CODE, ignore_errors=True)
