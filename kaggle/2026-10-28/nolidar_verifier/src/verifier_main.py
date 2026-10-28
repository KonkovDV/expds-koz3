# KOZ3 · nolidar_verifier (bot-9): верификатор «объект/фон» для кандидатов стека на участках БЕЗ лидара (sp_or).
# Рецепт как s4 (/workspace/kpull/s4/koz3-gpu-s4.py: ResNet18-энкодер + линейная голова, кропы вокруг кандидата, OOF по семьям),
# но кропы режутся здесь же из растров SpOr в S3. Две головы (отдельные модели): kurgan (povr+tselye) и gor (городища).
# Негативы — реальные ложные кандидаты стека; позитивы — кандидаты, сопоставленные с GT (+ GT sp_or с джиттером ТОЛЬКО в обучении).
# Инференс/оценка — только кандидаты, кроп по центру кандидата. Ключи S3 — только Kaggle Secrets, нигде не печатаются.
# ── Шаг 0. Зависимости: закреплённые версии, жёсткий импорт-чек ─────────────────────────────────────────────────────
import os, sys, subprocess, importlib
PINS = {"rasterio": "rasterio==1.4.3", "pyproj": "pyproj==3.7.1", "shapely": "shapely==2.1.1",
        "boto3": "boto3==1.40.61", "sklearn": "scikit-learn==1.6.1", "cv2": "opencv-python-headless==4.11.0.86"}
def _deps():
    miss = []
    for mod in PINS:
        try: importlib.import_module(mod)
        except Exception: miss.append(mod)
    if miss:
        print("pip install:", [PINS[m] for m in miss], flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", *[PINS[m] for m in miss]])
        if r.returncode: raise SystemExit(f"pip install не удался ({miss}); нужен enable_internet=true")
        importlib.invalidate_caches()
    bad = []
    for mod in ["torch", "torchvision", *PINS]:
        try: m = importlib.import_module(mod); print(f"  {mod} {getattr(m, '__version__', '?')}", flush=True)
        except Exception as e: bad.append(f"{mod}: {e!r}")
    if bad: raise SystemExit("импорт-чек не пройден: " + "; ".join(bad))
_deps()
import json, time, csv, io, gzip, base64, math, threading, argparse, collections, concurrent.futures as cf
from pathlib import Path
import numpy as np, torch, torch.nn as nn

AP = argparse.ArgumentParser()
AP.add_argument("--allow-cpu", action="store_true"); AP.add_argument("--epochs-kurgan", type=int, default=14)
AP.add_argument("--epochs-gor", type=int, default=30); AP.add_argument("--regions", default="")
ARGS, _ = AP.parse_known_args()
OUT = Path(os.environ.get("VER_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("VER_CACHE", "/tmp/s3cache"))
BUCKET, ENDPOINT = "train-expds-3", os.environ.get("VER_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
WIN_M = {"kurgan": 64.0, "gor": 400.0}   # окно на местности, м (курган GT медиана ~100 м², городище — сотни м)
PX, PXM = 96, 120                         # вход сети 96; храним 120 (поле для джиттера ±12 px)
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)

# ── Шаг 1. --require-gpu (по умолчанию) ─────────────────────────────────────────────────────────────────────────
NG = torch.cuda.device_count() if torch.cuda.is_available() else 0
log("GPU:", [torch.cuda.get_device_name(i) for i in range(NG)])
if NG == 0 and not ARGS.allow_cpu:
    raise SystemExit("--require-gpu: GPU нет (сессия получила CPU — второй одновременный GPU-кернел?). Останов.")
DEVS = [torch.device(f"cuda:{i}") for i in range(NG)] or [torch.device("cpu")]

# ── Шаг 2. Секреты и ранняя проверка доступа к S3 ─────────────────────────────────────────────────────────────
def _cred_from_inputs():
    # приватный датасет с cred.json (поля id/key или aws_access_key_id/aws_secret_access_key); значения не печатаются
    import glob, json as _j
    for _p in sorted(glob.glob("/kaggle/input/**/cred.json", recursive=True)):
        try:
            with open(_p, encoding="utf-8") as _fh: _c = _j.load(_fh)
        except Exception as _e:
            log("cred.json не читается:", type(_e).__name__); continue
        if not isinstance(_c, dict): continue
        _k = _c.get("id") or _c.get("aws_access_key_id"); _s = _c.get("key") or _c.get("aws_secret_access_key")
        if _k and _s:
            log("ключи S3 из приватного датасета"); return _k, _s
    return None, None
def s3_client():
    import boto3
    from botocore.config import Config
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient(); kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception as e:
        log("Kaggle Secrets недоступны:", type(e).__name__, "— пробую env / приватный датасет")
    if not kid or not sec:
        kid, sec = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not kid or not sec:
        kid, sec = _cred_from_inputs()
    if not kid or not sec:
        raise SystemExit(f"нет ключей S3: заведите Kaggle Secrets {SECRET_ID}, {SECRET_KEY} и включите их в Add-ons → Secrets ядра")
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=kid, aws_secret_access_key=sec, region_name="ru-7",
                        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=32))
CLI = s3_client()
try:
    CLI.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
except Exception as e:
    raise SystemExit(f"S3 недоступен ({type(e).__name__}): проверьте секреты/интернет. Детали скрыты, чтобы не светить ключи.")
log("S3 OK")

# ── Шаг 3. Кандидаты, фолды по семьям, проверка ────────────────────────────────────────────────────────────────
CLS = {"P": "kurgany_povrezhdennye", "T": "kurgany_tselye", "G": "gorodishcha"}
rows = list(csv.DictReader(io.StringIO(gzip.decompress(base64.b64decode(TABLE_B64)).decode("utf-8"))))
FAM = FOLDS["family_of"]; FOLD = {r: f["fold"] for f in FOLDS["folds"] for r in f["test_regions"]}
fam_folds = collections.defaultdict(set)
for r, f in FOLD.items(): fam_folds[FAM[r]].add(f)
bad = {k: v for k, v in fam_folds.items() if len(v) > 1}
assert not bad, f"семья в двух фолдах: {bad}"
if ARGS.regions: rows = [r for r in rows if r["region"] in ARGS.regions.split(",")]
for r in rows:
    r["class_name"] = CLS[r["cls"]]; r["group"] = "gor" if r["cls"] == "G" else "kurgan"
    r["family"] = FAM[r["region"]]; r["fold"] = FOLD[r["region"]]; r["label"] = int(r["label"])
log("строк", len(rows), collections.Counter((r["kind"], r["group"], r["label"]) for r in rows))

# ── Шаг 4. Растры SpOr из S3 (параллельно) и кропы ────────────────────────────────────────────────────────────
def list_keys(region):
    out, tok = [], None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=region + "/")
        if tok: kw["ContinuationToken"] = tok
        r = CLI.list_objects_v2(**kw); out += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"): return out
        tok = r["NextContinuationToken"]

def fetch(key, size):
    from boto3.s3.transfer import TransferConfig
    dst = CACHE / key
    if dst.exists() and dst.stat().st_size == size: return dst
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_name(dst.name + ".part")
    for a in range(5):
        try:
            CLI.download_file(BUCKET, key, str(tmp), Config=TransferConfig(max_concurrency=4)); tmp.rename(dst); return dst
        except Exception as e:
            log("повтор", a + 1, key, type(e).__name__); time.sleep(5 * (a + 1))
    raise RuntimeError("не скачан " + key)

def crops_for_region(region, items):
    import rasterio, cv2
    from pyproj import Transformer
    tifs = [(k, s) for k, s in list_keys(region) if "_SpOr_" in k and k.lower().endswith((".tif", ".tiff"))]
    res = {}
    for k, s in sorted(tifs):
        todo = [it for it in items if it[0] not in res]
        if not todo: break
        with rasterio.open(fetch(k, s)) as ds:
            crs = ds.crs; epsg = crs.to_epsg() if crs else None
            tr = Transformer.from_crs("EPSG:3857", crs, always_xy=True)
            to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            xs, ys = tr.transform([it[1] for it in todo], [it[2] for it in todo])
            inside = [i for i, (x, y) in enumerate(zip(xs, ys))
                      if ds.bounds.left <= x <= ds.bounds.right and ds.bounds.bottom <= y <= ds.bounds.top]
            if not inside: continue
            arr = ds.read(list(range(1, min(3, ds.count) + 1)))
            if arr.shape[0] < 3: arr = np.repeat(arr[:1], 3, 0)
            if arr.dtype != np.uint8:
                lo, hi = np.percentile(arr[arr > 0], [1, 99]) if (arr > 0).any() else (0, 1)
                arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            img = np.ascontiguousarray(arr.transpose(1, 2, 0)); inv = ~ds.transform; px_m = abs(ds.transform.a)
            for i in inside:
                key, x3, y3, grp = todo[i]
                # окно в единицах растра: для EPSG:3857 метр местности = 1/cos(lat) единиц
                k3857 = 1.0 / math.cos(math.radians(to_ll.transform(x3, y3)[1])) if epsg == 3857 else 1.0
                win_px = WIN_M[grp] * k3857 / px_m * PXM / PX
                c, r = inv * (xs[i], ys[i]); s = win_px / PXM
                M = np.float32([[s, 0, c - s * PXM / 2], [0, s, r - s * PXM / 2]])
                res[key] = cv2.warpAffine(img, M, (PXM, PXM), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if os.environ.get("VER_KEEP", "0") != "1":
        import shutil; shutil.rmtree(CACHE / region, ignore_errors=True)
    return res

by_reg = collections.defaultdict(list)
for i, r in enumerate(rows): by_reg[r["region"]].append((i, float(r["cx"]), float(r["cy"]), r["group"]))
X = np.zeros((len(rows), PXM, PXM, 3), np.uint8); HAVE = np.zeros(len(rows), bool)
with cf.ThreadPoolExecutor(8) as ex:
    futs = {ex.submit(crops_for_region, g, it): g for g, it in by_reg.items()}
    for n, fu in enumerate(cf.as_completed(futs)):
        try:
            for i, a in fu.result().items(): X[i] = a; HAVE[i] = True
        except Exception as e:
            log("участок", futs[fu], "ошибка", repr(e)[:200])
        if n % 20 == 0: log(f"кропы: участков {n + 1}/{len(futs)}, есть {int(HAVE.sum())}/{len(rows)}")
log("кропы готовы", int(HAVE.sum()), "из", len(rows), "без растра:", int((~HAVE).sum()))
np.save(OUT / "crops_preview.npy", X[np.where(HAVE)[0][:64]])

# ── Шаг 5. Обучение/инференс OOF по семьям, по моделям на GPU параллельно ──────────────────────────────────────
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255; STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
def make_model():
    import torchvision
    try: m = torchvision.models.resnet18(weights="IMAGENET1K_V1"); src = "imagenet"
    except Exception as e: m = torchvision.models.resnet18(weights=None); src = f"scratch ({type(e).__name__})"
    m.fc = nn.Linear(m.fc.in_features, 1); return m, src

def batch(idx, dev, train):
    xb = torch.from_numpy(X[idx]).to(dev).permute(0, 3, 1, 2).float()
    m = PXM - PX
    if train:
        oy, ox = np.random.randint(0, m + 1, 2); xb = xb[:, :, oy:oy + PX, ox:ox + PX]
        if np.random.rand() < .5: xb = xb.flip(3)
        if np.random.rand() < .5: xb = xb.flip(2)
        xb = torch.rot90(xb, int(np.random.randint(4)), (2, 3))
    else:
        xb = xb[:, :, m // 2:m // 2 + PX, m // 2:m // 2 + PX]
    return (xb - MEAN.to(dev)) / STD.to(dev)

def predict(model, idx, dev):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        for b in range(0, len(idx), 512):
            xb = batch(idx[b:b + 512], dev, False)
            p = sum(torch.sigmoid(model(t).float().squeeze(1)) for t in (xb, xb.flip(3), xb.flip(2), xb.flip(2).flip(3))) / 4
            out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None

P_OOF = np.full(len(rows), np.nan); JOBREP = []
lab = np.array([r["label"] for r in rows]); kind = np.array([r["kind"] for r in rows])
grp = np.array([r["group"] for r in rows]); fold = np.array([r["fold"] for r in rows]); fam = np.array([r["family"] for r in rows])

def job(head, k, dev):
    t = time.time(); torch.manual_seed(101 + k); np.random.seed(101 + k)
    base = HAVE & (grp == head)
    test = np.where(base & (fold == k) & (kind == "c"))[0]
    if len(test) == 0: return {"head": head, "fold": k, "skip": "нет кандидатов в тесте"}
    trpool = base & (fold != k)
    # val-семья (для выбора эпохи): семья обучения с долей позитивов, ближайшей к 15 % (≤30 %), детерминированно;
    # если семей с позитивами < 3 — без валидации (последняя эпоха)
    cnt = collections.Counter(fam[(trpool) & (kind == "c") & (lab == 1)]); tot = sum(cnt.values())
    ok = sorted((abs(v / tot - .15), f) for f, v in cnt.items() if v / tot <= .30) if len(cnt) >= 3 else []
    valf = ok[0][1] if ok else None
    va = np.where(trpool & (fam == valf) & (kind == "c"))[0] if valf else np.array([], int)
    tr = np.where(trpool & (fam != valf))[0]
    assert not (set(fam[tr]) | set(fam[va])) & set(fam[test]), "утечка семьи"
    if lab[tr].sum() == 0 or lab[tr].sum() == len(tr):
        return {"head": head, "fold": k, "skip": "в обучении один класс", "n_train": len(tr), "n_test": len(test)}
    model, src = make_model(); model.to(dev)
    npos = max(1, int(lab[tr].sum())); pw = torch.tensor(min(50.0, (len(tr) - npos) / npos), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    E = ARGS.epochs_gor if head == "gor" else ARGS.epochs_kurgan
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=E * max(1, math.ceil(len(tr) / 128)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda"); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, hist = -1, None, []
    for ep in range(E):
        model.train(); perm = np.random.permutation(tr)
        for b in range(0, len(perm), 128):
            idx = np.sort(perm[b:b + 128]); xb = batch(idx, dev, True)
            yb = torch.from_numpy(lab[idx].astype(np.float32)).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                l = lossf(model(xb).float().squeeze(1), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(l).backward(); scaler.step(opt); scaler.update(); sch.step()
        a = auc(lab[va], predict(model, va, dev)) if len(va) else None
        hist.append({"ep": ep, "val_auc": a})
        score = a if a is not None else ep  # без валидации — последняя эпоха
        if score >= best: best, best_state = score, {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    P_OOF[test] = predict(model, test, dev)
    torch.save(best_state, OUT / f"weights_{head}_fold{k}.pt")
    rep = {"head": head, "fold": k, "init": src, "val_family": valf, "n_train": len(tr), "pos_train": int(lab[tr].sum()),
           "n_val": len(va), "n_test": len(test), "pos_test": int(lab[test].sum()), "best_val_auc": best if valf else None,
           "test_auc": auc(lab[test], P_OOF[test]), "test_auc_stack_conf": auc(lab[test], [float(rows[i]["confidence"]) for i in test]),
           "sec": round(time.time() - t), "hist": hist}
    log("job", json.dumps({k2: v for k2, v in rep.items() if k2 != "hist"}, ensure_ascii=False)); return rep

jobs = [(h, k) for h in ("kurgan", "gor") for k in range(len(FOLDS["folds"]))]
q = list(jobs); lock = threading.Lock()
def runner(dev):
    while True:
        with lock:
            if not q: return
            h, k = q.pop(0)
        try: JOBREP.append(job(h, k, dev))
        except Exception as e:
            import traceback; JOBREP.append({"head": h, "fold": k, "error": repr(e), "tb": traceback.format_exc()[-1500:]}); log("ОШИБКА", h, k, repr(e))
ths = [threading.Thread(target=runner, args=(d,)) for d in DEVS[:2]]
for t in ths: t.start()
for t in ths: t.join()

# ── Шаг 6. Выход: OOF-вероятности на каждого кандидата + AUC по семьям ──────────────────────────────────────────
cand = np.where(kind == "c")[0]
with open(OUT / "oof_verifier_nolidar.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["stack_idx", "region", "family", "fold", "markup_type", "class_name", "group", "confidence", "cx", "cy", "label", "has_crop", "p_verifier"])
    for i in cand:
        r = rows[i]; w.writerow([r["stack_idx"], r["region"], r["family"], r["fold"], r["markup_type"], r["class_name"], r["group"],
                                 r["confidence"], r["cx"], r["cy"], r["label"], int(HAVE[i]), "" if np.isnan(P_OOF[i]) else round(float(P_OOF[i]), 6)])
per_fam = {}
for h in ("kurgan", "gor"):
    sel = cand[(grp[cand] == h) & ~np.isnan(P_OOF[cand])]
    per_fam[h] = {"all": {"n": len(sel), "pos": int(lab[sel].sum()), "auc": auc(lab[sel], P_OOF[sel]),
                          "auc_stack_conf": auc(lab[sel], [float(rows[i]["confidence"]) for i in sel])}}
    for f in sorted(set(fam[sel])):
        s2 = sel[fam[sel] == f]
        per_fam[h][f] = {"n": len(s2), "pos": int(lab[s2].sum()), "auc": auc(lab[s2], P_OOF[s2]),
                         "auc_stack_conf": auc(lab[s2], [float(rows[i]["confidence"]) for i in s2])}
json.dump({"auc": per_fam, "jobs": sorted(JOBREP, key=lambda r: (r["head"], r["fold"])), "n_rows": len(rows),
           "crops": int(HAVE.sum()), "gpus": NG, "total_sec": round(time.time() - T0), "win_m": WIN_M},
          open(OUT / "summary_verifier.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("ГОТОВО", json.dumps({h: v["all"] for h, v in per_fam.items()}, ensure_ascii=False))
