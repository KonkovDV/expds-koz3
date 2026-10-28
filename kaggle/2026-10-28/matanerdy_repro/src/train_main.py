# ==== Ядро 2 (GPU T4×2, интернет): обучение DeepLabV3+ R34 по фолдам семей + OOF-полигоны по полным растрам ====
# Гиперпараметры — configs/archaeology_5class_research_split_v1.yaml конкурента (80 эп., patience 12, batch 16, Adam 1e-4,
# plateau 0.5/5, clip 1.0, CE 0.7 (веса [0.2,1,1,1.4,1.8,1.8]) + Dice 0.3, 256 px, 1 канал, без предобучения, без аугментаций);
# постобработка Stage C: порог 0.3, мин. площадь 8 px, opening 3×3.
import sys, csv, argparse, subprocess, traceback
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
OUT = Path(os.environ.get("MN_OUT", "/kaggle/working"))
CFG = dict(epochs=int(os.environ.get("MN_EPOCHS", 80)), patience=12, batch=16, lr=1e-4, clip=1.0, ce_w=0.7, dice_w=0.3,
           class_weights=[0.2, 1.0, 1.0, 1.4, 1.8, 1.8], thr=0.3, min_area=8, opening=3,
           scales=[1, 4], win=256, stride=192, block=4096, seed=101, budget_h=11.0, train_deadline_h=8.0)


def find_prep():
    for base in [Path(os.environ.get("MN_PREP", "/kaggle/input"))]:
        for p in sorted(base.rglob("manifest.json")):
            if (p.parent / "X.npy").exists(): return p.parent
    raise SystemExit("не найден выход ядра подготовки (manifest.json + X.npy) в /kaggle/input")


def ensure_smp():
    try:
        import segmentation_models_pytorch  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"])


def build_model():
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=len(CLASSES))


def zscore(x):
    import torch
    m = x.mean(dim=(-1, -2), keepdim=True); s = x.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    return (x - m) / s


def loss_fn(logits, y, w):
    import torch, torch.nn.functional as F
    ce = F.cross_entropy(logits, y, weight=w)
    p = logits.softmax(1); oh = F.one_hot(y, p.shape[1]).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return CFG["ce_w"] * ce + CFG["dice_w"] * dice


def train_fold(k, dev, X, M, meta, folds, t_start, n_left):
    import torch
    torch.manual_seed(CFG["seed"] + k); np.random.seed(CFG["seed"] + k)
    fo = folds[k]; valf = fo["val_family"]
    tr = [i for i, r in enumerate(meta) if int(r["fold"]) != k and r["family"] != valf]
    va = [i for i, r in enumerate(meta) if int(r["fold"]) != k and r["family"] == valf]
    assert not set(fo["test_regions"]) & {meta[i]["region"] for i in tr + va}, "утечка: участок теста в обучении"
    model = build_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    w = torch.tensor(CFG["class_weights"], device=dev)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    deadline = t_start + CFG["train_deadline_h"] * 3600
    best, bad, hist, best_state = 1e9, 0, [], None
    for ep in range(CFG["epochs"]):
        te = time.time(); model.train(); perm = np.random.permutation(tr); tl = 0.0
        for b in range(0, len(perm), CFG["batch"]):
            idx = np.sort(perm[b:b + CFG["batch"]])
            xb = zscore(torch.from_numpy(X[idx].astype(np.float32))[:, None].to(dev))
            yb = torch.from_numpy(M[idx].astype(np.int64)).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                l = loss_fn(model(xb).float(), yb, w)
            opt.zero_grad(set_to_none=True); scaler.scale(l).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"]); scaler.step(opt); scaler.update()
            tl += float(l) * len(idx)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for b in range(0, len(va), 32):
                idx = va[b:b + 32]
                xb = zscore(torch.from_numpy(X[idx].astype(np.float32))[:, None].to(dev))
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    vl += float(loss_fn(model(xb).float(), torch.from_numpy(M[idx].astype(np.int64)).to(dev), w)) * len(idx)
        vl = vl / max(1, len(va)) if va else tl / max(1, len(tr)); sch.step(vl)
        hist.append({"ep": ep, "train": tl / max(1, len(tr)), "val": vl, "sec": round(time.time() - te, 1)})
        print(f"[fold {k}] ep {ep} train {hist[-1]['train']:.4f} val {vl:.4f} {hist[-1]['sec']} с", flush=True)
        if vl < best: best, bad, best_state = vl, 0, {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else: bad += 1
        ep_sec = hist[-1]["sec"]
        if bad >= CFG["patience"]: break
        if time.time() + ep_sec * n_left > deadline: print(f"[fold {k}] стоп по времени", flush=True); break
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "cfg": CFG, "fold": k, "val_family": valf, "test_regions": fo["test_regions"],
                "n_train": len(tr), "n_val": len(va), "hist": hist}, OUT / f"weights/fold{k}.pt")
    return model, {"fold": k, "n_train": len(tr), "n_val": len(va), "best_val": best, "epochs": len(hist)}


def postprocess(cls):
    import cv2
    def rm_small(m):
        out = m.copy()
        for c in range(1, len(CLASSES)):
            n, lab, st, _ = cv2.connectedComponentsWithStats((m == c).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < CFG["min_area"]: out[lab == i] = 0
        return out
    m = rm_small(cls)
    if CFG["opening"] > 1:
        ker = np.ones((CFG["opening"],) * 2, np.uint8); o = np.zeros_like(m)
        for c in range(1, len(CLASSES)):
            o[cv2.morphologyEx((m == c).astype(np.uint8), cv2.MORPH_OPEN, ker) > 0] = c
        m = rm_small(o)
    return m


def infer_raster(model, dev, path, crs, scale, region, markup, fold):
    """Полный растр блоками (с полем), окна 256 / шаг 192, z-score на окно, порог 0.3, полигоны → EPSG:3857."""
    import torch, rasterio, cv2
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    from rasterio.features import shapes
    from affine import Affine
    from shapely.geometry import shape as sshape
    from shapely.ops import transform as stransform
    from pyproj import Transformer
    to3857 = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform
    feats = []
    W, S = CFG["win"], CFG["stride"]
    with rasterio.open(path) as ds:
        nod = ds.nodata
        core, halo = CFG["block"] * scale, 128 * scale
        for r0 in range(0, ds.height, core):
            for c0 in range(0, ds.width, core):
                rr0, cc0 = max(0, r0 - halo), max(0, c0 - halo)
                rr1, cc1 = min(ds.height, r0 + core + halo), min(ds.width, c0 + core + halo)
                h, w = max(1, (rr1 - rr0) // scale), max(1, (cc1 - cc0) // scale)
                a = ds.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0), out_shape=(h, w),
                            resampling=Resampling.bilinear if scale > 1 else Resampling.nearest).astype(np.float32)
                valid = np.isfinite(a) & (a != 0) & ((a != nod) if nod is not None else True)
                if valid.mean() < 0.02: continue
                a = np.nan_to_num(a)
                ph, pw = max(h, W), max(w, W)
                A = np.zeros((ph, pw), np.float32); A[:h, :w] = a
                ys = list(range(0, ph - W + 1, S)); xs = list(range(0, pw - W + 1, S))
                if ys[-1] != ph - W: ys.append(ph - W)
                if xs[-1] != pw - W: xs.append(pw - W)
                P = torch.zeros((len(CLASSES), ph, pw), device=dev); C = torch.zeros((ph, pw), device=dev)
                At = torch.from_numpy(A).to(dev); Vt = torch.from_numpy(np.pad(valid, ((0, ph - h), (0, pw - w)))).to(dev)
                wins = [(y, x) for y in ys for x in xs if Vt[y:y + W, x:x + W].float().mean() > 0.1]
                with torch.no_grad():
                    for b in range(0, len(wins), 64):
                        bw = wins[b:b + 64]
                        xb = zscore(torch.stack([At[y:y + W, x:x + W] for y, x in bw])[:, None])
                        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                            pr = model(xb).float().softmax(1)
                        for (y, x), p in zip(bw, pr):
                            P[:, y:y + W, x:x + W] += p; C[y:y + W, x:x + W] += 1
                P = (P / C.clamp_min(1)).cpu().numpy()[:, :h, :w]
                mx, cl = P.max(0), P.argmax(0).astype(np.uint8)
                cl[(mx < CFG["thr"]) | ~valid] = 0
                cl = postprocess(cl)
                tr = ds.transform * Affine.translation(cc0, rr0) * Affine.scale(scale)
                # ядро блока в координатах уменьшенной сетки
                ky0, kx0 = (r0 - rr0) // scale, (c0 - cc0) // scale
                ky1, kx1 = ky0 + core // scale, kx0 + core // scale
                for c in range(1, len(CLASSES)):
                    n, lab = cv2.connectedComponents((cl == c).astype(np.uint8), connectivity=8)
                    if n <= 1: continue
                    conf = np.bincount(lab.ravel(), weights=P[c].ravel(), minlength=n) / np.maximum(np.bincount(lab.ravel(), minlength=n), 1)
                    for g, v in shapes(lab.astype(np.int32), mask=lab > 0, transform=tr):
                        geom = sshape(g)
                        cx, cy = ~tr * (geom.centroid.x, geom.centroid.y)  # (col, row) уменьшенной сетки
                        if not (ky0 <= cy < ky1 and kx0 <= cx < kx1): continue
                        g3 = stransform(to3857, geom)
                        if g3.geom_type != "Polygon": g3 = max(getattr(g3, "geoms", [g3]), key=lambda q: q.area)
                        feats.append({"type": "Feature", "geometry": g3.__geo_interface__,
                                      "properties": {"class_name": CLASSES[c], "confidence": round(float(conf[int(v)]), 5),
                                                     "region_name": region, "crs": "EPSG:3857", "markup_type": markup,
                                                     "original_crs": crs, "threshold_group": f"mn_{markup}",
                                                     "fold": fold, "scale": scale}})
    return feats


def wait_file(p, size, timeout=3600):
    t = time.time()
    while not (p.exists() and p.stat().st_size == size):
        if time.time() - t > timeout: raise TimeoutError(str(p))
        time.sleep(5)
    return p


def worker(g, fold_ids, t_start):
    import torch
    dev = torch.device(f"cuda:{g}" if torch.cuda.is_available() else "cpu")
    prep = find_prep(); man = json.load(open(prep / "manifest.json", encoding="utf-8"))
    folds = man["folds"]["folds"]
    X = np.load(prep / "X.npy", mmap_mode="r"); M = np.load(prep / "M.npy", mmap_mode="r")
    meta = list(csv.DictReader(open(prep / "meta.csv", encoding="utf-8")))
    for j, k in enumerate(fold_ids):
        try:
            model, rep = train_fold(k, dev, X, M, meta, folds, t_start, len(fold_ids) - j)
            model.eval(); feats = []; ti = time.time()
            for region in folds[k]["test_regions"]:
                info = man["regions"].get(region, {})
                for mod, lst in info.get("rasters", {}).items():
                    for r in lst:
                        if "crs" not in r: continue
                        p = wait_file(CACHE / r["key"], r["size"])
                        for sc in CFG["scales"]:
                            feats += infer_raster(model, dev, p, r["crs"], sc, region, MOD2MARKUP[mod], k)
                        if os.environ.get("MN_KEEP", "0") != "1": p.unlink()  # фолды не пересекаются — растр больше не нужен
            rep.update(infer_sec=round(time.time() - ti), n_polys=len(feats))
            json.dump({"type": "FeatureCollection", "features": feats}, open(OUT / f"oof/fold{k}.geojson", "w", encoding="utf-8"), ensure_ascii=False)
        except Exception as e:
            rep = {"fold": k, "error": repr(e), "tb": traceback.format_exc()[-2000:]}
        json.dump(rep, open(OUT / f"logs/fold{k}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[fold]", json.dumps(rep, ensure_ascii=False)[:500], flush=True)


def downloader(man, order):
    cli = s3_client()
    for region in order:
        for mod, lst in man["regions"].get(region, {}).get("rasters", {}).items():
            for r in lst:
                if "crs" in r: fetch(cli, r["key"], r["size"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--require-gpu", action="store_true", default=True)
    ap.add_argument("--allow-cpu", action="store_true"); ap.add_argument("--folds", default="0,1,2,3,4")
    a, _ = ap.parse_known_args()
    t_start = time.time()
    n = gpu_guard(require=not a.allow_cpu)
    ensure_smp()
    for d in ("weights", "oof", "logs"): (OUT / d).mkdir(parents=True, exist_ok=True)
    prep = find_prep(); man = json.load(open(prep / "manifest.json", encoding="utf-8"))
    fids = [int(x) for x in a.folds.split(",")]
    order = [r for k in fids for r in man["folds"]["folds"][k]["test_regions"]]
    th = threading.Thread(target=downloader, args=(man, order), daemon=True); th.start()
    import torch.multiprocessing as mp
    ng = max(1, min(n, 2))
    groups = [fids[g::ng] for g in range(ng)]
    if ng == 1: worker(0, groups[0], t_start)
    else:
        ps = [mp.get_context("spawn").Process(target=worker, args=(g, groups[g], t_start)) for g in range(ng)]
        for p in ps: p.start()
        for p in ps: p.join()
    feats = []
    for k in fids:
        f = OUT / f"oof/fold{k}.geojson"
        if f.exists(): feats += json.load(open(f, encoding="utf-8"))["features"]
    for i, f in enumerate(feats): f["properties"]["fid"] = i
    json.dump({"type": "FeatureCollection", "features": feats}, open(OUT / "oof_mn_deeplab.geojson", "w", encoding="utf-8"), ensure_ascii=False)
    logs = [json.load(open(p, encoding="utf-8")) for p in sorted((OUT / "logs").glob("fold*.json"))]
    json.dump({"cfg": CFG, "gpus": n, "total_sec": round(time.time() - t_start), "folds": logs, "n_polys": len(feats)},
              open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("ГОТОВО", len(feats), "полигонов,", round((time.time() - t_start) / 3600, 2), "ч", flush=True)


if __name__ == "__main__":
    main()
