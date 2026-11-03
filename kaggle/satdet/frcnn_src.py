# KOZ3 · W1-SATDET (bot-W1): satellite (SpOr) detector #2 = torchvision Faster R-CNN ResNet-50-FPN v2 (COCO weights, BSD-3).
# PREREG docs/PREREG_W1_SATDET.md (843d807). Clean-room: written from the torchvision detection API; no EN code, nothing from bots/bot3-*.
# Raster/chip/inference plumbing adapted from our own bot-9 kernel (kaggle/2026-10-29/dfine_spor), S3 removed: rasters from the private
# dataset elenakonkova/koz3-dino-gor-raw, GT (markup sp_or) from the private dataset elenakonkova/koz3-gt-train-v2. No credentials used.
# Honest OOF: 4 models, model j is trained on all regions outside group G_j (G = unions of the gate's 10 family folds T["folds"]) and predicts
# only the regions of G_j. Output: /kaggle/working/oof_frcnn_g{j}.geojson (+ merged oof_frcnn_spor.geojson), report frcnn_report.json.
# Weights are NOT saved (outputs only). Modes: SATDET_MODE=smoke (local CPU, few iterations) | full.
import os, sys, subprocess, importlib
PINS = {"rasterio": "rasterio==1.4.3", "pyproj": "pyproj==3.7.1", "shapely": "shapely==2.1.1"}
def _deps():
    need = []
    for mod, pin in PINS.items():
        try: importlib.import_module(mod)
        except Exception: need.append(pin)
    if need:
        print("pip install:", need, flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", *need])
        if r.returncode: raise SystemExit(f"pip install failed ({need})")
        importlib.invalidate_caches()
_deps()
import json, time, re, math, glob, random, threading, argparse, collections, io, zlib, concurrent.futures as cf
from pathlib import Path
import numpy as np, torch, torchvision
from PIL import Image, ImageFilter

REGION_FOLD = json.loads(r'''@@REGION_FOLD@@''')          # region -> gate T-fold (0..9)
GROUP_OF_FOLD = {0: 0, 2: 1, 4: 1, 9: 1, 3: 2, 5: 2, 1: 3, 6: 3, 7: 3, 8: 3}   # PREREG: G0={0} G1={2,4,9} G2={3,5} G3={1,6,7,8}
GROUP = {r: GROUP_OF_FOLD[f] for r, f in REGION_FOLD.items()}

MODE = os.environ.get("SATDET_MODE", "@@MODE@@")
AP = argparse.ArgumentParser(); AP.add_argument("--allow-cpu", action="store_true"); ARGS, _ = AP.parse_known_args()
E = lambda k, d: type(d)(os.environ.get(k, d))
CFG = dict(mode=MODE, res=0.5, tile=640, overlap=160, bs=E("SATDET_BS", 8), iters=E("SATDET_ITERS", 6 if MODE == "smoke" else 9000),
           lr=0.02, momentum=0.9, wd=1e-4, warmup=E("SATDET_WARMUP", 2 if MODE == "smoke" else 300), clip=10.0,
           obj_frac=0.6, cls_cap=4.0, shift=1 / 3, qaug=0.5, workers=E("SATDET_WORKERS", 2), max_obj_chips_region=400,
           anchors=[16, 32, 64, 128, 256], min_score=E("SATDET_MIN_SCORE", 0.05), nms_iou=0.5, max_det=300, infer_reserve_s=E("SATDET_INFER_RESERVE", 2400),
           deadline_h=E("SATDET_DEADLINE_H", 5.6), groups=os.environ.get("SATDET_GROUPS", "0,1,2,3"),
           max_windows=E("SATDET_MAX_WINDOWS", 0), regions=os.environ.get("SATDET_REGIONS", ""),
           raw=os.environ.get("SATDET_RAW", ""), gt=os.environ.get("SATDET_GT", ""))
OUT = Path(os.environ.get("SATDET_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
CHIPS = Path(os.environ.get("SATDET_CHIPS", "/tmp/chips_frcnn"))
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]   # labels 1..5, 0 = background
CI = {c: i for i, c in enumerate(CLASSES)}
T0 = time.time(); DEADLINE = T0 + CFG["deadline_h"] * 3600
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
log("CFG", json.dumps(CFG, ensure_ascii=False)); log("torch", torch.__version__, "torchvision", torchvision.__version__)

# ── 1. devices ──
NG = torch.cuda.device_count() if torch.cuda.is_available() else 0
log("GPU:", [torch.cuda.get_device_name(i) for i in range(NG)])
if NG == 0 and not ARGS.allow_cpu: raise SystemExit("--require-gpu: no GPU in this session. Stop.")
DEVS = [torch.device(f"cuda:{i}") for i in range(NG)][:2] or [torch.device("cpu")]
if not DEVS[0].type == "cuda": torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

# ── 2. inputs (private datasets) ──
def _find(pattern):
    for base in ("/kaggle/input", "/kaggle/input/datasets"):
        hits = sorted(glob.glob(f"{base}/**/{pattern}", recursive=True))
        if hits: return hits
    return []
RAW = Path(CFG["raw"]) if CFG["raw"] else next((Path(p) for p in _find("raw") if Path(p).is_dir()), None)
GTP = Path(CFG["gt"]) if CFG["gt"] else next((Path(p) for p in _find("gt_train_v2.geojson")), None)
if RAW is None or GTP is None: raise SystemExit(f"inputs not found: raw={RAW} gt={GTP}")
log("raw:", RAW, "gt:", GTP)
GT = collections.defaultdict(list)
for f in json.load(open(GTP, encoding="utf-8"))["features"]:
    p = f["properties"]
    if p.get("markup_type") == "sp_or" and p.get("class_name") in CI and f.get("geometry"):
        GT[p["region_name"]].append((f["geometry"], CI[p["class_name"]]))
INV = []
for d in sorted(RAW.iterdir()):
    if not d.is_dir() or d.name not in REGION_FOLD: continue
    if CFG["regions"] and d.name not in CFG["regions"].split(","): continue
    tifs = sorted(str(p) for p in d.glob("*SpOr*/*.tif*"))
    if tifs: INV.append({"region": d.name, "tifs": tifs, "group": GROUP[d.name], "fold": REGION_FOLD[d.name]})
log(f"regions with SpOr rasters: {len(INV)}; GT sp_or objects: {sum(len(GT[i['region']]) for i in INV)};",
    "per group regions", dict(collections.Counter(i["group"] for i in INV)),
    "GT per group", dict(collections.Counter(i["group"] for i in INV for _ in GT[i["region"]])))

# ── 3. model (early fail: weights download + one forward on CPU) ──
MODEL_LOCK = threading.Lock()
def make_model(seed=None):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    with MODEL_LOCK:
        if seed is not None: torch.manual_seed(seed)
        m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, trainable_backbone_layers=5,
                                       min_size=CFG["tile"], max_size=CFG["tile"], box_detections_per_img=CFG["max_det"],
                                       box_score_thresh=CFG["min_score"], box_nms_thresh=CFG["nms_iou"])
        m.rpn.anchor_generator = AnchorGenerator(tuple((a,) for a in CFG["anchors"]), ((0.5, 1.0, 2.0),) * len(CFG["anchors"]))
        assert m.rpn.head.cls_logits.out_channels == m.rpn.anchor_generator.num_anchors_per_location()[0]
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, len(CLASSES) + 1)
    return m
_t = time.time()
try:
    _m = make_model(0); _m.eval()
    with torch.no_grad(): _o = _m([torch.rand(3, 256, 256)])
    assert set(_o[0]) >= {"boxes", "scores", "labels"}
    _m.train(); _l = _m([torch.rand(3, 256, 256)], [{"boxes": torch.tensor([[10., 10., 40., 50.]]), "labels": torch.tensor([2])}])
    assert all(torch.isfinite(v) for v in _l.values()), _l
    PARAMS_M = round(sum(p.numel() for p in _m.parameters()) / 1e6, 2); del _m, _o, _l
except Exception as e:
    raise SystemExit(f"Faster R-CNN early check failed: {e!r}")
log(f"model OK (CPU): {PARAMS_M} M params, losses finite, {time.time() - _t:.0f} s")

# ── 4. scene: native SpOr raster warped on the fly to UTM 0.5 m ──
def utm_of(ds):
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(*ds.xy(ds.height // 2, ds.width // 2))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"
class Scene:
    def __init__(self, path):
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform
        self.src = rasterio.open(path); self.crs = utm_of(self.src)
        tr, w, h = calculate_default_transform(self.src.crs, self.crs, self.src.width, self.src.height, *self.src.bounds, resolution=CFG["res"])
        try: src_m = abs(calculate_default_transform(self.src.crs, self.crs, self.src.width, self.src.height, *self.src.bounds)[0].a)
        except Exception: src_m = CFG["res"]
        rs = Resampling.average if src_m < CFG["res"] else Resampling.bilinear
        self.vrt = WarpedVRT(self.src, crs=self.crs, transform=tr, width=w, height=h, resampling=rs, nodata=0)
        self.W, self.H, self.T = w, h, tr; self.bands = list(range(1, min(3, self.src.count) + 1)); self.stretch = None
        if self.src.dtypes[0] != "uint8":
            ov = self.src.read(1, out_shape=(min(1024, self.src.height), min(1024, self.src.width))); v = ov[ov > 0]
            self.stretch = tuple(np.percentile(v, [1, 99])) if v.size else (0, 1)
    def read(self, x0, y0, n):
        from rasterio.windows import Window
        a = np.zeros((len(self.bands), n, n), dtype=self.src.dtypes[0])
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(self.W, x0 + n), min(self.H, y0 + n)
        if cx1 > cx0 and cy1 > cy0:
            a[:, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = self.vrt.read(self.bands, window=Window(cx0, cy0, cx1 - cx0, cy1 - cy0))
        if a.shape[0] < 3: a = np.repeat(a[:1], 3, 0)
        if self.stretch:
            lo, hi = self.stretch; a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return np.ascontiguousarray(a.astype(np.uint8).transpose(1, 2, 0))
    def to_px(self, b):
        inv = ~self.T; (x0, y1), (x1, y0) = inv * (b[0], b[1]), inv * (b[2], b[3])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    def close(self): self.vrt.close(); self.src.close()
def gt_boxes(region, crs):
    from shapely.geometry import shape
    from shapely.ops import transform as st
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:3857", crs, always_xy=True).transform; out = []
    for g, c in GT[region]:
        g = st(tr, shape(g)); g = g if g.is_valid else g.buffer(0)
        if not g.is_empty: out.append((g.bounds, c))
    return out

# ── 5. chip bank: object-centred windows (BIG = tile·(1+2·shift), object kept inside) + random background windows ──
BIG = int(round(CFG["tile"] * (1 + 2 * CFG["shift"])))
def _clamp0(v, n): return int(min(max(v, 0), n - BIG)) if n >= BIG else int((n - BIG) // 2)
def build_chips(inv):
    rng = random.Random(zlib.crc32(inv["region"].encode())); rec = []; drop = collections.Counter()
    for path in inv["tifs"]:
        try:
            sc = Scene(path); boxes = [(sc.to_px(b), c) for b, c in gt_boxes(inv["region"], sc.crs)]
            boxes = [(b, c) for b, c in boxes if b[2] > 0 and b[3] > 0 and b[0] < sc.W and b[1] < sc.H]
            objs = rng.sample(boxes, min(len(boxes), CFG["max_obj_chips_region"]))
            nb = max(2, int(len(objs) * (1 - CFG["obj_frac"]) / CFG["obj_frac"]))
            cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, c, b) for b, c in objs] + [(rng.uniform(0, sc.W), rng.uniform(0, sc.H), -1, None) for _ in range(nb)]
            for j, (cx, cy, c, ob) in enumerate(cents):
                x0, y0 = _clamp0(cx - BIG / 2, sc.W), _clamp0(cy - BIG / 2, sc.H); img = sc.read(x0, y0, BIG); valid = img.max(2) > 0
                if ob is None:
                    if valid.sum() < 0.5 * CFG["tile"] ** 2: drop["bg_nodata"] += 1; continue
                else:
                    a0, b0 = max(0, int(ob[0] - x0)), max(0, int(ob[1] - y0)); a1, b1 = min(BIG, int(math.ceil(ob[2] - x0))), min(BIG, int(math.ceil(ob[3] - y0)))
                    if a1 <= a0 or b1 <= b0 or valid[b0:b1, a0:a1].mean() < 0.5: drop["obj_nodata"] += 1; continue
                bx = [[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0, cc] for b, cc in boxes if b[2] > x0 and b[3] > y0 and b[0] < x0 + BIG and b[1] < y0 + BIG]
                p = CHIPS / f"{inv['region']}__{Path(path).stem}__{j}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(p, quality=95)
                rec.append({"path": str(p), "region": inv["region"], "group": inv["group"], "cls": c, "boxes": bx,
                            "c": (cx - x0, cy - y0) if ob is not None else None})
            sc.close()
        except Exception as e:
            log("chips: error", inv["region"], Path(path).name, repr(e)[:200])
    if drop: log("chips:", inv["region"], "dropped", dict(drop))
    return rec

# ── 6. training stream with augmentation (flips, rot90, colour/gamma, gray, blur, quality aug) ──
def _aug(img, bx, rs, T):
    if rs.rand() < 0.3: img = np.asarray(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rs.uniform(0.5, 1.5)))).astype(np.float32)
    img = img * rs.uniform(0.75, 1.3) * rs.uniform(0.85, 1.15, 3) + rs.uniform(-20, 20); img = np.clip(img, 0, 255)
    if rs.rand() < 0.3: img = 255 * (img / 255) ** rs.uniform(0.7, 1.4)
    if rs.rand() < 0.15: img = np.repeat(img.mean(2, keepdims=True), 3, 2)
    if rs.rand() < CFG["qaug"]:
        f = rs.uniform(1.5, 3.2); s = max(8, int(T / f))
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((s, s), Image.BOX).resize((T, T), Image.BILINEAR)
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=int(rs.uniform(15, 60))); buf.seek(0)
        img = np.asarray(Image.open(buf).convert("RGB")).astype(np.float32) + rs.normal(0, rs.uniform(0, 6), (T, T, 3))
    if rs.rand() < 0.5: img = img[:, ::-1]; bx[:, [0, 2]] = T - bx[:, [2, 0]]
    if rs.rand() < 0.5: img = img[::-1]; bx[:, [1, 3]] = T - bx[:, [3, 1]]
    if rs.rand() < 0.5: img = img.transpose(1, 0, 2); bx = bx[:, [1, 0, 3, 2, 4]].copy()     # transpose == rot90 ∘ flip
    return img, bx
class ChipSet(torch.utils.data.IterableDataset):
    def __init__(self, recs, seed):
        self.seed = seed; n = collections.Counter(r["cls"] for r in recs if r["cls"] >= 0); nmax = max(n.values()) if n else 1
        self.obj = [r for r in recs if r["cls"] >= 0]; self.bg = [r for r in recs if r["cls"] < 0]
        w = np.array([min(CFG["cls_cap"], math.sqrt(nmax / n[r["cls"]])) for r in self.obj], float); self.w = w / w.sum() if len(w) else w
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rs = np.random.RandomState(self.seed + (wi.id if wi else 0)); T = CFG["tile"]
        while True:
            use_obj = self.obj and (not self.bg or rs.rand() < CFG["obj_frac"])
            r = self.obj[rs.choice(len(self.obj), p=self.w)] if use_obj else self.bg[rs.randint(len(self.bg))]
            img = np.asarray(Image.open(r["path"]).convert("RGB")); S = img.shape[0]
            if r.get("c") is not None:
                sh = CFG["shift"] * T; ox, oy = [int(min(max(v - T / 2 + rs.uniform(-sh, sh), 0), S - T)) for v in r["c"]]
            else:
                ox, oy = rs.randint(0, S - T + 1), rs.randint(0, S - T + 1)
            img = img[oy:oy + T, ox:ox + T].astype(np.float32); bx = []
            for x0, y0, x1, y1, c in r["boxes"]:
                a = max(1e-6, (x1 - x0) * (y1 - y0)); cx0, cy0, cx1, cy1 = max(x0 - ox, 0), max(y0 - oy, 0), min(x1 - ox, T), min(y1 - oy, T)
                if cx1 > cx0 and cy1 > cy0 and (cx1 - cx0) * (cy1 - cy0) >= 0.5 * a: bx.append([cx0, cy0, cx1, cy1, c])
            bx = np.array(bx, np.float32).reshape(-1, 5); img, bx = _aug(img, bx, rs, T)
            keep = (bx[:, 2] - bx[:, 0] >= 1) & (bx[:, 3] - bx[:, 1] >= 1); bx = bx[keep]
            x = torch.from_numpy(np.ascontiguousarray(np.clip(img, 0, 255).transpose(2, 0, 1), dtype=np.float32) / 255.0)
            yield x, {"boxes": torch.from_numpy(bx[:, :4].copy()), "labels": torch.from_numpy(bx[:, 4].astype(np.int64) + 1)}
def batches(recs, seed, dev):
    dl = torch.utils.data.DataLoader(ChipSet(recs, seed), batch_size=CFG["bs"], num_workers=CFG["workers"], collate_fn=lambda b: b,
                                     persistent_workers=CFG["workers"] > 0, prefetch_factor=4 if CFG["workers"] else None)
    for b in dl: yield [x.to(dev, non_blocking=True) for x, _ in b], [{k: v.to(dev) for k, v in t.items()} for _, t in b]

def train_loop(model, it, dev, budget_s, tag):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=CFG["lr"], momentum=CFG["momentum"], weight_decay=CFG["wd"])
    total = CFG["iters"]; wu = CFG["warmup"]
    lr_at = lambda s, tot: CFG["lr"] * (min(1.0, (s + 1) / wu) if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, tot - wu))))
    amp = dev.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp); model.train()
    t0 = time.time(); step = 0; tm = []; hist = []; nan_run = 0; run = collections.defaultdict(float)
    while step < total:
        x, y = next(it); ts = time.time()
        for g in opt.param_groups: g["lr"] = lr_at(step, total)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            ld = model(x, y)
        loss = sum(v.float() for v in ld.values())
        if not torch.isfinite(loss):
            nan_run += 1; opt.zero_grad(set_to_none=True)
            if nan_run > 50: raise RuntimeError("non-finite loss 50 times in a row")
            continue
        nan_run = 0; opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, CFG["clip"]); scaler.step(opt); scaler.update(); step += 1
        for k, v in ld.items(): run[k] += float(v)
        if dev.type == "cuda": torch.cuda.synchronize(dev)
        tm.append(time.time() - ts)
        if step == 50 and budget_s:
            spi = float(np.median(tm[10:])); fit = int(budget_s / spi)
            if fit < total: log(f"{tag} budget: {spi:.3f} s/it → iters {total} → {max(step + 10, fit)}"); total = max(step + 10, fit)
        every = 1 if MODE == "smoke" else 200
        if step % every == 0 or step == total:
            hist.append({"it": step, "lr": round(lr_at(step - 1, total), 5), "s_it": round(float(np.median(tm[-50:])), 4),
                         **{k: round(v / every, 4) for k, v in run.items()}}); run.clear(); log(tag, json.dumps(hist[-1]))
        if time.time() > DEADLINE - CFG["infer_reserve_s"]: log(tag, "STOP (deadline) at it", step); break
    return {"iters": step, "planned": CFG["iters"], "s_it_median": float(np.median(tm[min(5, len(tm) - 1):])) if tm else None,
            "hist": hist, "sec": round(time.time() - t0)}

# ── 7. inference on full rasters (windows 640 / overlap 160, hflip TTA, class-wise NMS) → inscribed 32-gon ellipses EPSG:3857 ──
@torch.no_grad()
def infer_region(model, inv, dev):
    from torchvision.ops import batched_nms
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as st
    from pyproj import Transformer
    model.eval(); T, S = CFG["tile"], CFG["tile"] - CFG["overlap"]; feats = []
    for path in inv["tifs"]:
        sc = Scene(path); B, L, Cc = [], [], []
        xs = list(range(0, max(1, sc.W - CFG["overlap"]), S)) or [0]; ys = list(range(0, max(1, sc.H - CFG["overlap"]), S)) or [0]
        wins = [(x, y) for y in ys for x in xs]
        if CFG["max_windows"]: wins = wins[:CFG["max_windows"]]
        t_r = time.time()
        for b in range(0, len(wins), CFG["bs"]):
            chunk, imgs = [], []
            for x, y in wins[b:b + CFG["bs"]]:
                im = sc.read(x, y, T)
                if (im.max(2) == 0).mean() > 0.95: continue
                chunk.append((x, y)); imgs.append(torch.from_numpy(im.transpose(2, 0, 1).astype(np.float32) / 255))
            if not imgs: continue
            px = [t.to(dev) for t in imgs]
            for flip in (False, True):
                inp = [t.flip(2) for t in px] if flip else px
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    outs = model(inp)
                for (x, y), o in zip(chunk, outs):
                    bb = o["boxes"].float()
                    if flip: bb = torch.stack([T - bb[:, 2], bb[:, 1], T - bb[:, 0], bb[:, 3]], 1)
                    B.append((bb + torch.tensor([x, y, x, y], device=bb.device)).cpu()); L.append(o["scores"].float().cpu()); Cc.append((o["labels"] - 1).cpu())
        log(f"[infer] {inv['region']} {Path(path).name[:40]}: {sc.W}x{sc.H} px, {len(wins)} windows, {time.time() - t_r:.0f} s")
        if B:
            B, L, Cc = torch.cat(B), torch.cat(L), torch.cat(Cc); k = batched_nms(B, L, Cc, CFG["nms_iou"]); B, L, Cc = B[k], L[k], Cc[k]
            to3857 = Transformer.from_crs(sc.crs, "EPSG:3857", always_xy=True).transform; t = np.linspace(0, 2 * np.pi, 33)[:-1]
            for (x0, y0, x1, y1), s, c in zip(B.numpy(), L.numpy(), Cc.numpy()):
                cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, .5), max((y1 - y0) / 2, .5)
                ring = [sc.T * (cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in t]
                feats.append({"type": "Feature", "geometry": mapping(st(to3857, Polygon(ring))),
                              "properties": {"class_name": CLASSES[int(c)], "confidence": round(float(s), 5), "region_name": inv["region"],
                                             "crs": "EPSG:3857", "markup_type": "sp_or", "original_crs": sc.crs, "group": "sp_det2",
                                             "fold": inv["fold"], "model_group": inv["group"], "tif": Path(path).name}})
        sc.close()
    return feats

# ── 8. orchestration ──
groups_run = [int(g) for g in CFG["groups"].split(",")]
t = time.time()
with cf.ThreadPoolExecutor(4 if MODE == "smoke" else 8) as ex: RECS = [r for rs in ex.map(build_chips, INV) for r in rs]
log(f"chips: {len(RECS)} (object {sum(r['cls'] >= 0 for r in RECS)}), {time.time() - t:.0f} s;",
    "classes:", dict(collections.Counter(CLASSES[r['cls']] for r in RECS if r['cls'] >= 0)))
if not RECS: raise SystemExit("empty chip bank")
REPORT, lock = {"cfg": CFG, "model": "torchvision fasterrcnn_resnet50_fpn_v2 COCO_V1", "params_M": PARAMS_M, "groups": {}}, threading.Lock()
queue = list(groups_run)
def gpu_worker(gi, dev):
    while True:
        with lock:
            if not queue: return
            j = queue.pop(0); left = len(queue)
        tag = f"[gpu{gi} g{j}]"
        try:
            tr = [r for r in RECS if r["group"] != j]; test = [i for i in INV if i["group"] == j]
            held_folds = {f for f, g in GROUP_OF_FOLD.items() if g == j}
            assert not {REGION_FOLD[r["region"]] for r in tr} & held_folds, "leak: training chip from a held fold"
            per_gpu_left = math.ceil((left + 1) / len(DEVS))
            budget = (DEADLINE - time.time() - 600) / per_gpu_left - CFG["infer_reserve_s"]
            model = make_model(seed=100 + j).to(dev)
            log(tag, f"train chips {len(tr)}, test regions {len(test)}, train budget {budget / 3600:.2f} h")
            rep = train_loop(model, batches(tr, 1000 + j, dev), dev, budget, tag)
            ti = time.time(); feats = []
            for inv in test:
                if time.time() > DEADLINE: log(tag, "infer: deadline, skip", inv["region"]); rep.setdefault("infer_skipped", []).append(inv["region"]); continue
                try: feats += infer_region(model, inv, dev)
                except Exception as e: log(tag, "infer: error", inv["region"], repr(e)[:200]); rep.setdefault("infer_errors", []).append(inv["region"])
            json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": feats},
                      open(OUT / f"oof_frcnn_g{j}.geojson", "w", encoding="utf-8"), ensure_ascii=False)
            rep.update({"n_train_chips": len(tr), "test_regions": [i["region"] for i in test], "n_det": len(feats), "infer_sec": round(time.time() - ti)})
            if dev.type == "cuda": rep["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
            log(tag, "done", json.dumps({a: b for a, b in rep.items() if a not in ("hist", "test_regions")}, ensure_ascii=False))
            del model
            if dev.type == "cuda": torch.cuda.empty_cache()
        except Exception as e:
            import traceback; rep = {"error": repr(e), "tb": traceback.format_exc()[-2000:]}; log(tag, "ERROR", repr(e))
        with lock:
            REPORT["groups"][j] = rep; json.dump(REPORT, open(OUT / "frcnn_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ths = [threading.Thread(target=gpu_worker, args=(gi, d)) for gi, d in enumerate(DEVS)]
for th in ths: th.start()
for th in ths: th.join()
allf = []
for j in groups_run:
    p = OUT / f"oof_frcnn_g{j}.geojson"
    if p.exists(): allf += json.load(open(p, encoding="utf-8"))["features"]
for i, f in enumerate(allf): f["properties"]["fid"] = i
json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": allf},
          open(OUT / "oof_frcnn_spor.geojson", "w", encoding="utf-8"), ensure_ascii=False)
REPORT["total_sec"] = round(time.time() - T0); REPORT["n_det"] = len(allf)
json.dump(REPORT, open(OUT / "frcnn_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
errs = [j for j, v in REPORT["groups"].items() if "error" in v]
log("DONE", len(allf), "detections; group errors:", errs)
if errs: raise SystemExit(f"groups with errors: {errs}")
