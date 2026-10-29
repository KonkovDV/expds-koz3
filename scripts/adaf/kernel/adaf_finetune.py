#!/usr/bin/env python3
"""КОЗ №3 / bot-2: fine-tune ADAF Faster R-CNN (Irish ALS, CC-BY-SA-4.0) on our lidar SLRM tiles,
family out-of-fold, and export bot-9-compatible maps npz + OOF geojson.

Own code (torchvision only, no aitlas / ADAF code copied). Model architecture = torchvision
fasterrcnn_resnet50_fpn_v2, which is what the ADAF checkpoints were trained with (Čož et al. 2026,
JAS: Reports 71:105733; weights doi:10.5281/zenodo.15848663, CC-BY-SA-4.0). Any fine-tuned weights
written by this script are an adaptation of those weights and are therefore CC-BY-SA-4.0 too
(see README.md, "Licence").

Inputs (all paths env-overridable):
  ADAF_SLRM   dir with <region>.npz from scripts/adaf/prep_slrm.py: slrm_u8 (H,W) u8, valid (H,W) u8,
              left, top, res (0.5), crs (JSON string, e.g. "urn:ogc:def:crs:EPSG::32636")
  ADAF_GT     gt_train_v2.geojson (EPSG:3857; props class_name, region_name)
  ADAF_FOLDS  region_fold.json (bot-9 family folds; keys region_fold, region_family)
  ADAF_INIT   OD_barrow .tar/.pt (dict with 'state_dict'); downloaded from Zenodo if missing
Outputs in ADAF_OUT:
  maps/maps_adaf_{tag}_{region}.npz  — ONLY for the region's test fold (bot-9 HEAD_INTERFACE format)
  oof_adaf_{tag}.geojson             — EPSG:3857 circles, markup_type "adaf", threshold_group "adaf_ft"
  region_fold.json                   — copy of folds used (+ jobs)
  weights/adaf_ft_f{k}.pt            — CC-BY-SA-4.0 (with ATTRIBUTION.txt)
  report.json                        — per fold/class raw recall/AP on the test fold (box centre in GT polygon)
"""
from __future__ import annotations

import hashlib, json, math, os, random, sys, time, urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.ops import batched_nms

E = os.environ.get
SMOKE = E("ADAF_SMOKE", "0") == "1"
SLRM = Path(E("ADAF_SLRM", "/kaggle/input/koz3-adaf-slrm/slrm"))
GT = Path(E("ADAF_GT", "/kaggle/input/koz3-gt/gt_train_v2.geojson"))
FOLDS = Path(E("ADAF_FOLDS", str(Path(__file__).with_name("region_fold.json"))))
INIT = Path(E("ADAF_INIT", "/kaggle/working/OD_barrow.tar"))
OUT = Path(E("ADAF_OUT", "/kaggle/working/out"))
TAG = E("ADAF_TAG", "ft1")
FOLD_LIST = [int(x) for x in E("ADAF_FOLD_LIST", "0,1,2,3,4").split(",")]
ITERS = int(E("ADAF_ITERS", "4000"))
BATCH = int(E("ADAF_BATCH", "4"))
LR = float(E("ADAF_LR", "0.01"))
CROP = int(E("ADAF_CROP", "512"))          # ADAF trained on 512 px patches (0.5 m)
MIN_SIZE = int(E("ADAF_MIN_SIZE", "800"))  # torchvision default resize, as in ADAF inference
TILE, TSTEP = int(E("ADAF_TILE", "512")), int(E("ADAF_TSTEP", "384"))
SEED = int(E("ADAF_SEED", "0"))
ONLY_REGIONS = [r for r in E("ADAF_ONLY_REGIONS", "").split(",") if r]
ZENODO = ("https://zenodo.org/records/15848663/files/OD_barrow.tar?download=1",
          "b9afba2246821035ea7007e08a64ab80")
# label 1..K; hm channels follow bot-9: 0=tselye, 1=povr; mask channels: 0=gor, 1=fort, 2=arch
CLASSES = E("ADAF_CLASSES", "kurgany_tselye,kurgany_povrezhdennye,fortifikatsii").split(",")
HM_CH = {"kurgany_tselye": 0, "kurgany_povrezhdennye": 1}
MASK_CH = {"gorodishcha": 0, "fortifikatsii": 1}
STRIDE, RES = 4, 0.5
DEV = torch.device("cuda" if torch.cuda.is_available() and not SMOKE else "cpu")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- data
def epsg_of(crs) -> str:
    s = crs if isinstance(crs, str) else str(crs)
    try:
        s = json.loads(s)
        s = s if isinstance(s, str) else s.get("crs", s)
    except Exception:
        pass
    return "EPSG:" + str(s).split(":")[-1]


def load_region(reg: str) -> dict:
    z = np.load(SLRM / f"{reg}.npz")
    return {"img": z["slrm_u8"], "valid": z["valid"].astype(bool), "left": float(z["left"]),
            "top": float(z["top"]), "res": float(z["res"]), "crs": str(z["crs"]), "epsg": epsg_of(str(z["crs"]))}


def gt_boxes(gt_feats: list, R: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """GT of one region → pixel boxes (x0,y0,x1,y1), labels 1..K, and UTM polygons (for recall).
    Only objects whose representative point lies on a valid SLRM pixel."""
    import shapely, shapely.ops
    from shapely.geometry import shape
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:3857", R["epsg"], always_xy=True)
    boxes, labels, polys = [], [], []
    H, W = R["valid"].shape
    for f in gt_feats:
        c = f["properties"]["class_name"]
        if c not in CLASSES or f["geometry"] is None:
            continue
        g = shapely.ops.transform(lambda x, y, z=None: tr.transform(x, y), shape(f["geometry"]))
        if g.is_empty:
            continue
        px, py = g.representative_point().coords[0]
        i, j = int((R["top"] - py) / R["res"]), int((px - R["left"]) / R["res"])
        if not (0 <= i < H and 0 <= j < W and R["valid"][i, j]):
            continue
        x0, y0, x1, y1 = g.bounds
        b = [(x0 - R["left"]) / R["res"], (R["top"] - y1) / R["res"],
             (x1 - R["left"]) / R["res"], (R["top"] - y0) / R["res"]]
        if b[2] - b[0] < 4:  # points / tiny: 4 px (2 m) minimum box
            cx = (b[0] + b[2]) / 2; b[0], b[2] = cx - 2, cx + 2
        if b[3] - b[1] < 4:
            cy = (b[1] + b[3]) / 2; b[1], b[3] = cy - 2, cy + 2
        boxes.append(b); labels.append(CLASSES.index(c) + 1); polys.append((c, g))
    return (np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(labels, np.int64), polys)


def d8(img: np.ndarray, boxes: np.ndarray, k: int, flip: bool):
    """rot90 k times (counter-clockwise, np.rot90) + optional horizontal flip; boxes in pixel coords."""
    S = img.shape[0]
    b = boxes.copy()
    for _ in range(k):
        img = np.rot90(img)
        # (x,y) -> (y, S-x)
        b = np.stack([b[:, 1], S - b[:, 2], b[:, 3], S - b[:, 0]], 1)
    if flip:
        img = img[:, ::-1]
        b = np.stack([S - b[:, 2], b[:, 1], S - b[:, 0], b[:, 3]], 1)
    return np.ascontiguousarray(img), b


class Sampler:
    """Random CROP×CROP crops from training regions: p_obj centred near a random GT object, else uniform
    over crops with ≥30 % valid pixels. Boxes clipped to the crop; keep if ≥50 % of area remains."""

    def __init__(self, regions: dict, p_obj: float = 0.7, rng=None):
        self.R = regions; self.p_obj = p_obj; self.rng = rng or np.random.default_rng(SEED)
        self.names = [r for r in regions if regions[r]["img"].shape[0] >= 64 and regions[r]["img"].shape[1] >= 64]
        self.obj = [(r, k) for r in self.names for k in range(len(regions[r]["boxes"]))]

    def one(self):
        rng = self.rng
        for _ in range(50):
            if self.obj and rng.random() < self.p_obj:
                r, k = self.obj[rng.integers(len(self.obj))]
                b = self.R[r]["boxes"][k]
                cx, cy = (b[0] + b[2]) / 2 + rng.uniform(-CROP / 3, CROP / 3), (b[1] + b[3]) / 2 + rng.uniform(-CROP / 3, CROP / 3)
            else:
                r = self.names[rng.integers(len(self.names))]
                H, W = self.R[r]["img"].shape
                cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            R = self.R[r]; H, W = R["img"].shape
            x0 = int(np.clip(cx - CROP / 2, 0, max(0, W - CROP))); y0 = int(np.clip(cy - CROP / 2, 0, max(0, H - CROP)))
            img = np.zeros((CROP, CROP), np.uint8); val = np.zeros((CROP, CROP), bool)
            sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
            h, w = R["img"][sl].shape
            img[:h, :w] = R["img"][sl]; val[:h, :w] = R["valid"][sl]
            if val.mean() < 0.3:
                continue
            bx = R["boxes"] - np.array([x0, y0, x0, y0], np.float32); lb = R["labels"]
            if len(bx):
                area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
                cb = np.clip(bx, 0, CROP)
                carea = np.clip(cb[:, 2] - cb[:, 0], 0, None) * np.clip(cb[:, 3] - cb[:, 1], 0, None)
                keep = carea >= 0.5 * area
                bx, lb = cb[keep], lb[keep]
                keep = ((bx[:, 2] - bx[:, 0]) >= 2) & ((bx[:, 3] - bx[:, 1]) >= 2)
                bx, lb = bx[keep], lb[keep]
            img, bx = d8(img, bx, int(rng.integers(4)), bool(rng.integers(2)))
            return img, bx, lb
        raise RuntimeError("sampler: no valid crop")

    def batch(self, n):
        ims, tg = [], []
        for _ in range(n):
            img, bx, lb = self.one()
            t = torch.from_numpy(img).float().div_(255.0)[None].expand(3, -1, -1).contiguous()
            ims.append(t.to(DEV))
            tg.append({"boxes": torch.from_numpy(np.asarray(bx, np.float32)).reshape(-1, 4).to(DEV),
                       "labels": torch.from_numpy(np.asarray(lb, np.int64)).to(DEV)})
        return ims, tg


# ---------------------------------------------------------------- model
def fetch_init() -> Path:
    if INIT.exists():
        return INIT
    url, md5 = ZENODO
    log("downloading ADAF OD_barrow from Zenodo (CC-BY-SA-4.0)")
    INIT.parent.mkdir(parents=True, exist_ok=True)
    tmp = INIT.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    h = hashlib.md5()
    with open(tmp, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    if h.hexdigest() != md5:
        raise RuntimeError(f"md5 mismatch {h.hexdigest()} != {md5}")
    tmp.rename(INIT)
    return INIT


def build_model(init_sd: dict | None) -> torch.nn.Module:
    """fasterrcnn_resnet50_fpn_v2 with 1+K classes; everything from ADAF barrow except the predictor,
    whose rows for every new class are initialised from the ADAF 'barrow' row (bg row from bg)."""
    K = len(CLASSES)
    m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=K + 1, min_size=MIN_SIZE)
    if init_sd is None:
        return m
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in init_sd.items()}
    pred = {k: sd.pop(k) for k in list(sd) if k.startswith("roi_heads.box_predictor.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("roi_heads.box_predictor.")]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    with torch.no_grad():
        cw, cb = pred["roi_heads.box_predictor.cls_score.weight"], pred["roi_heads.box_predictor.cls_score.bias"]
        bw, bb = pred["roi_heads.box_predictor.bbox_pred.weight"], pred["roi_heads.box_predictor.bbox_pred.bias"]
        P = m.roi_heads.box_predictor
        P.cls_score.weight[0] = cw[0]; P.cls_score.bias[0] = cb[0]
        P.bbox_pred.weight[0:4] = bw[0:4]; P.bbox_pred.bias[0:4] = bb[0:4]
        for c in range(1, K + 1):
            P.cls_score.weight[c] = cw[1] + 1e-3 * torch.randn_like(cw[1]); P.cls_score.bias[c] = cb[1]
            P.bbox_pred.weight[4 * c:4 * c + 4] = bw[4:8]; P.bbox_pred.bias[4 * c:4 * c + 4] = bb[4:8]
    return m


def train_fold(model, sampler: Sampler, iters: int) -> list:
    model.to(DEV).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
    warm = min(300, max(1, iters // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, iters - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")
    hist = []
    for it in range(iters):
        ims, tg = sampler.batch(BATCH)
        with torch.autocast(device_type=DEV.type, enabled=DEV.type == "cuda"):
            losses = model(ims, tg)
            loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 10.0)
        scaler.step(opt); scaler.update(); sched.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append({"it": it, "loss": float(loss), **{k: float(v) for k, v in losses.items()}})
            log("it", it, "loss %.4f" % float(loss), "lr %.5f" % opt.param_groups[0]["lr"])
    return hist


# ---------------------------------------------------------------- inference + export
def tile_starts(n: int) -> list:
    if n <= TILE:
        return [0]
    s = list(range(0, n - TILE, TSTEP)) + [n - TILE]
    return sorted(set(s))


@torch.no_grad()
def predict_region(model, R: dict, min_score: float = 0.05) -> dict:
    model.to(DEV).eval()
    model.roi_heads.score_thresh = min_score
    model.roi_heads.detections_per_img = 300
    H, W = R["img"].shape
    B, S, L = [], [], []
    for y0 in tile_starts(H):
        for x0 in tile_starts(W):
            sl = (slice(y0, y0 + TILE), slice(x0, x0 + TILE))
            v = R["valid"][sl]
            if v.mean() < 0.05:
                continue
            t = torch.zeros((TILE, TILE)); a = R["img"][sl]
            t[:a.shape[0], :a.shape[1]] = torch.from_numpy(a).float() / 255.0
            o = model([t[None].expand(3, -1, -1).to(DEV)])[0]
            b = o["boxes"].cpu() + torch.tensor([x0, y0, x0, y0], dtype=torch.float32)
            B.append(b); S.append(o["scores"].cpu()); L.append(o["labels"].cpu())
    if not B:
        return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32), "labels": np.zeros(0, np.int64)}
    b, s, l = torch.cat(B), torch.cat(S), torch.cat(L)
    keep = batched_nms(b, s, l, 0.3)
    b, s, l = b[keep].numpy(), s[keep].numpy(), l[keep].numpy()
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    ok = R["valid"][np.clip(cy.astype(int), 0, H - 1), np.clip(cx.astype(int), 0, W - 1)]
    return {"boxes": b[ok], "scores": s[ok], "labels": l[ok]}


def write_maps(path: Path, R: dict, det: dict, meta: dict):
    """bot-9 HEAD_INTERFACE maps npz: valid u8 (H,W); valid_h u8 (h,w); hm f32 (2,h,w) 0=tselye 1=povr;
    wh f16 (1,h,w) log(radius in head px); off f16 (2,h,w); mask u8 (3,H,W) 0=gor 1=fort 2=arch; meta JSON.
    Each detection = a single peak (value = score) at its centre cell of the stride-4 grid, so a peak decoder
    (pool 3, centre = left + (col+0.5+off_x)*step_head) reproduces box centres; masks = score*255 in the box."""
    H, W = R["valid"].shape; h, w = -(-H // STRIDE), -(-W // STRIDE)
    hm = np.zeros((2, h, w), np.float32); wh = np.zeros((1, h, w), np.float16); off = np.zeros((2, h, w), np.float16)
    mask = np.zeros((3, H, W), np.uint8)
    order = np.argsort(det["scores"])  # ascending: higher score written last wins
    for k in order:
        x0, y0, x1, y1 = det["boxes"][k]; s = float(det["scores"][k]); c = CLASSES[int(det["labels"][k]) - 1]
        if c in HM_CH:
            gx, gy = (x0 + x1) / 2 / STRIDE, (y0 + y1) / 2 / STRIDE
            col, row = int(gx), int(gy)
            if 0 <= row < h and 0 <= col < w:
                ch = HM_CH[c]
                if s >= hm[ch, row, col]:
                    hm[ch, row, col] = s
                    off[0, row, col] = gx - col - 0.5; off[1, row, col] = gy - row - 0.5
                    wh[0, row, col] = math.log(max(max(x1 - x0, y1 - y0) / 2 / STRIDE, 1e-3))
        if c in MASK_CH:
            a, b_ = (slice(max(0, int(y0)), min(H, int(math.ceil(y1)))), slice(max(0, int(x0)), min(W, int(math.ceil(x1)))))
            m = mask[MASK_CH[c]][a, b_]; np.maximum(m, np.uint8(round(s * 255)), out=m)
            m2 = mask[2][a, b_]; np.maximum(m2, np.uint8(round(s * 255)), out=m2)
    valid = R["valid"].astype(np.uint8)
    vh = valid[::STRIDE, ::STRIDE][:h, :w]
    meta = dict(meta, left=R["left"], top=R["top"], step=R["res"], step_head=R["res"] * STRIDE, crs=R["crs"],
                H=H, W=W, stride=STRIDE, head="adaf_frcnn", classes=CLASSES)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, valid=valid, valid_h=vh, hm=hm, wh=wh, off=off, mask=mask, meta=json.dumps(meta, ensure_ascii=False))


def det_features(R: dict, det: dict, reg: str, fold: int, fid0: int) -> list:
    from pyproj import Transformer
    tr = Transformer.from_crs(R["epsg"], "EPSG:3857", always_xy=True)
    feats = []
    for k in range(len(det["scores"])):
        x0, y0, x1, y1 = det["boxes"][k]
        X = R["left"] + (x0 + x1) / 2 * R["res"]; Y = R["top"] - (y0 + y1) / 2 * R["res"]
        r = max(x1 - x0, y1 - y0) / 2 * R["res"]
        mx, my = tr.transform(X, Y)
        lat = math.degrees(2 * math.atan(math.exp(my / 6378137.0)) - math.pi / 2)
        rr = 1.0 / math.cos(math.radians(lat))  # 1 m in Web-Mercator units (stack convention, adaf_splice)
        ring = [[mx + rr * math.cos(2 * math.pi * i / 16), my + rr * math.sin(2 * math.pi * i / 16)] for i in range(16)]
        ring.append(ring[0])
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"fid": fid0 + k, "class_name": CLASSES[int(det["labels"][k]) - 1],
                                     "confidence": float(det["scores"][k]), "region_name": reg, "crs": "EPSG:3857",
                                     "markup_type": "adaf", "threshold_group": "adaf_ft", "original_crs": R["epsg"],
                                     "fold": fold, "radius_m": float(r)}})
    return feats


def raw_recall(R: dict, det: dict, polys: list) -> dict:
    """box centre within GT polygon, greedy one-to-one by score within class."""
    import shapely
    out = {}
    for ci, c in enumerate(CLASSES, 1):
        ps = [g for cc, g in polys if cc == c]
        sel = np.where(det["labels"] == ci)[0]; sel = sel[np.argsort(-det["scores"][sel])]
        used, hits = set(), []
        tree = shapely.STRtree(ps) if ps else None
        for k in sel:
            x0, y0, x1, y1 = det["boxes"][k]
            p = shapely.Point(R["left"] + (x0 + x1) / 2 * R["res"], R["top"] - (y0 + y1) / 2 * R["res"])
            hit = 0
            if tree is not None:
                for gi in tree.query(p, predicate="within"):
                    if gi not in used:
                        used.add(int(gi)); hit = 1; break
            hits.append(hit)
        out[c] = {"n_gt": len(ps), "scores": det["scores"][sel].tolist(), "hits": hits}
    return out


def ap_of(scores, hits, n_gt):
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    o = np.argsort(-np.asarray(scores)); hh = np.asarray(hits)[o].astype(float)
    tp = np.cumsum(hh); prec = tp / np.arange(1, len(hh) + 1); rec = tp / n_gt
    mprec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum((rec - np.concatenate([[0], rec[:-1]])) * mprec))


# ---------------------------------------------------------------- main
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if DEV.type == "cpu":
        torch.set_num_threads(int(E("OMP_NUM_THREADS", "1")))
    OUT.mkdir(parents=True, exist_ok=True)
    folds = json.load(open(FOLDS))
    rf, fam = folds["region_fold"], folds.get("region_family", {})
    regs = sorted(r for r in rf if (SLRM / f"{r}.npz").exists() and (not ONLY_REGIONS or r in ONLY_REGIONS))
    log("device", DEV, "regions", len(regs), "classes", CLASSES, "folds", FOLD_LIST)
    gt_by = {}
    for f in json.load(open(GT))["features"]:
        gt_by.setdefault(f["properties"].get("region_name"), []).append(f)
    R = {}
    for r in regs:
        R[r] = load_region(r)
        R[r]["boxes"], R[r]["labels"], R[r]["polys"] = gt_boxes(gt_by.get(r, []), R[r])
    log("gt boxes", {c: int(sum((R[r]["labels"] == i + 1).sum() for r in regs)) for i, c in enumerate(CLASSES)})
    ck = torch.load(fetch_init(), map_location="cpu", weights_only=False)
    init_sd = ck["state_dict"] if "state_dict" in ck else ck
    feats, report, jobs = [], {"tag": TAG, "classes": CLASSES, "iters": ITERS, "folds": {}}, []
    acc = {c: {"scores": [], "hits": [], "n_gt": 0} for c in CLASSES}
    for k in FOLD_LIST:
        test = [r for r in regs if rf[r] == k]
        train = {r: R[r] for r in regs if rf[r] != k}
        if not test or not train:
            log("fold", k, "skipped (test", len(test), "train", len(train), ")"); continue
        log("fold", k, "train", len(train), "test", len(test))
        torch.manual_seed(SEED + k)
        model = build_model(init_sd)
        hist = train_fold(model, Sampler(train, rng=np.random.default_rng(SEED + k)), ITERS)
        wdir = OUT / "weights"; wdir.mkdir(exist_ok=True)
        if not SMOKE:
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "fold": k, "tag": TAG,
                        "licence": "CC-BY-SA-4.0, adapted from ADAF OD_barrow (doi:10.5281/zenodo.15848663)"},
                       wdir / f"adaf_ft_f{k}.pt")
        fr = {"train_regions": len(train), "test_regions": test, "loss": hist, "per_region": {}}
        for r in test:
            det = predict_region(model, R[r])
            meta = {"region": r, "fold": k, "family": fam.get(r), "tag": TAG}
            write_maps(OUT / "maps" / f"maps_adaf_{TAG}_{r}.npz", R[r], det, meta)
            feats += det_features(R[r], det, r, k, 30_000_000 + len(feats))
            rr = raw_recall(R[r], det, R[r]["polys"])
            fr["per_region"][r] = {c: {"n_gt": v["n_gt"], "n_det": len(v["scores"]), "hit05": int(sum(v["hits"]))} for c, v in rr.items()}
            for c, v in rr.items():
                acc[c]["scores"] += v["scores"]; acc[c]["hits"] += v["hits"]; acc[c]["n_gt"] += v["n_gt"]
            log("  ", r, {c: (v["n_gt"], int(sum(v["hits"]))) for c, v in rr.items()})
        report["folds"][k] = fr; jobs.append({"variant": f"adaf-{TAG}", "fold": k})
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    report["oof"] = {c: {"n_gt": v["n_gt"], "AP": ap_of(v["scores"], v["hits"], v["n_gt"]),
                         "recall@0.05": (sum(v["hits"]) / v["n_gt"]) if v["n_gt"] else None,
                         "recall@0.3": (sum(h for s, h in zip(v["scores"], v["hits"]) if s >= 0.3) / v["n_gt"]) if v["n_gt"] else None}
                     for c, v in acc.items()}
    json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
               "features": feats}, open(OUT / f"oof_adaf_{TAG}.geojson", "w"), ensure_ascii=False)
    json.dump(dict(folds, jobs=jobs, note=folds.get("note", "") + f" | adaf {TAG}: maps only for the test fold"),
              open(OUT / "region_fold.json", "w"), ensure_ascii=False, indent=1)
    json.dump(report, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1)
    (OUT / "weights").mkdir(exist_ok=True)
    (OUT / "weights" / "ATTRIBUTION.txt").write_text(
        "Fine-tuned from ADAF OD_barrow weights by Kostovska, Kocev, Čož, Kokalj, Curran, Corns, "
        "doi:10.5281/zenodo.15848663, licensed CC-BY-SA-4.0 (https://creativecommons.org/licenses/by-sa/4.0/). "
        "Changes: new 1+K class predictor, fine-tuned on KOZ-3 lidar SLRM tiles. These adapted weights are "
        "licensed CC-BY-SA-4.0. Paper: Čož et al. 2026, JAS: Reports 71, 105733.\n")
    log("done", json.dumps(report["oof"], ensure_ascii=False))


if __name__ == "__main__":
    main()
