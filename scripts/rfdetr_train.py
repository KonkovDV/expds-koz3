"""RF-DETR-M/L. По умолчанию (--init coco) — дообучение чекпойнта RF-DETR COCO (rf-detr-medium.pth / rf-detr-large-2026.pth,
Apache-2.0, веса разрешены организаторами). Запасные режимы: --init dinov2 (только энкодер DINOv2-S, декодер с нуля), --init scratch.

Написано bot-5. Из пакета rfdetr (Apache-2.0) берутся только сборка модели, сопоставитель и критерий
(rfdetr.models.lwdetr.build_model_from_config / build_criterion_from_config); цикл обучения, данные, MAL,
flat-cosine, copy-paste курганов и перенос весов DINOv2 — свои. Код конкурентов не использовался.

Почему перенос весов вручную: у M/L patch_size 16 и сетка позиций ≠ 37, поэтому rfdetr при pretrain_weights=None
строит энкодер СЛУЧАЙНЫМ (rfdetr/models/backbone/dinov2.py:164-180, :198-203). Здесь ядро patch-эмбеддинга 14×14
пересэмплируется в 16×16 (псевдообратное «PI-resize», FlexiViT), позиции 37×37 — бикубически в сетку модели.

Вход — COCO от scripts/coco_from_cache.py --layout rfdetr; выход — preds_<split>.json (COCO results, px PNG,
category_id 1..K) для scripts/coco_preds_to_stack.py и timing.json.

    python scripts/rfdetr_train.py --data det/f0 --out runs/rf_smoke --smoke           # 10 итераций
    python scripts/rfdetr_train.py --data det/f0 --out runs/rf/f0 --model rfdetr-m --size 640 --iters 8000 --bs 4 --accum 4 --fp16
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from det_train_hf import Boxes, assert_no_meta, jitter, peak_rss_mb, read_split  # noqa: E402  (свои функции из det_train_hf.py)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DINO = "facebook/dinov2-small"  # Apache-2.0, https://huggingface.co/facebook/dinov2-small


# ---------------- модель ----------------
def make_configs(a, K: int):
    from rfdetr.config import RFDETRBaseConfig, RFDETRLargeConfig, RFDETRMediumConfig, TrainConfig
    MC = {"rfdetr-m": RFDETRMediumConfig, "rfdetr-l": RFDETRLargeConfig, "rfdetr-b": RFDETRBaseConfig}[a.model]
    kw = dict(pretrain_weights=None, num_classes=K, device=a.device, group_detr=a.group_detr)
    if a.size:
        kw["resolution"] = a.size  # PE подстраивается сам (config.py:749-780)
    mc = MC(**kw)
    tc = TrainConfig(dataset_dir=".", output_dir=".")
    return mc, tc


def load_coco(model, mc, a) -> dict:
    """Чекпойнт RF-DETR COCO штатным rfdetr.models.weights.load_pretrain_weights (md5, PE→сетка --size,
    срез групп запросов, голова классов → K+1 заново). Возвращает, сколько тензоров модели реально заменено."""
    import hashlib
    import torch
    from rfdetr.models.weights import load_pretrain_weights
    name = os.path.basename(str(type(mc).model_fields["pretrain_weights"].default))
    path = a.coco_weights or os.path.join(os.environ.get("RF_HOME") or os.path.expanduser("~/.cache/rfdetr"), name)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    def sig(v):  # дешёвая подпись тензора вместо копии (память бокса)
        v = v.detach().double()
        return (tuple(v.shape), float(v.sum()), float(v.abs().sum()))
    before = {k: sig(v) for k, v in model.state_dict().items()}
    import rfdetr.utilities.io as rio
    orig = rio._safe_torch_load
    # mmap: распакованный чекпойнт (405 МБ, model+ema+args) не оседает в куче процесса; файл проверен по md5 rfdetr
    rio._safe_torch_load = lambda p_, trust=False: torch.load(p_, map_location="cpu", mmap=True, weights_only=False)
    try:
        load_pretrain_weights(model, mc.model_copy(update={"pretrain_weights": os.path.abspath(path)}), trust=True)
    finally:
        rio._safe_torch_load = orig
    after = model.state_dict()
    changed = sum(1 for k, v in after.items() if before.get(k) != sig(v))
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    del before
    import ctypes, gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # вернуть ОС память от распакованного чекпойнта (405 МБ)
    except OSError:
        pass
    return {"weights": os.path.abspath(path), "md5": h.hexdigest(), "tensors": len(after), "changed": changed,
            "head_out": int(after["class_embed.bias"].shape[0])}


def pi_resize_kernel(w: "torch.Tensor", new: int) -> "torch.Tensor":
    """Ядро патча k×k → new×new так, чтобы <x_resized, w_new> ≈ <x, w> (псевдообратная матрица билинейного ресайза)."""
    import torch
    import torch.nn.functional as F
    k = w.shape[-1]
    if k == new:
        return w
    eye = torch.eye(k * k, dtype=torch.float64).reshape(k * k, 1, k, k)
    B = F.interpolate(eye, size=(new, new), mode="bilinear", align_corners=False).reshape(k * k, new * new).T  # (n², k²)
    Binv = torch.linalg.pinv(B)  # (k², n²)
    flat = w.double().reshape(-1, k * k)  # (C_out·C_in, k²)
    return (flat @ Binv).reshape(*w.shape[:2], new, new).to(w.dtype)


def load_dinov2(model, hf_dir: str | None) -> dict:
    """Переносит веса DINOv2-S в энкодер RF-DETR. Возвращает отчёт (сколько тензоров, что пересэмплировано)."""
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    if hf_dir is None:
        from huggingface_hub import snapshot_download
        hf_dir = snapshot_download(DINO, allow_patterns=["*.json", "*.safetensors", "README.md"])
    src = load_file(os.path.join(hf_dir, "model.safetensors"))
    enc = model.backbone[0].encoder.encoder
    dst = enc.state_dict()
    out, rep = {}, {"copied": 0, "resampled": [], "missing": [], "source": hf_dir}
    for k, v in dst.items():
        if k not in src:
            rep["missing"].append(k); continue
        s = src[k]
        if k.endswith("patch_embeddings.projection.weight") and s.shape != v.shape:
            s = pi_resize_kernel(s, v.shape[-1]); rep["resampled"].append(f"{k} {tuple(src[k].shape)}->{tuple(s.shape)}")
        elif k.endswith("position_embeddings") and s.shape != v.shape:
            cls, grid = s[:, :1], s[:, 1:]
            g0, g1 = int(round(math.sqrt(grid.shape[1]))), int(round(math.sqrt(v.shape[1] - 1)))
            grid = grid.reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
            grid = F.interpolate(grid.float(), size=(g1, g1), mode="bicubic", align_corners=False)
            s = torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(1, g1 * g1, -1).to(s.dtype)], 1)
            rep["resampled"].append(f"{k} {tuple(src[k].shape)}->{tuple(s.shape)}")
        assert s.shape == v.shape, (k, s.shape, v.shape)
        out[k] = s; rep["copied"] += 1
    enc.load_state_dict(out, strict=False)
    return rep


# ---------------- лосс MAL (DEIM, arXiv 2412.04234) ----------------
def install_mal(criterion, gamma: float):
    """Подменяет классификационную часть критерия rfdetr на Matchability-Aware Loss:
    положительная пара (запрос, объект класса c): BCE с целью q^γ (q — IoU с объектом), вес 1;
    все прочие логиты: BCE с целью 0 и весом p^γ (как фокальная). Нормировка — на число объектов."""
    import types
    import torch
    import torch.nn.functional as F
    from rfdetr.utilities import box_ops

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, matched_targets=None):
        logits = outputs["pred_logits"]
        if matched_targets is None:
            idx = self._get_src_permutation_idx(indices)
            cls_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            tb = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], 0)
        else:
            idx, cls_o, tb = matched_targets.source_indices, matched_targets.labels, matched_targets.boxes
        valid = self._matched_valid_mask(targets, indices)
        q, _ = box_ops.elementwise_box_iou(box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][idx].detach()),
                                           box_ops.box_cxcywh_to_xyxy(tb))
        q = q.detach().clamp(min=0)
        if valid is not None:
            q = q * valid
        onehot = torch.zeros_like(logits)
        pos = tuple(list(idx) + [cls_o])
        onehot[pos] = 1.0 if valid is None else valid.to(logits.dtype)
        target = torch.zeros_like(logits); target[pos] = q.pow(gamma).to(logits.dtype)
        p = logits.sigmoid().detach()
        weight = p.pow(gamma) * (1 - onehot) + onehot
        loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum") / num_boxes
        return {"loss_ce": loss}

    criterion.loss_labels = types.MethodType(loss_labels, criterion)
    criterion._compile_batched_detection_losses = False  # быстрый CUDA-путь criterion.py:1068-1090 обошёл бы патч
    return criterion


# ---------------- copy-paste курганов (вместо Mosaic) ----------------
class Bank:
    """Банк вырезок объектов заданных классов (с полем контекста) из train-окон."""

    def __init__(self, split: dict, cat_pos: dict, classes: set[int], margin: float, max_items: int, seed: int):
        rng = random.Random(seed)
        cand = [(im, a) for im in split["images"] for a in split["anns"].get(int(im["id"]), [])
                if int(a["category_id"]) in classes and a["bbox"][2] > 2 and a["bbox"][3] > 2]
        rng.shuffle(cand)
        self.items = []
        for im, a in cand[:max_items]:
            img = Image.open(split["dir"] / im["file_name"]).convert("RGB")
            x, y, w, h = a["bbox"]; m = margin * max(w, h)
            x0, y0 = max(0, int(x - m)), max(0, int(y - m))
            x1, y1 = min(img.width, int(math.ceil(x + w + m))), min(img.height, int(math.ceil(y + h + m)))
            patch = np.asarray(img.crop((x0, y0, x1, y1)))
            self.items.append((patch, [x - x0, y - y0, w, h], cat_pos[int(a["category_id"])]))

    def paste(self, img: np.ndarray, boxes: np.ndarray, labels: list, n: int, rng: random.Random):
        if not self.items or n <= 0:
            return img, boxes, labels
        H, W = img.shape[:2]
        img = img.copy(); bx = [list(b) for b in boxes]; lb = list(labels)
        for _ in range(n):
            patch, (px, py, pw, ph), l = self.items[rng.randrange(len(self.items))]
            if rng.random() < 0.5:
                patch = patch[:, ::-1]; px = patch.shape[1] - px - pw
            ph_, pw_ = patch.shape[:2]
            if pw_ >= W or ph_ >= H:
                continue
            for _try in range(10):
                ox, oy = rng.randrange(0, W - pw_), rng.randrange(0, H - ph_)
                nb = [ox + px, oy + py, pw, ph]
                if all(nb[0] + nb[2] < b[0] or b[0] + b[2] < nb[0] or nb[1] + nb[3] < b[1] or b[1] + b[3] < nb[1] for b in bx):
                    break
            else:
                continue
            # мягкий край: альфа 1 внутри рамки, спад к краю вырезки
            yy, xx = np.mgrid[0:ph_, 0:pw_].astype(np.float32)
            dx = np.maximum(0, np.maximum(px - xx, xx - (px + pw))) / max(1.0, px)
            dy = np.maximum(0, np.maximum(py - yy, yy - (py + ph))) / max(1.0, py)
            al = np.clip(1 - np.maximum(dx, dy), 0, 1)[..., None]
            reg = img[oy:oy + ph_, ox:ox + pw_].astype(np.float32)
            # выравнивание яркости вырезки под фон
            pf = patch.astype(np.float32); pf = pf - pf.mean((0, 1)) + reg.mean((0, 1))
            img[oy:oy + ph_, ox:ox + pw_] = np.clip(al * pf + (1 - al) * reg, 0, 255).astype(np.uint8)
            bx.append(nb); lb.append(l)
        return img, np.array(bx, np.float32).reshape(-1, 4), lb


def to_tensor(img: np.ndarray, size: int):
    import torch
    h, w = img.shape[:2]
    if (h, w) != (size, size):
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), (h, w)


def make_batch(ds: Boxes, idx, a, bank: Bank | None, rng: random.Random, aug_on: bool):
    import torch
    xs, tg = [], []
    for i in idx:
        im = ds.items[i]
        img = np.asarray(Image.open(ds.s["dir"] / im["file_name"]).convert("RGB"))
        anns = [an for an in ds.s["anns"].get(int(im["id"]), []) if an["bbox"][2] > 1 and an["bbox"][3] > 1]
        boxes = np.array([an["bbox"] for an in anns], np.float32).reshape(-1, 4)
        labels = [ds.cat_pos[int(an["category_id"])] for an in anns]
        if aug_on and bank is not None and rng.random() < a.cp_prob:
            img, boxes, labels = bank.paste(img, boxes, labels, rng.randint(1, a.cp_max), rng)
        img, boxes = jitter(img, boxes, rng, a.strong_aug and aug_on)
        h, w = img.shape[:2]
        x, _ = to_tensor(img, a.size)
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        cxcywh = torch.stack([(b[:, 0] + b[:, 2] / 2) / w, (b[:, 1] + b[:, 3] / 2) / h, b[:, 2] / w, b[:, 3] / h], 1)
        xs.append(x); tg.append({"labels": torch.tensor(labels, dtype=torch.int64), "boxes": cxcywh.clamp(0, 1)})
    return torch.stack(xs), tg


def f32(o):
    """Рекурсивно переводит выходы модели в float32 (для лосса под autocast)."""
    import torch
    if isinstance(o, torch.Tensor):
        return o.float()
    if isinstance(o, dict):
        return {k: f32(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(f32(v) for v in o)
    return o


def lr_factor(it: int, a) -> float:
    """flat-cosine: линейный разгон warmup → плато до flat·iters → косинус до min_lr·lr."""
    if it < a.warmup_iters:
        return (it + 1) / a.warmup_iters
    f0 = max(a.warmup_iters, int(a.flat * a.iters))
    if it < f0:
        return 1.0
    t = (it - f0) / max(1, a.iters - f0)
    return a.min_lr + (1 - a.min_lr) * 0.5 * (1 + math.cos(math.pi * t))


def predict(a, model, post, split: dict, cat_ids: list[int], dev, limit: int | None):
    import torch
    from torchvision.ops import batched_nms
    model.eval(); K = len(cat_ids); res = []
    items = split["images"][:limit] if limit else split["images"]
    with torch.no_grad():
        for k in range(0, len(items), a.pred_bs):
            chunk = items[k:k + a.pred_bs]
            arrs = [np.asarray(Image.open(split["dir"] / im["file_name"]).convert("RGB")) for im in chunk]
            sizes = [x.shape[:2] for x in arrs]
            views = [arrs] + ([[x[:, ::-1] for x in arrs]] if a.tta_flip else [])
            per = [[] for _ in chunk]
            for v, va in enumerate(views):
                xb = torch.stack([to_tensor(x, a.size)[0] for x in va]).to(dev)
                with torch.autocast(dev.type, enabled=a.fp16 and dev.type == "cuda"):
                    out = model(xb)
                dets = post({"pred_logits": out["pred_logits"].float(), "pred_boxes": out["pred_boxes"].float()},
                            torch.tensor(sizes, device=dev))
                for j, d in enumerate(dets):
                    keep = (d["scores"] >= a.min_score) & (d["labels"] < K)
                    bx = d["boxes"][keep].float().cpu()
                    if v == 1:
                        w = sizes[j][1]
                        bx = torch.stack([w - bx[:, 2], bx[:, 1], w - bx[:, 0], bx[:, 3]], 1)
                    per[j].append((bx, d["scores"][keep].float().cpu(), d["labels"][keep].cpu()))
            for im, parts in zip(chunk, per):
                bx = torch.cat([p[0] for p in parts]); sc = torch.cat([p[1] for p in parts]); lb = torch.cat([p[2] for p in parts])
                if len(bx):
                    kp = batched_nms(bx, sc, lb, a.nms_iou)[: a.max_det]
                    bx, sc, lb = bx[kp], sc[kp], lb[kp]
                for b, s, l in zip(bx.tolist(), sc.tolist(), lb.tolist()):
                    res.append({"image_id": int(im["id"]), "category_id": int(cat_ids[int(l)]),
                                "bbox": [round(b[0], 2), round(b[1], 2), round(b[2] - b[0], 2), round(b[3] - b[1], 2)],
                                "score": round(float(s), 5)})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="rfdetr-m", choices=["rfdetr-m", "rfdetr-l", "rfdetr-b"],
                    help="rfdetr-b (Base, patch 14, сторона кратна 56) — только для --coco-weights rf-detr-base-o365.pth (чистый Objects365)")
    ap.add_argument("--size", type=int, default=640, help="сторона входа (кратна 32); 0 — родная (M 576, L 704)")
    ap.add_argument("--dinov2-dir", default=None, help="локальная папка facebook/dinov2-small (иначе скачать с HF)")
    ap.add_argument("--no-dinov2", action="store_true", help="контроль: энкодер случайный (= --init scratch)")
    ap.add_argument("--init", default="coco", choices=["coco", "dinov2", "scratch"],
                    help="coco (по умолчанию): весь детектор из чекпойнта RF-DETR COCO (M: rf-detr-medium.pth, "
                         "L: rf-detr-large-2026.pth), голова классов заново; dinov2: только энкодер DINOv2-S; scratch: всё с нуля")
    ap.add_argument("--coco-weights", default=None,
                    help="локальный .pth (Kaggle Dataset); иначе $RF_HOME/<имя> (по умолчанию ~/.cache/rfdetr), скачивается с md5")
    ap.add_argument("--iters", type=int, default=8000); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="накопление градиента (эфф. батч bs·accum)")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr-encoder", type=float, default=1.5e-4)
    ap.add_argument("--layer-decay", type=float, default=0.8, help="множитель lr на слой ViT от верхнего к нижнему")
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--warmup-iters", type=int, default=500); ap.add_argument("--flat", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=0.05, help="доля lr в конце косинуса")
    ap.add_argument("--no-aug-last", type=float, default=0.15, help="доля последних итераций без copy-paste/сильных ауг.")
    ap.add_argument("--loss", default="mal", choices=["mal", "iabce"], help="mal (DEIM) или родной IoU-aware BCE rfdetr")
    ap.add_argument("--mal-gamma", type=float, default=1.5)
    ap.add_argument("--group-detr", type=int, default=13)
    ap.add_argument("--ema", type=float, default=0.999, help="0 — без EMA")
    ap.add_argument("--cp-prob", type=float, default=0.5); ap.add_argument("--cp-max", type=int, default=4)
    ap.add_argument("--cp-classes", default="kurgany_povrezhdennye,kurgany_tselye")
    ap.add_argument("--cp-margin", type=float, default=0.3); ap.add_argument("--cp-bank", type=int, default=3000)
    ap.add_argument("--pos-share", type=float, default=0.6); ap.add_argument("--class-cap", type=float, default=4.0)
    ap.add_argument("--strong-aug", action="store_true"); ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pred-split", default="valid"); ap.add_argument("--pred-data", default=None)
    ap.add_argument("--load", default=None, help="папка с model.pt и rf_config.json: только предсказание")
    ap.add_argument("--pred-bs", type=int, default=8); ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.05); ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--nms-iou", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="10 итераций, 8 окон train, 4 окна на предсказание")
    ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args()

    import torch
    T: dict = {"t0": time.time()}
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); a.device = dev.type
    if a.smoke:
        a.iters, a.log_every, a.warmup_iters = 10, 1, 2
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.load:  # конфиг обучения берётся из сохранения
        saved = json.loads((Path(a.load) / "rf_config.json").read_text(encoding="utf-8"))
        a.model, a.size = saved["model"], saved["size"]
    root = Path(a.data)
    tr = read_split(root, "train"); pr = read_split(Path(a.pred_data) if a.pred_data else root, a.pred_split)
    cats = sorted(tr["categories"], key=lambda c: c["id"]); cat_ids = [int(c["id"]) for c in cats]; K = len(cats)
    if a.smoke:
        tr = {**tr, "images": [im for im in tr["images"] if tr["anns"].get(int(im["id"]))][:8] or tr["images"][:8]}
    ds = Boxes(tr, cat_ids, True, a.strong_aug, a.seed)

    from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
    mc, tc = make_configs(a, K)
    a.size = int(mc.resolution)
    model = build_model_from_config(mc, tc)
    crit, post = build_criterion_from_config(mc, tc)
    if a.loss == "mal":
        install_mal(crit, a.mal_gamma)
    T["build_s"] = time.time() - T["t0"]
    if a.load:
        model.load_state_dict(torch.load(Path(a.load) / "model.pt", map_location="cpu"))
        T["dinov2"] = "from --load"
    elif a.init == "coco" and not a.no_dinov2:
        T["coco"] = load_coco(model, mc, a)
        print("coco:", T["coco"], flush=True)
    elif a.init == "dinov2" and not a.no_dinov2:
        rep = load_dinov2(model, a.dinov2_dir); T["dinov2"] = {k: (v if k != "missing" else len(v)) for k, v in rep.items()}
        print("dinov2:", T["dinov2"], flush=True)
    assert_no_meta(model, a.model)  # энкодер rfdetr — классы transformers (dinov2.py:203); пин transformers==5.21.0
    model.to(dev); crit.to(dev)
    T["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model {a.model} res {a.size} params {T['params_M']:.1f}M device {dev} build {T['build_s']:.1f}s "
          f"rss {peak_rss_mb():.0f}MB", flush=True)

    if a.load:
        t = time.time()
        preds = predict(a, model, post, pr, cat_ids, dev, 4 if a.smoke else None)
        (out / f"preds_{a.pred_split}.json").write_text(json.dumps(preds), encoding="utf-8")
        T["pred_s"] = time.time() - t; T["pred_images"] = 4 if a.smoke else len(pr["images"]); T["pred_boxes"] = len(preds)
    else:
        enc = model.backbone[0].encoder.encoder
        nl = len(enc.encoder.layer)
        groups: dict = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("backbone.0.encoder.encoder."):
                sub = n[len("backbone.0.encoder.encoder."):]
                depth = int(sub.split(".")[2]) + 1 if sub.startswith("encoder.layer.") else (0 if sub.startswith("embeddings") else nl)
                lr = a.lr_encoder * a.layer_decay ** (nl - depth)
                key = f"enc{depth}"
            else:
                lr, key = a.lr, "rest"
            nd = p.ndim == 1 or n.endswith(".bias")
            g = groups.setdefault(key + ("_nd" if nd else ""), {"params": [], "lr": lr, "weight_decay": 0.0 if nd else a.wd})
            g["params"].append(p)
        pg = list(groups.values())
        for g in pg:
            g["base_lr"] = g["lr"]
        opt = torch.optim.AdamW(pg)
        ema = None
        if a.ema > 0:
            ema = copy.deepcopy(model).eval()
            for p in ema.parameters():
                p.requires_grad_(False)
        scaler = torch.amp.GradScaler("cuda", enabled=a.fp16 and dev.type == "cuda")
        w = ds.sample_weights(a.pos_share, a.class_cap)
        cp_cls = {int(c["id"]) for c in cats if c["name"] in set(a.cp_classes.split(","))}
        bank = Bank(tr, ds.cat_pos, cp_cls, a.cp_margin, 64 if a.smoke else a.cp_bank, a.seed) if a.cp_prob > 0 else None
        T["cp_bank"] = len(bank.items) if bank else 0
        nrng = np.random.default_rng(a.seed); prng = random.Random(a.seed)
        bs = min(a.bs, 2) if a.smoke else a.bs
        accum = 1 if a.smoke else a.accum
        aug_stop = int((1 - a.no_aug_last) * a.iters)
        model.train(); t = time.time(); hist = []
        for it in range(a.iters):
            f = lr_factor(it, a)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            opt.zero_grad(set_to_none=True); tot = 0.0
            for _ in range(accum):
                idx = nrng.choice(len(ds), size=bs, replace=True, p=w)
                xb, tg = make_batch(ds, idx, a, bank, prng, it < aug_stop)
                tg = [{k: v.to(dev) for k, v in d.items()} for d in tg]
                with torch.autocast(dev.type, enabled=a.fp16 and dev.type == "cuda"):
                    o = model(xb.to(dev), tg)
                L = crit(f32(o), tg)  # лосс в fp32 даже при --fp16
                loss = sum(L[k] * crit.weight_dict[k] for k in L if k in crit.weight_dict) / accum
                scaler.scale(loss).backward(); tot += float(loss.detach())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt); scaler.update()
            if ema is not None:
                with torch.no_grad():
                    d = min(a.ema, (1 + it) / (10 + it))
                    for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                        if pe.dtype.is_floating_point:
                            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                        else:
                            pe.copy_(pm)
            hist.append(tot)
            if (it + 1) % a.log_every == 0 or it == a.iters - 1:
                el = time.time() - t
                print(f"it {it + 1}/{a.iters} loss {np.mean(hist[-a.log_every:]):.4f} lr×{f:.3f} {el / (it + 1):.2f}s/it "
                      f"rss {peak_rss_mb():.0f}MB", flush=True)
        T["train_s"] = time.time() - t; T["s_per_iter"] = T["train_s"] / a.iters
        T["loss_first"], T["loss_last"] = hist[0], hist[-1]; T["bs"] = bs; T["accum"] = accum; T["iters"] = a.iters
        final = ema if ema is not None else model
        if not a.smoke:
            (out / "model").mkdir(exist_ok=True)
            torch.save(final.state_dict(), out / "model" / "model.pt")
            (out / "model" / "rf_config.json").write_text(json.dumps({"model": a.model, "size": a.size, "classes": cats}), encoding="utf-8")
        t = time.time()
        if pr is not None:
            preds = predict(a, final, post, pr, cat_ids, dev, 4 if a.smoke else None)
            (out / f"preds_{a.pred_split}.json").write_text(json.dumps(preds), encoding="utf-8")
            T["pred_s"] = time.time() - t; T["pred_images"] = 4 if a.smoke else len(pr["images"]); T["pred_boxes"] = len(preds)
    T["peak_rss_MB"] = peak_rss_mb()
    if dev.type == "cuda":
        T["cuda_peak_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    T["total_s"] = time.time() - T.pop("t0")
    T.update(model=a.load or a.model, size=a.size, loss=a.loss, device=str(dev), smoke=a.smoke, torch=torch.__version__)
    (out / "timing.json").write_text(json.dumps(T, indent=1, default=str), encoding="utf-8")
    print(json.dumps(T, default=str), flush=True)


if __name__ == "__main__":
    main()
