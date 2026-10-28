"""Обучение коробочного детектора (D-FINE-L, по желанию RT-DETRv2) на COCO из scripts/coco_from_cache.py.

Написано bot-5 с нуля на публичном API Hugging Face transformers (AutoModelForObjectDetection /
AutoImageProcessor); код конкурентов не использовался. Вход — папка RF-DETR-раскладки
(train/valid[/test] с _annotations.coco.json и PNG). Выход — результаты COCO
[{"image_id","category_id","bbox":[x,y,w,h],"score"}] в пикселях PNG для scripts/coco_preds_to_stack.py
(category_id 1..K, как в categories файла) и timing.json (время фаз, итер/с, пик памяти).

    # смоук: 10 итераций, крошечное подмножество
    python scripts/det_train_hf.py --data COCO_fold0 --out runs/f0 --smoke
    # полный прогон одного фолда на T4
    python scripts/det_train_hf.py --data COCO_fold0 --out runs/f0 --model dfine-l --iters 8000 --bs 8 --size 640 --fp16
"""
from __future__ import annotations

import argparse
import json
import math
import random
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

MODELS = {
    "dfine-l": "ustc-community/dfine-large-coco",
    "dfine-s": "ustc-community/dfine-small-coco",
    "rtdetrv2-r50": "PekingU/rtdetr_v2_r50vd",
}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def read_split(root: Path, split: str) -> dict | None:
    p = root / split / "_annotations.coco.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    anns: dict[int, list] = {}
    for a in d.get("annotations", []):
        anns.setdefault(int(a["image_id"]), []).append(a)
    return {"dir": root / split, "images": d["images"], "anns": anns, "categories": d["categories"]}


# ---------------- аугментации (свои, на numpy/PIL) ----------------
def jitter(img: np.ndarray, boxes: np.ndarray, rng: random.Random, strong: bool) -> tuple[np.ndarray, np.ndarray]:
    """img HxWx3 uint8, boxes Nx4 xywh в пикселях. Геометрия: rot90 и отражение; фотометрия: усиление,
    сдвиг, гамма, серый, размытие; при strong — порча качества (JPEG-подобное уменьшение-увеличение)."""
    h, w = img.shape[:2]
    x = img.astype(np.float32)
    b = boxes.astype(np.float32).copy()
    if rng.random() < 0.5:  # отражение по горизонтали
        x = x[:, ::-1]
        if len(b):
            b[:, 0] = w - b[:, 0] - b[:, 2]
    k = rng.randrange(4)
    for _ in range(k):  # поворот на 90° против часовой: (x, y) -> (y, W - x)
        hh, ww = x.shape[:2]
        x = np.rot90(x)
        if len(b):
            nx, ny = b[:, 1].copy(), ww - b[:, 0] - b[:, 2]
            b = np.stack([nx, ny, b[:, 3], b[:, 2]], 1)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.3:
        x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.1:
        x = np.repeat(x.mean(2, keepdims=True), 3, 2)
    out = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    if rng.random() < 0.2:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if strong and rng.random() < 0.3:
        f = rng.uniform(0.4, 0.8)
        sz = out.size
        out = out.resize((max(8, int(sz[0] * f)), max(8, int(sz[1] * f))), Image.BILINEAR).resize(sz, Image.BILINEAR)
    return np.asarray(out), b


class Boxes:
    """Набор окон с рамками. Метки модели 0..K-1 = позиция category_id в categories."""

    def __init__(self, split: dict, cat_ids: list[int], train: bool, strong_aug: bool, seed: int):
        self.s, self.train, self.strong = split, train, strong_aug
        self.cat_pos = {c: i for i, c in enumerate(cat_ids)}
        self.items = split["images"]
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def load(self, i: int):
        im = self.items[i]
        img = np.asarray(Image.open(self.s["dir"] / im["file_name"]).convert("RGB"))
        anns = [a for a in self.s["anns"].get(int(im["id"]), []) if a["bbox"][2] > 1 and a["bbox"][3] > 1]
        boxes = np.array([a["bbox"] for a in anns], np.float32).reshape(-1, 4)
        labels = [self.cat_pos[int(a["category_id"])] for a in anns]
        if self.train:
            img, boxes = jitter(img, boxes, self.rng, self.strong)
        target = {"image_id": int(im["id"]), "annotations": [
            {"bbox": [float(v) for v in bb], "category_id": int(l), "area": float(bb[2] * bb[3]), "iscrowd": 0}
            for bb, l in zip(boxes, labels)]}
        return img, target

    def sample_weights(self, pos_share: float, cap: float) -> np.ndarray:
        """Вес окна: доля pos_share на окна с объектами (внутри — по редкости класса 1/√n, не больше cap×),
        остальное — окнам без объектов. Если пустых окон нет, всё идёт на окна с объектами."""
        n_cls: dict[int, int] = {}
        for anns in self.s["anns"].values():
            for a in anns:
                n_cls[int(a["category_id"])] = n_cls.get(int(a["category_id"]), 0) + 1
        cw = {c: min(cap, math.sqrt(max(n_cls.values()) / n)) if n_cls else 1.0 for c, n in n_cls.items()}
        w = np.zeros(len(self.items))
        has = np.zeros(len(self.items), bool)
        for i, im in enumerate(self.items):
            anns = self.s["anns"].get(int(im["id"]), [])
            if anns:
                has[i] = True
                w[i] = max(cw[int(a["category_id"])] for a in anns)
        if has.all() or not has.any():
            return np.where(has, w, 1.0) / np.where(has, w, 1.0).sum()
        w[has] = pos_share * w[has] / w[has].sum()
        w[~has] = (1 - pos_share) / (~has).sum()
        return w


TRANSFORMERS_PIN = "5.21.0"  # проверено на боксе; см. kaggle/requirements_dfine.txt


def assert_no_meta(model, where: str) -> None:
    """Падает, если после загрузки хоть один параметр/буфер остался на meta (не материализован).
    transformers 5.x на время from_pretrained подменяет tie_weights на весь процесс: при загрузке двух моделей
    в параллельных потоках class_embed/bbox_embed D-FINE могут остаться на meta. Наши фолды — отдельные процессы."""
    bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if bad:
        import transformers
        raise RuntimeError(f"{where}: {len(bad)} тензор(ов) на meta после загрузки (transformers {transformers.__version__}, "
                           f"ожидается {TRANSFORMERS_PIN}; модели не грузить параллельно в потоках): {bad[:8]}")


def build(args, K: int, id2label: dict):
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    ckpt = MODELS.get(args.model, args.model)
    proc = AutoImageProcessor.from_pretrained(ckpt)
    proc.size = {"height": args.size, "width": args.size}
    extra = {}
    if args.num_dn is not None:
        extra["num_denoising"] = args.num_dn
    model = AutoModelForObjectDetection.from_pretrained(
        ckpt, num_labels=K, id2label=id2label, label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True, **extra)
    return proc, model


def batch(proc, ds: Boxes, idx):
    imgs, tg = zip(*(ds.load(i) for i in idx))
    enc = proc(images=list(imgs), annotations=list(tg), return_tensors="pt")
    return enc["pixel_values"], enc["labels"]


def predict(args, proc, model, split: dict, cat_ids: list[int], dev, limit: int | None):
    import torch
    from torchvision.ops import batched_nms
    model.eval()
    res = []
    items = split["images"][:limit] if limit else split["images"]
    with torch.no_grad():
        for k in range(0, len(items), args.pred_bs):
            chunk = items[k:k + args.pred_bs]
            imgs = [Image.open(split["dir"] / im["file_name"]).convert("RGB") for im in chunk]
            sizes = [(im.height, im.width) for im in imgs]
            views = [imgs] + ([[im.transpose(Image.FLIP_LEFT_RIGHT) for im in imgs]] if args.tta_flip else [])
            per = [[] for _ in chunk]
            for v, vimgs in enumerate(views):
                pv = proc(images=vimgs, return_tensors="pt")["pixel_values"].to(dev)
                with torch.autocast(dev.type, enabled=args.fp16 and dev.type == "cuda"):
                    out = model(pixel_values=pv)
                dets = proc.post_process_object_detection(out, threshold=args.min_score, target_sizes=sizes)
                for j, d in enumerate(dets):
                    bx = d["boxes"].float().cpu()
                    if v == 1:  # вернуть отражение
                        w = sizes[j][1]
                        bx = torch.stack([w - bx[:, 2], bx[:, 1], w - bx[:, 0], bx[:, 3]], 1)
                    per[j].append((bx, d["scores"].float().cpu(), d["labels"].cpu()))
            for im, parts in zip(chunk, per):
                bx = torch.cat([p[0] for p in parts]); sc = torch.cat([p[1] for p in parts]); lb = torch.cat([p[2] for p in parts])
                if len(bx):
                    keep = batched_nms(bx, sc, lb, args.nms_iou)[: args.max_det]
                    bx, sc, lb = bx[keep], sc[keep], lb[keep]
                for b, s, l in zip(bx.tolist(), sc.tolist(), lb.tolist()):
                    res.append({"image_id": int(im["id"]), "category_id": int(cat_ids[int(l)]),
                                "bbox": [round(b[0], 2), round(b[1], 2), round(b[2] - b[0], 2), round(b[3] - b[1], 2)],
                                "score": round(float(s), 5)})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="папка coco_from_cache.py --layout rfdetr (train/valid[/test])")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="dfine-l", help=f"{list(MODELS)} или id/путь чекпойнта HF")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--size", type=int, default=640, help="сторона входа модели, px")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05, help="pct_start OneCycle")
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--num-dn", type=int, default=None, help="num_denoising (0 — если обучение разваливается)")
    ap.add_argument("--pos-share", type=float, default=0.6, help="доля выборки — окна с объектами")
    ap.add_argument("--class-cap", type=float, default=4.0, help="потолок веса редкого класса 1/√n")
    ap.add_argument("--strong-aug", action="store_true", help="добавить порчу качества снимка")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2, help="torch.set_num_threads на CPU")
    ap.add_argument("--pred-split", default="valid")
    ap.add_argument("--pred-data", default=None, help="другая папка COCO для предсказания (сплошная сетка фолда); по умолчанию --data")
    ap.add_argument("--load", default=None, help="папка save_pretrained: без обучения, только предсказание")
    ap.add_argument("--pred-bs", type=int, default=8)
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="10 итераций, 8 окон train, 4 окна на предсказание")
    ap.add_argument("--dry-run", action="store_true", help="только модель, загрузчик и один проход вперёд на 1 окне")
    ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args()

    import torch
    T: dict = {"t0": time.time()}
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.smoke:
        a.iters, a.log_every = 10, 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data)
    tr, pr = read_split(root, "train"), read_split(Path(a.pred_data) if a.pred_data else root, a.pred_split)
    cats = sorted(tr["categories"], key=lambda c: c["id"])
    cat_ids = [int(c["id"]) for c in cats]
    id2label = {i: c["name"] for i, c in enumerate(cats)}
    if a.smoke or a.dry_run:
        tr = {**tr, "images": [im for im in tr["images"] if tr["anns"].get(int(im["id"]))][:8] or tr["images"][:8]}
    ds = Boxes(tr, cat_ids, True, a.strong_aug, a.seed)

    if a.load:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        proc = AutoImageProcessor.from_pretrained(a.load); model = AutoModelForObjectDetection.from_pretrained(a.load)
    else:
        proc, model = build(a, len(cats), id2label)
    assert_no_meta(model, a.load or a.model)
    model.to(dev)
    T["build_s"] = time.time() - T["t0"]
    T["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model {MODELS.get(a.model, a.model)} params {T['params_M']:.1f}M device {dev} build {T['build_s']:.1f}s "
          f"rss {peak_rss_mb():.0f}MB", flush=True)

    if a.dry_run:
        t = time.time()
        pv, lab = batch(proc, ds, [0])
        model.train()
        o = model(pixel_values=pv.to(dev), labels=[{k: v.to(dev) for k, v in l.items()} for l in lab])
        T["dry_forward_s"] = time.time() - t
        T["dry_loss"] = float(o.loss)
        T["dry_input"] = list(pv.shape)
    elif a.load:
        t = time.time(); model.eval()
        preds = predict(a, proc, model, pr, cat_ids, dev, 4 if a.smoke else None)
        (out / f"preds_{a.pred_split}.json").write_text(json.dumps(preds), encoding="utf-8")
        T["pred_s"] = time.time() - t; T["pred_images"] = 4 if a.smoke else len(pr["images"]); T["pred_boxes"] = len(preds)
    else:
        bb = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
        rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        lrs = [a.lr * a.backbone_lr_mult, a.lr]
        opt = torch.optim.AdamW([{"params": bb, "lr": lrs[0]}, {"params": rest, "lr": lrs[1]}], weight_decay=a.wd)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lrs, total_steps=a.iters, pct_start=a.warmup)
        scaler = torch.amp.GradScaler("cuda", enabled=a.fp16 and dev.type == "cuda")
        w = ds.sample_weights(a.pos_share, a.class_cap)
        rng = np.random.default_rng(a.seed)
        bs = min(a.bs, 2) if a.smoke else a.bs
        model.train()
        t = time.time(); hist = []
        for it in range(a.iters):
            idx = rng.choice(len(ds), size=bs, replace=True, p=w)
            pv, lab = batch(proc, ds, idx)
            with torch.autocast(dev.type, enabled=a.fp16 and dev.type == "cuda"):
                o = model(pixel_values=pv.to(dev), labels=[{k: v.to(dev) for k, v in l.items()} for l in lab])
            opt.zero_grad(set_to_none=True)
            scaler.scale(o.loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt); scaler.update(); sch.step()
            hist.append(float(o.loss))
            if (it + 1) % a.log_every == 0 or it == a.iters - 1:
                el = time.time() - t
                print(f"it {it + 1}/{a.iters} loss {np.mean(hist[-a.log_every:]):.4f} {el / (it + 1):.2f}s/it "
                      f"rss {peak_rss_mb():.0f}MB", flush=True)
        T["train_s"] = time.time() - t; T["s_per_iter"] = T["train_s"] / a.iters
        T["loss_first"], T["loss_last"] = hist[0], hist[-1]; T["bs"] = bs; T["iters"] = a.iters
        if not a.smoke:
            model.save_pretrained(out / "model"); proc.save_pretrained(out / "model")
        t = time.time()
        if pr is not None:
            preds = predict(a, proc, model, pr, cat_ids, dev, 4 if a.smoke else None)
            (out / f"preds_{a.pred_split}.json").write_text(json.dumps(preds), encoding="utf-8")
            T["pred_s"] = time.time() - t; T["pred_images"] = 4 if a.smoke else len(pr["images"]); T["pred_boxes"] = len(preds)
    T["peak_rss_MB"] = peak_rss_mb()
    if dev.type == "cuda":
        T["cuda_peak_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    T["total_s"] = time.time() - T.pop("t0")
    T.update(model=a.load or MODELS.get(a.model, a.model), size=a.size, device=str(dev), smoke=a.smoke, dry_run=a.dry_run,
             torch=torch.__version__)
    (out / "timing.json").write_text(json.dumps(T, indent=1), encoding="utf-8")
    print(json.dumps(T), flush=True)


if __name__ == "__main__":
    main()
