#!/usr/bin/env python3
"""Второй этап по картинке: кандидат стека проверяется классификатором окна вокруг него, вне обучения по семьям.

Стекер (`rerank_oof.py`) видит только числа стека: соседей, группы, уверенность. Картинку вокруг
кандидата он не видит. Проверка кропа классификатором — приём, который встречается в трёх местах:
- полевая проверка курганов Алто-Миньо: ViT по ортофото отсеивает ложные срабатывания детектора;
- открытый образ КОЗ-2: DINOv2 по окну с контекстом ×1,8 выбрасывает «фон»;
- образ КОЗ-3 сильнейшей открытой команды: отдельный классификатор городищ после детектора.
Здесь то же на нашем стеке и нашими окнами: `expds.tiles.iter_sources` режет кроп тем же чтением
и теми же каналами, что кэш обучения (снимок — RGB, лидар — каналы рельефа из LAS).

Метка — официальное правило попадания с эталоном своего класса (как `rerank_oof.py --labels any`):
классификатор отвечает, есть ли здесь объект, а не первый ли это круг на нём. Дубли снимают
пороги групп. Модель — энкодер smp (по умолчанию ResNet-18, веса ImageNet; `--no-pretrained` — без них)
и линейная голова; повороты на 90° и отражения. Каждая семья участков оценивается моделью без неё.

Новая уверенность (`--blend`):
- geo — среднее геометрическое уверенности стека и вероятности кропа (по умолчанию);
- p — только вероятность кропа;
- none — уверенность прежняя, вероятность в свойстве `verify_p`.
Кандидат группы, у которого кропа нет (ниже `--min-conf`, вне снимка), получает вместо p медиану p
своей группы и класса: шкала группы не рвётся. Старая уверенность — в `confidence_base`.

    python scripts/scoreboard.py <флаги раздела 0> --dump-stack F:/expds3/work/stack_base.geojson
    python scripts/crop_verify.py --root F:/2/train-expds-3 --stack F:/expds3/work/stack_base.geojson \
        --gt F:/expds3/work/gt_train_v2.geojson --markup sp_or --groups sp_focal sp_focal2 \
        --classes kurgany_povrezhdennye kurgany_tselye --res 0.5 --tile 96 --min-conf 0.05 \
        --out F:/expds3/work/stack_verify_sp.geojson --report notes/verify_sp.json
    python scripts/scoreboard.py <те же флаги> --rerank F:/expds3/work/stack_verify_sp.geojson --honest 10 --honest-by region

Лидар — `--markup li --groups li li_dev li_mosaic --res 0.5 --tile 64`: ЦМР строится заново из LAS,
это минуты на облако. Приёмка — только табло; различимость в отчёте — подсказка, не доказательство.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("crop_verify")


def load_rerank():
    spec = importlib.util.spec_from_file_location("rerank_oof", ROOT / "scripts" / "rerank_oof.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select(stack: dict, classes: set[str], markup: str, groups: set[str] | None, min_conf: float) -> list[int]:
    """Номера кандидатов стека, которые проверяются: класс, тип данных или группа, уверенность."""
    out = []
    for i, f in enumerate(stack.get("features", [])):
        p = f.get("properties") or {}
        if p.get("class_name") not in classes:
            continue
        grp = p.get("threshold_group") or p.get("markup_type")
        if groups is not None and grp not in groups:
            continue
        if groups is None and p.get("markup_type") != markup:
            continue
        if float(p.get("confidence", 0.0)) < min_conf:
            continue
        out.append(i)
    return out


def extract_crops(root: str, stack: dict, picked: list[int], markup: str, res: float, tile: int,
                  li_tags: tuple[str, ...], dem: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Кроп tile×tile с центром в точке кандидата: то же чтение и те же каналы, что у кэша обучения."""
    from shapely.geometry import shape

    from expds.catalog import scan_root
    from expds.crs import WEB_MERCATOR, to_crs
    from expds.tiles import fix_channels, iter_sources

    X = np.zeros((len(picked), 3, tile, tile), np.uint8)
    ok = np.zeros(len(picked), bool)
    by_region: dict[str, dict[int, object]] = defaultdict(dict)
    feats = stack["features"]
    for k, i in enumerate(picked):
        p = feats[i].get("properties") or {}
        by_region[str(p.get("region_name"))][k] = shape(feats[i]["geometry"]).centroid
    regions = {r.name: r for r in scan_root(root)}
    for name, todo in by_region.items():
        reg = regions.get(name)
        if reg is None:
            log.warning("участка %s нет в %s", name, root)
            continue
        for _, crs, bnd, read in iter_sources(reg, markup, list(li_tags), "las", res, tile, dem):
            for k, pt in list(todo.items()):
                q = to_crs(pt, WEB_MERCATOR, crs)
                if not bnd.contains(q):
                    continue
                img, _ = read(q.x, q.y)
                if not img.any():
                    continue
                X[k] = fix_channels(np.asarray(img), 3)
                ok[k] = True
                del todo[k]
            if not todo:
                break
    return X, ok


def make_model(encoder: str, pretrained: bool):
    import segmentation_models_pytorch as smp
    import torch

    enc = smp.encoders.get_encoder(encoder, in_channels=3, depth=5, weights="imagenet" if pretrained else None)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.head = torch.nn.Linear(enc.out_channels[-1], 1)

        def forward(self, x):
            return self.head(self.enc(x)[-1].mean(dim=(2, 3))).squeeze(1)

    return Net()


def _tensor(x: np.ndarray, device):
    import torch

    return (torch.from_numpy(np.ascontiguousarray(x)).to(device).float() / 255.0 - 0.5) / 0.25


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, epochs: int = 8, batch: int = 64,
                lr: float = 3e-4, encoder: str = "resnet18", pretrained: bool = True, seed: int = 0,
                device: str | None = None) -> np.ndarray:
    """Учит классификатор кропа на (Xtr, ytr) и возвращает вероятности для Xte (среднее по 4 отражениям)."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = make_model(encoder, pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = max(1, int(ytr.sum()))
    neg = max(1, len(ytr) - pos)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=dev))
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(ytr))
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            x = _tensor(Xtr[b], dev)
            k = int(rng.integers(4))
            x = torch.rot90(x, k, dims=(2, 3))
            if rng.random() < 0.5:
                x = x.flip(-1)
            y = torch.from_numpy(ytr[b].astype(np.float32)).to(dev)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()
    model.eval()
    out = np.zeros(len(Xte), np.float64)
    with torch.inference_mode():
        for s in range(0, len(Xte), batch):
            x = _tensor(Xte[s:s + batch], dev)
            p = sum(torch.sigmoid(model(v)) for v in (x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2))) / 4
            out[s:s + batch] = p.cpu().numpy()
    return out


def oof_verify(X: np.ndarray, y: np.ndarray, regions: list[str], folds: int, **kw) -> np.ndarray:
    """Вероятности вне обучения по семьям участков; кандидат фолда без обеих меток в обучении — NaN."""
    from expds.metric import family_folds
    from expds.naming import site_family

    p = np.full(len(y), np.nan)
    split = family_folds(sorted(set(regions)), site_family, folds)
    fold_of = {r: k for k, regs in enumerate(split) for r in regs}
    fold = np.array([fold_of[r] for r in regions])
    for k in sorted(set(fold)):
        te = fold == k
        tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        p[te] = fit_predict(X[tr], y[tr], X[te], seed=int(k), **kw)
    return p


def blend(conf: np.ndarray, p: np.ndarray, mode: str) -> np.ndarray:
    if mode == "p":
        return p
    if mode == "geo":
        return np.sqrt(np.clip(conf, 0, 1) * np.clip(p, 0, 1))
    if mode == "none":
        return conf
    raise ValueError(f"blend: geo, p или none, не {mode}")


def run(root: str, stack: dict, gt: dict, markup: str, classes: set[str], groups: set[str] | None,
        min_conf: float, res: float, tile: int, li_tags: tuple[str, ...], folds: int, mode: str,
        metric_path: str | None = None, **kw) -> tuple[dict, dict]:
    rr = load_rerank()
    picked = select(stack, classes, markup, groups, min_conf)
    if not picked:
        raise SystemExit("нет кандидатов под фильтр: проверьте --classes, --groups, --min-conf")
    sub = {"type": "FeatureCollection", "features": [stack["features"][i] for i in picked]}
    rows, y_rows = rr.collect(sub, gt, metric_path, labels="any")
    y = np.zeros(len(picked), np.int8)
    for r, v in zip(rows, y_rows):
        y[r["idx"]] = v
    X, ok = extract_crops(root, stack, picked, markup, res, tile, li_tags)
    regions = [str((stack["features"][i].get("properties") or {}).get("region_name")) for i in picked]
    p = np.full(len(picked), np.nan)
    idx = np.flatnonzero(ok)
    if idx.size:
        p[idx] = oof_verify(X[idx], y[idx], [regions[i] for i in idx], folds, **kw)
    feats = [dict(f, properties=dict(f.get("properties") or {})) for f in stack["features"]]
    conf = np.array([float(feats[i]["properties"].get("confidence", 0.0)) for i in picked])
    keys = [(feats[i]["properties"].get("threshold_group") or feats[i]["properties"].get("markup_type"),
             feats[i]["properties"].get("class_name")) for i in picked]
    med: dict[tuple, float] = {}
    for key in set(keys):
        vals = [p[k] for k in range(len(picked)) if keys[k] == key and np.isfinite(p[k])]
        med[key] = float(np.median(vals)) if vals else 0.5
    report: dict = {"candidates": len(picked), "with_crop": int(ok.sum()), "verified": int(np.isfinite(p).sum()),
                    "hits": int(y.sum()), "blend": mode, "groups": {}}
    m = np.isfinite(p)
    if m.any() and 0 < y[m].sum() < m.sum():
        report["auc_stack"] = round(rr.auc(conf[m], y[m]), 4)
        report["auc_crop"] = round(rr.auc(p[m], y[m]), 4)
        report["auc_blend"] = round(rr.auc(blend(conf[m], p[m], "geo"), y[m]), 4)
    for f in feats:
        f["properties"]["confidence_base"] = f["properties"].get("confidence")
    # Вся группа и класс, затронутые проверкой, переводятся на новую шкалу, включая кандидатов без кропа.
    touched = set(keys)
    for f in feats:
        pr = f["properties"]
        key = (pr.get("threshold_group") or pr.get("markup_type"), pr.get("class_name"))
        if key in touched:
            pr["confidence"] = round(float(blend(np.array([float(pr.get("confidence") or 0.0)]),
                                                 np.array([med[key]]), mode)[0]), 6)
    for k, i in enumerate(picked):
        pr = feats[i]["properties"]
        pk = p[k] if np.isfinite(p[k]) else med[keys[k]]
        pr["verify_p"] = round(float(pk), 6)
        pr["confidence"] = round(float(blend(np.array([conf[k]]), np.array([pk]), mode)[0]), 6)
    for key in sorted(touched, key=str):
        report["groups"][f"{key[0]}:{key[1]}"] = {"median_p": round(med[key], 4)}
    return {"type": "FeatureCollection", "features": feats}, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="корень трейна (те же участки, что в стеке)")
    ap.add_argument("--stack", required=True, help="стек до порогов (scoreboard.py --dump-stack)")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--markup", required=True, help="откуда резать кроп: sp_or, ae, or, li")
    ap.add_argument("--classes", nargs="+", default=["kurgany_povrezhdennye", "kurgany_tselye"])
    ap.add_argument("--groups", nargs="*", default=None, help="группы порога; без них — все кандидаты --markup")
    ap.add_argument("--min-conf", type=float, default=0.05, help="кандидаты слабее не проверяются (их порог всё равно выше)")
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--li-tags", nargs="*", default=None, help="каналы рельефа для --markup li (по умолчанию как у кэша)")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--blend", choices=["geo", "p", "none"], default="geo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from expds.lidar import RELIEF_TAGS

    stack = json.loads(Path(args.stack).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    out, report = run(args.root, stack, gt, args.markup, set(args.classes),
                      set(args.groups) if args.groups else None, args.min_conf, args.res, args.tile,
                      tuple(args.li_tags or RELIEF_TAGS), args.folds, args.blend, args.metric,
                      epochs=args.epochs, batch=args.batch, lr=args.lr, encoder=args.encoder,
                      pretrained=not args.no_pretrained)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
