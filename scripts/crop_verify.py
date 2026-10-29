#!/usr/bin/env python3
"""Второй этап по картинке: кандидат стека проверяется классификатором окна вокруг него, вне обучения по семьям.

Стекер (`rerank_oof.py`) видит только числа стека: соседей, группы, уверенность. Картинку вокруг
кандидата он не видит. Проверка кропа классификатором — приём, который встречается в трёх местах:
- полевая проверка курганов Алто-Миньо: ViT по ортофото отсеивает ложные срабатывания детектора;
- открытый образ КОЗ-2: DINOv2 по окну с контекстом ×1,8 выбрасывает «фон»;
- образ КОЗ-3 сильнейшей открытой команды: отдельный классификатор городищ после детектора.
Здесь то же на нашем стеке и нашими окнами: `expds.tiles.iter_sources` режет кроп тем же чтением
и теми же каналами, что кэш обучения (снимок — RGB, лидар — каналы рельефа из LAS).

Метка для обучения — «есть ли здесь объект». Метка «первый по метрике» считается по всему стеку
и идёт только в различимость отчёта. Прогон 27.10 ещё обнулял второй круг на эталоне, глядя
в сам эталон: это утечка, и его табло (0,311053 → 0,318330) не честное. Теперь по умолчанию
выход — только вероятность картинки; старое поведение — `--demote-with-gt`, только для
диагностики. Если ни одного кропа нет, уверенность группы не меняется. `--save-model` пишет
веса на всех кропах для `[verify] weights` (`expds.crop_model`). Модель — энкодер smp (по умолчанию ResNet-18, веса ImageNet; `--encoder hf:facebook/dinov2-with-registers-base` — замороженный
DINOv2 с регистрами и линейная проба на [CLS, среднее патчей], только OOF без `--save-model`;
`--no-pretrained` — без них) и линейная голова; повороты на 90° и отражения. Каждая семья
участков оценивается моделью без неё.

Новая уверенность (`--blend`):
- geo — среднее геометрическое уверенности стека и вероятности кропа (по умолчанию);
- p — только вероятность кропа;
- none — уверенность прежняя, вероятность в свойстве `verify_p`.
Кандидат, который выбран для проверки, но кропа не имеет, получает медиану своей группы,
если хоть одно окно группы прочитано. Остальные круги группы не двигаются: иначе слабый
круг поднимается к медиане и на низком пороге лидара становится ложным. Старая уверенность
проверенных — в `confidence_base`.

    python scripts/scoreboard.py <флаги раздела 0> --dump-stack F:/expds3/work/stack_base.geojson
    python scripts/crop_verify.py --root F:/2/train-expds-3 --stack F:/expds3/work/stack_base.geojson \
        --gt F:/expds3/work/gt_train_v2.geojson --markup sp_or --groups sp_focal sp_focal2 \
        --classes kurgany_povrezhdennye kurgany_tselye --res 0.5 --tile 96 --min-conf 0.05 \
        --out F:/expds3/work/stack_verify_sp.geojson --report notes/verify_sp.json
    python scripts/scoreboard.py <те же флаги> --rerank F:/expds3/work/stack_verify_sp.geojson --honest 10

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
    fill = np.zeros(len(picked), np.int64)
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
            for k, pt in todo.items():
                q = to_crs(pt, WEB_MERCATOR, crs)
                if not bnd.contains(q):
                    continue
                img, _ = read(q.x, q.y)
                spatial = np.any(img, axis=0) if getattr(img, "ndim", 0) == 3 else img
                n = int(np.count_nonzero(spatial))
                if n <= fill[k]:
                    continue
                X[k] = fix_channels(np.asarray(img), 3)
                ok[k] = True
                fill[k] = n
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


HF_PREFIX = "hf:"
_HF_CACHE: dict = {}


def hf_embed(X: np.ndarray, model_id: str, device: str | None = None, batch: int = 32, px: int = 224) -> np.ndarray:
    """Замороженный ViT из HF (DINOv2 / DINOv2 с регистрами, Apache-2.0): [CLS, среднее патч-токенов].

    Регистры отбрасываются (`num_register_tokens` из конфига). Кроп растягивается до px (кратно 14),
    нормировка ImageNet, признак — среднее по четырём отражениям, как `predict` у smp-ветки.
    `model_id` — id HF или локальная папка снапшота; `id@ревизия` пинует ревизию.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    mid, _, rev = model_id.partition("@")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (mid, rev, str(dev))
    if key not in _HF_CACHE:
        cfg = AutoConfig.from_pretrained(mid, revision=rev or None)
        m = AutoModel.from_pretrained(mid, revision=rev or None).eval().to(dev)
        if dev.type == "cuda":
            m = m.half()
        _HF_CACHE[key] = (m, 1 + int(getattr(cfg, "num_register_tokens", 0) or 0))
    m, npre = _HF_CACHE[key]
    dt = next(m.parameters()).dtype
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for s in range(0, len(X), batch):
            x = torch.from_numpy(np.ascontiguousarray(X[s:s + batch])).to(dev).float() / 255.0
            x = torch.nn.functional.interpolate(x, size=(px, px), mode="bilinear", align_corners=False)
            x = ((x - mean) / std).to(dt)
            acc = 0
            for v in (x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)):
                h = m(pixel_values=v).last_hidden_state.float()
                acc = acc + torch.cat([h[:, 0], h[:, npre:].mean(1)], 1)
            out.append((acc / 4).cpu().numpy())
            log.info("кроп %s: %d/%d", mid, min(s + batch, len(X)), len(X))
    return np.concatenate(out).astype(np.float32)


def lin_fit_predict(Ftr: np.ndarray, ytr: np.ndarray, Fte: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """Линейная проба на замороженных признаках: стандартизация по обучению, логрегрессия с L2 (l2 зафиксирован
    заранее, не тюнится), веса классов сбалансированы, LBFGS до сходимости."""
    import torch

    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    A = torch.from_numpy(((Ftr - mu) / sd).astype(np.float64))
    B = torch.from_numpy(((Fte - mu) / sd).astype(np.float64))
    y = torch.from_numpy(ytr.astype(np.float64))
    pos = max(1.0, float(y.sum())); neg = max(1.0, float(len(y) - y.sum()))
    w_s = torch.where(y > 0, len(y) / (2 * pos), len(y) / (2 * neg))
    w = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = A @ w + b
        loss = (w_s * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).mean() + 0.5 * l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(B @ w + b).numpy()


def _tensor(x: np.ndarray, device):
    import torch

    return (torch.from_numpy(np.ascontiguousarray(x)).to(device).float() / 255.0 - 0.5) / 0.25


def fit(Xtr: np.ndarray, ytr: np.ndarray, epochs: int = 8, batch: int = 64, lr: float = 3e-4,
        encoder: str = "resnet18", pretrained: bool = True, seed: int = 0, device: str | None = None):
    """Классификатор кропа, обученный на (Xtr, ytr). Модель в режиме оценки."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = make_model(encoder, pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = max(1, int(ytr.sum()))
    neg = max(1, len(ytr) - pos)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=dev))
    for epoch in range(epochs):
        log.info("кроп: семя %s, эпоха %d/%d, окон %d, устройство %s", seed, epoch + 1, epochs, len(ytr), dev)
        model.train()
        order = rng.permutation(len(ytr))
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            if len(b) < 2:
                continue  # BatchNorm в режиме обучения не принимает пачку из одного окна
            x = _tensor(Xtr[b], dev)
            k = int(rng.integers(4))
            x = torch.rot90(x, k, dims=(2, 3))
            if rng.random() < 0.5:
                x = x.flip(-1)
            y = torch.from_numpy(ytr[b].astype(np.float32)).to(dev)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()
    return model.eval()


def predict(model, X: np.ndarray, batch: int = 64) -> np.ndarray:
    """Вероятность кропа: среднее по четырём отражениям, как `expds.crop_model` на инференсе."""
    import torch

    dev = next(model.parameters()).device
    out = np.zeros(len(X), np.float64)
    with torch.inference_mode():
        for s in range(0, len(X), batch):
            x = _tensor(X[s:s + batch], dev)
            p = sum(torch.sigmoid(model(v)) for v in (x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2))) / 4
            out[s:s + batch] = p.cpu().numpy()
    return out


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, epochs: int = 8, batch: int = 64,
                lr: float = 3e-4, encoder: str = "resnet18", pretrained: bool = True, seed: int = 0,
                device: str | None = None) -> np.ndarray:
    """Учит классификатор кропа на (Xtr, ytr) и возвращает вероятности для Xte."""
    model = fit(Xtr, ytr, epochs=epochs, batch=batch, lr=lr, encoder=encoder, pretrained=pretrained,
                seed=seed, device=device)
    return predict(model, Xte, batch)


def save_model(path: str, X: np.ndarray, y: np.ndarray, markup: str, res: float, tile: int,
               encoder: str = "resnet18", pretrained: bool = True, li_tags: tuple[str, ...] = (), **kw) -> dict:
    """Веса на всех кропах для `[verify] weights`. Метка — «есть объект», без эталона на выходе.

    В файл пишутся тип данных, шаг, окно и каналы рельефа: сдача режет кроп с ними же.
    """
    import torch

    if encoder.startswith(HF_PREFIX):
        raise SystemExit("--save-model для hf:-энкодера пока не поддержан: expds.crop_model его не читает (только замер OOF)")
    model = fit(X, y, encoder=encoder, pretrained=pretrained, **kw)
    blob = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "encoder": encoder, "pretrained": False, "markup": markup, "res": res, "tile": tile,
            "li_tags": list(li_tags),
            "positives": int(np.sum(y)), "crops": int(len(y))}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return {k: v for k, v in blob.items() if k != "state_dict"}


def oof_verify(X: np.ndarray, y: np.ndarray, regions: list[str], folds: int, **kw) -> np.ndarray:
    """Вероятности вне обучения по семьям участков; кандидат фолда без обеих меток в обучении — NaN."""
    from expds.metric import family_folds
    from expds.naming import site_family

    p = np.full(len(y), np.nan)
    split = family_folds(sorted(set(regions)), site_family, folds)
    fold_of = {r: k for k, regs in enumerate(split) for r in regs}
    fold = np.array([fold_of[r] for r in regions])
    enc = str(kw.get("encoder", ""))
    F = hf_embed(X, enc[len(HF_PREFIX):], kw.get("device"), int(kw.get("batch", 32))) if enc.startswith(HF_PREFIX) else None
    for k in sorted(set(fold)):
        te = fold == k
        tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        if F is not None:
            p[te] = lin_fit_predict(F[tr], y[tr], F[te])
            continue
        p[te] = fit_predict(X[tr], y[tr], X[te], seed=int(k), **kw)
    return p


def labels_of(rr, stack: dict, gt: dict, metric_path: str | None, picked: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Метки «есть объект» и «первый по метрике» для picked, по всему стеку.

    Сопоставление один к одному идёт по всем кругам класса. Если считать его только
    внутри проверяемой группы, круг спутника усилится, хотя эталон уже забрал лидар.
    Обе метки считаются за один проход геометрии: стек — около полутора миллионов кругов.
    """
    n = len(stack.get("features", []))
    y_any = np.zeros(n, np.int8)
    y_first = np.zeros(n, np.int8)
    rows, vals_any, vals_first = rr.collect_pair(stack, gt, metric_path)
    for row, any_hit, first_hit in zip(rows, vals_any, vals_first):
        y_any[int(row["idx"])] = int(bool(any_hit))
        y_first[int(row["idx"])] = int(bool(first_hit))
    ix = np.asarray(picked, np.int64)
    return y_any[ix], y_first[ix]


def demote_duplicates(p: np.ndarray, y_any: np.ndarray, y_first: np.ndarray) -> np.ndarray:
    """Второй круг на эталоне, который метрика уже отдала более уверенному, не усиливать.

    Метка «есть объект» учит классификатор картинке. Порог группы такие дубли не снимает:
    у них та же картинка и после смешения та же высокая уверенность, а сопоставление
    один к одному считает их ложными. Поэтому вероятность дубля обнуляется.
    """
    out = np.array(p, np.float64, copy=True)
    out[(np.asarray(y_any) == 1) & (np.asarray(y_first) == 0)] = 0.0
    return out


def group_median(p: np.ndarray, keys: list[tuple]) -> dict[tuple, float]:
    """Медиана вероятности кропа по группе и классу. Нет ни одного кропа — ключа нет.

    Подставлять 0,5 значит переписать уверенность всей группы, хотя картинку никто не видел.
    """
    med: dict[tuple, float] = {}
    for key in set(keys):
        vals = [float(p[k]) for k in range(len(p)) if keys[k] == key and np.isfinite(p[k])]
        if vals:
            med[key] = float(np.median(vals))
    return med


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
        metric_path: str | None = None, demote_with_gt: bool = False, save_path: str | None = None,
        **kw) -> tuple[dict, dict]:
    rr = load_rerank()
    picked = select(stack, classes, markup, groups, min_conf)
    if not picked:
        raise SystemExit("нет кандидатов под фильтр: проверьте --classes, --groups, --min-conf")
    y, y_first = labels_of(rr, stack, gt, metric_path, picked)
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
    med = group_median(p, keys)
    # Обнуление дубля смотрит в эталон проверяемого участка. На площадке эталона нет,
    # а на табло это утечка: ложные круги снимаются знанием ответа.
    scored = demote_duplicates(p, y, y_first) if demote_with_gt else p
    report: dict = {"candidates": len(picked), "with_crop": int(ok.sum()), "verified": int(np.isfinite(p).sum()),
                    "hits": int(y_first.sum()), "blend": mode, "demote_with_gt": bool(demote_with_gt),
                    "groups": {}}
    if save_path and idx.size:
        report["saved"] = save_model(save_path, X[idx], y[idx], markup, res, tile, li_tags=li_tags, **kw)
    m = np.isfinite(scored)
    if m.any() and 0 < y_first[m].sum() < m.sum():
        report["auc_stack"] = round(rr.auc(conf[m], y_first[m]), 4)
        report["auc_crop"] = round(rr.auc(np.nan_to_num(p[m]), y_first[m]), 4)
        report["auc_blend"] = round(rr.auc(blend(conf[m], scored[m], "geo"), y_first[m]), 4)
    for f in feats:
        f["properties"]["confidence_base"] = f["properties"].get("confidence")
    for k, i in enumerate(picked):
        pr = feats[i]["properties"]
        if np.isfinite(scored[k]):
            pk = float(scored[k])
        elif keys[k] in med:
            pk = med[keys[k]]
        else:
            continue
        pr["verify_p"] = round(pk, 6)
        pr["confidence"] = round(float(blend(np.array([conf[k]]), np.array([pk]), mode)[0]), 6)
    for key in sorted(med, key=str):
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
    ap.add_argument("--encoder", default="resnet18",
                    help="энкодер smp (дообучается) или hf:<id или папка>[@ревизия] — замороженный DINOv2(-reg) + линейная проба")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--blend", choices=["geo", "p", "none"], default="geo")
    ap.add_argument("--demote-with-gt", action="store_true",
                    help="диагностика: обнулить второй круг на эталоне по самому эталону. Утечка, не для табло приёмки")
    ap.add_argument("--save-model", default=None,
                    help="веса на всех кропах для [verify] weights (models/sp_crop.pt)")
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
                      demote_with_gt=args.demote_with_gt, save_path=args.save_model,
                      epochs=args.epochs, batch=args.batch, lr=args.lr, encoder=args.encoder,
                      pretrained=not args.no_pretrained)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
