"""Проверка кропа на инференсе. По умолчанию выключена: без файла весов сдача не меняется.

Проверка на трейне (`scripts/crop_verify.py`) учит модель вне обучения и ещё обнуляет
второй круг по эталону. На площадке эталона нет, поэтому здесь только смешение
уверенности стека с вероятностью картинки. Веса — отдельный файл, в конфиг их
ставить после своего табло.

    [verify]
    enabled = true
    weights = "models/sp_crop.pt"
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def blend_geo(conf: float, prob: float) -> float:
    """Среднее геометрическое уверенности стека и вероятности кропа."""
    return float(np.sqrt(max(0.0, conf) * max(0.0, min(1.0, prob))))


def maybe_verify(feats: list[dict], regions: dict, cfg: dict) -> list[dict] | None:
    """Новый список, если проверка реально прошла. None — оставить объекты как есть."""
    opts = cfg.get("verify") or {}
    if not opts.get("enabled"):
        return None
    weights = opts.get("weights")
    path = Path(weights) if weights else None
    if path is None or not path.is_file():
        log.warning("проверка кропа включена, файла весов нет (%s) — уверенности не меняю", weights)
        return None
    groups = set(opts.get("groups") or ())
    classes = set(opts.get("classes") or ())
    markup = str(opts.get("markup") or "sp_or")
    min_conf = float(opts.get("min_conf") or 0.0)
    chosen = []
    for i, feat in enumerate(feats):
        if feat.get("markup_type") != markup:
            continue
        if classes and feat.get("class_name") not in classes:
            continue
        if groups and (feat.get("threshold_group") or feat.get("markup_type")) not in groups:
            continue
        if float(feat.get("confidence") or 0.0) < min_conf:
            continue
        if feat.get("region_name") not in regions:
            continue
        chosen.append(i)
    if not chosen:
        log.info("проверка кропа: кандидатов нет")
        return None
    probs = _predict(feats, chosen, regions, path, markup, opts.get("tile"), opts.get("res"))
    if probs is None:
        return None
    probs = fill_group_median(feats, chosen, probs)
    out = [dict(f) for f in feats]
    for i, prob in zip(chosen, probs):
        if prob is None or not np.isfinite(prob):
            continue
        out[i]["confidence_base"] = out[i].get("confidence")
        out[i]["verify_p"] = round(float(prob), 6)
        out[i]["confidence"] = round(blend_geo(float(out[i].get("confidence") or 0.0), float(prob)), 6)
    log.info("проверка кропа: смешано %d кругов", sum(p is not None and np.isfinite(p) for p in probs))
    return out


def fill_group_median(feats: list[dict], chosen: list[int], probs: list) -> list:
    """Выбранный круг без окна получает медиану своей группы и класса, как в `crop_verify.py`.

    Табло мерило именно такое правило; без него сдача и табло расходятся. Группа, где ни
    одно окно не прочитано, остаётся без вероятности — уверенности её кругов не меняются.
    """
    keys = [(feats[i].get("threshold_group") or feats[i].get("markup_type"), feats[i].get("class_name"))
            for i in chosen]
    vals: dict[tuple, list[float]] = {}
    for key, p in zip(keys, probs):
        if p is not None and np.isfinite(p):
            vals.setdefault(key, []).append(float(p))
    med = {k: float(np.median(v)) for k, v in vals.items()}
    return [p if (p is not None and np.isfinite(p)) else med.get(k) for k, p in zip(keys, probs)]


def _predict(feats, indexes, regions, weights: Path, markup: str, tile, res, batch: int = 64):
    """Вероятности кропа. Ошибка чтения или торча — None, сдача остаётся прежней.

    Шаг, окно и каналы рельефа по умолчанию — из файла весов: те же, на которых учился
    классификатор. Из нескольких снимков участка берётся тот, где окно заполнено сильнее,
    как в `scripts/crop_verify.extract_crops`. Окна одного участка идут в сеть пачками.
    """
    try:
        import torch

        from .tiles import fix_channels, iter_sources
    except Exception:
        log.exception("проверка кропа не загрузилась")
        return None
    try:
        blob = torch.load(weights, map_location="cpu", weights_only=False)
        model = _network(str(blob.get("encoder") or "resnet18"), False)
        model.load_state_dict(blob["state_dict"])
    except Exception:
        log.exception("веса проверки кропа не прочитаны: %s", weights)
        return None
    tile = int(tile or blob.get("tile") or 96)
    res = float(res or blob.get("res") or 0.5)
    li_tags = list(blob.get("li_tags") or ())
    if markup == "li" and not li_tags:
        from .lidar import RELIEF_TAGS

        li_tags = list(RELIEF_TAGS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    probs = [None] * len(indexes)
    by_region: dict[str, list[int]] = {}
    for n, i in enumerate(indexes):
        by_region.setdefault(str(feats[i].get("region_name")), []).append(n)
    from .crs import WEB_MERCATOR, to_crs

    for name, todo in by_region.items():
        reg = regions.get(name)
        if reg is None:
            continue
        best: dict[int, tuple[int, np.ndarray]] = {}
        try:
            for _src, crs, bnd, read in iter_sources(reg, markup, li_tags, "las", res, tile):
                for n in todo:
                    geom = feats[indexes[n]].get("geom")
                    if geom is None:
                        continue
                    pt = geom if getattr(geom, "geom_type", "") == "Point" else geom.centroid
                    q = to_crs(pt, WEB_MERCATOR, crs)
                    if not bnd.contains(q):
                        continue
                    img, _tr = read(q.x, q.y)
                    img = np.asarray(img)
                    filled = int(np.count_nonzero(np.any(img, axis=0) if img.ndim == 3 else img))
                    if filled > best.get(n, (0, None))[0]:
                        best[n] = (filled, fix_channels(img, 3))
        except Exception:
            log.exception("%s: окна проверки кропа не прочитаны", name)
        order = sorted(best)
        for s in range(0, len(order), batch):
            part = order[s:s + batch]
            for n, p in zip(part, _probabilities(model, np.stack([best[n][1] for n in part]), device)):
                probs[n] = p
    return probs


def _network(encoder: str, pretrained: bool):
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


def _probabilities(model, images: np.ndarray, device) -> list[float]:
    """Вероятности пачки окон N×3×H×W: среднее по четырём отражениям, как в `crop_verify.predict`."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(images)).to(device).float()
    x = (x / 255.0 - 0.5) / 0.25
    with torch.inference_mode():
        views = (x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2))
        return [float(v) for v in (sum(torch.sigmoid(model(v)) for v in views) / 4).cpu()]


def _probability(model, image: np.ndarray, device) -> float:
    return _probabilities(model, image[None], device)[0]
