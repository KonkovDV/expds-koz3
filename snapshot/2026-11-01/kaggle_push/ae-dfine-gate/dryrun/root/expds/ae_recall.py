"""Точечные правки полноты аэрофото. Все выключены, пока в конфиге нет [ae_recall].

Две отдельные причины, которые не чинит понижение порога маски фортификаций:

* Целые копии курганов с мозаики и грубой сетки пишутся в группу ``li_ae``,
  а у неё ``kurgany_tselye = 1.000001`` — копия не проходит никогда. На снимках
  1940-х распаханный позже курган часто ещё целый, и метрика требует тот класс.
  ``mosaic_intact`` переносит только эти целые копии в группу ``ae``
  (порог 0.802255). Повреждённые остаются в ``li_ae``.

* D-FINE Ae/Or режет дубли внутри одного снимка. На участке несколько кадров
  (101 — два, 103 — три), и один курган становится несколькими объектами:
  один TP и остальные FP. ``cross_tif_dedup_m`` оставляет самый уверенный
  внутри (участок, класс, группа dfine_ae/dfine_or). 0 — выключено.

Слияние видов D-FINE (WBF делит уверенность одиночного вида на число видов,
OOF ядра оставляет максимум через NMS) включается отдельно: ``[dfine_ae_tile]``.
"""
from __future__ import annotations

import math

from .post import _ground_scale

DFINE_GROUPS = ("dfine_ae", "dfine_or", "ae_dfine", "or_dfine")


def _block(cfg: dict | None) -> dict:
    raw = (cfg or {}).get("ae_recall")
    return raw if isinstance(raw, dict) else {}


def mosaic_intact_on(cfg: dict | None) -> bool:
    return bool(_block(cfg).get("mosaic_intact", False))


def dedup_m(cfg: dict | None) -> float:
    try:
        return float(_block(cfg).get("cross_tif_dedup_m") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def cross_tif_dedup(feats: list[dict], dist_m: float, groups: tuple[str, ...] = DFINE_GROUPS) -> list[dict]:
    """Убирает менее уверенный объект той же группы, класса и участка ближе dist_m метров на земле.

    dist_m <= 0 возвращает тот же список. Порядок оставшихся не меняется.
    """
    if not dist_m or dist_m <= 0 or not feats:
        return feats
    wanted = set(groups)
    kept: dict[tuple, list[tuple[float, float]]] = {}
    drop: set[int] = set()
    order = sorted(range(len(feats)), key=lambda i: -float(feats[i].get("confidence") or 0.0))
    for i in order:
        f = feats[i]
        group = f.get("threshold_group")
        if group not in wanted or f.get("geom") is None:
            continue
        c = f["geom"].centroid
        s = _ground_scale(c.y)
        xy = (c.x * s, c.y * s)
        key = (f.get("region_name"), f.get("class_name"), group)
        near = kept.get(key)
        if near and any(math.hypot(xy[0] - p[0], xy[1] - p[1]) <= dist_m for p in near):
            drop.add(i)
            continue
        kept.setdefault(key, []).append(xy)
    if not drop:
        return feats
    return [f for i, f in enumerate(feats) if i not in drop]
