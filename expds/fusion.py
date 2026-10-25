"""Слияние слоёв: уверенные курганы лидара — ещё и объекты слоя аэрофото.

Эталон размечен по слоям: курган, видимый и на лидаре, и на аэрофото, — два
эталонных объекта, а метрика сопоставляет по классу без учёта слоя. На
обучающих участках с лидаром и аэрофото половина курганов аэрофото совпадает
с лидарными. Класс на аэрофото — по состоянию на дату съёмки: на немецких
снимках 1940-х распаханные позже курганы ещё целые, поэтому копия «целый»
получает p_повр + p_целый.

OOF лидара на 15 участках (эталон всех слоёв): F1 повреждённых 0.450 → 0.476,
целых 0.449 → 0.530. Если в тесте дубли между слоями убраны — потеря ~0.02
балла этих участков; отключается [fusion] li_to_ae = false.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .crs import WEB_MERCATOR, transformer, usable_raster_crs

log = logging.getLogger(__name__)

POVR, TSEL = "kurgany_povrezhdennye", "kurgany_tselye"

#: Группа порога копии по группе исходного кургана лидара. Так копии подписаны в табло 0,353793:
#: копии мелкой сетки — без своей группы, то есть ae (0,99053 / 0,802255); копии мозаики — li_ae
#: (scripts/scoreboard.py, mosaic_kurgans). Копий грубой сетки в табло не было; li_ae — строгий порог.
COPY_GROUP = {None: "ae", "": "ae", "li_mosaic": "li_ae", "li_coarse": "li_ae"}


@dataclass
class FusionParams:
    li_to_ae: bool = True
    povr_min: float = 0.99       # копия «повреждённый» при p_повр ≥
    tselye_min: float = 0.8      # копия «целый» при p_повр + p_целый ≥
    coverage_side: int = 1024


@lru_cache(maxsize=32)
def coverage(path: str | Path, max_side: int = 1024, fallback_crs: str | None = None):
    """Маска непустых пикселей снимка на обзоре ≤ max_side, её привязка и CRS.

    Безымянная LOCAL_CS с метрами зоны получает fallback_crs. Снимок с рабочей CRS
    возвращает её же строку, что и раньше.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        f = max(1, math.ceil(max(ds.width, ds.height) / max_side))
        n = min(ds.count, 3)
        a = ds.read(list(range(1, n + 1)), out_shape=(n, max(1, ds.height // f), max(1, ds.width // f)),
                    resampling=Resampling.nearest)
        valid = (a != 0).any(axis=0)
        tr = ds.transform @ ds.transform.scale(ds.width / valid.shape[1], ds.height / valid.shape[0])
        declared = ds.crs.to_string() if ds.crs else None
        return valid, tr, usable_raster_crs(declared, ds.bounds, fallback_crs)


def covered(x: np.ndarray, y: np.ndarray, rasters: list, max_side: int = 1024,
            fallback_crs: str | None = None) -> np.ndarray:
    """Какие точки EPSG:3857 лежат на непустых пикселях хотя бы одного снимка.

    Снимок, который не переводится, пропускается. Остальные снимки участка остаются.
    """
    hit = np.zeros(len(x), bool)
    for path in rasters:
        valid, tr, crs = coverage(str(path), max_side, fallback_crs)
        if crs is None:
            continue
        try:
            sx, sy = transformer(WEB_MERCATOR, crs).transform(x, y)
        except Exception:
            log.warning("%s: CRS снимка не переводится, покрытие пропущено", Path(path).name)
            continue
        col, row = ~tr @ (np.asarray(sx), np.asarray(sy))
        r, c = np.floor(row).astype(int), np.floor(col).astype(int)
        ok = (r >= 0) & (r < valid.shape[0]) & (c >= 0) & (c < valid.shape[1])
        hit[ok] |= valid[r[ok], c[ok]]
    return hit


def li_to_ae(feats: list[dict], rasters: list, p: FusionParams | None = None,
             fallback_crs: str | None = None) -> list[dict]:
    """Копии курганов нейросети лидара (обе вероятности кандидата) для слоя ae в покрытии снимков."""
    p = p or FusionParams()
    groups: dict[tuple, dict] = {}
    for f in feats:
        # li_coarse и li_mosaic — те же вероятности нейросети, только порог выдачи другой. Запасной
        # детектор (li_cpu) и уже скопированные объекты не копируются.
        if (f.get("markup_type") != "li" or f.get("threshold_group") not in COPY_GROUP
                or f["class_name"] not in (POVR, TSEL)):
            continue
        c = f["geom"].centroid
        groups.setdefault((f.get("source"), round(c.x, 1), round(c.y, 1)), {})[f["class_name"]] = f
    if not groups or not rasters:
        return []
    keys = list(groups)
    hit = covered(np.array([k[1] for k in keys]), np.array([k[2] for k in keys]), rasters, p.coverage_side,
                  fallback_crs)
    out = []
    for key, ok in zip(keys, hit):
        if not ok:
            continue
        g = groups[key]
        pp = g[POVR]["confidence"] if POVR in g else 0.0
        pt = g[TSEL]["confidence"] if TSEL in g else 0.0
        base = next(iter(g.values()))
        group = COPY_GROUP[base.get("threshold_group")]
        if pp >= p.povr_min:
            out.append(dict(base, class_name=POVR, confidence=float(pp), markup_type="ae", threshold_group=group))
        if pp + pt >= p.tselye_min:
            out.append(dict(base, class_name=TSEL, confidence=float(min(pp + pt, 1.0)), markup_type="ae",
                            threshold_group=group))
    return out


@dataclass
class CopyRule:
    """Копия объекта в другой слой там, где есть снимок этого слоя. По умолчанию правил нет."""

    src: str = "li"
    dst: str = "sp_or"
    group: str = "li_sp"
    min_conf: dict | None = None
    src_groups: tuple = ("", "li", "li_coarse", "li_mosaic", "li_mask")

    @classmethod
    def from_cfg(cls, raw: dict) -> CopyRule:
        src, dst = raw.get("src", "li"), raw.get("dst", "sp_or")
        return cls(src=src, dst=dst, group=raw.get("group") or f"{src}_{dst}",
                   min_conf=dict(raw.get("min") or {}),
                   src_groups=tuple(raw.get("src_groups") or ("", "li", "li_coarse", "li_mosaic", "li_mask")))


def copy_layers(feats: list[dict], rasters_by_markup: dict, rules: list, max_side: int = 1024,
                fallback_crs: str | None = None) -> list[dict]:
    """Копии по [[fusion.copy]]. Пустой список правил ничего не добавляет."""
    out: list[dict] = []
    for rule in rules or ():
        if isinstance(rule, dict):
            rule = CopyRule.from_cfg(rule)
        rasters = rasters_by_markup.get(rule.dst) or []
        if not rasters or not rule.min_conf:
            continue
        cand = [f for f in feats
                if f.get("markup_type") == rule.src and f["class_name"] in rule.min_conf
                and (f.get("threshold_group") or "") in rule.src_groups
                and float(f.get("confidence", 1.0)) >= float(rule.min_conf[f["class_name"]])]
        if not cand:
            continue
        pts = [f["geom"].representative_point() for f in cand]
        hit = covered(np.array([p.x for p in pts]), np.array([p.y for p in pts]), rasters, max_side,
                      fallback_crs)
        out += [dict(f, markup_type=rule.dst, threshold_group=rule.group) for f, ok in zip(cand, hit) if ok]
    return out
