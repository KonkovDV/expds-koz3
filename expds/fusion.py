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

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .crs import WEB_MERCATOR, transformer

POVR, TSEL = "kurgany_povrezhdennye", "kurgany_tselye"


@dataclass
class FusionParams:
    li_to_ae: bool = True
    povr_min: float = 0.99       # копия «повреждённый» при p_повр ≥
    tselye_min: float = 0.8      # копия «целый» при p_повр + p_целый ≥
    coverage_side: int = 1024


@lru_cache(maxsize=32)
def coverage(path: str | Path, max_side: int = 1024):
    """Маска непустых пикселей снимка на обзоре ≤ max_side, её привязка и CRS."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        f = max(1, math.ceil(max(ds.width, ds.height) / max_side))
        n = min(ds.count, 3)
        a = ds.read(list(range(1, n + 1)), out_shape=(n, max(1, ds.height // f), max(1, ds.width // f)),
                    resampling=Resampling.nearest)
        valid = (a != 0).any(axis=0)
        tr = ds.transform @ ds.transform.scale(ds.width / valid.shape[1], ds.height / valid.shape[0])
        return valid, tr, ds.crs.to_string() if ds.crs else None


def covered(x: np.ndarray, y: np.ndarray, rasters: list, max_side: int = 1024) -> np.ndarray:
    """Какие точки EPSG:3857 лежат на непустых пикселях хотя бы одного снимка."""
    hit = np.zeros(len(x), bool)
    for path in rasters:
        valid, tr, crs = coverage(str(path), max_side)
        if crs is None:
            continue
        sx, sy = transformer(WEB_MERCATOR, crs).transform(x, y)
        col, row = ~tr @ (np.asarray(sx), np.asarray(sy))
        r, c = np.floor(row).astype(int), np.floor(col).astype(int)
        ok = (r >= 0) & (r < valid.shape[0]) & (c >= 0) & (c < valid.shape[1])
        hit[ok] |= valid[r[ok], c[ok]]
    return hit


def li_to_ae(feats: list[dict], rasters: list, p: FusionParams | None = None) -> list[dict]:
    """Копии курганов нейросети лидара (обе вероятности кандидата) для слоя ae в покрытии снимков."""
    p = p or FusionParams()
    groups: dict[tuple, dict] = {}
    for f in feats:
        # li_coarse — те же вероятности нейросети, только порог выдачи другой. Запасной
        # детектор (li_cpu) и уже скопированные объекты не копируются.
        if (f.get("markup_type") != "li" or f.get("threshold_group") not in (None, "", "li_coarse")
                or f["class_name"] not in (POVR, TSEL)):
            continue
        c = f["geom"].centroid
        groups.setdefault((f.get("source"), round(c.x, 1), round(c.y, 1)), {})[f["class_name"]] = f
    if not groups or not rasters:
        return []
    keys = list(groups)
    hit = covered(np.array([k[1] for k in keys]), np.array([k[2] for k in keys]), rasters, p.coverage_side)
    out = []
    for key, ok in zip(keys, hit):
        if not ok:
            continue
        g = groups[key]
        pp = g[POVR]["confidence"] if POVR in g else 0.0
        pt = g[TSEL]["confidence"] if TSEL in g else 0.0
        base = next(iter(g.values()))
        if pp >= p.povr_min:
            out.append(dict(base, class_name=POVR, confidence=float(pp), markup_type="ae", threshold_group="li_ae"))
        if pp + pt >= p.tselye_min:
            out.append(dict(base, class_name=TSEL, confidence=float(min(pp + pt, 1.0)), markup_type="ae",
                            threshold_group="li_ae"))
    return out
