"""Кандидаты городищ по форме рельефа, без обучения.

Мысовое городище лесной зоны — поднятая площадка 0,1–6 га на слиянии рек: с двух-трёх
сторон крутой естественный склон, с напольной стороны вал и ров. На ЦМР это связная
пологая область выше окрестности, по краю которой уклон резко растёт. Холм с пологими
склонами (морена, бугор) по краю обрыва не имеет, распаханная терраса не поднята над
окрестностью, поэтому оценка — доля обрывистого края и высота площадки.

Выход — маленький круг в полюсе недоступности площадки: метрика засчитывает городище,
если внутренняя точка предсказания лежит в эталоне и круг не меньше чем наполовину
внутри него. Контур площадки не нужен.

Цена: на фолдах одно найденное городище стоит как 18 повреждённых курганов
(`docs/STRATEGY_2026-10-25.md`), выгодно добавлять уже при 7,8 % попаданий. Сдача
включает генератор строкой `[hillfort] enabled = true` (группа порога `gor_phys`).
Принятое табло 26.10 — один проход: в конфиге и в умолчании `relief_radius_l_m = 0`.
Второе окно 300 м на синтетике держит широкий мыс, но табло 27.10 его не приняло
(0,338881 → 0,337312). Повторно не запускать.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import numpy as np

from . import accel

log = logging.getLogger(__name__)


@dataclass
class HillfortParams:
    work_res: float = 1.0            # шаг ЦМР для поиска, м
    relief_radius_m: float = 60.0    # окно «окрестности», м
    relief_radius_l_m: float = 0.0      # второй проход, м; 0 — один проход. 150 замерено и не принято
    min_rel_height_m: float = 1.5    # площадка выше окрестности хотя бы на столько
    flat_max_deg: float = 12.0       # внутри площадки уклон не круче
    scarp_min_deg: float = 18.0      # обрыв по краю — круче
    ring_m: float = 6.0              # ширина пояса края, где ищется обрыв
    min_area_m2: float = 800.0       # 0,08 га
    max_area_m2: float = 60000.0     # 6 га
    min_enclosure: float = 0.25      # доля обрывистого края, ниже — не кандидат
    core_m: float = 3.0              # радиус выходного круга, м
    max_per_dem: int = 5
    dem_res: float = 0.5             # ЦМР из LAS: один рецепт в табло (hillfort_oof.py) и в сдаче
    dem_fill: str = "pyramid"


def params_from(cfg: dict | None) -> HillfortParams:
    names = {f.name for f in fields(HillfortParams)}
    return HillfortParams(**{k: v for k, v in (cfg or {}).items() if k in names})


def build_dem(las, crs: str | None, p: HillfortParams):
    """ЦМР генератора. Табло и сдача зовут одну функцию, иначе табло не доказывает конфиг."""
    from .lidar import build_dem as _build

    return _build(las, res=p.dem_res, fill=p.dem_fill, crs=crs)


def score(enclosure: float, height_m: float) -> float:
    """Уверенность 0…1: доля обрывистого края, ослабленная для низких площадок."""
    return float(enclosure * (1.0 - np.exp(-max(height_m, 0.0) / 3.0)))


def _plateaus(valid, slope, lr, step, p: HillfortParams, tag: int) -> tuple[list[dict], np.ndarray]:
    """Площадки одного окна окрестности: кандидаты и карта меток (для снятия дублей между окнами)."""
    from scipy import ndimage

    plateau = valid & (np.nan_to_num(lr, nan=-1e9) >= p.min_rel_height_m) & (slope <= p.flat_max_deg)
    plateau = ndimage.binary_closing(plateau, iterations=2) & valid
    lab, n = ndimage.label(plateau)
    if n == 0:
        return [], lab
    cell = step * step
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * cell
    ring_px = max(1, round(p.ring_m / step))
    boxes = ndimage.find_objects(lab)
    out = []
    for k in np.nonzero((sizes >= p.min_area_m2) & (sizes <= p.max_area_m2))[0]:
        sl = boxes[k]
        pad = ring_px + 1
        r0, r1 = max(0, sl[0].start - pad), min(lab.shape[0], sl[0].stop + pad)
        c0, c1 = max(0, sl[1].start - pad), min(lab.shape[1], sl[1].stop + pad)
        mask = lab[r0:r1, c0:c1] == k + 1
        ring = ndimage.binary_dilation(mask, iterations=ring_px) & ~mask & valid[r0:r1, c0:c1]
        if not ring.any():
            continue
        enclosure = float(np.mean(slope[r0:r1, c0:c1][ring] >= p.scarp_min_deg))
        if enclosure < p.min_enclosure:
            continue
        height = float(np.nanmedian(lr[r0:r1, c0:c1][mask]))
        dist = ndimage.distance_transform_edt(mask)
        rr, cc = np.unravel_index(int(np.argmax(dist)), dist.shape)
        out.append({"row": r0 + rr, "col": c0 + cc, "area_m2": float(sizes[k]), "enclosure": enclosure,
                    "height_m": height, "inscribed_m": float(dist[rr, cc] * step),
                    "confidence": score(enclosure, height), "_pass": tag, "_label": k + 1})
    return out, lab


def find(z: np.ndarray, step: float, p: HillfortParams) -> list[dict]:
    """Площадки-кандидаты на сетке z (NaN — нет данных): строка, столбец полюса и признаки.

    Окно окрестности 120 м держит мыс до ~1 га, но середина площадки шире окна «не выше
    окрестности»: маска — подкова у края, площадь втрое меньше, уверенность падает с
    размером. Окно 300 м держит площадку 1–5 га, но сливает малый мыс с напольным плато
    через дно рва. Поэтому два прохода. Если полюс кандидата лежит на площадке кандидата
    другого окна (или наоборот), это одна форма рельефа, и остаётся более обрывистый край.
    """
    valid = np.isfinite(z)
    if valid.sum() < 16:
        return []
    mean = np.float32(np.nanmean(z, dtype=np.float64))
    filled = np.where(valid, z, mean).astype(np.float32)
    slope = accel.slope_deg(filled, step)
    zn = np.where(valid, z, np.nan).astype(np.float32)
    radii = [p.relief_radius_m] + ([p.relief_radius_l_m] if p.relief_radius_l_m > p.relief_radius_m else [])
    found: list[dict] = []
    labs: list[np.ndarray] = []
    for tag, radius in enumerate(radii):
        lr = accel.local_relief(zn, max(1, round(radius / step)))
        cands, lab = _plateaus(valid, slope, lr, step, p, tag)
        found += cands
        labs.append(lab)
    # Из двух площадок одной формы остаётся та, у которой обрывистее край: подкова большой
    # площадки и слияние мыса с напольным плато имеют пологий участок края, сама площадка — нет.
    found.sort(key=lambda c: (-c["enclosure"], -c["confidence"]))
    kept: list[dict] = []
    for c in found:
        same = any(k["_pass"] != c["_pass"]
                   and (labs[k["_pass"]][c["row"], c["col"]] == k["_label"]
                        or labs[c["_pass"]][k["row"], k["col"]] == c["_label"]) for k in kept)
        if not same:
            kept.append(c)
    kept.sort(key=lambda c: -c["confidence"])
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in kept[: p.max_per_dem]]


def candidates(d, region: str, source: str, p: HillfortParams | None = None) -> list[dict]:
    """ЦМР (lidar.Dem) → круги класса gorodishcha в EPSG:3857, группа gor_phys."""
    from .candidates import to_work_grid
    from .raster import circle_3857

    p = p or HillfortParams()
    if d.crs is None:
        return []
    z, step = to_work_grid(d.dem, d.res, p.work_res)
    found = find(z, step, p)
    out = []
    for c in found:
        x = d.left + (c["col"] + 0.5) * step
        y = d.top - (c["row"] + 0.5) * step
        r = min(p.core_m, max(1.0, 0.5 * c["inscribed_m"]))
        out.append({"geom": circle_3857(x, y, r, d.crs), "class_name": "gorodishcha",
                    "confidence": round(c["confidence"], 6), "region_name": region, "markup_type": "li",
                    "source": source, "original_crs": d.crs, "threshold_group": "gor_phys",
                    "hillfort_area_m2": round(c["area_m2"], 1), "hillfort_enclosure": round(c["enclosure"], 3),
                    "hillfort_height_m": round(c["height_m"], 2)})
    log.info("%s: кандидатов городищ по рельефу %d", source, len(out))
    return out
