"""Крупные объекты (>5k м²): пороги по площади, защита при склейке, ядра, грубые кандидаты.

Всё за одним флагом [large_objects] enabled (по умолчанию false). При выключенном флаге
код постобработки и пайплайна не вызывает эти функции — поведение байт-в-байт как v4.

Не повторяет отвергнутый bot-2 F1 (отдельные группы порогов под sweep — переобучение на
малых группах): здесь те же калиброванные τ родителя, только ослабление по площади
кандидата. F2 (ядра глобально) — только для крупных городищ.

Опора: cbd4804 (v4 + gor_dense/hillfort). Не дублирует gor_dense (отдельный флаг).
"""
from __future__ import annotations

import logging
from dataclasses import fields

from . import post

log = logging.getLogger(__name__)

#: Классы, для которых ослабляем порог и защищаем при склейке.
DEFAULT_CLASSES = ("gorodishcha", "fortifikatsii")


def enabled(cfg: dict | None) -> bool:
    return bool(((cfg or {}).get("large_objects") or {}).get("enabled"))


def spec(cfg: dict | None) -> dict:
    """Параметры [large_objects]; пустой dict, если секции нет."""
    return dict((cfg or {}).get("large_objects") or {})


def area_m2(feat: dict) -> float:
    """Площадь объекта в м². Учитывает EPSG:3857 через post._ground_scale; иначе geom.area."""
    geom = feat.get("geom")
    if geom is None or geom.is_empty:
        return 0.0
    # hillfort уже кладёт площадь площадки в метаданные (круг-выход маленький).
    stored = feat.get("hillfort_area_m2")
    if stored is not None:
        try:
            return float(stored)
        except (TypeError, ValueError):
            pass
    minx, miny, maxx, maxy = geom.bounds
    # Эвристика как в post.split_large: 3857 имеет огромные координаты.
    if abs(maxx) > 1_000_000 or abs(maxy) > 1_000_000:
        scale = post._ground_scale((miny + maxy) / 2.0)
        return float(geom.area) * scale * scale
    return float(geom.area)


def is_large(feat: dict, lo: dict) -> bool:
    classes = set(lo.get("classes") or DEFAULT_CLASSES)
    if feat.get("class_name") not in classes:
        return False
    return area_m2(feat) >= float(lo.get("min_area_m2", 5000.0))


def area_tau(base_tau, lo: dict):
    """Обёртка threshold_fn: для крупных ослабляет τ (линейный разгон от min_area до 2×min_area).

    Не вводит новых групп под sweep — база та же калиброванная τ.
    """
    min_area = float(lo.get("min_area_m2", 5000.0))
    relax = float(lo.get("tau_relax", 0.15))
    floor = float(lo.get("tau_floor", 0.05))
    classes = set(lo.get("classes") or DEFAULT_CLASSES)

    def tau(f: dict) -> float:
        t = float(base_tau(f))
        if f.get("class_name") not in classes:
            return t
        a = area_m2(f)
        if a < min_area or relax <= 0.0:
            return t
        frac = min(1.0, (a - min_area) / max(min_area, 1.0))
        return max(floor, t - relax * frac)

    return tau


def apply_thresholds(feats: list[dict], thresholds: dict, default: float, by_markup: dict | None,
                     lo: dict) -> list[dict]:
    """Как post.apply_thresholds, но с area_tau при включённом флаге."""
    base = post.threshold_fn(thresholds, default, by_markup)
    tau = area_tau(base, lo)
    return [f for f in feats if f.get("confidence", 1.0) >= tau(f)]


def merge_across_sources_protect(items: list[dict], lo: dict, source_key: str = "source") -> list[dict]:
    """Склейка как post.merge_across_sources, но крупный не сливается с мелким.

    Крупные склеиваются только с крупными, мелкие — с мелкими. Пересечение крупный↔мелкий
    не создаёт ребро: оба остаются (защита TP на >5k от поглощения мелким осколком).
    """
    if len(items) < 2:
        return items
    large_idx = {i for i, it in enumerate(items) if is_large(it, lo)}
    if not large_idx or len(large_idx) == len(items):
        # Все мелкие или все крупные — обычная склейка.
        return post.merge_across_sources(items, source_key=source_key)

    from collections import defaultdict
    from shapely import STRtree
    from shapely.ops import unary_union

    geoms = [i["geom"] for i in items]
    tree = STRtree(geoms)
    parent = list(range(len(items)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, g in enumerate(geoms):
        for j in tree.query(g):
            j = int(j)
            if j <= i or items[j].get(source_key) == items[i].get(source_key):
                continue
            # Не связываем крупный с мелким.
            if (i in large_idx) != (j in large_idx):
                continue
            if g.intersection(geoms[j]).area > 0:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(items[members[0]])
            continue
        best = max(members, key=lambda k: items[k].get("confidence", 1.0))
        geom = unary_union([geoms[k] for k in members])
        for part in post.polygons_of(geom):
            out.append(dict(items[best], geom=part))
    return out


def merge_within_markup(feats: list[dict], isolate, lo: dict) -> list[dict]:
    """Как post.merge_within_markup, но с protect-склейкой."""
    from collections import defaultdict

    iso = set(isolate or ())
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for raw in feats:
        feat = post._repaired(raw)
        if feat is not None:
            tg = feat.get("threshold_group")
            groups[(feat.get("region_name"), feat.get("markup_type"), feat["class_name"],
                    tg if tg in iso else None)].append(feat)
    out = []
    for key, items in groups.items():
        out.extend(items if key[1] == "etgr" else merge_across_sources_protect(items, lo))
    return out


def shape_cores_large(feats: list[dict], lo: dict) -> list[dict]:
    """Ядра только у крупных городищ (не глобальный F2 bot-2)."""
    if not lo.get("shape_cores", True):
        return feats
    min_area = float(lo.get("min_area_m2", 5000.0))
    core_rule = lo.get("core") or {"frac": 0.35, "min_r_m": 2.0}
    if not isinstance(core_rule, dict):
        core_rule = {"frac": 0.35, "min_r_m": 2.0}
    # shape_cores ждёт {класс: rule}
    spec_cores = {"gorodishcha": dict(core_rule)}
    large, rest = [], []
    for f in feats:
        if f.get("class_name") == "gorodishcha" and area_m2(f) >= min_area:
            large.append(f)
        else:
            rest.append(f)
    if not large:
        return feats
    return rest + post.shape_cores(large, spec_cores)


def fort_to_gor(feats: list[dict], lo: dict) -> list[dict]:
    """Копии крупных fortifikatsii → gorodishcha (группа gor_from_fort) до порогов.

    Закрывает часть no_cand, где есть крепость/окоп, но нет городища (ae без gor-модели,
    спутник без gor). Масштаб conf < 1, чтобы не забить выдачу.
    """
    if not lo.get("fort_to_gor", True):
        return feats
    min_area = float(lo.get("min_area_m2", 5000.0))
    scale = float(lo.get("fort_to_gor_scale", 0.9))
    group = str(lo.get("fort_to_gor_group", "gor_from_fort"))
    extra = []
    for f in feats:
        if f.get("class_name") != "fortifikatsii":
            continue
        if area_m2(f) < min_area:
            continue
        extra.append(dict(
            f,
            class_name="gorodishcha",
            threshold_group=group,
            confidence=round(float(f.get("confidence", 1.0)) * scale, 6),
            large_objects_seed="fort_to_gor",
        ))
    if extra:
        log.info("large_objects: fort→gor копий %d", len(extra))
    return list(feats) + extra


def coarse_hillfort_params(lo: dict | None = None):
    """Параметры грубого прохода hillfort для площадок ≥5k м² (и до ~20 га)."""
    from .hillfort import HillfortParams

    lo = lo or {}
    overrides = {
        "work_res": float(lo.get("coarse_work_res", 3.0)),
        "relief_radius_m": float(lo.get("coarse_relief_radius_m", 180.0)),
        "relief_radius_l_m": float(lo.get("coarse_relief_radius_l_m", 300.0)),
        "min_area_m2": float(lo.get("min_area_m2", 5000.0)),
        "max_area_m2": float(lo.get("coarse_max_area_m2", 200_000.0)),
        "max_per_dem": int(lo.get("coarse_max_per_dem", 3)),
        "min_enclosure": float(lo.get("coarse_min_enclosure", 0.20)),
        "min_rel_height_m": float(lo.get("coarse_min_rel_height_m", 1.2)),
        "flat_max_deg": float(lo.get("coarse_flat_max_deg", 14.0)),
        "scarp_min_deg": float(lo.get("coarse_scarp_min_deg", 16.0)),
    }
    base = {f.name: getattr(HillfortParams(), f.name) for f in fields(HillfortParams)}
    base.update(overrides)
    return HillfortParams(**base)


def coarse_hillfort_candidates(dem, region: str, source: str, lo: dict | None = None) -> list[dict]:
    """Грубый проход hillfort → группа gor_coarse_dem. Пусто, если ЦМР без CRS."""
    from . import hillfort

    lo = lo or {}
    if not lo.get("coarse_hillfort", True):
        return []
    p = coarse_hillfort_params(lo)
    group = str(lo.get("coarse_group", "gor_coarse_dem"))
    found = hillfort.candidates(dem, region, source, p)
    out = []
    for f in found:
        out.append(dict(f, threshold_group=group, large_objects_seed="coarse_hillfort",
                        source=f"{source}#lo_coarse"))
    if out:
        log.info("%s: large_objects coarse hillfort %d", source, len(out))
    return out
