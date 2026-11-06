"""Крупные объекты (>5k м²): пороги по площади, защита при склейке, ядра, грубые кандидаты.

Мастер-флаг [large_objects] enabled (по умолчанию false) + независимые sub-flags
(area_tau / protect_merge / shape_cores / fort_to_gor / coarse_hillfort /
dfine_ae_gor_area_tau), тоже false.
При enabled=false или всех sub-flags=false — поведение байт-в-байт как v4 (cbd4804).
Ablation: enabled=true + один sub-flag.

Fix C (dfine_ae_gor_area_tau): абсолютный τ для крупных (box_area_m2≥min) gorodishcha
только группы dfine_ae/ae_dfine и markup ae — li не трогаем. Нужен stash box_area_m2
из D-FINE (круги r=4 иначе ~50 м²).

Не повторяет отвергнутый F1 (отдельные группы порогов под sweep — переобучение на
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


def flag(cfg: dict | None, name: str) -> bool:
    """Мастер enabled И sub-flag name (по умолчанию false)."""
    if not enabled(cfg):
        return False
    return bool(((cfg or {}).get("large_objects") or {}).get(name))


def spec(cfg: dict | None) -> dict:
    """Параметры [large_objects]; пустой dict, если секции нет."""
    return dict((cfg or {}).get("large_objects") or {})


def area_m2(feat: dict) -> float:
    """Площадь объекта в м². Учитывает EPSG:3857 через post._ground_scale; иначе geom.area.

    Приоритет метаданных: box_area_m2 (D-FINE бокс до круга) → hillfort_area_m2 → geom.
    """
    for key in ("box_area_m2", "hillfort_area_m2"):
        stored = feat.get(key)
        if stored is not None:
            try:
                return float(stored)
            except (TypeError, ValueError):
                pass
    geom = feat.get("geom")
    if geom is None or geom.is_empty:
        return 0.0
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


#: Группы порога D-FINE Ae (OOF иногда пишет ae_dfine).
DFINE_AE_GROUPS = frozenset({"dfine_ae", "ae_dfine"})


def dfine_ae_gor_area_tau(base_tau, lo: dict):
    """Fix C: для крупных gorodishcha группы dfine_ae на ae — абсолютный τ_large.

    Иначе база (обычно 1.01 = OFF). li / прочие группы / мелкие — без изменений.
    Площадь: box_area_m2 (бокс D-FINE), не круг r=4.
    """
    min_area = float(lo.get("min_area_m2", 5000.0))
    tau_large = float(lo.get("dfine_ae_gor_tau_large", 0.20))
    classes = set(lo.get("dfine_ae_gor_classes") or ("gorodishcha",))
    markups = set(lo.get("dfine_ae_gor_markups") or ("ae",))
    groups = set(lo.get("dfine_ae_gor_groups") or DFINE_AE_GROUPS)

    def tau(f: dict) -> float:
        t = float(base_tau(f))
        if f.get("class_name") not in classes:
            return t
        if f.get("markup_type") not in markups:
            return t
        tg = f.get("threshold_group")
        if tg not in groups:
            return t
        if area_m2(f) < min_area:
            return t
        return min(t, tau_large)

    return tau


def threshold_for(base_tau: float, area: float, class_name: str, group: str, markup: str,
                  lo: dict, *, area_on: bool, fixc_on: bool) -> float:
    """Порог одного объекта при уже подобранной базовой τ группы.

    area_on — ослабление area_tau. fixc_on — абсолютный τ Fix C. База не переобучается:
    её даёт sweep на других фолдах, сюда она приходит готовой.
    """
    feat = {"class_name": class_name, "threshold_group": group, "markup_type": markup,
            "box_area_m2": area}
    t = float(base_tau)
    if area_on:
        t = float(area_tau(lambda f: t, lo)(feat))
    if fixc_on:
        t = float(dfine_ae_gor_area_tau(lambda f: t, lo)(feat))
    return t


def apply_thresholds(feats: list[dict], thresholds: dict, default: float, by_markup: dict | None,
                     lo: dict) -> list[dict]:
    """Как post.apply_thresholds + опционально area_tau и/или Fix C (dfine_ae_gor_area_tau)."""
    base = post.threshold_fn(thresholds, default, by_markup)
    tau = base
    if lo.get("area_tau"):
        tau = area_tau(tau, lo)
    if lo.get("dfine_ae_gor_area_tau"):
        tau = dfine_ae_gor_area_tau(tau, lo)
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
    """Ядра только у крупных городищ (не глобальный F2)."""
    if not lo.get("shape_cores"):
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

    Оригинал fortifikatsii НЕ переименовывается (keep-original-label). Копия не
    добавляется, если пересекается с уже имеющимся gorodishcha того же региона
    (иначе FP поверх существующего TP/кандидата). Доп. порог fort_to_gor_min_conf.
    """
    if not lo.get("fort_to_gor"):
        return feats
    min_area = float(lo.get("min_area_m2", 5000.0))
    scale = float(lo.get("fort_to_gor_scale", 0.8))
    min_conf = float(lo.get("fort_to_gor_min_conf", 0.5))
    group = str(lo.get("fort_to_gor_group", "gor_from_fort"))
    gors = [f for f in feats if f.get("class_name") == "gorodishcha" and f.get("geom") is not None]
    extra = []
    for f in feats:
        if f.get("class_name") != "fortifikatsii":
            continue
        if area_m2(f) < min_area:
            continue
        conf = float(f.get("confidence", 1.0))
        if conf < min_conf:
            continue
        geom = f.get("geom")
        if geom is not None and not geom.is_empty:
            # не копировать, если fort уже лежит на городище
            if any(g.get("region_name") == f.get("region_name")
                   and g["geom"] is not None and geom.intersects(g["geom"])
                   for g in gors):
                continue
        extra.append(dict(
            f,
            class_name="gorodishcha",
            threshold_group=group,
            confidence=round(conf * scale, 6),
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
        "max_per_dem": int(lo.get("coarse_max_per_dem", 1)),
        "min_enclosure": float(lo.get("coarse_min_enclosure", 0.35)),
        "min_rel_height_m": float(lo.get("coarse_min_rel_height_m", 1.8)),
        "flat_max_deg": float(lo.get("coarse_flat_max_deg", 12.0)),
        "scarp_min_deg": float(lo.get("coarse_scarp_min_deg", 18.0)),
    }
    base = {f.name: getattr(HillfortParams(), f.name) for f in fields(HillfortParams)}
    base.update(overrides)
    return HillfortParams(**base)


def coarse_hillfort_candidates(dem, region: str, source: str, lo: dict | None = None) -> list[dict]:
    """Грубый проход hillfort → группа gor_coarse_dem. Пусто, если ЦМР без CRS."""
    from . import hillfort

    lo = lo or {}
    if not lo.get("coarse_hillfort"):
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
