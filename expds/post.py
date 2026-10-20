"""Постобработка предсказаний перед записью.

Правила (ТР КОЗ №3 и разъяснения организаторов):
* внутри одного (регион, markup_type, класс) пересекающиеся полигоны с РАЗНЫХ
  снимков объединяются в один объект;
* между разными markup_type ничего не объединяется, даже при пересечении:
  эталон размечен по модальностям, и два попадания дают два TP;
* MultiPolygon метрика оценивает только по самой большой части, поэтому
  всё разбивается на отдельные Polygon;
* не больше 3000 вершин на полигон и 20 000 объектов на файл.
"""
from __future__ import annotations

from collections import defaultdict

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

from .constants import CLASS_WEIGHTS, CLASSES, MAX_FEATURES, MAX_VERTICES


def polygons_of(geom: BaseGeometry) -> list[BaseGeometry]:
    """Только непустые Polygon (MultiPolygon и коллекции разбираются)."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(polygons_of(g))
        return out
    return []


def merge_across_sources(items: list[dict], source_key: str = "source") -> list[dict]:
    """Объединяет пересекающиеся полигоны разных источников (снимков) одной группы.

    Полигоны одного источника не сливаются между собой. Уверенность
    объединённого объекта — максимум по членам.
    """
    if len(items) < 2:
        return items
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
        for part in polygons_of(geom):
            out.append(dict(items[best], geom=part))
    return out


def merge_within_markup(feats: list[dict]) -> list[dict]:
    """Объединение по группам (регион, markup_type, класс); разные markup_type не трогаются."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for f in feats:
        groups[(f.get("region_name"), f.get("markup_type"), f["class_name"])].append(f)
    out = []
    for items in groups.values():
        out.extend(merge_across_sources(items))
    return out


def limit_vertices(geom: BaseGeometry, max_vertices: int = MAX_VERTICES) -> BaseGeometry:
    """Упрощает полигон, пока число вершин не станет меньше лимита метрики."""
    def count(g):
        return len(g.exterior.coords) + sum(len(r.coords) for r in g.interiors)

    if count(geom) < max_vertices:
        return geom
    tol = max(geom.length / 5000.0, 1e-3)
    g = geom
    for _ in range(30):
        g = geom.simplify(tol, preserve_topology=True)
        if g.geom_type == "Polygon" and count(g) < max_vertices:
            return g
        tol *= 1.6
    return geom.convex_hull


def finalize(feats: list[dict], min_area: float = 0.01) -> list[dict]:
    """Валидные отдельные Polygon с допустимым числом вершин."""
    out = []
    for f in feats:
        g = f["geom"]
        if not g.is_valid:
            g = make_valid(g)
        for part in polygons_of(g):
            if part.area < min_area:
                continue
            out.append(dict(f, geom=limit_vertices(part)))
    return out


def apply_thresholds(feats: list[dict], thresholds: dict[str, float], default: float = 0.15,
                     by_markup: dict[str, dict[str, float]] | None = None) -> list[dict]:
    """Оставляет объекты с уверенностью не ниже порога своего класса.

    by_markup[группа][класс] переопределяет общий порог: у моделей разных
    модальностей своя шкала уверенности. Группа — threshold_group объекта
    (например, li_cpu у запасного детектора), иначе его markup_type.
    """
    by_markup = by_markup or {}

    def tau(f: dict) -> float:
        own = by_markup.get(f.get("threshold_group") or f.get("markup_type"), {})
        return own.get(f["class_name"], thresholds.get(f["class_name"], default))

    return [f for f in feats if f.get("confidence", 1.0) >= tau(f)]


def cap_features(feats: list[dict], max_n: int = MAX_FEATURES) -> list[dict]:
    """Если объектов больше лимита — квота класса пропорциональна его весу в метрике.

    Внутри класса остаются самые уверенные. Общий отбор по уверенности отдаёт весь
    лимит классу с самой высокой шкалой и вытесняет более тяжёлые классы.
    """
    if len(feats) <= max_n:
        return feats
    groups: dict[str, list[dict]] = {}
    for f in feats:
        groups.setdefault(f["class_name"], []).append(f)
    for g in groups.values():
        g.sort(key=lambda f: f.get("confidence", 1.0), reverse=True)
    weight = {c: CLASS_WEIGHTS.get(c, 1.0) for c in groups}
    total_w = sum(weight.values())
    raw = {c: max_n * weight[c] / total_w for c in groups}
    quota = {c: min(len(groups[c]), int(raw[c])) for c in groups}
    left = max_n - sum(quota.values())
    order = sorted(groups, key=lambda c: raw[c] - int(raw[c]), reverse=True)
    while left > 0:
        progressed = False
        for c in order:
            if quota[c] < len(groups[c]):
                quota[c] += 1
                left -= 1
                progressed = True
                if left <= 0:
                    break
        if not progressed:
            break
    out: list[dict] = []
    for c in groups:
        out.extend(groups[c][: quota[c]])
    return out


def only_known_classes(feats: list[dict]) -> list[dict]:
    """Отбрасывает объекты с неизвестным классом (иначе платформа отклонит файл)."""
    return [f for f in feats if f["class_name"] in CLASSES]
