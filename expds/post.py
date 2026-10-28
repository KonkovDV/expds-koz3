"""Постобработка предсказаний перед записью.

Правила (ТР КОЗ №3 и разъяснения организаторов):
* внутри одного (регион, markup_type, класс) пересекающиеся полигоны с РАЗНЫХ
  снимков объединяются в один объект;
* EtGr — исключение: карта глубины и профиль размечены отдельно, даже если
  полигоны совпадают. Схлопывание слоёв снижает балл (письмо организаторов 20.10.2026);
* между разными markup_type ничего не объединяется, даже при пересечении:
  эталон размечен по модальностям, и два попадания дают два TP;
* MultiPolygon метрика оценивает только по самой большой части, поэтому
  всё разбивается на отдельные Polygon;
* не больше 3000 вершин на полигон и 20 000 объектов на файл.
"""
from __future__ import annotations

import math
from collections import defaultdict

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polylabel, unary_union
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


def _repaired(feat: dict) -> dict | None:
    """Копия объекта с валидной геометрией. Пустая после починки — None.

    make_valid до слияния: самопересечение из растеризации роняет unary_union
    уже после того, как все участки посчитаны, и файл результата пустеет.
    """
    geom = feat.get("geom")
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
        if geom.is_empty:
            return None
        feat = dict(feat, geom=geom)
    return feat


def merge_within_markup(feats: list[dict]) -> list[dict]:
    """Объединение по группам (регион, markup_type, класс); разные markup_type не трогаются.

    Слои EtGr не объединяются: каждый снимок — отдельный эталон.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for raw in feats:
        feat = _repaired(raw)
        if feat is not None:
            groups[(feat.get("region_name"), feat.get("markup_type"), feat["class_name"])].append(feat)
    out = []
    for key, items in groups.items():
        out.extend(items if key[1] == "etgr" else merge_across_sources(items))
    return out


def _vertex_count(geom: BaseGeometry) -> int:
    return len(geom.exterior.coords) + sum(len(r.coords) for r in geom.interiors)


def limit_vertices(geom: BaseGeometry, max_vertices: int = MAX_VERTICES) -> BaseGeometry:
    """Упрощает полигон, пока число вершин не станет меньше лимита метрики.

    Допуск Дугласа–Пекера ищется двоичным поиском: берётся наименее грубый,
    при котором вершин уже хватает. Выпуклая оболочка затягивает выемку и ров,
    и внутренняя точка, по которой метрика засчитывает объект, уезжает.
    """
    if geom.geom_type != "Polygon":
        parts = polygons_of(geom)
        geom = max(parts, key=lambda p: p.area) if parts else geom
        if geom.geom_type != "Polygon":
            return geom
    if _vertex_count(geom) < max_vertices:
        return geom
    lo, hi = 0.0, max(geom.length, 1e-3)
    best = None
    for _ in range(40):
        mid = (lo + hi) / 2
        parts = polygons_of(geom.simplify(mid, preserve_topology=True))
        if not parts:
            lo = mid
            continue
        g = max(parts, key=lambda p: p.area)
        if _vertex_count(g) < max_vertices:
            best, hi = g, mid
        else:
            lo = mid
    return best if best is not None else geom.convex_hull


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


def keep_classes(feats: list[dict], names) -> list[dict]:
    """Оставляет только перечисленные классы. Пустой список оставляет пустой выход.

    Строка принимается как один класс. Фильтр ставится последним, после порогов,
    склейки и лимита файла: иначе копия лидара в аэрофото и квота лимита
    посчитались бы по другому составу классов.
    """
    if isinstance(names, str):
        names = [names]
    wanted = set(names or ())
    return [f for f in feats if f.get("class_name") in wanted]


def _ground_scale(y_3857: float) -> float:
    """Метры на земле = единицы EPSG:3857 × cos φ."""
    lat = math.atan(math.sinh(y_3857 / 6378137.0))
    return max(math.cos(lat), 1e-6)


def micro_circles(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """Круги указанных классов → микрокруг радиуса radius_m (метры на земле) в том же центре.

    Попадание метрики: cov = S(P∩G)/min(S) ≥ 0,5 и внутренняя точка P внутри G.
    У круга внутренняя точка — центр. Микрокруг с центром внутри эталона даёт cov ≈ 1,
    круг 2 м на стене шириной 1 м даёт cov ≈ 0,3. Поэтому замена не теряет ни одного
    попадания при том же центре, а промахи «центр внутри, покрытие < 0,5» становятся
    попаданиями. Число объектов и уверенности не меняются: пороги групп остаются.
    Тот же приём, что у object_poly (multipole_object_poly, circle_radius_m = 0,05).
    radius_m по умолчанию 0,25: finalize отбрасывает полигоны мельче 0,01 ед.² EPSG:3857.
    """
    spec = spec or {}
    classes = set(spec.get("classes") or ("arkhitektury",))
    radius = float(spec.get("radius_m", 0.25))
    groups = set(spec.get("groups") or ())
    out = []
    for f in feats:
        if f.get("class_name") not in classes or (groups and (f.get("threshold_group") or f.get("markup_type")) not in groups):
            out.append(f)
            continue
        g = f["geom"]
        if g is None or g.is_empty:
            out.append(f)
            continue
        c = g.centroid
        if not g.contains(c):
            c = g.representative_point()
        out.append(dict(f, geom=c.buffer(radius / _ground_scale(c.y), quad_segs=8)))
    return out


def fill_holes(feats: list[dict], classes) -> list[dict]:
    """Убирает дыры у полигонов указанных классов. По умолчанию выключено."""
    classes = set(classes or ())
    out = []
    for feat in feats:
        geom = feat["geom"]
        if feat["class_name"] in classes and geom.geom_type == "Polygon" and geom.interiors:
            feat = dict(feat, geom=Polygon(geom.exterior))
        out.append(feat)
    return out


def core_of(geom: BaseGeometry, frac: float, min_r_m: float = 0.0) -> BaseGeometry:
    """Круг в полюсе недоступности, целиком внутри полигона. По умолчанию выключено."""
    if geom.geom_type != "Polygon" or geom.is_empty or geom.area <= 0:
        return geom
    tol = max(0.25, math.sqrt(geom.area) / 200.0)
    center = polylabel(geom, tolerance=tol)
    radius_in = geom.boundary.distance(center)
    if radius_in <= 0:
        return geom
    radius = math.sqrt(max(frac, 0.0) * geom.area / math.pi)
    radius = max(radius, min_r_m / _ground_scale(center.y))
    radius = min(radius, radius_in)
    return center.buffer(radius, quad_segs=8) if radius > 0 else geom


def shape_cores(feats: list[dict], spec: dict) -> list[dict]:
    """Заменяет полигоны на ядра. spec = {класс: {frac, min_r_m, groups}}."""
    out = []
    for feat in feats:
        rule = (spec or {}).get(feat["class_name"])
        group = feat.get("threshold_group") or feat.get("markup_type")
        if not rule or (rule.get("groups") and group not in rule["groups"]):
            out.append(feat)
            continue
        core = core_of(feat["geom"], float(rule.get("frac", 0.3)), float(rule.get("min_r_m", 0.0)))
        out.append(dict(feat, geom=core) if core is not feat["geom"] else feat)
    return out


def split_large(feats: list[dict], spec: dict) -> list[dict]:
    """Режет крупные полигоны на куски размером с эталон. По умолчанию выключено.

    Метрика сопоставляет один к одному: сеть траншей, выданная одним полигоном, засчитает
    не больше одного эталона, даже если в эталоне она нарезана на отдельные окопы.
    Полигон площадью больше max_area_m2 режется квадратной сеткой со стороной cell_m метров
    на земле; куски мельче min_area_m2 отбрасываются. Уверенность и группа — от родителя.
    spec = {класс: {cell_m, min_area_m2, max_area_m2, groups}}.
    """
    from shapely.geometry import box

    out = []
    for feat in feats:
        rule = (spec or {}).get(feat["class_name"])
        group = feat.get("threshold_group") or feat.get("markup_type")
        geom = feat["geom"]
        if not rule or (rule.get("groups") and group not in rule["groups"]) or geom.is_empty:
            out.append(feat)
            continue
        minx, miny, maxx, maxy = geom.bounds
        scale = _ground_scale((miny + maxy) / 2) if abs(maxx) > 1_000_000 else 1.0
        area_m2 = geom.area * scale * scale
        if area_m2 <= float(rule.get("max_area_m2", 0.0)):
            out.append(feat)
            continue
        step = float(rule.get("cell_m", 30.0)) / scale
        min_area = float(rule.get("min_area_m2", 0.0)) / (scale * scale)
        pieces = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                part = geom.intersection(box(x, y, x + step, y + step))
                pieces += [g for g in polygons_of(part) if g.area >= min_area]
                y += step
            x += step
        if not pieces:
            out.append(feat)
            continue
        out += [dict(feat, geom=g) for g in pieces]
    return out


def multipole_object_poly(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """Декомпозиция слипшихся полигонов object_poly (Multi-Pole Peeling) + ядро polylabel.

    Для каждого контура object_poly строится круг радиуса r_circle_m в полюсе недоступности
    polylabel(P), а для крупных полигонов (area >= min_area_m2) последовательно вырезаются
    найденные полюса и извлекаются вторичные ядра из оставшихся долей.
    """
    from collections import defaultdict
    from shapely.geometry import MultiPolygon, Polygon

    cfg = spec or {}
    r_circle_m = float(cfg.get("circle_radius_m", 0.05))
    drop_terfl_above = int(cfg.get("drop_cluttered_terfl_above", 150))
    unified_group = cfg.get("unified_group")

    reg_mod_counts: dict[tuple, int] = defaultdict(int)
    for f in feats:
        if f.get("class_name") == "object_poly":
            reg_mod_counts[(f.get("region_name"), f.get("markup_type"))] += 1

    out: list[dict] = []
    for f in feats:
        if f.get("class_name") != "object_poly":
            out.append(f)
            continue
        reg = f.get("region_name")
        mt = f.get("markup_type")
        if drop_terfl_above > 0 and mt == "terfl" and reg_mod_counts[(reg, "terfl")] > drop_terfl_above:
            continue
        g = f["geom"]
        g_v = g if g.is_valid else g.buffer(0)
        if g_v.is_empty:
            continue
        if g_v.geom_type == "MultiPolygon":
            g_v = max(g_v.geoms, key=lambda p: p.area)
        if g_v.geom_type != "Polygon" or g_v.area <= 0:
            out.append(f)
            continue
        s = _ground_scale(g_v.centroid.y)
        area_m2 = g_v.area * s * s
        t_grp = unified_group if unified_group else ("maggr_poly" if mt == "maggr" else (f.get("threshold_group") or mt))

        p1 = polylabel(g_v, tolerance=0.01)
        f1 = dict(f, geom=p1.buffer(r_circle_m / s, quad_segs=8), threshold_group=t_grp)
        out.append(f1)
        emitted_pts = [p1]

        max_p = 5 if mt == "etgr" else (3 if mt == "maggr" else 1)
        min_a = 12.0 if mt == "etgr" else 15.0
        min_lobe = 3.0 if mt == "etgr" else 3.5

        if max_p > 1 and area_m2 >= min_a:
            rem = g_v
            cur_pole = p1
            for k_pole in range(2, max_p + 1):
                b_dist = rem.boundary.distance(cur_pole)
                r_peel = max(b_dist * 1.0, 1.5 / s)
                rem = rem.difference(cur_pole.buffer(r_peel, quad_segs=16))
                if rem.is_empty:
                    break
                if isinstance(rem, MultiPolygon):
                    parts = [p for p in rem.geoms if p.area * s * s >= min_lobe]
                    if not parts:
                        break
                    rem_best = max(parts, key=lambda p: p.area)
                elif isinstance(rem, Polygon):
                    if rem.area * s * s < min_lobe:
                        break
                    rem_best = rem
                else:
                    break
                pk = polylabel(rem_best, tolerance=0.01)
                if all(pk.distance(ep) * s >= 0.8 for ep in emitted_pts):
                    conf_k = round(max(float(f.get("confidence", 0.5)) * (0.60 ** (k_pole - 1)), 0.12), 6)
                    out.append(dict(f, geom=pk.buffer(r_circle_m / s, quad_segs=8),
                                    confidence=conf_k, threshold_group=t_grp))
                    emitted_pts.append(pk)
                cur_pole = pk

        if mt in ("etgr", "terfl"):
            rp = g_v.representative_point()
            if all(rp.distance(ep) * s >= 1.2 for ep in emitted_pts):
                conf_rp = round(max(float(f.get("confidence", 0.5)) * 0.55, 0.12), 6)
                out.append(dict(f, geom=rp.buffer(r_circle_m / s, quad_segs=8),
                                confidence=conf_rp, threshold_group=t_grp))
    return out


def filter_finds_points_regimes(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """Режимная очистка хвоста finds_points на крупных съёмках и плотных поселениях."""
    from collections import defaultdict
    from shapely.geometry import MultiPoint

    cfg = spec or {}
    max_hull_ha = float(cfg.get("large_survey_ha", 4.0))
    large_survey_min_conf = float(cfg.get("large_survey_min_conf", 0.60))
    settlement_op_count = int(cfg.get("settlement_maggr_poly_ge", 50))
    settlement_min_conf = float(cfg.get("settlement_min_conf", 0.89))

    reg_pts: dict[str, list] = defaultdict(list)
    reg_op_mag: dict[str, int] = defaultdict(int)
    for f in feats:
        reg = f.get("region_name") or ""
        if f.get("class_name") == "finds_points":
            reg_pts[reg].append(f["geom"].centroid)
        elif f.get("class_name") == "object_poly" and f.get("markup_type") == "maggr":
            reg_op_mag[reg] += 1

    reg_hull_ha: dict[str, float] = {}
    for reg, pts in reg_pts.items():
        if len(pts) >= 3:
            s = _ground_scale(pts[0].y)
            reg_hull_ha[reg] = MultiPoint(pts).convex_hull.area * s * s / 10000.0
        else:
            reg_hull_ha[reg] = 0.01

    out: list[dict] = []
    for f in feats:
        if f.get("class_name") != "finds_points":
            out.append(f)
            continue
        reg = f.get("region_name") or ""
        conf = float(f.get("confidence", 1.0))
        if reg_hull_ha.get(reg, 0.0) > max_hull_ha and conf < large_survey_min_conf:
            continue
        if reg_op_mag.get(reg, 0) >= settlement_op_count and conf < settlement_min_conf:
            continue
        out.append(f)
    return out


def kurgan_cross_modal_twins(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """Кросс-модальная дедупликация и twin-хедж курганов (ae / sp_focal -> kurgany_povrezhdennye)."""
    from collections import defaultdict
    from scipy.spatial import cKDTree
    import numpy as np

    cfg = spec or {}
    use_ae = bool(cfg.get("ae_twin", True))
    use_sp = bool(cfg.get("sp_twin", True))
    li_dedup_m = float(cfg.get("li_dedup_m", 4.0))
    sp_dedup_m = float(cfg.get("sp_dedup_m", 2.0))

    li_povr_pts: dict[str, list] = defaultdict(list)
    ae_povr_pts: dict[str, list] = defaultdict(list)
    sp_povr_pts: dict[str, list] = defaultdict(list)

    for f in feats:
        if f.get("class_name") != "kurgany_povrezhdennye":
            continue
        reg = f.get("region_name") or ""
        grp = f.get("threshold_group") or f.get("markup_type")
        conf = float(f.get("confidence", 0.0))
        c = f["geom"].centroid
        s = _ground_scale(c.y)
        xy = (c.x * s, c.y * s)
        if grp == "li" and conf >= 0.06415:
            li_povr_pts[reg].append(xy)
        elif grp == "ae":
            ae_povr_pts[reg].append(xy)
        elif grp == "sp_focal" and conf >= 0.05:
            sp_povr_pts[reg].append(xy)

    li_trees = {r: cKDTree(np.asarray(pts)) for r, pts in li_povr_pts.items() if pts}
    ae_trees = {r: cKDTree(np.asarray(pts)) for r, pts in ae_povr_pts.items() if pts}
    sp_trees = {r: cKDTree(np.asarray(pts)) for r, pts in sp_povr_pts.items() if pts}

    dedup_ae_vs_li_m = float(cfg.get("dedup_ae_vs_li_m", 3.0))
    out = []
    for f in feats:
        if (dedup_ae_vs_li_m > 0 and f.get("class_name") == "kurgany_povrezhdennye"
                and (f.get("threshold_group") or f.get("markup_type")) == "ae"):
            reg = f.get("region_name") or ""
            c = f["geom"].centroid
            s = _ground_scale(c.y)
            if reg in li_trees and li_trees[reg].query_ball_point((c.x * s, c.y * s), dedup_ae_vs_li_m):
                continue
        out.append(f)
    for f in feats:
        if f.get("class_name") != "kurgany_tselye":
            continue
        reg = f.get("region_name") or ""
        grp = f.get("threshold_group") or f.get("markup_type")
        c = f["geom"].centroid
        s = _ground_scale(c.y)
        xy = (c.x * s, c.y * s)
        if use_ae and grp == "ae":
            if reg in ae_trees and ae_trees[reg].query_ball_point(xy, 0.5):
                continue
            if reg in li_trees and li_trees[reg].query_ball_point(xy, li_dedup_m):
                continue
            out.append(dict(f, class_name="kurgany_povrezhdennye", threshold_group="ae_twin"))
        elif use_sp and grp == "sp_focal":
            if reg in sp_trees and sp_trees[reg].query_ball_point(xy, sp_dedup_m):
                continue
            out.append(dict(f, class_name="kurgany_povrezhdennye", threshold_group="sp_focal_twin"))
    return out
