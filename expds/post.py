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


#: bot-6: группы порогов, которые склеиваются только сами с собой (D-FINE Ae/Or, [dfine_aeor]).
ISOLATED_GROUPS = ("dfine_ae", "dfine_or")


def merge_within_markup(feats: list[dict], isolate=ISOLATED_GROUPS) -> list[dict]:
    """Объединение по группам (регион, markup_type, класс); разные markup_type не трогаются.

    Слои EtGr не объединяются: каждый снимок — отдельный эталон.
    bot-6: у объектов с threshold_group из isolate ключ ещё и по группе — dfine_ae склеивается только
    с dfine_ae (не с прочими ae). У остальных четвёртый элемент ключа None — их группы как в v3.
    """
    iso = set(isolate or ())
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for raw in feats:
        feat = _repaired(raw)
        if feat is not None:
            tg = feat.get("threshold_group")
            groups[(feat.get("region_name"), feat.get("markup_type"), feat["class_name"],
                    tg if tg in iso else None)].append(feat)
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


GEO_HULL_MARKUPS = ("etgr", "terfl")


def convex_hull_object_poly(feats: list[dict], spec=None) -> list[dict]:
    """Выпуклая оболочка object_poly только для геофизики etgr/terfl ([post] geo_convex_hull, по умолчанию выкл.).

    Честная проверка bot-6 (GEO.md, work/geo/geo_val.py, LOSO внутри Гнёздово): +0,0017 к исходной
    геометрии, P 0,998. maggr и остальные классы не трогаются.
    """
    cfg = spec if isinstance(spec, dict) else {}
    marks = set(cfg.get("markups", GEO_HULL_MARKUPS))
    out = []
    for f in feats:
        if f.get("class_name") == "object_poly" and f.get("markup_type") in marks and not f["geom"].is_empty:
            hull = f["geom"].convex_hull
            if hull.geom_type == "Polygon":
                f = dict(f, geom=hull)
        out.append(f)
    return out


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
    tau = threshold_fn(thresholds, default, by_markup)
    return [f for f in feats if f.get("confidence", 1.0) >= tau(f)]


def threshold_fn(thresholds: dict[str, float], default: float = 0.15,
                 by_markup: dict[str, dict[str, float]] | None = None):
    """Порог объекта, как в apply_thresholds: группа порога (или markup_type) → класс → общий → default."""
    by_markup = by_markup or {}

    def tau(f: dict) -> float:
        own = by_markup.get(f.get("threshold_group") or f.get("markup_type"), {})
        return own.get(f["class_name"], thresholds.get(f["class_name"], default))

    return tau


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
    radius_m — число (для всех classes) или словарь {класс: радиус}; во втором случае
    classes по умолчанию = ключи словаря. Курганы (bot-7, 28.10): 0,5 м вместо 2 м /
    4 м (sp_focal) — тот же центр, cov ≥ 0,5 почти всегда, если центр внутри.
    """
    spec = spec or {}
    rad = spec.get("radius_m", 0.25)
    per_class = dict(rad) if isinstance(rad, dict) else {}
    classes = set(spec.get("classes") or (per_class.keys() if per_class else ("arkhitektury",)))
    groups = set(spec.get("groups") or ())
    anchor = str(spec.get("anchor", "centroid"))
    out = []
    for f in feats:
        if f.get("class_name") not in classes or (groups and (f.get("threshold_group") or f.get("markup_type")) not in groups):
            out.append(f)
            continue
        g = f["geom"]
        if g is None or g.is_empty:
            out.append(f)
            continue
        if anchor == "rp":
            # W0-seams: метрика проверяет representative_point(P) ∈ G; диск r ≪ G в этой же точке
            # засчитывается всегда, когда засчитан P (cov = 1), — слабое доминирование для любого P.
            c = g.representative_point()
        else:
            c = g.centroid
            if not g.contains(c):
                c = g.representative_point()
        radius = float(per_class.get(f.get("class_name"), rad if not per_class else 0.25))
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
        # bot-2 bigobj F2: min_area_m2 — ядро только у полигонов не мельче (м² на земле); нет ключа — у всех.
        if rule.get("min_area_m2") and _area_m2(feat["geom"]) < float(rule["min_area_m2"]):
            out.append(feat)
            continue
        core = core_of(feat["geom"], float(rule.get("frac", 0.3)), float(rule.get("min_r_m", 0.0)))
        out.append(dict(feat, geom=core) if core is not feat["geom"] else feat)
    return out


def _area_m2(geom: BaseGeometry) -> float:
    """Площадь на земле, м²: EPSG:3857 с поправкой cos(широты) по середине охвата, иначе как есть."""
    if geom is None or geom.is_empty:
        return 0.0
    minx, miny, maxx, maxy = geom.bounds
    scale = _ground_scale((miny + maxy) / 2) if abs(maxx) > 1_000_000 else 1.0
    return geom.area * scale * scale


def big_area_groups(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """bot-2 bigobj F1: крупные прогнозы — в свою группу порога. По умолчанию выключено.

    Уверенность масок городищ и фортификаций — пик вероятности экземпляра (infer_nn.probs_to_features),
    одна шкала у мелкого пятна и у маски на весь памятник; порог группы (sp_or, ae, li_mask …) ставят
    многочисленные мелкие ложные, и крупные попадания с уверенностью 0,93–0,999 отсекаются
    (work/bigobj/A/gt_rows.csv: 113 из 204 пропущенных эталонов > 5000 м² — ниже честного порога).
    Прогноз класса из spec.classes площадью ≥ min_area_m2 (м² на земле) получает threshold_group
    «<группа><suffix>»; порог такой группы подбирается отдельно (big_group_thresholds).
    spec = {min_area_m2 = 5000, classes = [gorodishcha, fortifikatsii], groups = [] (все), suffix = "_big"}.
    Ставить ДО apply_thresholds (порог зависит от группы), геометрию не меняет.
    """
    spec = spec or {}
    min_a = float(spec.get("min_area_m2", 5000.0))
    suffix = str(spec.get("suffix", "_big"))
    classes = set(spec.get("classes") or ("gorodishcha", "fortifikatsii"))
    groups = set(spec.get("groups") or ())
    out = []
    for feat in feats:
        group = feat.get("threshold_group") or feat.get("markup_type")
        if (feat.get("class_name") in classes and (not groups or group in groups)
                and not str(group).endswith(suffix) and _area_m2(feat.get("geom")) >= min_a):
            feat = dict(feat, threshold_group=f"{group}{suffix}")
        out.append(feat)
    return out


def big_group_thresholds(by_markup: dict | None, spec: dict | None = None) -> dict:
    """thresholds_by_markup с группами «<g><suffix>» из big_area_groups.

    Свои пороги — из spec.thresholds = {"<g>_big": {класс: tau}}; группы и классы без своего порога
    наследуют порог родителя <g> (тогда выдача та же, что без F1). Исходный словарь не меняется.
    """
    spec = spec or {}
    suffix = str(spec.get("suffix", "_big"))
    classes = set(spec.get("classes") or ("gorodishcha", "fortifikatsii"))
    out = {g: dict(v) for g, v in (by_markup or {}).items()}
    for g, v in list(out.items()):
        if g.endswith(suffix):
            continue
        own = out.setdefault(f"{g}{suffix}", {})
        for c in classes:
            if c in v:
                own.setdefault(c, v[c])
    for g, v in (spec.get("thresholds") or {}).items():
        out.setdefault(g, {}).update({c: float(t) for c, t in v.items()})
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


# Карта центров спутника: в табло группа подписана sp_focal, в сдаче читается из markup_type = sp_or
# (pipeline.py, комментарий у KEPT_KURGAN_CONF). Без sp_or ветка sp_twin в сдаче не срабатывает.
SP_TWIN_GROUPS = ("sp_focal", "sp_or")


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

    # Один масштаб cos φ на участок (по среднему y его объектов). Масштаб для каждой точки, s(y)·y, не изометрия:
    # на 55° расстояния по меридиану сжимаются примерно в 16 раз (bot-10 REPORT6_twins.md §2, ae-двойников 5 вместо 85).
    ys: dict[str, list] = defaultdict(list)
    for f in feats:
        if f.get("class_name") in ("kurgany_povrezhdennye", "kurgany_tselye"):
            ys[f.get("region_name") or ""].append(f["geom"].centroid.y)
    scale = {r: _ground_scale(sum(v) / len(v)) for r, v in ys.items()}

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
        s = scale[reg]
        xy = (c.x * s, c.y * s)
        if grp == "li" and conf >= 0.06415:
            li_povr_pts[reg].append(xy)
        elif grp == "ae":
            ae_povr_pts[reg].append(xy)
        elif grp in SP_TWIN_GROUPS and conf >= 0.05:
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
            s = scale[reg]
            if reg in li_trees and li_trees[reg].query_ball_point((c.x * s, c.y * s), dedup_ae_vs_li_m):
                continue
        out.append(f)
    for f in feats:
        if f.get("class_name") != "kurgany_tselye":
            continue
        reg = f.get("region_name") or ""
        grp = f.get("threshold_group") or f.get("markup_type")
        c = f["geom"].centroid
        s = scale[reg]
        xy = (c.x * s, c.y * s)
        if use_ae and grp == "ae":
            if reg in ae_trees and ae_trees[reg].query_ball_point(xy, 0.5):
                continue
            if reg in li_trees and li_trees[reg].query_ball_point(xy, li_dedup_m):
                continue
            out.append(dict(f, class_name="kurgany_povrezhdennye", threshold_group="ae_twin"))
        elif use_sp and grp in SP_TWIN_GROUPS:
            if reg in sp_trees and sp_trees[reg].query_ball_point(xy, sp_dedup_m):
                continue
            out.append(dict(f, class_name="kurgany_povrezhdennye", threshold_group="sp_focal_twin"))
    return out


SP_FUSION_GROUPS = ("sp_focal", "sp_focal2")
SP_FUSION_CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye")


def sp_score_fusion(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """Слияние уверенностей двух спутниковых групп для курганов (bot-6 ORSP, PREREG_ORSP.md).

    Для кургана (povr или tselye) из sp_focal / sp_focal2: o = наибольшая уверенность курганов
    (любого из двух классов) ДРУГОЙ sp-группы в радиусе radius_m метров на земле на том же
    участке, иначе 0; новая уверенность = (conf + o) / 2. Объекты не добавляются и не
    удаляются (кроме помеченных sp_partner_only — они только соседи); применять ДО порогов,
    на сырых кандидатах (pipeline: gen_floor вместо порога выдачи), пороги — из spec.thresholds
    (watchdog.finish подставляет их в thresholds_by_markup). Работает и с плоскими
    признаками (geom, поля в корне), и с GeoJSON (geometry, properties).
    """
    from collections import defaultdict
    from scipy.spatial import cKDTree
    import numpy as np
    from shapely.geometry import shape

    cfg = spec or {}
    radius = float(cfg.get("radius_m", 8.0))
    groups = tuple(cfg.get("groups", SP_FUSION_GROUPS))
    if len(groups) != 2:
        raise ValueError("sp_score_fusion: нужны ровно две группы")

    def _p(f):
        return f["properties"] if isinstance(f.get("properties"), dict) else f

    def _grp(p):
        # В сдаче карта центров идёт с markup_type sp_or, в табло — с группой sp_focal.
        g = p.get("threshold_group") or p.get("markup_type")
        return "sp_focal" if g == "sp_or" else g

    def _c(f):
        g = f.get("geom")
        if g is None:
            g = shape(f["geometry"])
        return g.centroid

    idx = []
    for i, f in enumerate(feats):
        p = _p(f)
        grp = _grp(p)
        if p.get("class_name") in SP_FUSION_CLASSES and grp in groups:
            c = _c(f)
            idx.append((i, p.get("region_name") or "", grp, c.x, c.y, float(p.get("confidence", 0.0))))
    if not idx:
        return [f for f in feats if not _p(f).get("sp_partner_only")]
    # Один cos φ на участок (как в kurgan_cross_modal_twins): метры на земле.
    ys = defaultdict(list)
    for _, reg, _, _, y, _ in idx:
        ys[reg].append(y)
    scale = {reg: _ground_scale(float(np.median(v))) for reg, v in ys.items()}
    pts = defaultdict(list)
    for i, reg, grp, x, y, conf in idx:
        pts[(reg, grp)].append((x * scale[reg], y * scale[reg], conf))
    trees = {k: (cKDTree(np.asarray([(a, b) for a, b, _ in v])), np.asarray([c for _, _, c in v]))
             for k, v in pts.items()}
    new_conf = {}
    for i, reg, grp, x, y, conf in idx:
        other = groups[1] if grp == groups[0] else groups[0]
        o = 0.0
        t = trees.get((reg, other))
        if t is not None:
            hit = t[0].query_ball_point((x * scale[reg], y * scale[reg]), radius)
            if hit:
                o = float(t[1][hit].max())
        new_conf[i] = (conf + o) / 2.0
    out = []
    for i, f in enumerate(feats):
        if _p(f).get("sp_partner_only"):
            continue  # кандидат только как сосед для слияния (его выкинул отбор 12 м), не выдаётся
        if i in new_conf:
            f = dict(f)
            if isinstance(f.get("properties"), dict):
                f["properties"] = dict(f["properties"], confidence=new_conf[i])
            else:
                f["confidence"] = new_conf[i]
        out.append(f)
    return out


KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")


def _group_of(f: dict) -> str:
    return f.get("threshold_group") or f.get("markup_type") or ""


def _region_scales(feats: list[dict]) -> dict[str, float]:
    """Один cos φ на участок: метры на земле = единицы EPSG:3857 × масштаб (как в kurgan_cross_modal_twins)."""
    ys: dict[str, list[float]] = defaultdict(list)
    for f in feats:
        g = f.get("geom")
        if g is not None and not g.is_empty:
            ys[f.get("region_name") or ""].append(g.centroid.y)
    return {r: _ground_scale(sum(v) / len(v)) for r, v in ys.items()}


def kurgans_outside_hillforts(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """Курганы, чья внутренняя точка внутри выданного городища того же участка, убираются.

    Правило разметки организаторов: насыпи внутри городищ курганами не размечаются. На трейне
    внутри городищ 7 из 974 курганов (все на одном участке); у соперника кандидаты-насыпи внутри
    лидарных городищ — курганы в 2 % случаев против 45 % снаружи (DATA_INSIGHTS EN, разобрано в
    docs/research/OSINT_2026-10-29.md). Безубыточность добавления — F/(4−F) ≈ 0,1–0,14, так что такой
    курган в ожидании отнимает балл. Шаг идёт после порогов: убирает только выданное.

    spec:
      classes      — классы курганов (по умолчанию оба);
      groups       — только эти группы порога курганов (пусто — все);
      min_gor_conf — городища слабее этой уверенности не маскируют (по умолчанию 0: всё выданное);
      gor_groups   — только городища этих групп (пусто — все);
      buffer_m     — расширить (+) или сжать (−) городище, метры на земле (по умолчанию 0);
      min_area_m2 / max_area_m2 — городища вне диапазона не маскируют (ядро-круг или ложная гигантская
                     маска); по умолчанию 500 и 400 000 м² (p90 эталона спутника ≈ 57 000, аэрофото ≈ 83 000).
    """
    spec = spec or {}
    classes = set(spec.get("classes") or KURGANS)
    groups = set(spec.get("groups") or ())
    gor_groups = set(spec.get("gor_groups") or ())
    min_conf = float(spec.get("min_gor_conf", 0.0))
    buf_m = float(spec.get("buffer_m", 0.0))
    lo, hi = float(spec.get("min_area_m2", 500.0)), float(spec.get("max_area_m2", 400_000.0))
    scale = _region_scales(feats)
    masks: dict[str, list] = defaultdict(list)
    for f in feats:
        if f.get("class_name") != "gorodishcha" or float(f.get("confidence", 1.0)) < min_conf:
            continue
        if gor_groups and _group_of(f) not in gor_groups:
            continue
        g = f.get("geom")
        if g is None or g.is_empty:
            continue
        s = scale.get(f.get("region_name") or "", 1.0)
        if not lo <= g.area * s * s <= hi:
            continue
        masks[f.get("region_name") or ""].append(g.buffer(buf_m / s) if buf_m else g)
    if not masks:
        return feats
    trees = {r: (STRtree(gs), gs) for r, gs in masks.items()}
    out = []
    for f in feats:
        reg = f.get("region_name") or ""
        if f.get("class_name") in classes and reg in trees and (not groups or _group_of(f) in groups):
            pt = f["geom"].representative_point()
            tree, gs = trees[reg]
            if any(gs[int(i)].covers(pt) for i in tree.query(pt)):
                continue
        out.append(f)
    return out

def sp_fusion_thresholds(by_markup: dict | None, spec) -> dict:
    """thresholds_by_markup с порогами курганов sp из [post.sp_score_fusion.thresholds] (sp_focal = sp_or в сдаче)."""
    out = {k: dict(v) for k, v in (by_markup or {}).items()}
    th = (spec or {}).get("thresholds", {}) if isinstance(spec, dict) else {}
    for g, cls in th.items():
        for key in (("sp_or", "sp_focal") if g == "sp_focal" else (g,)):
            out.setdefault(key, {}).update({c: float(t) for c, t in cls.items()})
    return out


def sp_fusion_gen_floors(floors: dict[str, float], spec) -> dict[str, float]:
    """Пол генерации курганов sp при включённом слиянии: сырые кандидаты от gen_floor (по умолчанию 0,05)."""
    g = float((spec if isinstance(spec, dict) else {}).get("gen_floor", 0.05))
    return {c: (min(t, g) if c in SP_FUSION_CLASSES else t) for c, t in floors.items()}


# --- Калибровка по группам + WBF-слияние разных групп (bot-2, п. 5 REPORT.md) ---
# Флаг выключен по умолчанию. Честная офлайн-проверка (LOFO по семьям, 7 вариантов)
# не прошла honest_accept: все Δ < 0 (см. koz3_shared/bot-2/REPORT2.md, раздел п. 5).
FUSION_CALIB_ENABLED = False


KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")


def _group_of(f: dict) -> str:
    return f.get("threshold_group") or f.get("markup_type") or ""


def calibrate_group_conf(feats: list[dict], calib: dict | None = None,
                         enabled: bool | None = None) -> list[dict]:
    """Пересчёт уверенности по (класс, группа).

    calib[(класс, группа)] = ("platt", a, b): c' = sigmoid(a*logit(c)+b)
                           | ("iso", xs, ys): кусочно-линейная интерполяция с отсечкой.
    Нет записи — уверенность не меняется. Исходная сохраняется в confidence_raw.
    """
    if not (FUSION_CALIB_ENABLED if enabled is None else enabled) or not calib:
        return feats
    import bisect
    out = []
    for f in feats:
        spec = calib.get((f["class_name"], _group_of(f)))
        c = float(f.get("confidence", 1.0))
        if spec is None:
            out.append(f); continue
        if spec[0] == "platt":
            x = min(max(c, 1e-6), 1 - 1e-6)
            z = spec[1] * math.log(x / (1 - x)) + spec[2]
            cn = 1.0 / (1.0 + math.exp(-z))
        else:
            xs, ys = spec[1], spec[2]
            if c <= xs[0]:
                cn = ys[0]
            elif c >= xs[-1]:
                cn = ys[-1]
            else:
                k = bisect.bisect_right(xs, c)
                x0, x1, y0, y1 = xs[k - 1], xs[k], ys[k - 1], ys[k]
                cn = y0 if x1 == x0 else y0 + (y1 - y0) * (c - x0) / (x1 - x0)
        out.append(dict(f, confidence=float(cn), confidence_raw=c))
    return out


def fuse_cross_group(feats: list[dict], how: str = "wbf",
                     enabled: bool | None = None) -> list[dict]:
    """WBF-подобное слияние ТОЛЬКО кластеров, где есть объекты из РАЗНЫХ групп.

    Внутри (регион, markup_type, класс): union-find по пересечению с площадью > 0
    между объектами разных групп. Остаётся лучший член (геометрия и группа его),
    уверенность: how="wbf" — mean_g(max c группы) * min(N,T)/T (N — групп в
    кластере, T — групп в (регион, markup, класс)); how="max" — max c.
    Кластеры одной группы и EtGr не трогаются.
    """
    if not (FUSION_CALIB_ENABLED if enabled is None else enabled):
        return feats
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, f in enumerate(feats):
        buckets[(f.get("region_name"), f.get("markup_type"), f["class_name"])].append(i)
    drop: set[int] = set()
    new_conf: dict[int, float] = {}
    for key, idx in buckets.items():
        if key[1] == "etgr" or len(idx) < 2:
            continue
        T = len({_group_of(feats[i]) for i in idx})
        if T < 2:
            continue
        geoms = [feats[i]["geom"] for i in idx]
        tree = STRtree(geoms)
        parent = list(range(len(idx)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, g in enumerate(geoms):
            ga = _group_of(feats[idx[a]])
            for b in tree.query(g):
                b = int(b)
                if b <= a or _group_of(feats[idx[b]]) == ga:
                    continue
                if g.intersection(geoms[b]).area > 0:
                    parent[find(a)] = find(b)
        comps: dict[int, list[int]] = defaultdict(list)
        for a in range(len(idx)):
            comps[find(a)].append(idx[a])
        for mem in comps.values():
            if len(mem) < 2:
                continue
            mem.sort(key=lambda i: (-float(feats[i].get("confidence", 1.0)), i))
            best = mem[0]
            if how == "max":
                val = float(feats[best].get("confidence", 1.0))
            else:
                gmax: dict[str, float] = {}
                for i in mem:
                    g = _group_of(feats[i])
                    gmax[g] = max(gmax.get(g, -1.0), float(feats[i].get("confidence", 1.0)))
                n = len(gmax)
                val = sum(gmax.values()) / n * min(n, T) / T
            new_conf[best] = val
            drop.update(mem[1:])
    return [dict(f, confidence=new_conf[i], fused=True) if i in new_conf else f
            for i, f in enumerate(feats) if i not in drop]


def _region_scales(feats: list[dict]) -> dict[str, float]:
    """Один cos φ на участок: метры на земле = единицы EPSG:3857 × масштаб (как в kurgan_cross_modal_twins)."""
    ys: dict[str, list[float]] = defaultdict(list)
    for f in feats:
        g = f.get("geom")
        if g is not None and not g.is_empty:
            ys[f.get("region_name") or ""].append(g.centroid.y)
    return {r: _ground_scale(sum(v) / len(v)) for r, v in ys.items()}


def kurgans_outside_hillforts(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """Курганы, чья внутренняя точка внутри выданного городища того же участка, убираются.

    Правило разметки организаторов: насыпи внутри городищ курганами не размечаются. На трейне
    внутри городищ 7 из 974 курганов (все на одном участке); у соперника кандидаты-насыпи внутри
    лидарных городищ — курганы в 2 % случаев против 45 % снаружи (DATA_INSIGHTS EN, разобрано в
    docs/research/OSINT_2026-10-29.md). Безубыточность добавления — F/(4−F) ≈ 0,1–0,14, так что такой
    курган в ожидании отнимает балл. Шаг идёт после порогов: убирает только выданное.

    spec:
      classes      — классы курганов (по умолчанию оба);
      groups       — только эти группы порога курганов (пусто — все);
      min_gor_conf — городища слабее этой уверенности не маскируют (по умолчанию 0: всё выданное);
      gor_groups   — только городища этих групп (пусто — все);
      buffer_m     — расширить (+) или сжать (−) городище, метры на земле (по умолчанию 0);
      min_area_m2 / max_area_m2 — городища вне диапазона не маскируют (ядро-круг или ложная гигантская
                     маска); по умолчанию 500 и 400 000 м² (p90 эталона спутника ≈ 57 000, аэрофото ≈ 83 000).
    """
    spec = spec or {}
    classes = set(spec.get("classes") or KURGANS)
    groups = set(spec.get("groups") or ())
    gor_groups = set(spec.get("gor_groups") or ())
    min_conf = float(spec.get("min_gor_conf", 0.0))
    buf_m = float(spec.get("buffer_m", 0.0))
    lo, hi = float(spec.get("min_area_m2", 500.0)), float(spec.get("max_area_m2", 400_000.0))
    scale = _region_scales(feats)
    masks: dict[str, list] = defaultdict(list)
    for f in feats:
        if f.get("class_name") != "gorodishcha" or float(f.get("confidence", 1.0)) < min_conf:
            continue
        if gor_groups and _group_of(f) not in gor_groups:
            continue
        g = f.get("geom")
        if g is None or g.is_empty:
            continue
        s = scale.get(f.get("region_name") or "", 1.0)
        if not lo <= g.area * s * s <= hi:
            continue
        masks[f.get("region_name") or ""].append(g.buffer(buf_m / s) if buf_m else g)
    if not masks:
        return feats
    trees = {r: (STRtree(gs), gs) for r, gs in masks.items()}
    out = []
    for f in feats:
        reg = f.get("region_name") or ""
        if f.get("class_name") in classes and reg in trees and (not groups or _group_of(f) in groups):
            pt = f["geom"].representative_point()
            tree, gs = trees[reg]
            if any(gs[int(i)].covers(pt) for i in tree.query(pt)):
                continue
        out.append(f)
    return out


def kurgan_neighbour_prior(feats: list[dict], spec: dict | None = None, tau=None) -> list[dict]:
    """Уверенность кургана × (floor + slope·clip(n / k, 0, cap)), n — число соседних курганов в радиусе.

    Курганы стоят могильниками: у соперника этот множитель на лидаре дал nested 0,5765 → 0,5831 и на LB
    recall +0,039 при precision −0,007 (COMPETITORS_DEEP §2.2). Одиночная «кочка» теряет уверенность,
    насыпь внутри цепочки её набирает. Шаг идёт ДО порогов: пороги групп из spec.groups нужно заново
    подобрать по OOF после того же преобразования (scripts/oof_transform.py neigh).

    Сосед — другое место кургана (оба класса в одной точке считаются одним местом) того же участка
    из групп count_groups (по умолчанию = groups), прошедшее опору: уверенность ≥ support_conf, если он
    задан, иначе ≥ порога своей группы (tau(feat) — функция порога, как в apply_thresholds).
    Расстояния — метры на земле, один cos φ на участок.

    spec: groups (по умолчанию ["li", "li_coarse", "li_mosaic"]), count_groups, radius_m (200),
    k (3), floor (0,5), slope (0,5), cap (2), support_conf (None), self_m (1,0 — ближе этого — то же место).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    spec = spec or {}
    groups = set(spec.get("groups") or ("li", "li_coarse", "li_mosaic"))
    count_groups = set(spec.get("count_groups") or groups)
    radius = float(spec.get("radius_m", 200.0))
    k = max(float(spec.get("k", 3.0)), 1e-9)
    floor, slope, cap = float(spec.get("floor", 0.5)), float(spec.get("slope", 0.5)), float(spec.get("cap", 2.0))
    self_m = float(spec.get("self_m", 1.0))
    support = spec.get("support_conf")
    if support is None and tau is None:
        raise ValueError("kurgan_neighbour_prior: нужен support_conf или функция порога tau")
    scale = _region_scales(feats)

    def ground_xy(f):
        c = f["geom"].centroid
        s = scale.get(f.get("region_name") or "", 1.0)
        return c.x * s, c.y * s

    sup: dict[str, list] = defaultdict(list)
    for f in feats:
        if f.get("class_name") not in KURGANS or _group_of(f) not in count_groups:
            continue
        thr = float(support) if support is not None else tau(f)
        if float(f.get("confidence", 0.0)) >= thr:
            sup[f.get("region_name") or ""].append(ground_xy(f))
    trees = {}
    for r, pts in sup.items():
        # одно место — одна опора: два класса в одной точке (и почти совпадающие копии) не считаются дважды
        a = np.round(np.asarray(pts, np.float64) / max(self_m, 1e-3))
        _, first = np.unique(a, axis=0, return_index=True)
        trees[r] = cKDTree(np.asarray(pts, np.float64)[np.sort(first)])
    out = []
    for f in feats:
        if f.get("class_name") not in KURGANS or _group_of(f) not in groups:
            out.append(f)
            continue
        reg = f.get("region_name") or ""
        n = 0
        if reg in trees:
            xy = ground_xy(f)
            near = trees[reg].query_ball_point(xy, radius)
            if near:
                d = np.hypot(*(trees[reg].data[near] - np.asarray(xy)).T)
                n = int((d > self_m).sum())
        factor = floor + slope * min(max(n / k, 0.0), cap)
        out.append(dict(f, confidence=float(f.get("confidence", 0.0)) * factor, neighbours=n,
                        conf_raw=float(f.get("confidence", 0.0))))
    return out


def scene_drel(pt, boxes) -> float | None:
    """Положение точки в кадре: max(|dx|/полуширина, |dy|/полувысота) от центра сцены, 0 — центр, 1 — край.

    Из нескольких сцен, накрывающих точку, берётся наименьшее значение; точка вне всех сцен — None.
    boxes — рамки сцен в той же CRS, что и точка (EPSG:3857: масштаб внутри одной сцены общий).
    """
    best = None
    for b in boxes:
        x0, y0, x1, y1 = b.bounds if hasattr(b, "bounds") else b
        hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        if hw <= 0 or hh <= 0 or not (x0 <= pt.x <= x1 and y0 <= pt.y <= y1):
            continue
        d = max(abs(pt.x - (x0 + hw)) / hw, abs(pt.y - (y0 + hh)) / hh)
        best = d if best is None else min(best, d)
    return best


def annotate_scene_drel(feats: list[dict], boxes_by_markup: dict[str, list]) -> list[dict]:
    """Поле scene_drel у объектов тех типов снимков, для которых даны рамки сцен участка."""
    for f in feats:
        boxes = boxes_by_markup.get(f.get("markup_type") or "")
        if boxes and f.get("geom") is not None and not f["geom"].is_empty:
            d = scene_drel(f["geom"].representative_point(), boxes)
            if d is not None:
                f["scene_drel"] = float(d)
    return feats


def frame_prior(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """Уверенность объекта у края крупной сцены снижается: метки есть только около памятника в центре кадра.

    Публичная выборка: 17 из 30 спутниковых участков — крупные сцены (57 % против 15 % в трейне). У соперника
    precision спутника падает от центра к углам с 0,38 до 0,05, а приор кадра дал спутниковый OOF 0,279 → 0,297
    в 4/4 фолдах (OSINT 29.10 §1.2). Шаг идёт ДО порогов: пороги групп из spec.groups подбирать заново по OOF
    после scripts/oof_transform.py frame. Множитель не больше 1, поэтому ранний пол выдачи не мешает.

    spec: groups (по умолчанию ["sp_or", "sp_focal", "sp_focal2"]), classes (пусто — все), d0 (0,5 — до него
    множитель 1), edge (0,5 — множитель на краю кадра, линейно от d0). Объект без scene_drel не меняется.
    """
    spec = spec or {}
    groups = set(spec.get("groups") or ("sp_or", "sp_focal", "sp_focal2"))
    classes = set(spec.get("classes") or ())
    d0 = min(max(float(spec.get("d0", 0.5)), 0.0), 0.999)
    edge = min(max(float(spec.get("edge", 0.5)), 0.0), 1.0)
    out = []
    for f in feats:
        d = f.get("scene_drel")
        if d is None or _group_of(f) not in groups or (classes and f.get("class_name") not in classes) or d <= d0:
            out.append(f)
            continue
        m = 1.0 - (1.0 - edge) * min((float(d) - d0) / (1.0 - d0), 1.0)
        out.append(dict(f, confidence=float(f.get("confidence", 0.0)) * m))
    return out


CALIB_MODES = ("rank", "quantile", "mix")


def site_calibration(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """Калибровка уверенности по участку без эталона: шкала каждой (участок, группа, класс) — по её же объектам.

    Одна сеть на разных участках откалибрована по-разному: ConvNeXt лидара (bot-9 CONVNEXT.md §2) имеет лучший
    порог повреждённых 0,329 на одной семье и 0,064 на другой; честный порог группы, снятый на одной семье,
    дал на другой 5142 FP против 678 у базы. Геофизика решает то же рангом внутри карты (geophys.rank_within_source).
    Шаг идёт ДО порогов; пороги групп из spec.groups подбирать заново после scripts/oof_transform.py calib.

    Режимы (s — уверенность, n — объектов участка с s ≥ floor):
      rank     — r = доля объектов участка со s не выше (0…1]: шкала карты не важна, только порядок;
      quantile — s / q, q — квантиль q_level распределения участка (обрезка до 1): поднимает недоуверенный
                 участок и прижимает переуверенный хвост;
      mix      — s^(1−alpha) · r^alpha: участок без объектов (весь шум с низким s) не выходит наверх по рангу.
    spec: groups (обязательно), classes (по умолчанию курганы), mode ("mix"), alpha (0,5), q_level (0,99),
    floor (0,05: объекты ниже не калибруются и не входят в распределение — тот же набор, что в OOF),
    min_n (30: на участке с меньшим числом объектов уверенность не меняется).
    """
    import numpy as np

    spec = spec or {}
    groups = set(spec.get("groups") or ())
    if not groups:
        raise ValueError("site_calibration: нужен список groups")
    classes = set(spec.get("classes") or KURGANS)
    mode = spec.get("mode", "mix")
    if mode not in CALIB_MODES:
        raise ValueError(f"site_calibration: неизвестный режим {mode}")
    alpha, q_level = float(spec.get("alpha", 0.5)), float(spec.get("q_level", 0.99))
    floor, min_n = float(spec.get("floor", 0.05)), int(spec.get("min_n", 30))
    sites: dict[tuple, list[int]] = defaultdict(list)
    for i, f in enumerate(feats):
        if (f.get("class_name") in classes and _group_of(f) in groups
                and float(f.get("confidence", 0.0)) >= floor):
            sites[(f.get("region_name") or "", _group_of(f), f["class_name"])].append(i)
    out = list(feats)
    for idx in sites.values():
        if len(idx) < min_n:
            continue
        s = np.array([float(feats[i]["confidence"]) for i in idx])
        r = np.searchsorted(np.sort(s), s, side="right") / len(s)
        if mode == "rank":
            new = r
        elif mode == "quantile":
            new = np.minimum(s / max(float(np.quantile(s, q_level)), 1e-6), 1.0)
        else:
            new = s ** (1.0 - alpha) * r ** alpha
        for i, v in zip(idx, new):
            out[i] = dict(feats[i], confidence=float(v), conf_raw=float(feats[i]["confidence"]))
    return out
