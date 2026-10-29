"""Геометрия предсказаний под правило метрики koz3-1.5.0 (bot-7, 2026-10-28).

Правило попадания полигона P в эталон G:
    cov = |P∩G| / min(|P|, |G|) ≥ 0,5   И   P.representative_point() ∈ G (или на границе).
Отсюда (см. bot-7/REPORT.md, work/mc_shapes.json):
* P(попадание) ≤ P(якорь ∈ G), где якорь — representative_point P;
* маленький диск (r ≪ размера эталона) достигает этой границы: внутри G cov = 1,
  на прямой границе — ровно 0,5 при якоре на границе. Огромный диск — тоже (cov = 1 к G);
* хуже всего фигура размера эталона: при сдвиге центра cov падает ниже 0,5 раньше,
  чем якорь выходит из G; копия контура эталона со сдвигом — худший случай;
* форма (круг/эллипс/квадрат) при малом размере не важна; вытянутая хуже при r ~ R.
Поэтому генератор: «диск-якорь» малого радиуса в лучшей оценке центра, плюс
разделение (несколько якорей на крупный кандидат) и склейка (подавление дублей по якорю).

Объекты — словари пайплайна: {"geom": shapely в EPSG:3857, "class_name", "confidence",
"threshold_group", "markup_type", "region_name", ...}. Все радиусы — метры на земле.
По умолчанию ничего не включено: spec без класса — объект не меняется.
"""
from __future__ import annotations

import math
from collections import defaultdict

from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import polylabel, unary_union
from shapely.strtree import STRtree

from .post import _ground_scale, polygons_of


def _largest(geom):
    parts = polygons_of(geom)
    return max(parts, key=lambda p: p.area) if parts else None


def anchor_of(geom, mode: str = "rp") -> Point | None:
    """Якорь кандидата: rp — representative_point самой большой части (его и проверяет метрика),
    polylabel — полюс недоступности, centroid — центр масс, если он внутри, иначе rp."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Point":
        return geom
    part = _largest(geom)
    if part is None:
        return None
    if mode == "polylabel":
        tol = max(0.01, math.sqrt(part.area) / 200.0)
        return polylabel(part, tolerance=tol)
    if mode == "centroid":
        c = part.centroid
        return c if part.covers(c) else part.representative_point()
    return part.representative_point()


def _rule(spec, feat):
    rule = (spec or {}).get(feat.get("class_name"))
    if rule is None:
        return None
    grp = feat.get("threshold_group") or feat.get("markup_type")
    if rule.get("groups") and grp not in rule["groups"]:
        return None
    return rule


def reshape(feats: list[dict], spec: dict) -> list[dict]:
    """Замена геометрии на диск в якоре. spec = {класс: {r_m, anchor, groups, r_frac, r_max_m}}.

    r_m — радиус в метрах; r_frac > 0 — радиус = r_frac · радиус вписанной окружности части
    (не больше r_max_m), так диск гарантированно лежит внутри исходного кандидата.
    """
    out = []
    for f in feats:
        rule = _rule(spec, f)
        if rule is None:
            out.append(f)
            continue
        a = anchor_of(f["geom"], rule.get("anchor", "rp"))
        if a is None:
            out.append(f)
            continue
        s = _ground_scale(a.y)
        r_m = float(rule.get("r_m", 0.1))
        if float(rule.get("r_frac", 0.0)) > 0 and f["geom"].geom_type != "Point":
            part = _largest(f["geom"])
            r_in = part.exterior.distance(a) * s if part is not None else r_m
            r_m = max(r_m, min(float(rule["r_frac"]) * r_in, float(rule.get("r_max_m", 1e9))))
        out.append(dict(f, geom=a.buffer(r_m / s, quad_segs=int(rule.get("quad_segs", 4)))))
    return out


def split_poles(feats: list[dict], spec: dict) -> list[dict]:
    """Разделение: крупный кандидат → до max_k дисков-якорей (полюса с вырезанием, как
    post.multipole_object_poly, но для любого класса). spec = {класс: {min_area_m2, max_k,
    min_lobe_m2, min_sep_m, decay, r_m, groups, keep_parent}}.
    Первый якорь — с уверенностью родителя, k-й — conf·decay^(k-1)."""
    out = []
    for f in feats:
        rule = _rule(spec, f)
        g = f.get("geom")
        if rule is None or g is None or g.geom_type == "Point":
            out.append(f)
            continue
        part = _largest(g if g.is_valid else g.buffer(0))
        if part is None:
            out.append(f)
            continue
        s = _ground_scale(part.centroid.y)
        if part.area * s * s < float(rule.get("min_area_m2", 0.0)):
            out.append(f)
            continue
        r_m = float(rule.get("r_m", 0.1))
        max_k, min_lobe = int(rule.get("max_k", 3)), float(rule.get("min_lobe_m2", 3.0))
        min_sep, decay = float(rule.get("min_sep_m", 1.0)), float(rule.get("decay", 0.6))
        tol = max(0.01, math.sqrt(part.area) / 200.0)
        p = polylabel(part, tolerance=tol)
        poles, rem = [p], part
        conf = float(f.get("confidence", 1.0))
        if rule.get("keep_parent"):
            out.append(f)
        else:
            out.append(dict(f, geom=p.buffer(r_m / s, quad_segs=4)))
        for k in range(2, max_k + 1):
            rem = rem.difference(poles[-1].buffer(max(rem.boundary.distance(poles[-1]), 1.5 / s), quad_segs=8))
            lobes = [q for q in polygons_of(rem) if q.area * s * s >= min_lobe]
            if not lobes:
                break
            rem = max(lobes, key=lambda q: q.area)
            pk = polylabel(rem, tolerance=max(0.01, math.sqrt(rem.area) / 200.0))
            if all(pk.distance(e) * s >= min_sep for e in poles):
                out.append(dict(f, geom=pk.buffer(r_m / s, quad_segs=4), confidence=round(conf * decay ** (k - 1), 6)))
            poles.append(pk)
    return out


def dedup_anchors(feats: list[dict], spec: dict) -> list[dict]:
    """Склейка без объединения контуров: в (регион, класс) по убыванию уверенности объект
    выбрасывается, если его якорь ближе radius_m к якорю уже оставленного или лежит внутри
    его геометрии (inside=True). cross_group=False — только внутри своей группы порога.
    spec = {класс: {radius_m, inside, within: all|group|markup, groups}}.
    Эталон содержит один объект в нескольких слоях markup_type (work/gt_dups.txt), поэтому
    склеивать безопасно только внутри одного markup_type.

    Почему не unary_union: у объединения якорь может уйти в промежуток между объектами, а
    одно предсказание всё равно закрывает не больше одного эталона."""
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, f in enumerate(feats):
        rule = _rule(spec, f)
        if rule is None:
            continue
        key = (f.get("region_name"), f.get("class_name"))
        within = rule.get("within", "all" if rule.get("cross_group", True) else "group")
        if within == "group":
            key += (f.get("threshold_group") or f.get("markup_type"),)
        elif within == "markup":   # эталон размечен по модальностям: дубль li/ae — это два разных эталона
            key += (f.get("markup_type"),)
        buckets[key].append(i)
    drop: set[int] = set()
    for key, idx in buckets.items():
        rule = spec[key[1]]
        rad, inside = float(rule.get("radius_m", 1.0)), bool(rule.get("inside", False))
        idx.sort(key=lambda i: -float(feats[i].get("confidence", 1.0)))
        cell = max(rad, 1e-3)
        grid: dict[tuple, list[int]] = defaultdict(list)   # сетка в метрах: ячейка = radius_m
        kept_pts, kept_geoms = [], []
        for i in idx:
            a = anchor_of(feats[i]["geom"], "rp")
            if a is None:
                continue
            s = _ground_scale(a.y)
            x, y = a.x * s, a.y * s
            cx, cy = int(math.floor(x / cell)), int(math.floor(y / cell))
            dup = any(math.hypot(kept_pts[j][0] - x, kept_pts[j][1] - y) < rad
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in grid.get((cx + dx, cy + dy), ()))
            if not dup and inside:
                dup = any(g.covers(a) for g in kept_geoms if g.bounds[0] <= a.x <= g.bounds[2]
                          and g.bounds[1] <= a.y <= g.bounds[3])
            if dup:
                drop.add(i)
            else:
                grid[(cx, cy)].append(len(kept_pts))
                kept_pts.append((x, y))
                kept_geoms.append(feats[i]["geom"])
    return [f for i, f in enumerate(feats) if i not in drop]


def merge_union(feats: list[dict], spec: dict) -> list[dict]:
    """Классическая склейка для сравнения: пересекающиеся объекты класса (все группы) →
    unary_union, уверенность — максимум, группа — от лучшего. Затем, если r_m задан,
    объединение заменяется диском в его якоре."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    out = []
    for i, f in enumerate(feats):
        if _rule(spec, f) is None:
            out.append(f)
        else:
            groups[(f.get("region_name"), f.get("class_name"))].append(i)
    for key, idx in groups.items():
        geoms = [feats[i]["geom"] for i in idx]
        tree = STRtree(geoms)
        parent = list(range(len(idx)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a, g in enumerate(geoms):
            for b in tree.query(g):
                b = int(b)
                if b > a and g.intersection(geoms[b]).area > 0:
                    parent[find(a)] = find(b)
        comp = defaultdict(list)
        for a in range(len(idx)):
            comp[find(a)].append(a)
        rule = spec[key[1]]
        for members in comp.values():
            best = max(members, key=lambda a: float(feats[idx[a]].get("confidence", 1.0)))
            u = unary_union([geoms[a] for a in members]) if len(members) > 1 else geoms[best]
            nf = dict(feats[idx[best]], geom=_largest(u) or u)
            out.append(nf)
    if any("r_m" in r for r in (spec or {}).values()):
        out = reshape(out, {c: r for c, r in spec.items() if "r_m" in r})
    return out


def twins(feats: list[dict], spec: dict) -> list[dict]:
    """Двойники: в эталоне есть дубли (один объект размечен дважды, IoU>0.5 — work/gt_dups.txt),
    а сопоставление один-к-одному, поэтому копия предсказания может закрыть второй эталон.
    Копия получает собственную группу порога "<группа>~twin" (порог для неё подбирается
    sweep_groups отдельно, честно по фолдам), уверенность conf*decay.
    spec = {класс: {decay=1.0, groups}}."""
    out = list(feats)
    for f in feats:
        rule = _rule(spec, f)
        if rule is None:
            continue
        grp = f.get("threshold_group") or f.get("markup_type") or "none"
        out.append(dict(f, threshold_group=f"{grp}~twin",
                        confidence=round(float(f.get("confidence", 1.0)) * float(rule.get("decay", 1.0)), 6)))
    return out


def optimize(feats: list[dict], cfg: dict) -> list[dict]:
    """Порядок: разделение → форма → склейка. cfg = {"split": {...}, "reshape": {...}, "dedup": {...}}."""
    if cfg.get("split"):
        feats = split_poles(feats, cfg["split"])
    if cfg.get("reshape"):
        feats = reshape(feats, cfg["reshape"])
    if cfg.get("merge_union"):
        feats = merge_union(feats, cfg["merge_union"])
    if cfg.get("dedup"):
        feats = dedup_anchors(feats, cfg["dedup"])
    if cfg.get("twins"):
        feats = twins(feats, cfg["twins"])
    return feats


# Рекомендация bot-7 (REPORT.md). PRE — до порогов (меняет только форму, кривые/пороги
# надо подбирать уже на преобразованном стеке), POST — после apply_thresholds.
_KUR_DISC = {"r_m": 0.5, "anchor": "centroid"}
RECOMMENDED_PRE = {"reshape": {"kurgany_povrezhdennye": dict(_KUR_DISC), "kurgany_tselye": dict(_KUR_DISC)}}
RECOMMENDED_PRE_GOR = {"reshape": {"gorodishcha": {"r_m": 0.5, "anchor": "polylabel", "r_frac": 0.5}}}  # спорно, см. REPORT
RECOMMENDED_POST = {"split": {"object_poly": {"min_area_m2": 12, "max_k": 3, "min_lobe_m2": 3,
                                              "min_sep_m": 0.8, "decay": 0.6, "r_m": 0.05}}}
