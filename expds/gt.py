"""Сборка эталона трейна в формате валидационной выборки и статистика.

Эталон: один FeatureCollection в EPSG:3857 с полями class_name, region_name,
markup_type, source (файл разметки). Пересекающиеся объекты одного класса
из РАЗНЫХ файлов одного markup_type объединяются (правило организаторов для
выхода; так же, вероятно, собран эталон валидации — флаг --no-merge отключает).
Исключение — EtGr: по письму организаторов от 20.10 каждый срез и профиль —
отдельный эталон, как и в post.merge_within_markup.

Статистика отвечает на вопрос «куда вкладываться»: число объектов по
классам и модальностям и доля объектов, размеченных в двух модальностях.

    python -m expds.gt --root D:/expds3 --out work/gt_train.geojson --stats work/gt_stats
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

from .catalog import Region, read_utm_json, scan_root
from .constants import CLASSES
from .crs import WEB_MERCATOR, guess_geojson_crs, to_crs, transformer, utm_for_lonlat
from .naming import code_is_cyrillic

log = logging.getLogger(__name__)


def _polygons(geom: BaseGeometry) -> list[BaseGeometry]:
    """Разбивает на Polygon/Point; линии и пустые части отбрасываются."""
    if geom.is_empty:
        return []
    t = geom.geom_type
    if t in ("Polygon", "Point"):
        return [geom]
    if t in ("MultiPolygon", "MultiPoint", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(_polygons(g))
        return out
    return []


def read_annotation(path: Path, cls: str, markup_type: str, region: Region,
                    report: list[dict] | None = None) -> list[dict]:
    """Объекты одного файла разметки в EPSG:3857.

    report (если передан) получает строку диагностики файла: какая CRS выбрана,
    сколько объектов, типы геометрий, что отброшено и почему.
    """
    rep = {"file": str(path), "region": region.name, "class_name": cls, "markup_type": markup_type,
           "cyrillic_code": code_is_cyrillic(path),
           "crs": None, "features": 0, "objects": 0, "lines_dropped": 0, "invalid_fixed": 0,
           "geom_types": {}, "problems": []}
    if report is not None:
        report.append(rep)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        rep["problems"].append(f"не читается: {type(exc).__name__}")
        return []
    feats = data.get("features", []) or []
    rep["features"] = len(feats)
    if not feats:
        rep["problems"].append("нет объектов")
        return []
    src = guess_geojson_crs(data, read_utm_json(region.utm_json))
    rep["crs"] = src
    if src is None:
        rep["problems"].append("координаты локальные/пиксельные или нет UTM.json — файл пропущен")
        log.warning("%s: CRS не определена (локальные координаты?) — пропуск", path.name)
        return []
    if rep["cyrillic_code"]:
        rep["problems"].append("код модальности набран кириллицей (учтён)")
    out = []
    for feat in feats:
        if not feat.get("geometry"):
            continue
        try:
            geom = shape(feat["geometry"])
            rep["geom_types"][geom.geom_type] = rep["geom_types"].get(geom.geom_type, 0) + 1
            if geom.geom_type not in ("Point", "MultiPoint") and not geom.is_valid:
                geom = make_valid(geom)
                rep["invalid_fixed"] += 1
            geom = to_crs(geom, src, WEB_MERCATOR)
        except Exception as exc:
            log.warning("%s: объект пропущен (%s)", path.name, exc)
            continue
        parts = _polygons(geom)
        if not parts:
            rep["lines_dropped"] += 1
        for g in parts:
            out.append({"geom": g, "class_name": cls, "markup_type": markup_type,
                        "region_name": region.name, "source": path.name, "src_crs": src})
    rep["objects"] = len(out)
    if rep["lines_dropped"]:
        rep["problems"].append(f"{rep['lines_dropped']} линий/прочих геометрий: метрика их не видит")
    return out




#: Слои, объекты которых из разных файлов не объединяются (срезы и профили электроразведки).
UNMERGED_MARKUP = frozenset({"etgr"})


def merge_across_sources(items: list[dict]) -> tuple[list[dict], int]:
    """Объединяет пересекающиеся полигоны из разных источников (файлов/снимков).

    items — объекты одной группы (регион, markup_type, класс). Полигоны
    одного источника между собой не сливаются: соседние курганы на одном
    снимке остаются отдельными объектами. Возвращает (объекты, число слияний).
    """
    polys = [i for i in items if i["geom"].geom_type == "Polygon"]
    others = [i for i in items if i["geom"].geom_type != "Polygon"]
    if len(polys) < 2:
        return items, 0
    tree = STRtree([p["geom"] for p in polys])
    parent = list(range(len(polys)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, p in enumerate(polys):
        for j in tree.query(p["geom"]):
            j = int(j)
            if j <= i or polys[j]["source"] == p["source"]:
                continue
            if p["geom"].intersection(polys[j]["geom"]).area > 0:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = defaultdict(list)
    for i, p in enumerate(polys):
        groups[find(i)].append(p)
    merged, merges = [], 0
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        merges += len(members) - 1
        geom = unary_union([m["geom"] for m in members])
        for g in _polygons(geom):
            merged.append(dict(members[0], geom=g, source="+".join(sorted({m["source"] for m in members}))))
    return merged + others, merges


def build_gt(regions: list[Region], merge: bool = True,
             report: list[dict] | None = None) -> tuple[list[dict], dict]:
    """Эталон по всем регионам, счётчик слияний (всего и по markup_type) и отчёт по файлам."""
    out: list[dict] = []
    merges_total = 0
    merges_by_mt: dict[str, int] = defaultdict(int)
    for reg in regions:
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for path, cls, mt in reg.annotations:
            for item in read_annotation(path, cls, mt, reg, report):
                groups[(mt, cls)].append(item)
        for (mt, _), items in groups.items():
            if merge and mt not in UNMERGED_MARKUP:
                items, m = merge_across_sources(items)
                merges_total += m
                merges_by_mt[mt] += m
            out.extend(items)
    return out, {"merges": merges_total, "merges_by_markup_type": dict(merges_by_mt)}


def stats(items: list[dict], dup_radius_m: float = 5.0) -> dict:
    """Число объектов по классам и модальностям и доля дублей между модальностями."""
    count = defaultdict(int)
    regions = defaultdict(set)
    for it in items:
        key = (it["class_name"], it["markup_type"])
        count[key] += 1
        regions[key].add(it["region_name"])
    # Дубли: объекты одного класса в одном регионе из разных модальностей ближе dup_radius_m.
    by_rc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        by_rc[(it["region_name"], it["class_name"])].append(it)
    dup = defaultdict(int)
    for (_, cls), its in by_rc.items():
        by_mt: dict[str, list[dict]] = defaultdict(list)
        for i in its:
            by_mt[i["markup_type"]].append(i)
        if len(by_mt) < 2:
            continue
        lon, lat = transformer(WEB_MERCATOR, "EPSG:4326").transform(*its[0]["geom"].representative_point().coords[0])
        tr = transformer(WEB_MERCATOR, utm_for_lonlat(lon, lat))
        pts = {mt: np.array([tr.transform(*i["geom"].representative_point().coords[0]) for i in lst])
               for mt, lst in by_mt.items()}
        mts = sorted(pts)
        for ai, a in enumerate(mts):
            for b in mts[ai + 1:]:
                pairs = cKDTree(pts[a]).query_ball_tree(cKDTree(pts[b]), dup_radius_m)
                dup[(cls, a, b)] += sum(1 for p in pairs if p)
    # Один курган — разные классы (целый/повреждённый): обычно в разных модальностях
    # (на аэрофото 1940-х целый, на современном лидаре распахан).
    conflicts = defaultdict(int)
    by_r: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if it["class_name"] in ("kurgany_tselye", "kurgany_povrezhdennye"):
            by_r[it["region_name"]].append(it)
    for its in by_r.values():
        tsel = [i for i in its if i["class_name"] == "kurgany_tselye"]
        povr = [i for i in its if i["class_name"] == "kurgany_povrezhdennye"]
        if not tsel or not povr:
            continue
        lon, lat = transformer(WEB_MERCATOR, "EPSG:4326").transform(*its[0]["geom"].representative_point().coords[0])
        tr = transformer(WEB_MERCATOR, utm_for_lonlat(lon, lat))
        pa = np.array([tr.transform(*i["geom"].representative_point().coords[0]) for i in tsel])
        pb = np.array([tr.transform(*i["geom"].representative_point().coords[0]) for i in povr])
        for ia, near in enumerate(cKDTree(pa).query_ball_tree(cKDTree(pb), dup_radius_m)):
            for ib in near:
                conflicts[tuple(sorted((tsel[ia]["markup_type"], povr[ib]["markup_type"])))] += 1
    # Размеры объектов в метрах (площадь по UTM своей зоны).
    sizes: dict[tuple[str, str], list[float]] = defaultdict(list)
    points = defaultdict(int)
    for it in items:
        key = (it["class_name"], it["markup_type"])
        g = it["geom"]
        if g.geom_type == "Point":
            points[key] += 1
            continue
        c = g.representative_point()
        lon, lat = transformer(WEB_MERCATOR, "EPSG:4326").transform(c.x, c.y)
        sizes[key].append(float(to_crs(g, WEB_MERCATOR, utm_for_lonlat(lon, lat)).area))

    def q(v: list[float], p: float) -> float | None:
        return round(float(np.percentile(v, p)), 1) if v else None

    return {
        "objects": [{"class_name": c, "markup_type": m, "objects": n, "regions": len(regions[(c, m)]),
                     "points": points[(c, m)],
                     "area_m2_p10": q(sizes[(c, m)], 10), "area_m2_p50": q(sizes[(c, m)], 50),
                     "area_m2_p90": q(sizes[(c, m)], 90)}
                    for (c, m), n in sorted(count.items(), key=lambda kv: (CLASSES.index(kv[0][0]), -kv[1]))],
        "cross_modality_pairs": [{"class_name": c, "a": a, "b": b, "pairs": n}
                                 for (c, a, b), n in sorted(dup.items())],
        "kurgan_class_conflicts": [{"a": a, "b": b, "pairs": n} for (a, b), n in sorted(conflicts.items())],
    }


def to_feature_collection(items: list[dict]) -> dict:
    """Объекты → GeoJSON FeatureCollection (EPSG:3857)."""
    feats = []
    for k, it in enumerate(items):
        feats.append({"type": "Feature", "geometry": mapping(it["geom"]),
                      "properties": {"fid": k, "class_name": it["class_name"], "region_name": it["region_name"],
                                     "markup_type": it["markup_type"], "source": it["source"]}})
    return {"type": "FeatureCollection", "name": "gt", "features": feats}


def _md(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |\n"
                          for r in rows)


def render_summary(st: dict, report: list[dict], anonymous: bool) -> str:
    """Сводка эталона в markdown. anonymous=True — без имён участков и путей (можно пересылать)."""
    out = ["# Эталон обучающей выборки: сводка\n",
           (f"Регионов: {st['regions']}; файлов разметки: {len(report)}; объектов: "
            f"{sum(r['objects'] for r in st['objects'])}; слияний внутри markup_type: {st['merges']} "
            f"({st.get('merges_by_markup_type', {})}).\n"),
           "\n## Объекты по классам и модальностям\n",
           _md(st["objects"], ["class_name", "markup_type", "objects", "regions", "points",
                               "area_m2_p10", "area_m2_p50", "area_m2_p90"]),
           "\n## Дубли между модальностями (центры ближе 5 м)\n",
           _md(st["cross_modality_pairs"], ["class_name", "a", "b", "pairs"]),
           "\n## Один курган — разные классы (целый/повреждённый)\n",
           _md(st["kurgan_class_conflicts"], ["a", "b", "pairs"])]
    probs: dict[str, int] = defaultdict(int)
    for r in report:
        for p in r["problems"]:
            probs[p.split(":")[0].split("(")[0].strip()] += 1
    crs_src: dict[str, int] = defaultdict(int)
    for r in report:
        crs_src[str(r["crs"])] += 1
    out += ["\n## CRS файлов разметки\n", _md([{"crs": k, "files": v} for k, v in sorted(crs_src.items())],
                                                ["crs", "files"]),
            "\n## Проблемы\n", _md([{"problem": k, "files": v} for k, v in sorted(probs.items())],
                                    ["problem", "files"])]
    if not anonymous:
        out += ["\n## Файлы с проблемами\n",
                _md([{"file": r["file"], "problems": "; ".join(r["problems"])} for r in report if r["problems"]],
                    ["file", "problems"])]
    return "".join(out)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Сборка эталона трейна и статистика")
    ap.add_argument("--root", required=True, help="корень обучающей выборки")
    ap.add_argument("--out", required=True, help="выходной GeoJSON эталона")
    ap.add_argument("--stats", default=None,
                    help="префикс файлов статистики: .csv, .json, _files.csv, _summary.md, _share.md")
    ap.add_argument("--only", nargs="*", default=None, help="номера или имена регионов")
    ap.add_argument("--no-merge", action="store_true", help="не объединять объекты разных файлов")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    regions = scan_root(args.root)
    if args.only:
        regions = [r for r in regions if any(r.name.startswith(o) or r.name == o for o in args.only)]
    report: list[dict] = []
    items, info = build_gt(regions, merge=not args.no_merge, report=report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(to_feature_collection(items), ensure_ascii=False), encoding="utf-8")
    st = stats(items)
    st.update(info)
    st["regions"] = len(regions)
    st["skipped_files"] = sum(len(r.skipped) for r in regions)
    st["files_with_problems"] = sum(1 for r in report if r["problems"])
    if args.stats:
        prefix = Path(args.stats)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
        with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["class_name", "markup_type", "objects", "regions", "points",
                        "area_m2_p10", "area_m2_p50", "area_m2_p90"])
            for row in st["objects"]:
                w.writerow([row[k] for k in ("class_name", "markup_type", "objects", "regions", "points",
                                             "area_m2_p10", "area_m2_p50", "area_m2_p90")])
        with open(f"{prefix}_files.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            cols = ["file", "region", "class_name", "markup_type", "cyrillic_code", "crs", "features",
                    "objects", "lines_dropped", "invalid_fixed", "geom_types", "problems"]
            w.writerow(cols)
            for r in report:
                w.writerow([r[c] if c != "problems" else " | ".join(r[c]) for c in cols])
        Path(f"{prefix}_summary.md").write_text(render_summary(st, report, anonymous=False), encoding="utf-8")
        Path(f"{prefix}_share.md").write_text(render_summary(st, report, anonymous=True), encoding="utf-8")
    print(json.dumps({k: st[k] for k in ("regions", "merges", "skipped_files", "files_with_problems")},
                     ensure_ascii=False))
    for row in st["objects"]:
        print(f'{row["class_name"]:24s} {row["markup_type"]:6s} {row["objects"]:7d}  регионов {row["regions"]}')


if __name__ == "__main__":
    _cli()
