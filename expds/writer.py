"""Запись и проверка выходного GeoJSON по требованиям ТР КОЗ №3.

Формат: FeatureCollection, EPSG:3857, geometry только Polygon,
properties.class_name — одно из 7 значений; confidence, region_name,
markup_type, fid — рекомендуемые поля.

    python -m expds.writer check result.geojson
"""
from __future__ import annotations

import argparse
import itertools
import logging
import json
import math
import os
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping, shape

from .constants import CLASSES, MARKUP_TYPES, MAX_FEATURES, MAX_FILE_MB, MAX_VERTICES

log = logging.getLogger(__name__)


def _polygon_coords(geoms: list) -> list | None:
    """Координаты колец всех полигонов одним вызовом GEOS, округлённые как раньше (round Python).

    То же, что mapping(g)["coordinates"] с округлением, но без кортежей shapely на каждую
    вершину. None — есть не Polygon или пустая геометрия: тогда старый путь.
    """
    import shapely

    if not geoms:
        return []
    arr = np.asarray(geoms, dtype=object)
    if not (shapely.get_type_id(arr) == 3).all() or shapely.is_empty(arr).any():
        return None
    _, coords, (ring_off, poly_off) = shapely.to_ragged_array(arr, include_z=False)
    flat = [round(v, 2) for v in coords.ravel().tolist()]
    pts = [flat[i:i + 2] for i in range(0, len(flat), 2)]
    rings = [pts[a:b] for a, b in zip(ring_off[:-1].tolist(), ring_off[1:].tolist())]
    return [rings[a:b] for a, b in zip(poly_off[:-1].tolist(), poly_off[1:].tolist())]


def to_feature_collection(feats: list[dict]) -> dict:
    """Объекты пайплайна → FeatureCollection (координаты округлены до 0,01 м)."""
    out = []
    # speedup-2026-10-28 (харвест 29.10): координаты всех полигонов одним вызовом GEOS — по флагу
    # fast_writer ([speed], по умолчанию ВЫКЛ.); побайтно тот же JSON (tests/test_writer_fast.py).
    # Сверка 29.10: быстрый путь берётся, только если кольцо после округления до 2 знаков валидно;
    # иначе — общий путь с _rounded_coords (Прил. 3 ТР).
    from . import speed

    fast = _polygon_coords([f["geom"] for f in feats]) if speed.on("fast_writer") else None
    for k, f in enumerate(feats):
        coords = None
        if fast is not None and _ring_ok(fast[k]):
            geom = {"type": "Polygon", "coordinates": fast[k]}
            coords = fast[k]
        else:
            geom = mapping(f["geom"])
        if coords is None:
            if not all(math.isfinite(v) for ring in geom["coordinates"] for xy in ring for v in xy[:2]):
                continue                # bot-8: NaN/inf дают невалидный JSON — объект отбрасывается
            coords = _rounded_coords(geom["coordinates"])
            if coords is None:
                log.warning("объект %s/%s отброшен: геометрия невалидна и после округления",
                            f.get("region_name"), f.get("class_name"))
                continue                # Прил. 3 ТР: самопересечение — замечание валидатора; > 50 — отказ
        geom["coordinates"] = coords
        props = {"fid": len(out), "class_name": f["class_name"],
                 "confidence": round(min(1.0, max(0.0, float(f.get("confidence", 1.0)))), 5),
                 "region_name": f.get("region_name", "default_source"),
                 "crs": "EPSG:3857"}
        if f.get("markup_type"):
            props["markup_type"] = f["markup_type"]
        if f.get("original_crs"):
            props["original_crs"] = f["original_crs"]
        out.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "features": out}


#: Знаки после запятой: 2 (0,01 м) — как раньше; больше — только если округление ломает кольцо.
#: Прил. 3 ТР: не более 15 знаков после запятой, иначе замечание валидатора.
_ROUND_DIGITS = (2, 4, 6)


def _ring_ok(rings: list) -> bool:
    """Кольца уже округлены: конечные координаты, валидный полигон с площадью > 0."""
    if not all(math.isfinite(v) for ring in rings for xy in ring for v in xy[:2]):
        return False
    try:
        poly = Polygon(rings[0], rings[1:])
    except Exception:
        return False
    return bool(poly.is_valid and poly.area > 0)


def _rounded_coords(rings: list) -> list | None:
    """Кольца с координатами, округлёнными до 2 знаков, если полигон после этого валиден.

    Округление до 0,01 м может сделать мелкий полигон самопересекающимся или нулевой
    площади (замечания валидатора Прил. 3). Тогда берётся 4, затем 6 знаков; если
    невалиден и исходник — None (объект не пишется).
    """
    for nd in _ROUND_DIGITS:
        out = [[[round(x, nd), round(y, nd)] for x, y, *_ in ring] for ring in rings]
        try:
            poly = Polygon(out[0], out[1:])
        except Exception:
            continue
        if poly.is_valid and poly.area > 0:
            return out
    return None


#: Номер записи в процессе: сторож и основной поток не делят временный файл.
_TMP_SEQ = itertools.count()


def tmp_path_for(path: Path) -> Path:
    """Временный файл рядом с ответом.

    Имя не содержит «result» и не кончается на .geojson: площадка может забрать
    первый появившийся result.geojson, в том числе по префиксу.
    """
    return path.parent / f"writing.{os.getpid()}.{next(_TMP_SEQ)}.tmp"


def enforce_limits(fc: dict) -> dict:
    """Последний рубеж перед записью (bot-7): только Polygon, ≤ MAX_VERTICES вершин, ≤ MAX_FEATURES объектов.

    В норме ничего не делает: post.finalize/limit_vertices и post.cap_features уже держат лимиты.
    Срабатывает, если в обход них прошло лишнее (запасная оболочка limit_vertices, новый шаг после cap).
    Лишнее отбрасывается с предупреждением в лог, ответ остаётся допустимым для метрики.
    """
    feats = fc.get("features", [])
    good = []
    bad_type = too_many_vertices = 0
    for f in feats:
        geom = f.get("geometry") or {}
        if geom.get("type") != "Polygon":
            bad_type += 1
            continue
        if sum(len(ring) for ring in geom["coordinates"]) > MAX_VERTICES:
            too_many_vertices += 1
            continue
        good.append(f)
    over = max(0, len(good) - MAX_FEATURES)
    if over:   # cap_features уже оставил квоты по классам; здесь просто срез хвоста
        good = good[:MAX_FEATURES]
    if bad_type or too_many_vertices or over:
        log.warning("лимиты выдачи: отброшено не-Polygon %d, вершин > %d: %d, сверх %d объектов: %d",
                    bad_type, MAX_VERTICES, too_many_vertices, MAX_FEATURES, over)
        for k, f in enumerate(good):
            f["properties"]["fid"] = k
    return {"type": "FeatureCollection", "features": good}


#: Прил. 3 ТР: файл > 100 МБ отклоняется. Бюджет — 97 % от 100·10⁶ байт (строже и 2²⁰-трактовки).
SIZE_BUDGET_BYTES = int(MAX_FILE_MB * 1_000_000 * 0.97)


def _dumps(feats: list[dict]) -> str:
    return json.dumps(enforce_limits(to_feature_collection(feats)), ensure_ascii=False, allow_nan=False)


def _fit_size(feats: list[dict], budget: int | None = None) -> str:
    """Текст GeoJSON не больше бюджета байт.

    В норме (файл в разы меньше лимита) возвращает тот же текст, что и раньше.
    Если больше — объектов становится на 10 % меньше за шаг по тем же квотам классов,
    что у post.cap_features (внутри класса остаются самые уверенные).
    """
    budget = SIZE_BUDGET_BYTES if budget is None else budget
    text = _dumps(feats)
    n = len(feats)
    while len(text.encode("utf-8")) > budget and n > 0:
        from .post import cap_features   # post не импортирует writer: цикла нет

        n = int(n * 0.9)
        feats = cap_features(feats, n) if n else []
        text = _dumps(feats)
        log.warning("лимит размера: файл > %d байт, оставляю %d объектов", budget, len(feats))
    return text


def write_geojson(feats: list[dict], path: str | Path) -> Path:
    """Атомарная запись: временный файл, fsync, затем переименование."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path_for(path)
    try:
        text = _fit_size(feats)
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


#: Прил. 3 ТР: больше 50 замечаний валидатора геометрии — файл отклоняется.
TR_MAX_REMARKS = 50
#: Прил. 3 ТР: площадь полигона должна быть > 0 и не больше 1e12 м²; |координата| ≤ 1e8.
TR_MAX_AREA = 1e12
TR_MAX_ABS_COORD = 1e8
_MANY_DECIMALS = None


def tr_geometry_remarks(fc: dict, raw_text: str | None = None) -> list[str]:
    """Замечания валидатора геометрии по списку Прил. 3 ТР (по одному на объект и вид).

    Незамкнутое кольцо; кольцо длиннее MAX_VERTICES точек; самопересечение/невалидность;
    площадь ≤ 0 или > 1e12; координата вне ±1e8, NaN/Inf; точка (0, 0);
    больше 15 знаков после запятой (по тексту файла, если он передан).
    Площадь берётся в единицах файла (EPSG:3857 завышает м² — проверка строже).
    """
    global _MANY_DECIMALS
    import re

    out: list[str] = []
    if raw_text is not None:
        if _MANY_DECIMALS is None:
            _MANY_DECIMALS = re.compile(r"\d\.\d{16,}")
        n = len(_MANY_DECIMALS.findall(raw_text))
        if n:
            out.append(f"чисел с > 15 знаками после запятой: {n}")
    for i, f in enumerate(fc.get("features", [])):
        geom = (f or {}).get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        rings = geom.get("coordinates") or []
        bad_xy = False
        for ring in rings:
            if not ring or ring[0] != ring[-1]:
                out.append(f"#{i}: незамкнутое кольцо")
            if len(ring) > MAX_VERTICES:
                out.append(f"#{i}: кольцо {len(ring)} точек > {MAX_VERTICES}")
            for xy in ring:
                x, y = float(xy[0]), float(xy[1])
                if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > TR_MAX_ABS_COORD or abs(y) > TR_MAX_ABS_COORD:
                    bad_xy = True
                elif x == 0.0 and y == 0.0:
                    bad_xy = True
        if bad_xy:
            out.append(f"#{i}: координата вне ±1e8, NaN/Inf или (0, 0)")
            continue
        try:
            g = shape(geom)
        except Exception:
            out.append(f"#{i}: геометрия не читается")
            continue
        if not g.is_valid:
            out.append(f"#{i}: самопересечение или невалидный полигон")
        if not (0.0 < g.area <= TR_MAX_AREA):
            out.append(f"#{i}: площадь {g.area:.3g} вне (0, 1e12]")
    return out


def validate(fc: dict | str | Path) -> dict:
    """Проверяет файл или словарь; возвращает {'ok': bool, 'errors': [...], 'warnings': [...], 'summary': {...}}."""
    errors, warnings = [], []
    size_mb = None
    raw_text = None
    if not isinstance(fc, dict):
        size_bytes = Path(fc).stat().st_size
        size_mb = size_bytes / 2 ** 20
        raw_text = Path(fc).read_text(encoding="utf-8")
        fc = json.loads(raw_text)
        if size_bytes > MAX_FILE_MB * 1_000_000:      # строже: 100·10⁶ байт, а не 100·2²⁰
            errors.append(f"файл {size_bytes / 1e6:.1f} МБ (10⁶ байт) > {MAX_FILE_MB} МБ")
    if fc.get("type") != "FeatureCollection":
        errors.append("type должен быть FeatureCollection")
    feats = fc.get("features", [])
    if len(feats) > MAX_FEATURES:
        errors.append(f"объектов {len(feats)} > {MAX_FEATURES}")
    by_class: dict[str, int] = {}
    small_x = 0
    for i, f in enumerate(feats):
        props = f.get("properties") or {}
        cls = props.get("class_name")
        if cls not in CLASSES:
            errors.append(f"#{i}: неизвестный class_name {cls!r}")
            continue
        by_class[cls] = by_class.get(cls, 0) + 1
        geom = f.get("geometry") or {}
        if geom.get("type") != "Polygon":
            errors.append(f"#{i}: geometry {geom.get('type')!r}, нужен Polygon")
            continue
        try:
            g = shape(geom)
        except Exception as exc:
            errors.append(f"#{i}: геометрия не читается ({exc})")
            continue
        n = len(g.exterior.coords) + sum(len(r.coords) for r in g.interiors)
        if n > MAX_VERTICES:
            errors.append(f"#{i}: вершин {n} > {MAX_VERTICES}")
        if not g.is_valid or g.is_empty:
            warnings.append(f"#{i}: невалидная или пустая геометрия")
        x = g.centroid.x
        if not math.isfinite(x):
            errors.append(f"#{i}: нечисловые координаты")
        elif abs(x) <= 1_000_000:
            small_x += 1
        conf = props.get("confidence")
        if conf is not None and not (0.0 <= float(conf) <= 1.0):
            errors.append(f"#{i}: confidence {conf} вне [0, 1]")
        mt = props.get("markup_type")
        if mt is not None and mt not in MARKUP_TYPES:
            warnings.append(f"#{i}: markup_type {mt!r} вне списка ТР")
    remarks = tr_geometry_remarks(fc, raw_text)
    if len(remarks) > TR_MAX_REMARKS:
        errors.append(f"замечаний валидатора геометрии {len(remarks)} > {TR_MAX_REMARKS} (Прил. 3 ТР: файл отклоняется)")
    elif remarks:
        warnings.append(f"замечаний валидатора геометрии {len(remarks)} (≤ {TR_MAX_REMARKS} допускается)")
    warnings += remarks[:20]
    if small_x:
        warnings.append(f"{small_x} объектов с |x| ≤ 1e6: метрика не пересчитает их из EPSG:3857 в UTM "
                        "(для России так быть не должно — проверьте CRS)")
    return {"ok": not errors, "errors": errors[:50], "warnings": warnings[:50],
            "summary": {"features": len(feats), "by_class": by_class, "tr_remarks": len(remarks),
                        "size_mb": None if size_mb is None else round(size_mb, 2)}}


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Проверка выходного GeoJSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("path")
    args = ap.parse_args()
    rep = validate(args.path)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    _cli()
