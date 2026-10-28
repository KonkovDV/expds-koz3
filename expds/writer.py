"""Запись и проверка выходного GeoJSON по требованиям ТР КОЗ №3.

Формат: FeatureCollection, EPSG:3857, geometry только Polygon,
properties.class_name — одно из 7 значений; confidence, region_name,
markup_type, fid — рекомендуемые поля.

    python -m expds.writer check result.geojson
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

from shapely.geometry import mapping, shape

from .constants import CLASSES, MARKUP_TYPES, MAX_FEATURES, MAX_FILE_MB, MAX_VERTICES


def to_feature_collection(feats: list[dict]) -> dict:
    """Объекты пайплайна → FeatureCollection (координаты округлены до 0,01 м)."""
    out = []
    for f in feats:
        geom = mapping(f["geom"])
        geom["coordinates"] = [[[round(x, 2), round(y, 2)] for x, y, *_ in ring] for ring in geom["coordinates"]]
        if not all(math.isfinite(v) for ring in geom["coordinates"] for xy in ring for v in xy):
            continue                    # bot-8: NaN/inf дают невалидный JSON — объект отбрасывается
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


#: Номер записи в процессе: сторож и основной поток не делят временный файл.
_TMP_SEQ = itertools.count()


def tmp_path_for(path: Path) -> Path:
    """Временный файл рядом с ответом.

    Имя не содержит «result» и не кончается на .geojson: площадка может забрать
    первый появившийся result.geojson, в том числе по префиксу.
    """
    return path.parent / f"writing.{os.getpid()}.{next(_TMP_SEQ)}.tmp"


def write_geojson(feats: list[dict], path: str | Path) -> Path:
    """Атомарная запись: временный файл, fsync, затем переименование."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path_for(path)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(to_feature_collection(feats), ensure_ascii=False, allow_nan=False))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def validate(fc: dict | str | Path) -> dict:
    """Проверяет файл или словарь; возвращает {'ok': bool, 'errors': [...], 'warnings': [...], 'summary': {...}}."""
    errors, warnings = [], []
    size_mb = None
    if not isinstance(fc, dict):
        size_mb = Path(fc).stat().st_size / 2 ** 20
        fc = json.loads(Path(fc).read_text(encoding="utf-8"))
        if size_mb > MAX_FILE_MB:
            errors.append(f"файл {size_mb:.1f} МБ > {MAX_FILE_MB} МБ")
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
    if small_x:
        warnings.append(f"{small_x} объектов с |x| ≤ 1e6: метрика не пересчитает их из EPSG:3857 в UTM "
                        "(для России так быть не должно — проверьте CRS)")
    return {"ok": not errors, "errors": errors[:50], "warnings": warnings[:50],
            "summary": {"features": len(feats), "by_class": by_class,
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
