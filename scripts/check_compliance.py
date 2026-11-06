#!/usr/bin/env python3
"""Проверка ФИНАЛЬНОГО ответа КОЗ №3 (result.geojson или zip с ним) —.

LESSONS §5: validate the submission output scored on LB, NEVER a code/weights kit zip.
LB scores e.g. submission_bot7cand_v2_eb268d6.zip → result.geojson (FeatureCollection).

Лимиты ТР / compute_metrics / Прил. 3:
  zip/файл ≤ 100 МБ; ≤ 20 000 объектов; ≤ 3000 вершин на объект;
  EPSG:3857 (по координатам / properties.crs); только Polygon;
  ≤ 50 замечаний валидатора геометрии.

Kit-zip REJECT (exit 2): members with model weights (.pt/.pth/.onnx/.safetensors)
or size >100 MB together with such members / a models/ tree. Loud FAIL message.

    python scripts/check_compliance.py path/to/submission.zip
    python scripts/check_compliance.py path/to/result.geojson

Код возврата: 0 — ok; 1 — есть FAIL; 2 — ошибка аргументов / чтения / kit-zip.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path

MAX_FEATURES = 20_000
MAX_VERTICES = 3_000
MAX_FILE_MB = 100.0
MAX_FILE_BYTES = int(MAX_FILE_MB * 1_000_000)  # 100·10⁶, как на платформе
TR_MAX_REMARKS = 50
TR_MAX_AREA = 1e12
TR_MAX_ABS_COORD = 1e8
CLASSES = (
    "kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury",
    "object_poly", "finds_points", "fortifikatsii",
)
_MANY_DECIMALS = re.compile(r"\d\.\d{16,}")
WEIGHT_SUFFIXES = (".pt", ".pth", ".onnx", ".safetensors", ".bin", ".ckpt")
KIT_NAME_HINTS = ("v3on", "kit", "models/", "weights/")


def looks_like_kit_zip(path: Path) -> str | None:
    """Return reason string if path is a code/weights kit, else None.

    Loud reject for LESSONS §5: compliance must target submission output, not kit.
    """
    size = path.stat().st_size
    name_l = path.name.lower()
    # GeoJSON submission is never a kit
    if path.suffix.lower() in (".geojson", ".json"):
        return None
    if path.suffix.lower() != ".zip":
        return None
    weight_hits: list[str] = []
    has_models_dir = False
    has_result = False
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            for n in names:
                nl = n.lower()
                if nl.endswith("result.geojson") or Path(nl).name == "result.geojson":
                    has_result = True
                if "/models/" in nl or nl.startswith("models/"):
                    has_models_dir = True
                if nl.endswith(WEIGHT_SUFFIXES):
                    weight_hits.append(n)
                    if len(weight_hits) >= 5:
                        break
    except zipfile.BadZipFile as exc:
        return f"не читается как zip: {exc}"
    if weight_hits:
        sample = ", ".join(weight_hits[:3])
        return (
            f"KIT-ZIP REJECT: в архиве веса модели ({sample}"
            f"{'…' if len(weight_hits) > 3 else ''}). "
            f"Нужен submission output (result.geojson / zip ≤100 МБ без .pt/.onnx), "
            f"не kit. См. LESSONS §5. path={path}"
        )
    if has_models_dir and not has_result:
        return (
            f"KIT-ZIP REJECT: есть models/ без result.geojson. "
            f"Нужен финальный ответ пайплайна. path={path}"
        )
    if size > MAX_FILE_BYTES and (has_models_dir or any(h in name_l for h in KIT_NAME_HINTS)):
        return (
            f"KIT-ZIP REJECT: размер {size/1e6:.1f} МБ > {MAX_FILE_MB} МБ "
            f"и признаки kit (models/ или имя). path={path}"
        )
    return None



def _load(path: Path) -> tuple[dict, str, int]:
    """→ (FeatureCollection, raw_text, size_bytes_of_payload)."""
    size = path.stat().st_size
    if path.suffix.lower() == ".zip":
        if size > MAX_FILE_BYTES:
            # всё равно попробуем прочитать для диагностики
            pass
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/") and not n.startswith("__MACOSX")]
            geo = [n for n in names if n.lower().endswith(".geojson") or n.lower().endswith("result.json")]
            if not geo:
                # одиночный .json с FeatureCollection
                geo = [n for n in names if n.lower().endswith(".json")]
            if not geo:
                raise ValueError(f"в zip нет .geojson (файлы: {names[:10]})")
            # предпочитаем result.geojson
            prefer = [n for n in geo if Path(n).name.lower() == "result.geojson"]
            name = prefer[0] if prefer else geo[0]
            raw = zf.read(name).decode("utf-8")
            payload_size = size  # лимит ТР — на весь zip ответа
    else:
        raw = path.read_text(encoding="utf-8")
        payload_size = size
    fc = json.loads(raw)
    return fc, raw, payload_size


def geometry_remarks(fc: dict, raw_text: str) -> list[str]:
    """Замечания Прил. 3 ТР (по одному на объект и вид) — без зависимости от writer main."""
    out: list[str] = []
    n_dec = len(_MANY_DECIMALS.findall(raw_text))
    if n_dec:
        out.append(f"чисел с > 15 знаками после запятой: {n_dec}")
    try:
        from shapely.geometry import shape
    except ImportError:
        shape = None
    for i, f in enumerate(fc.get("features") or []):
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
        if shape is None:
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


def check(fc: dict, raw: str, size_bytes: int) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    size_mb_si = size_bytes / 1_000_000
    if size_bytes > MAX_FILE_BYTES:
        errors.append(f"размер {size_mb_si:.1f} МБ (10⁶) > {MAX_FILE_MB} МБ")

    if fc.get("type") != "FeatureCollection":
        errors.append("type должен быть FeatureCollection")
    feats = fc.get("features") or []
    if len(feats) > MAX_FEATURES:
        errors.append(f"объектов {len(feats)} > {MAX_FEATURES}")

    small_x = 0
    non_poly = 0
    max_vertices = 0
    by_class: dict[str, int] = {}
    try:
        from shapely.geometry import shape
    except ImportError:
        shape = None

    for i, f in enumerate(feats):
        props = (f or {}).get("properties") or {}
        cls = props.get("class_name")
        if cls is not None and cls not in CLASSES:
            errors.append(f"#{i}: неизвестный class_name {cls!r}")
        elif cls in CLASSES:
            by_class[cls] = by_class.get(cls, 0) + 1
        crs = props.get("crs")
        crs_norm = str(crs).upper().replace(" ", "") if crs is not None else ""
        if crs_norm not in ("EPSG:3857", "3857"):
            errors.append(f"#{i}: properties.crs {crs!r}, нужен EPSG:3857")
        geom = (f or {}).get("geometry") or {}
        gtype = geom.get("type")
        if gtype != "Polygon":
            non_poly += 1
            errors.append(f"#{i}: geometry {gtype!r}, нужен Polygon")
            continue
        if shape is None:
            # без shapely — считаем вершины по кольцам
            n = sum(len(ring) for ring in (geom.get("coordinates") or []))
            max_vertices = max(max_vertices, n)
            if n > MAX_VERTICES:
                errors.append(f"#{i}: вершин {n} > {MAX_VERTICES}")
            continue
        try:
            g = shape(geom)
        except Exception as exc:
            errors.append(f"#{i}: геометрия не читается ({exc})")
            continue
        n = len(g.exterior.coords) + sum(len(r.coords) for r in g.interiors)
        max_vertices = max(max_vertices, n)
        if n > MAX_VERTICES:
            errors.append(f"#{i}: вершин {n} > {MAX_VERTICES}")
        x = g.centroid.x
        if not math.isfinite(x):
            errors.append(f"#{i}: нечисловые координаты")
        elif abs(x) <= 1_000_000:
            small_x += 1

    if non_poly == 0 and feats:
        # явная отметка Polygon-only
        pass
    if small_x:
        warnings.append(
            f"{small_x} объектов с |x| ≤ 1e6: похоже не EPSG:3857 "
            "(метрика не пересчитает такие координаты в UTM)"
        )
        if small_x == len(feats):
            errors.append("все объекты с |x| ≤ 1e6 — CRS не EPSG:3857")

    remarks = geometry_remarks(fc, raw)
    if len(remarks) > TR_MAX_REMARKS:
        errors.append(
            f"замечаний валидатора геометрии {len(remarks)} > {TR_MAX_REMARKS} (Прил. 3 ТР)"
        )
    elif remarks:
        warnings.append(f"замечаний валидатора геометрии {len(remarks)} (≤ {TR_MAX_REMARKS})")
    warnings.extend(remarks[:20])

    crs_ok = small_x == 0 and not any("crs" in e.lower() for e in errors)
    checks = {
        "size": size_bytes <= MAX_FILE_BYTES,
        "features": len(feats) <= MAX_FEATURES,
        "vertices": max_vertices <= MAX_VERTICES,
        "remarks": len(remarks) <= TR_MAX_REMARKS,
        "crs": crs_ok,
        "polygon": non_poly == 0,
    }
    return {
        "ok": not errors,
        "errors": errors[:50],
        "warnings": warnings[:50],
        "summary": {
            "features": len(feats),
            "by_class": by_class,
            "tr_remarks": len(remarks),
            "max_vertices": max_vertices,
            "size_mb": round(size_mb_si, 3),
            "polygon_only": non_poly == 0,
            "epsg_3857_ok": crs_ok,
            "checks": checks,
        },
    }


def format_report(rep: dict) -> str:
    """Шесть пунктов ТР отдельными строками. Итог PASS, только если все пункты PASS."""
    s = rep["summary"]
    ch = s["checks"]
    rows = [
        ("size", ch["size"], f"{s['size_mb']} МБ ≤ {MAX_FILE_MB:g}"),
        ("features", ch["features"], f"{s['features']} ≤ {MAX_FEATURES}"),
        ("vertices", ch["vertices"], f"макс. {s['max_vertices']} ≤ {MAX_VERTICES}"),
        ("remarks", ch["remarks"], f"{s['tr_remarks']} ≤ {TR_MAX_REMARKS}"),
        ("crs", ch["crs"], "EPSG:3857"),
        ("polygon", ch["polygon"], "только Polygon"),
    ]
    lines = [f"[{'PASS' if ok else 'FAIL'}] {name:10} {detail}" for name, ok, detail in rows]
    lines.append("ИТОГ " + ("PASS" if rep["ok"] and all(ch.values()) else "FAIL"))
    for e in rep["errors"]:
        lines.append(f"  ERROR: {e}")
    for w in rep["warnings"][:10]:
        lines.append(f"  WARN: {w}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Проверка финального ответа КОЗ №3")
    ap.add_argument("path", help="submission.zip или result.geojson")
    ap.add_argument("--json", action="store_true", help="печатать полный отчёт JSON")
    args = ap.parse_args(argv)
    path = Path(args.path)
    if not path.exists():
        print(f"FAIL path: нет файла {path}", file=sys.stderr)
        return 2
    kit_reason = looks_like_kit_zip(path)
    if kit_reason:
        print(f"FAIL kit: {kit_reason}", file=sys.stderr)
        return 2
    try:
        fc, raw, size = _load(path)
    except Exception as exc:
        print(f"FAIL read: {exc}", file=sys.stderr)
        return 2
    rep = check(fc, raw, size)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print(format_report(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    # Windows host cp1251 crash on Cyrillic/unicode in PASS/FAIL lines (lease 35ca3df).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
