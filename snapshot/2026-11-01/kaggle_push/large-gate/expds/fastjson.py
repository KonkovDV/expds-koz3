"""Быстрое чтение и фильтрация больших GeoJSON (w0, ускорение узких мест на боксе).

* load_json(path)  — orjson, если установлен (≈1.5× быстрее json на 300 МБ OOF), иначе json.
  Результат равен json.loads (проверено ==); BOM (utf-8-sig) срезается.
  Выключить: EXPDS_FASTJSON=0.
* filter_conf(src, dst, thr) — потоковый фильтр по confidence (score, иначе 1.0) ≥ thr.
  Rust `expds_core.filter_geojson_conf` (≈7.8× быстрее ijson на 3.27 ГБ ae-OOF: 60.5 → 7.8 с,
  те же 173 038 объектов, json-равенство по объектам); без модуля / EXPDS_BACKEND=numpy /
  EXPDS_FASTJSON=0 — прежний ijson-путь (как cpu/filter_oof.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_BOM = b"\xef\xbb\xbf"


def _fast_on() -> bool:
    return os.environ.get("EXPDS_FASTJSON", "1") != "0"


def load_json(path) -> dict:
    raw = Path(path).read_bytes()
    if raw.startswith(_BOM):
        raw = raw[3:]
    if _fast_on():
        try:
            import orjson  # type: ignore
            return orjson.loads(raw)
        except ImportError:
            pass
    return json.loads(raw)


def _rust():
    if os.environ.get("EXPDS_BACKEND", "auto").lower() == "numpy" or not _fast_on():
        return None
    try:
        import expds_core  # type: ignore
        return expds_core if hasattr(expds_core, "filter_geojson_conf") else None
    except ImportError:
        return None


def filter_conf(src, dst, thr: float) -> tuple[int, int]:
    """(всего, оставлено). Выход — FeatureCollection в EPSG:3857 (как cpu/filter_oof.py)."""
    rs = _rust()
    if rs is not None:
        return tuple(rs.filter_geojson_conf(str(src), str(dst), float(thr)))
    import ijson  # type: ignore

    out, n = [], 0
    with open(src, "rb") as f:
        for feat in ijson.items(f, "features.item", use_float=True):
            n += 1
            p = feat.get("properties") or {}
            c = p.get("confidence", p.get("score", 1.0))
            if c >= thr:
                out.append(feat)
    with open(dst, "w") as f:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
                   "features": out}, f)
    return n, len(out)


if __name__ == "__main__":
    import sys
    import time

    t = time.time()
    n, k = filter_conf(sys.argv[1], sys.argv[2], float(sys.argv[3]))
    print(f"done {n} {k} {time.time() - t:.1f}s backend={'rust' if _rust() else 'ijson'}")
