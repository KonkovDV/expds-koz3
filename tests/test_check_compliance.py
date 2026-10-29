"""Синтетические zip/geojson для scripts/check_compliance.py (bot-8)."""
from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "b8_check_compliance", ROOT / "scripts" / "check_compliance.py"
)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def _poly(x0: float, y0: float, w: float = 50.0) -> dict:
    # WebMercator-ish coords for Russia (~4e6, 7e6)
    ring = [
        [x0, y0], [x0 + w, y0], [x0 + w, y0 + w], [x0, y0 + w], [x0, y0],
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "fid": 0,
            "class_name": "kurgany_tselye",
            "confidence": 0.9,
            "region_name": "t",
            "crs": "EPSG:3857",
            "markup_type": "sp_or",
        },
    }


def _fc(feats: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": feats}


def _zip_bytes(fc: dict, name: str = "result.geojson") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, json.dumps(fc, ensure_ascii=False))
    return buf.getvalue()


def test_ok_geojson(tmp_path: Path):
    p = tmp_path / "result.geojson"
    p.write_text(json.dumps(_fc([_poly(4_200_000.0, 7_500_000.0)])), encoding="utf-8")
    fc, raw, size = cc._load(p)
    rep = cc.check(fc, raw, size)
    assert rep["ok"], rep["errors"]
    assert rep["summary"]["features"] == 1
    assert rep["summary"]["polygon_only"] is True


def test_ok_zip(tmp_path: Path):
    p = tmp_path / "submission.zip"
    p.write_bytes(_zip_bytes(_fc([_poly(4_200_000.0, 7_500_000.0)])))
    assert cc.main([str(p)]) == 0


def test_fail_too_many_features(tmp_path: Path):
    feats = [_poly(4_200_000.0 + i, 7_500_000.0) for i in range(cc.MAX_FEATURES + 1)]
    # перенумеруем fid
    for i, f in enumerate(feats):
        f["properties"]["fid"] = i
    p = tmp_path / "big.geojson"
    p.write_text(json.dumps(_fc(feats)), encoding="utf-8")
    fc, raw, size = cc._load(p)
    rep = cc.check(fc, raw, size)
    assert not rep["ok"]
    assert any("объектов" in e for e in rep["errors"])


def test_fail_multipolygon(tmp_path: Path):
    feat = _poly(4_200_000.0, 7_500_000.0)
    feat["geometry"] = {
        "type": "MultiPolygon",
        "coordinates": [feat["geometry"]["coordinates"]],
    }
    p = tmp_path / "mp.geojson"
    p.write_text(json.dumps(_fc([feat])), encoding="utf-8")
    assert cc.main([str(p)]) == 1


def test_fail_not_3857(tmp_path: Path):
    # UTM-like small coords
    p = tmp_path / "utm.geojson"
    p.write_text(json.dumps(_fc([_poly(500_000.0, 6_100_000.0, w=10.0)])), encoding="utf-8")
    fc, raw, size = cc._load(p)
    rep = cc.check(fc, raw, size)
    assert not rep["ok"]
    assert any("EPSG:3857" in e or "1e6" in e for e in rep["errors"])


def test_fail_too_many_vertices(tmp_path: Path):
    # кольцо > 3000 точек
    n = cc.MAX_VERTICES + 5
    ring = [[4_200_000.0 + i * 0.01, 7_500_000.0] for i in range(n - 1)]
    ring.append(ring[0])
    feat = _poly(4_200_000.0, 7_500_000.0)
    feat["geometry"]["coordinates"] = [ring]
    p = tmp_path / "fat.geojson"
    p.write_text(json.dumps(_fc([feat])), encoding="utf-8")
    fc, raw, size = cc._load(p)
    rep = cc.check(fc, raw, size)
    assert not rep["ok"]
    assert any("вершин" in e or "кольцо" in e for e in rep["errors"] + rep["warnings"])


def test_fail_zip_oversize(tmp_path: Path, monkeypatch):
    p = tmp_path / "huge.zip"
    p.write_bytes(_zip_bytes(_fc([_poly(4_200_000.0, 7_500_000.0)])))
    monkeypatch.setattr(cc, "MAX_FILE_BYTES", 10)  # 10 байт
    fc, raw, size = cc._load(p)
    rep = cc.check(fc, raw, size)
    assert not rep["ok"]
    assert any("МБ" in e for e in rep["errors"])


def test_cli_missing_file(tmp_path: Path):
    assert cc.main([str(tmp_path / "nope.zip")]) == 2
