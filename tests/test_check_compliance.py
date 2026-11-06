"""Синтетические zip/geojson для scripts/check_compliance.py."""
from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "check_compliance.py"
# На чужих ветках/частичных копиях скрипта может не быть — не роняем collection.
if not _SCRIPT.is_file():
    pytest.skip(
        "scripts/check_compliance.py отсутствует на этой ветке (bot-8 only)",
        allow_module_level=True,
    )
_spec = importlib.util.spec_from_file_location("b8_check_compliance", _SCRIPT)
if _spec is None or _spec.loader is None:
    pytest.skip(
        f"не удалось загрузить {_SCRIPT}",
        allow_module_level=True,
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


def test_reject_kit_zip_with_weights(tmp_path: Path):
    """LESSONS §5: kit with .pt must FAIL loudly (exit 2), not parse as submission."""
    p = tmp_path / "submission_bot7cand_v3on_kit.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("models/ae_fort_mask_r18.pt", b"FAKE_WEIGHTS" * 100)
        zf.writestr("inference/main.py", b"print(1)\n")
    p.write_bytes(buf.getvalue())
    reason = cc.looks_like_kit_zip(p)
    assert reason and "KIT-ZIP" in reason
    assert cc.main([str(p)]) == 2


def test_reject_kit_zip_models_dir_no_result(tmp_path: Path):
    p = tmp_path / "kit_no_result.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("models/readme.txt", b"no weights ext but models tree")
        zf.writestr("configs/default.toml", b"x=1\n")
    p.write_bytes(buf.getvalue())
    reason = cc.looks_like_kit_zip(p)
    assert reason and "KIT-ZIP" in reason
    assert cc.main([str(p)]) == 2


def test_ok_submission_zip_not_kit(tmp_path: Path):
    p = tmp_path / "submission_bot7cand_v2_eb268d6.zip"
    p.write_bytes(_zip_bytes(_fc([_poly(4_200_000.0, 7_500_000.0)])))
    assert cc.looks_like_kit_zip(p) is None
    assert cc.main([str(p)]) == 0


def test_geojson_never_kit(tmp_path: Path):
    p = tmp_path / "result.geojson"
    p.write_text(json.dumps(_fc([_poly(4_200_000.0, 7_500_000.0)])), encoding="utf-8")
    assert cc.looks_like_kit_zip(p) is None


def test_kit_zip_no_geojson(tmp_path: Path):
    """LESSONS §5: kit-zip (code/models) must FAIL loudly — not a submission."""
    p = tmp_path / "kit.zip"
    import zipfile
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("inference/main.py", "print(1)\n")
        zf.writestr("expds/pipeline.py", "x=1\n")
        zf.writestr("configs/default.toml", "[time]\n")
        zf.writestr("models/foo.pt", b"\0" * 16)
        zf.writestr("README.md", "# kit\n")
    with pytest.raises(ValueError, match="kit-zip|нет .geojson"):
        cc._load(p)




def test_validate_submission_v5_wrapper(tmp_path: Path):
    wrap = ROOT / "scripts" / "validate_submission_v5.py"
    assert wrap.is_file()
    spec = importlib.util.spec_from_file_location("validate_submission_v5", wrap)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = tmp_path / "result.geojson"
    p.write_text(__import__("json").dumps(_fc([_poly(4_200_000.0, 7_500_000.0)])), encoding="utf-8")
    assert mod.main([str(p)]) == 0
