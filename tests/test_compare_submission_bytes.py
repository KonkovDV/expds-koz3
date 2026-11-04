"""Byte-compare helper for OFF==v2 gate (geojson + submission zip refs)."""
from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_submission_bytes.py"
spec = importlib.util.spec_from_file_location("compare_submission_bytes", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _fc(extra: str = "") -> dict:
    ring = [
        [4_200_000.0, 7_500_000.0],
        [4_200_050.0, 7_500_000.0],
        [4_200_050.0, 7_500_050.0],
        [4_200_000.0, 7_500_050.0],
        [4_200_000.0, 7_500_000.0],
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "fid": 0,
                    "class_name": "kurgany_tselye",
                    "confidence": 0.9,
                    "region_name": "t" + extra,
                    "crs": "EPSG:3857",
                    "markup_type": "sp_or",
                },
            }
        ],
    }


def _write_geojson(path: Path, fc: dict) -> Path:
    path.write_text(json.dumps(fc, separators=(",", ":")), encoding="utf-8")
    return path


def _write_zip(path: Path, fc: dict, member: str = "result.geojson") -> Path:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, json.dumps(fc, separators=(",", ":")))
    path.write_bytes(buf.getvalue())
    return path


def test_identical_geojson(tmp_path: Path):
    a = _write_geojson(tmp_path / "a.geojson", _fc())
    b = _write_geojson(tmp_path / "b.geojson", _fc())
    assert mod.compare(a, b) == []
    assert mod.main([str(a), str(b)]) == 0


def test_mismatch_geojson(tmp_path: Path):
    a = _write_geojson(tmp_path / "a.geojson", _fc())
    b = _write_geojson(tmp_path / "b.geojson", _fc(extra="X"))
    errs = mod.compare(a, b)
    assert errs
    assert any("sha256" in e for e in errs)
    assert mod.main([str(a), str(b)]) == 1


def test_zip_ref_vs_geojson_cand(tmp_path: Path):
    fc = _fc()
    ref = _write_zip(tmp_path / "ref.zip", fc)
    cand = _write_geojson(tmp_path / "cand.geojson", fc)
    errs, meta = mod.compare_paths(ref, cand)
    assert errs == []
    assert meta["ref_kind"].startswith("zip:")
    assert meta["cand_kind"] == "geojson"
    assert meta["ref_file_sha256"]
    assert mod.main([str(ref), str(cand)]) == 0


def test_geojson_ref_vs_zip_cand(tmp_path: Path):
    fc = _fc()
    ref = _write_geojson(tmp_path / "ref.geojson", fc)
    cand = _write_zip(tmp_path / "cand.zip", fc)
    assert mod.compare(ref, cand) == []


def test_zip_vs_zip_mismatch(tmp_path: Path):
    ref = _write_zip(tmp_path / "ref.zip", _fc())
    cand = _write_zip(tmp_path / "cand.zip", _fc(extra="DIFF"))
    assert mod.main([str(ref), str(cand)]) == 1


def test_missing_ref(tmp_path: Path):
    b = _write_geojson(tmp_path / "b.geojson", _fc())
    assert mod.main([str(tmp_path / "nope.geojson"), str(b)]) == 1


def test_extract_scored_bytes_rejects_empty_zip(tmp_path: Path):
    p = tmp_path / "empty.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no geojson here")
    p.write_bytes(buf.getvalue())
    try:
        mod.extract_scored_bytes(p)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "no scored" in str(exc).lower() or "FeatureCollection" in str(exc)
