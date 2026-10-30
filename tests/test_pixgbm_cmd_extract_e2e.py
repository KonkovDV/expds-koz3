"""E2E: scripts/pixgbm.cmd_extract → G.pick_rasters (1-arg REF geo) on a tiny train-expds-3 tree.

Catches the 3f61ea8 regression where scripts called pick_rasters(files, rule, mt, crs)
but expds/geo_pixgbm.py @35e4331b still exposes pick_rasters(files) only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("rasterio")
pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py not in git (EN-similarity hold)")

from conftest import UTM, X0, Y0, write_mag_tif  # noqa: E402
from expds import geo_pixgbm as G  # noqa: E402


def _load_pixgbm():
    spec = importlib.util.spec_from_file_location("pixgbm_cli_e2e", ROOT / "scripts" / "pixgbm.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_object_poly(path: Path, ring_utm: list[list[float]]) -> None:
    """GeoJSON ObjectPoly in UTM 36N (matches UTM.json + MagGr CRS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": UTM}},
        "features": [{
            "type": "Feature",
            "properties": {"markup_type": "maggr", "class_name": "object_poly"},
            "geometry": {"type": "Polygon", "coordinates": [ring_utm]},
        }],
    }
    path.write_text(json.dumps(fc), encoding="utf-8")


def _tiny_train_tree(root: Path) -> Path:
    """Mirror F:/2/train-expds-3 layout for one geo region (200–217)."""
    reg = root / "200_SynthSite_1"
    maps = reg / "07_200_SynthSite_1_MagGr_карты"
    ann = reg / "06_200_SynthSite_1_разметка"
    maps.mkdir(parents=True)
    ann.mkdir(parents=True)
    (reg / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    # Gray + color copies of the same key — pick_rasters must keep gray only.
    write_mag_tif(maps / "01_200_SynthSite_1_MagGr_серый.tif", res=0.25, size_m=20.0, seed=3)
    write_mag_tif(maps / "02_200_SynthSite_1_MagGr_цвет.tif", res=0.25, size_m=20.0, seed=3)
    # Polygon covering left-bottom of the 20×20 m map (UTM, y up).
    ring = [
        [X0 + 2.0, Y0 + 2.0],
        [X0 + 8.0, Y0 + 2.0],
        [X0 + 8.0, Y0 + 8.0],
        [X0 + 2.0, Y0 + 8.0],
        [X0 + 2.0, Y0 + 2.0],
    ]
    _write_object_poly(ann / "200_SynthSite_1_MagGr_ObjectPoly.geojson", ring)
    return root


def test_pick_rasters_geo_is_one_arg():
    """Guard: REF geo signature must stay 1-arg (hp_energy 4-arg not vendored)."""
    import inspect
    sig = inspect.signature(G.pick_rasters)
    params = list(sig.parameters)
    assert params == ["files"], f"expected pick_rasters(files), got {params}"


def test_cmd_extract_pick_rasters_e2e(tmp_path):
    """Real cmd_extract path: scan → pick_rasters → load_map → features → npz."""
    mod = _load_pixgbm()
    # Source must call 1-arg (catches re-introduction of 4-arg call sites).
    src = (ROOT / "scripts" / "pixgbm.py").read_text(encoding="utf-8")
    assert "G.pick_rasters(reg.rasters.get(mt, []), None, mt, crs_region)" not in src
    assert src.count("G.pick_rasters(reg.rasters.get(mt, []))") == 3

    tree = _tiny_train_tree(tmp_path / "train-expds-3")
    out = tmp_path / "feats"
    args = SimpleNamespace(
        root=str(tree),
        out=str(out),
        config=str(ROOT / "configs" / "default.toml"),
        regions=None,
    )
    mod.cmd_extract(args)  # must not TypeError on pick_rasters

    log_path = out / "extract_log.json"
    assert log_path.is_file()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log, "extract_log empty — region/raster not seen"
    # One gray map processed (color twin collapsed by pick_rasters).
    ok = [r for r in log if r.get("skip") is None and r.get("n", 0) > 0]
    assert ok, f"no successful extract rows: {log}"
    assert ok[0]["mt"] == "maggr"
    assert "серый" in ok[0]["raster"]
    npzs = list(out.glob("*.npz"))
    assert npzs, "expected feature npz from cmd_extract"
    z = np.load(npzs[0])
    assert z["X"].ndim == 2 and z["X"].shape[0] == ok[0]["n"]
    assert int((z["yp"] == 1).sum()) >= 1
