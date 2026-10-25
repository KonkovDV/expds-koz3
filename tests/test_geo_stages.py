"""Счёт объектов геофизики по шагам сдачи (scripts/geo_stages.py)."""
import json
import subprocess
import sys
from pathlib import Path

from conftest import write_mag_tif

ROOT = Path(__file__).resolve().parents[1]


def run(root: Path, tmp_path: Path) -> list[dict]:
    out = tmp_path / "stages.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "geo_stages.py"), "--root", str(root), "--names",
                    "--out", str(out)], check=True, capture_output=True)
    return json.loads(out.read_text(encoding="utf-8"))


def test_noise_mask_over_the_whole_map_is_visible(tmp_path):
    reg = tmp_path / "root" / "region_0001"
    write_mag_tif(reg / "07_region_0001_MagGr_карты" / "01_region_0001_MagGr.tif")
    (reg / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    ring = [[32.99, 55.04], [33.01, 55.04], [33.01, 55.06], [32.99, 55.06], [32.99, 55.04]]
    (reg / "extra").mkdir()
    (reg / "extra" / "region_0001_MagGr_шум.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}), encoding="utf-8")
    (reg / "08_region_0001_неизвестно.tif").write_bytes(b"")
    rows = run(tmp_path / "root", tmp_path)
    skipped = [r for r in rows if "skipped_rasters" in r]
    assert skipped and skipped[0]["skipped_rasters"] == 1
    m = next(r for r in rows if r.get("markup") == "maggr")
    assert m["noise_cover"] == 1.0
    f = m["finds_points"]
    assert f["detector"] > 0 and f["after_boundary"] == f["detector"] and f["after_noise"] == 0


def test_noise_cap_drops_a_mask_that_covers_the_whole_map(tmp_path):
    from shapely.geometry import box

    from expds import geophys

    tif, _ = write_mag_tif(tmp_path / "01_r_MagGr.tif")
    fp = geophys.map_footprint(tif)
    whole = fp.buffer(100)
    part = box(fp.bounds[0], fp.bounds[1], fp.bounds[0] + 5, fp.bounds[1] + 5)
    assert geophys.sane_masks(tif, None, whole, {}) == (None, whole)            # по умолчанию как раньше
    assert geophys.sane_masks(tif, None, whole, {"noise_max_cover": 0.9}) == (None, None)
    assert geophys.sane_masks(tif, None, part, {"noise_max_cover": 0.9}) == (None, part)
    far = fp.buffer(10).difference(fp.buffer(5))                                 # граница мимо карты
    assert geophys.sane_masks(tif, far, None, {"boundary_min_cover": 0.05}) == (None, None)
    assert geophys.sane_masks(tif, whole, None, {"boundary_min_cover": 0.05}) == (whole, None)
