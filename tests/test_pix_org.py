"""GEO-ORG1/ORG2 в чистой переписи PIXGBM: флаги выключены по умолчанию, синтетика без данных соревнования."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

from expds import pix_clean as pc  # noqa: E402
from expds import pix_org as po  # noqa: E402

CRS = "EPSG:32636"


def _jet_raster(path: Path, n: int = 200, label=True) -> Path:
    lut = pc._jet_lut()
    yy, xx = np.mgrid[0:n, 0:n]
    t = (0.5 + 0.4 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 8.0 ** 2)) - 0.0 * xx)
    img = lut[np.clip((t * 255).astype(int), 0, 255)].transpose(2, 0, 1).copy()
    if label:
        img[:, 10:30, 10:60] = 128.0            # серая подпись: далеко от шкалы jet
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=CRS,
                       transform=from_origin(400000.0, 6070000.0, 0.1, 0.1)) as ds:
        ds.write(np.clip(img, 0, 255).astype(np.uint8))
    return path


def test_defaults_off():
    import tomllib
    cfg = tomllib.loads(Path(__file__).resolve().parents[1].joinpath("configs/default.toml").read_text(encoding="utf-8"))
    pix = cfg["geophys"]["pixgbm"]
    assert pix["org_maggr"] is False and pix["org2_mask"] is False and pix["org2_band_mode"] == "rgb"
    assert pc.OPTS == {"band_mode": "rgb", "org_mask": False}


def test_jet_scalar_roundtrip():
    lut = pc._jet_lut()
    t, d = pc.jet_scalar(lut.T.reshape(3, 1, 256).copy())
    assert np.abs(t.ravel() - np.arange(256) / 255.0).max() <= 2 / 255 and float(d.max()) < 1.0
    _, d2 = pc.jet_scalar(np.full((3, 1, 1), 128.0, np.float32))
    assert float(d2[0, 0]) >= pc.PALETTE_MAX_DIST


def test_prepare_default_unchanged_and_org2(tmp_path):
    p = _jet_raster(tmp_path / "e.tif")
    a = pc.prepare(p, "etgr")
    b = pc.prepare(p, "etgr", band_mode="rgb", org_mask=False)
    assert np.array_equal(a.z, b.z) and np.array_equal(a.valid, b.valid)
    m = pc.prepare(p, "etgr", org_mask=True)
    assert m.valid.sum() < a.valid.sum()
    j = pc.prepare(p, "etgr", band_mode="jet", org_mask=True)
    assert np.array_equal(j.z, j.z) and j.z.shape == a.z.shape and np.isfinite(j.z).all()
    # не EtGr: флаги ORG2 не действуют
    g = pc.prepare(p, "maggr", band_mode="jet", org_mask=True)
    g0 = pc.prepare(p, "maggr")
    assert np.array_equal(g.z, g0.z)


def test_nms_points_radius():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    keep = po.nms_points(xy, np.array([0.9, 0.8, 0.95, 0.1]), r_m=1.5)
    assert keep == [2, 0, 3]


def test_nms_polys_overlap():
    polys = [box(0, 0, 2, 2), box(0.1, 0.1, 2.1, 2.1), box(5, 5, 6, 6)]
    assert po.nms_polys(polys, np.array([0.5, 0.9, 0.2])) == [1, 2]


def test_pipeline_flag_gate():
    from expds import pipeline
    src = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert 'pix.get("org_maggr", False)' in src and 'impl == "clean"' in src
