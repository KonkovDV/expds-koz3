"""bot-2: аудит локализации D-FINE — tiling offsets, boxes_to_ellipses, utm_for.

На mock медиана pred→GT ~191 м при верном CRS (bot1). Здесь — регрессии координатной
математики; systematic 191 м на mock скорее domain/модель, не off-by-tile в порте.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point

from expds import sp_dfine


def test_windows_stride_overlap():
    # tile=640, overlap=160 → step=480
    wins = sp_dfine.windows(1000, 800, 640, 160)
    xs = sorted({x for x, _ in wins})
    ys = sorted({y for _, y in wins})
    assert xs == [0, 480]  # 0, then 480; 480+640=1120 > 1000 (pad)
    assert ys == [0, 480]
    assert (0, 0) in wins and (480, 480) in wins


def test_boxes_to_ellipses_xyxy_center_and_axes():
    # xyxy → центр / полуоси; affine 0.5 м/px north-up
    T = SimpleNamespace(a=0.5, b=0.0, c=500000.0, d=0.0, e=-0.5, f=6100000.0)
    B = np.array([[10.0, 20.0, 30.0, 40.0]])  # cx=20, cy=30, rx=10, ry=10 px
    ell = sp_dfine.boxes_to_ellipses(B, np.array([0.7]), np.array([1]), T, "EPSG:32636", "r", "a.tif", 0)
    from pyproj import Transformer
    e, n = 500000 + 20 * 0.5, 6100000 - 30 * 0.5
    x, y = Transformer.from_crs("EPSG:32636", "EPSG:3857", always_xy=True).transform(e, n)
    assert ell[0]["geom"].centroid.distance(Point(x, y)) < 0.05
    # полуоси ~10 px * 0.5 м = 5 м на земле (до ground_scale в 3857 — порядок метров)
    minx, miny, maxx, maxy = ell[0]["geom"].bounds
    assert 5.0 < (maxx - minx) < 20.0  # грубо, с учётом 3857 scale


def test_offset_window_cxcywh_to_map(monkeypatch):
    """Детекция в центре окна (x=576): центр сцены = 576+320 px → верный easting."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    class FakeScene:
        W = H = 1200
        T = SimpleNamespace(a=0.5, b=0.0, c=500000.0, d=0.0, e=-0.5, f=6100000.0)

        def read(self, x0, y0, n):
            import numpy as np
            a = np.full((n, n, 3), 120, np.uint8)
            return a

    # Одна «модель»: бокс в центре тайла (cx=cy=0.5), класс 1
    def fake_model(pixel_values):
        b = pixel_values.shape[0]
        logits = torch.full((b, 3, 5), -10.0)
        logits[:, 0, 1] = 4.0
        boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]] * 3).repeat(b, 1, 1)
        return SimpleNamespace(logits=logits, pred_boxes=boxes)

    st = dict(sp_dfine.DEFAULTS, flip_tta=False, batch=1, skip_empty=0.99)
    # Подменим windows → одно окно со смещением
    monkeypatch.setattr(sp_dfine, "windows", lambda W, H, tile, overlap: [(576, 0)])
    res, done, total = sp_dfine.detect_scene([fake_model], FakeScene(), st, torch.device("cpu"))
    B, L, C = res[0]
    assert len(L) == 1
    cx_px = (B[0, 0] + B[0, 2]) / 2
    assert abs(cx_px - (576 + 320)) < 1e-3
    ell = sp_dfine.boxes_to_ellipses(B, L, C, FakeScene.T, "EPSG:32636", "r", "a.tif", 0)
    from pyproj import Transformer
    e, n = 500000 + (576 + 320) * 0.5, 6100000 - 0  # cy=0.5 → 320 px down from window y=0
    # cy in tile = 0.5*640=320; window y=0 → scene y=320
    e, n = 500000 + (576 + 320) * 0.5, 6100000 - 320 * 0.5
    x, y = Transformer.from_crs("EPSG:32636", "EPSG:3857", always_xy=True).transform(e, n)
    assert ell[0]["geom"].centroid.distance(Point(x, y)) < 1.0


def test_utm_for_rejects_geographic_fallback(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    path = tmp_path / "a.tif"
    with rasterio.open(path, "w", driver="GTiff", width=32, height=32, count=1, dtype="uint8",
                       crs="EPSG:32636", transform=from_origin(500000, 6100000, 0.5, 0.5)) as ds:
        ds.write(np.ones((1, 32, 32), np.uint8) * 10)
    # geographic fallback must NOT be returned
    crs = sp_dfine.utm_for(path, "EPSG:4326")
    assert crs.startswith("EPSG:326") or crs.startswith("EPSG:327")
    assert not sp_dfine._crs_is_geographic(crs)
    # projected fallback kept
    assert sp_dfine.utm_for(path, "EPSG:32636") == "EPSG:32636"


def test_scene_rejects_geographic_crs(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    path = tmp_path / "a.tif"
    with rasterio.open(path, "w", driver="GTiff", width=32, height=32, count=1, dtype="uint8",
                       crs="EPSG:32636", transform=from_origin(500000, 6100000, 0.5, 0.5)) as ds:
        ds.write(np.ones((1, 32, 32), np.uint8) * 10)
    with pytest.raises(ValueError, match="географический"):
        sp_dfine.Scene(path, "EPSG:4326", 0.5)


def test_flip_tta_only_mirrors_x():
    """Горизонтальный flip: cx' = 1-cx; cy/w/h без изменений (как bot-9)."""
    torch = pytest.importorskip("torch")
    bx = torch.tensor([[[0.25, 0.4, 0.1, 0.2]]])
    flipped = bx.clone()
    flipped[..., 0] = 1 - flipped[..., 0]
    assert abs(float(flipped[0, 0, 0]) - 0.75) < 1e-6
    assert abs(float(flipped[0, 0, 1]) - 0.4) < 1e-6
