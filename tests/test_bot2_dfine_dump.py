"""bot-2: EXPDS_DFINE_DUMP — сырые xyxy + 3857 и GT; по умолчанию ВЫКЛ."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point

from expds import sp_dfine

ROOT = Path(__file__).resolve().parents[1]


def test_dump_root_off_by_default(monkeypatch):
    monkeypatch.delenv("EXPDS_DFINE_DUMP", raising=False)
    assert sp_dfine.dump_root() is None


def test_write_boxes_dump_pixel_and_3857(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPDS_DFINE_DUMP", str(tmp_path))
    assert sp_dfine.dump_root() == tmp_path
    # identity-ish affine at a known UTM point near Гнёздово (EPSG:32636 ≈ 30U)
    # 1 px = 0.5 m; origin easting/northing
    T = SimpleNamespace(a=0.5, b=0.0, c=500000.0, d=0.0, e=-0.5, f=6070000.0)
    B = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=float)
    L = np.array([0.7], dtype=float)
    Cc = np.array([1], dtype=int)  # kurgany_povrezhdennye
    ell = sp_dfine.boxes_to_ellipses(B, L, Cc, T, "EPSG:32636", "r1", "a.tif", 0)
    path = sp_dfine.write_boxes_dump(
        tmp_path, region="r1", source="a.tif", crs="EPSG:32636",
        transform=T, model_i=0, B=B, L=L, Cc=Cc, ellipses=ell,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["scene_crs"] == "EPSG:32636"
    assert data["target_crs"] == "EPSG:3857"
    assert data["n"] == 1
    row = data["boxes"][0]
    assert row["class_name"] == "kurgany_povrezhdennye"
    assert row["xyxy_px"] == [10.0, 20.0, 30.0, 40.0]
    assert abs(row["center_px"][0] - 20.0) < 1e-6
    # 3857 coords should be large web-mercator numbers
    assert abs(row["center_3857"][0]) > 1e5
    assert abs(row["center_3857"][1]) > 1e5
    assert row["ellipse_bounds_3857"] is not None


def test_write_gt_dump_empty_without_annotations(tmp_path):
    reg = SimpleNamespace(name="empty", annotations=[])
    assert sp_dfine.write_gt_dump(tmp_path, reg) is None
