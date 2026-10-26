"""Рамка из маски возвращается кругом в центр окна. Детектор не вызывается."""
import numpy as np
from shapely.geometry import Point

from expds.det_boxes import box_center_world, boxes_from_mask, circle_from_box
from expds.tiles import IGNORE_BIT


def test_ignore_bit_is_not_a_box_and_extent_is_inclusive():
    mask = np.zeros((8, 8), np.uint8)
    mask[3:7, 2:6] = 1
    mask[0, 0] = 1 << IGNORE_BIT
    boxes = boxes_from_mask(mask, ["kurgany_tselye"])
    assert len(boxes) == 1
    assert boxes[0]["bbox"] == [2, 3, 4, 4]
    assert boxes[0]["area_px"] == 16
    assert boxes[0]["centroid_px"] == [3.5, 4.5]


def test_full_window_box_sits_on_the_recorded_center():
    # Окно 4×4 при шаге 2 м. Рамка на весь кадр: её центр — записанный центр тайла.
    xy = box_center_world([0, 0, 4, 4], cx=10.0, cy=20.0, res=2.0, tile=4)
    assert xy == (10.0, 20.0)
    feat = circle_from_box([0, 0, 4, 4], 10.0, 20.0, 2.0, 4, radius_m=4.0, crs=None,
                           category="kurgany_tselye", confidence=0.5)
    assert feat["class_name"] == "kurgany_tselye"
    assert feat["confidence"] == 0.5
    assert abs(feat["geometry"].centroid.x - 10.0) < 1e-6
    assert abs(feat["geometry"].centroid.y - 20.0) < 1e-6
    assert feat["geometry"].contains(Point(13.0, 20.0))
    assert not feat["geometry"].contains(Point(15.5, 20.0))
