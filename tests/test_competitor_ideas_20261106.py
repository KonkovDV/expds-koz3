"""Идеи конкурентов, которых в коде не было: по умолчанию выключены, балл не меняют."""
import numpy as np
import pytest
from shapely.geometry import Point

torch = pytest.importorskip("torch")


def _dot(x, y, group="ae"):
    return {"geom": Point(x, y).buffer(0.5), "class_name": "kurgany_povrezhdennye",
            "markup_type": group, "region_name": "r", "confidence": 0.5}


def test_carpet_gate_keeps_a_cluster_and_drops_a_loner():
    from expds.post import carpet_gate

    cluster = [_dot(i * 10.0, 0.0) for i in range(4)]
    loner = _dot(10_000.0, 0.0)
    other = {"geom": Point(0, 0).buffer(1), "class_name": "fortifikatsii",
             "markup_type": "ae", "region_name": "r", "confidence": 0.9}
    out = carpet_gate(cluster + [loner, other], {"min_n": 2, "radius_m": 30})
    assert other in out
    assert loner not in out
    assert len([f for f in out if f["class_name"] == "kurgany_povrezhdennye"]) == 4


def test_pos_floor_does_not_change_the_default_loss():
    from expds.centers import focal_center_loss

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3, 3] = 1
    logits = torch.zeros(1, 1, 8, 8)
    base = float(focal_center_loss(logits, target))
    assert float(focal_center_loss(logits, target, pos_floor=0)) == base
    assert float(focal_center_loss(logits, target, pos_floor=50)) < base


def test_lidar_scale_list_identity_is_one_pass():
    from expds.lidar_cnx import _one_scale, _scale_list

    assert _one_scale(_scale_list(None)) or _one_scale(_scale_list([1.0]))
    assert _one_scale(_scale_list([1.0]))
    assert not _one_scale(_scale_list([0.8, 1.0, 1.25]))


def test_component_confidence_mean_is_not_the_peak():
    from rasterio.transform import Affine

    from expds.infer_nn import probs_to_features

    p = np.zeros((1, 6, 6), np.float32)
    p[0, 1:4, 1:4] = 0.2
    p[0, 2, 2] = 0.9
    tr = Affine(1, 0, 0, 0, -1, 0)
    mx = probs_to_features(p, ["fortifikatsii"], tr, "EPSG:3857", 1.0, "r", "li", "a", low=0.05)
    mean = probs_to_features(p, ["fortifikatsii"], tr, "EPSG:3857", 1.0, "r", "li", "a",
                             low=0.05, conf_stat="mean")
    assert mx and mean
    assert mx[0]["confidence"] == pytest.approx(0.9)
    assert mean[0]["confidence"] < 0.9
