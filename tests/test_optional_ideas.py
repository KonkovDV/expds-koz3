"""Выключенные по умолчанию идеи: ядро, заливка, копии, плоский вход, CRS."""
import json

from shapely.geometry import Polygon, box

from expds.catalog import scan_root
from expds.crs import valid_crs
from expds.fusion import copy_layers
from expds.post import core_of, fill_holes
from expds.watchdog import finish


def test_fill_holes_and_core_stay_inside():
    ring = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(2, 2), (2, 8), (8, 8), (8, 2)]])
    filled = fill_holes([{"class_name": "gorodishcha", "geom": ring}], ["gorodishcha"])[0]["geom"]
    assert not filled.interiors
    outer = box(0, 0, 10, 10)
    core = core_of(outer, 0.2, min_r_m=1.0)
    assert outer.contains(core)
    assert core.area < outer.area


def test_finish_without_post_keeps_the_polygon():
    feat = {"class_name": "gorodishcha", "geom": box(0, 0, 20, 20), "confidence": 0.99,
            "markup_type": "li", "region_name": "r", "threshold_group": "li_mask"}
    out = finish([feat], {"default_threshold": 0.15})
    assert len(out) == 1
    assert abs(out[0]["geom"].area - 400) < 1


def test_copy_rules_absent_add_nothing():
    feat = {"class_name": "kurgany_povrezhdennye", "geom": box(0, 0, 4, 4), "confidence": 0.9,
            "markup_type": "li", "threshold_group": "li"}
    assert copy_layers([feat], {"sp_or": []}, []) == []


def test_bare_epsg_is_not_a_crs():
    assert valid_crs("EPSG") is None
    assert valid_crs("EPSG:") is None
    assert valid_crs("EPSG:32636") == "EPSG:32636"


def test_flat_input_is_one_region(tmp_path):
    (tmp_path / "cloud.las").write_bytes(b"")
    regions = scan_root(tmp_path)
    assert len(regions) == 1
    assert regions[0].las


def test_read_utm_still_prefers_a_real_code(tmp_path):
    from expds.catalog import read_utm_json

    path = tmp_path / "UTM.json"
    path.write_text(json.dumps({"crs": "EPSG:32646"}), encoding="utf-8")
    assert read_utm_json(path) == "EPSG:32646"
