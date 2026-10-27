"""Граница большой сцены — по разрыву площади, не по первым одиночкам."""
import importlib.util

from conftest import ROOT


def load_script():
    spec = importlib.util.spec_from_file_location("scene_split", ROOT / "scripts" / "scene_split.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cut_follows_the_gap_after_the_large_cluster():
    ss = load_script()
    areas = [10, 6, 5.5, 5.2, 5.0, 4.8, 4.6, 4.4, 1.2, 1.1, 0.4, 0.2]
    cut = ss.scene_cut(areas)
    assert 1.2 < cut < 4.4
    big = [a for a in areas if a >= cut]
    assert len(big) == 8 and min(big) == 4.4


def test_assign_keeps_small_scenes_on_the_base_group():
    ss = load_script()
    feats = [
        {"properties": {"region_name": "big", "class_name": "kurgany_tselye"}},
        {"properties": {"region_name": "small", "class_name": "kurgany_povrezhdennye"}},
    ]
    counts = ss.assign(feats, {"big"})
    assert feats[0]["properties"]["threshold_group"] == "sp_focal_big"
    assert feats[1]["properties"]["threshold_group"] == "sp_focal"
    assert counts == {"features_big": 1, "features_base": 1, "regions_big": 1, "regions_base": 1}
