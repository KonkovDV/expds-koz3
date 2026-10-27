"""Вариант раздела 5: порог, при котором на трейне выдаётся в factor раз больше объектов."""
import importlib.util
import json

from conftest import ROOT


def load():
    spec = importlib.util.spec_from_file_location("density_thresholds", ROOT / "scripts" / "density_thresholds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_factor_one_and_a_half_keeps_half_as_many_again():
    dt = load()
    conf = {("li", "k"): [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]}
    rec = dt.density_variant(conf, {"li": {"k": 0.55}}, 1.5)["li"]["k"]
    assert rec["kept"] == 4 and rec["kept_new"] == 6 and rec["tau_new"] == 0.4 and not rec["short"]


def test_short_group_drops_to_the_weakest_candidate_and_off_groups_stay_off():
    dt = load()
    conf = {("ae", "k"): [0.995, 0.993, 0.991], ("li_ae", "k"): [0.9]}
    out = dt.density_variant(conf, {"ae": {"k": 0.990}, "li_ae": {"k": 1.000001}}, 1.5)
    assert out["ae"]["k"]["tau_new"] == 0.991 and out["ae"]["k"]["short"]
    assert out["li_ae"]["k"]["tau_new"] == 1.000001 and out["li_ae"]["k"]["kept_new"] == 0


def test_geophysics_is_left_to_the_census():
    dt = load()
    out = dt.density_variant({("etgr", "object_poly"): [0.5, 0.2, 0.1]}, {"etgr": {"object_poly": 0.096635}}, 1.5)
    assert out == {}


def test_board_groups_map_to_the_config_tables(tmp_path):
    dt = load()
    feats = [{"type": "Feature", "geometry": None,
              "properties": {"class_name": "kurgany_tselye", "confidence": 0.5, "threshold_group": "sp_focal",
                             "region_name": "r{1}"}},
             {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]},
              "properties": {"class_name": "arkhitektury", "confidence": 0.3, "threshold_group": "sp_arch"}}]
    path = tmp_path / "stack.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    conf = dt.collect(path, [])
    assert conf == {("sp_or", "kurgany_tselye"): [0.5], ("sp_or", "arkhitektury"): [0.3]}
