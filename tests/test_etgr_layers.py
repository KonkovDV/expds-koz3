"""Карта среза и профиль электроразведки различаются по имени файла."""
import importlib.util

from conftest import ROOT


def load_script():
    spec = importlib.util.spec_from_file_location("etgr_layers", ROOT / "scripts" / "etgr_layers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_layer_kind_splits_maps_and_profiles():
    mod = load_script()
    assert mod.layer_kind("участок_EtGr_карта_01_0.75.tif") == "map"
    assert mod.layer_kind("участок_EtGr_профиль_03.tif") == "profile"
    assert mod.layer_kind("участок_EtGr.tif") is None


def test_predictions_without_a_filename_are_compared_to_each_ground_truth():
    mod = load_script()
    pred = {"features": [{"properties": {"markup_type": "etgr", "class_name": "object_poly"}}]}
    got = mod.predictions_for(pred, "profile")
    assert len(got["features"]) == 1
