"""Выключенные ключи сдачи: проверка кропа и перепись одного класса."""
import importlib.util

from conftest import ROOT
from expds.crop_model import blend_geo, maybe_verify


def _submission():
    spec = importlib.util.spec_from_file_location("build_submission", ROOT / "scripts" / "build_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_crop_check_stays_off_without_the_flag():
    feats = [{"class_name": "kurgany_tselye", "confidence": 0.4, "markup_type": "sp_or",
              "region_name": "r", "threshold_group": "sp_focal"}]
    assert maybe_verify(feats, {}, {}) is None
    assert maybe_verify(feats, {}, {"verify": {"enabled": False, "weights": "missing.pt"}}) is None


def test_crop_check_without_weights_does_not_change_scores(tmp_path):
    feats = [{"class_name": "kurgany_tselye", "confidence": 0.4, "markup_type": "sp_or",
              "region_name": "r", "threshold_group": "sp_focal"}]
    cfg = {"verify": {"enabled": True, "weights": str(tmp_path / "no.pt"),
                      "groups": ["sp_focal"], "classes": ["kurgany_tselye"]}}
    assert maybe_verify(feats, {"r": object()}, cfg) is None
    assert feats[0]["confidence"] == 0.4


def test_geometric_blend():
    assert abs(blend_geo(0.16, 0.64) - 0.32) < 1e-9


def test_census_toml_appends_only_classes_and_leaves_the_rest():
    text = _submission().toml_with_only_classes("default_threshold = 0.15\n", ["gorodishcha"])
    assert "default_threshold = 0.15" in text
    assert 'only_classes = ["gorodishcha"]' in text
