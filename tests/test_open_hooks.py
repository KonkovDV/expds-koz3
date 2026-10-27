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


def test_census_from_an_uploaded_zip_changes_only_the_config(tmp_path):
    import tomllib
    import zipfile

    src, out = tmp_path / "sent.zip", tmp_path / "census.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("metadata.json", '{"image": "x/y:1"}')
        z.writestr("configs/default.toml", "default_threshold = 0.15\n# [post]\n# core = true\n")
        z.writestr("models/a.pt", b"\x00\x01weights")
    _submission().census_from_zip(src, out, ["finds_points"])
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(out) as b:
        assert a.namelist() == b.namelist()
        for name in a.namelist():
            if name != "configs/default.toml":
                assert a.read(name) == b.read(name)
        cfg = tomllib.loads(b.read("configs/default.toml").decode("utf-8"))
    assert cfg["post"]["only_classes"] == ["finds_points"] and cfg["default_threshold"] == 0.15


def test_threshold_variant_edits_only_the_listed_values_and_keeps_comments():
    import tomllib

    text = ("[thresholds_by_markup.li]\n# пояснение\nkurgany_povrezhdennye = 0.06415\nkurgany_tselye = 0.19422\n"
            "\n[thresholds_by_markup.sp_or]\nkurgany_povrezhdennye = 0.3645  # карта центров\n")
    sub = _submission()
    out = sub.toml_with_thresholds(text, {"li": {"kurgany_povrezhdennye": (0.06415, 0.031579)},
                                          "sp_or": {"kurgany_povrezhdennye": (0.3645, 0.29355)}})
    cfg = tomllib.loads(out)
    assert cfg["thresholds_by_markup"]["li"] == {"kurgany_povrezhdennye": 0.031579, "kurgany_tselye": 0.19422}
    assert cfg["thresholds_by_markup"]["sp_or"]["kurgany_povrezhdennye"] == 0.29355
    assert "# пояснение" in out and "# карта центров" in out


def test_threshold_variant_refuses_a_config_it_was_not_computed_from():
    import pytest

    with pytest.raises(SystemExit):
        _submission().toml_with_thresholds("[thresholds_by_markup.li]\nkurgany_tselye = 0.5\n",
                                           {"li": {"kurgany_tselye": (0.19422, 0.1)}})
    with pytest.raises(SystemExit):
        _submission().toml_with_thresholds("[thresholds_by_markup.li]\nkurgany_tselye = 0.5\n",
                                           {"li": {"gorodishcha": (0.99, 0.98)}})
