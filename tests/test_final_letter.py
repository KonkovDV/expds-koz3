"""Письмо организаторов 06.11 вечером: сдача ищет объекты, а не читает справочник."""
import importlib.util
import tomllib

from conftest import ROOT


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_submission_final", ROOT / "scripts" / "check_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _codes(mod):
    return [(s, c) for s, c, _ in mod.RESULTS if c == "final"]


def test_default_config_passes_the_final_letter():
    mod = _load()
    mod.RESULTS.clear()
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    mod.check_final_letter(cfg)
    assert _codes(mod) == [("PASS", "final")]
    assert cfg["orgate"]["enabled"] is False
    assert cfg["rerank_arch"]["enabled"] is False
    assert cfg["geophys"]["pixgbm"]["enabled"] is False
    assert cfg["geophys"]["pixgbm"]["union_name_re"] == ""
    assert "site_calibration" not in cfg["post"]


def test_orgate_with_site_coordinates_fails():
    mod = _load()
    mod.RESULTS.clear()
    mod.check_final_letter({"orgate": {"enabled": True}})
    assert _codes(mod)[0][0] == "FAIL"


def test_site_calibration_on_the_run_fails():
    mod = _load()
    mod.RESULTS.clear()
    mod.check_final_letter({"post": {"site_calibration": {"groups": ["li"]}}})
    assert _codes(mod)[0][0] == "FAIL"


def test_region_name_rule_fails_only_when_the_detector_is_on():
    mod = _load()
    mod.RESULTS.clear()
    mod.check_final_letter({"geophys": {"pixgbm": {"enabled": False, "union_name_re": "Гнёздово"}}})
    assert _codes(mod) == [("PASS", "final")]
    mod.RESULTS.clear()
    mod.check_final_letter({"geophys": {"pixgbm": {"enabled": True, "union_name_re": "Гнёздово"}}})
    assert _codes(mod)[0][0] == "FAIL"


def test_rerank_distances_are_not_a_gazetteer():
    """cx, cy реранка меряют соседей уже найденного объекта и в модель не входят."""
    mod = _load()
    mod.RESULTS.clear()
    mod.check_final_letter({"rerank_arch": {"enabled": True}})
    assert _codes(mod) == [("PASS", "final")]
    names = (ROOT / "expds" / "rerank_arch.py").read_text(encoding="utf-8")
    start = names.index("NAMES = [")
    block = names[start:names.index("]", start)]
    assert '"cx"' not in block and '"cy"' not in block and '"lat"' not in block
