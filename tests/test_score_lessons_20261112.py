"""Уроки чужой лестницы балла не включены в обычный конфиг и видны проверке архива."""
import importlib.util
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _cs():
    spec = importlib.util.spec_from_file_location(
        "check_submission_lessons", ROOT / "scripts" / "check_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RESULTS.clear()
    return mod


def test_default_config_does_not_enable_the_steps_that_dropped_the_score():
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert "kurgan_neighbour_prior" not in (cfg.get("post") or {})
    assert not (cfg.get("sp_yolo") or {}).get("enabled")
    text = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8")
    assert "0,48874→0,48634" in text


def test_checker_warns_when_those_steps_are_turned_on():
    cs = _cs()
    cs.check_public_lessons({"post": {"kurgan_neighbour_prior": {"radius_m": 200}}})
    assert any(r[0] == "WARN" and "0,48874" in r[2] for r in cs.RESULTS)
    cs.RESULTS.clear()
    cs.check_public_lessons({"sp_yolo": {"enabled": True, "replace_sp": True}})
    assert any(r[0] == "WARN" and "0,38441" in r[2] for r in cs.RESULTS)
    cs.RESULTS.clear()
    cs.check_public_lessons({})
    assert not [r for r in cs.RESULTS if r[0] == "WARN"]
