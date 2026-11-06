"""Замки измеренной сдачи. Числа порогов здесь не проверяются: их источник — конфиг.

0,43274 — база v2 плюс лидар c5-in в варианте R, его дописывает make_candidate.py.
В default.toml этот лидар выключен, чтобы обычная сборка не стала другой посылкой.
"""
import importlib.util
import tomllib

from conftest import ROOT


def _cfg():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def test_default_config_keeps_the_measured_switches():
    cfg = _cfg()
    assert cfg["li_cnx"]["enabled"] is False
    assert cfg["orgate"]["enabled"] is False
    assert cfg["rerank_arch"]["enabled"] is False
    assert cfg["geophys"]["pixgbm"]["enabled"] is False
    assert cfg["geophys"]["pixgbm"]["union_name_re"] == ""
    assert cfg["hillfort"]["enabled"] is True
    assert cfg["verify"]["enabled"] is True
    assert cfg["post"]["kurgan_twins_only"] is True
    assert cfg["post"]["micro_circles"]["radius_m"]["kurgany_povrezhdennye"] == 0.5
    assert cfg["post"]["micro_circles"]["radius_m"]["kurgany_tselye"] == 0.5
    assert "frame_prior" not in cfg["post"]
    assert "site_calibration" not in cfg["post"]


def test_c5in_variant_r_replaces_lidar_kurgans_and_does_not_add_yolo():
    path = ROOT / "kaggle" / "2026-11-05" / "li_cnx_in" / "make_candidate.py"
    spec = importlib.util.spec_from_file_location("make_candidate_r", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.cnx_section("R", ["w0.pt", "w1.pt", "w2.pt", "w3.pt", "w4.pt"],
                           {"kurgany_povrezhdennye": 0.2, "kurgany_tselye": 0.3})
    assert "enabled = true" in text
    assert 'decode = "head"' in text
    assert "replace_li_kurgans = true" in text
    assert "threshold_group = \"li_cnxin\"" in text
    assert "sp_yolo" not in text
    added = mod.cnx_section("K", ["w0.pt", "w1.pt", "w2.pt", "w3.pt", "w4.pt"],
                            {"kurgany_povrezhdennye": 0.2, "kurgany_tselye": 0.3})
    assert "replace_li_kurgans = false" in added
