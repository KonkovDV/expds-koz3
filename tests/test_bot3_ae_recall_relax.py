"""[ae_recall_relax] (bot-3): ослабление порогов ae/li_ae/fusion за флагом. Default OFF — байт-в-байт."""
import copy

import tomllib
from pathlib import Path

from expds import ae_recall_relax

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def test_default_off_noop():
    cfg = _cfg()
    assert cfg.get("ae_recall_relax", {}).get("enabled") is False
    assert not ae_recall_relax.enabled(cfg)
    before = copy.deepcopy(cfg["thresholds_by_markup"]["ae"])
    before_li = copy.deepcopy(cfg["thresholds_by_markup"]["li_ae"])
    before_fus = copy.deepcopy(cfg["fusion"])
    out = ae_recall_relax.apply(cfg)
    assert out is cfg  # same object when off
    assert out["thresholds_by_markup"]["ae"] == before
    assert out["thresholds_by_markup"]["li_ae"] == before_li
    assert out["fusion"] == before_fus


def test_on_lowers_ae_and_fusion():
    cfg = _cfg()
    cfg = dict(cfg)
    cfg["ae_recall_relax"] = {"enabled": True}
    base_ae_fort = cfg["thresholds_by_markup"]["ae"]["fortifikatsii"]
    base_li_tsel = cfg["thresholds_by_markup"]["li_ae"]["kurgany_tselye"]
    base_povr = cfg["fusion"]["povr_min"]
    out = ae_recall_relax.apply(cfg)
    assert out is not cfg
    assert out["thresholds_by_markup"]["ae"]["fortifikatsii"] < base_ae_fort
    assert out["thresholds_by_markup"]["li_ae"]["kurgany_tselye"] < base_li_tsel
    assert out["fusion"]["povr_min"] < base_povr
    # исходный cfg не тронут
    assert cfg["thresholds_by_markup"]["ae"]["fortifikatsii"] == base_ae_fort


def test_toml_overrides_defaults():
    cfg = {"ae_recall_relax": {"enabled": True, "thresholds_by_markup": {"ae": {"fortifikatsii": 0.42}},
                              "fusion": {"povr_min": 0.5}},
           "thresholds_by_markup": {"ae": {"fortifikatsii": 0.99}},
           "fusion": {"povr_min": 0.99, "li_to_ae": True}}
    out = ae_recall_relax.apply(cfg)
    assert out["thresholds_by_markup"]["ae"]["fortifikatsii"] == 0.42
    assert out["fusion"]["povr_min"] == 0.5
    assert out["fusion"]["li_to_ae"] is True


def test_dfine_aeor_still_off():
    cfg = _cfg()
    assert cfg.get("dfine_aeor", {}).get("enabled") is False
    assert "raster_fix" not in cfg  # без секции — выкл.
