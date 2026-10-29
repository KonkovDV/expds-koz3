"""bot-3 Fix B: [ae_recall_relax] default OFF; apply remaps ae/li_ae/fusion."""
from __future__ import annotations

from expds import ae_recall_relax as M


def test_defaults_off():
    st = M.settings({})
    assert st["enabled"] is False
    assert abs(float(st["thresholds_by_markup"]["ae"]["fortifikatsii"]) - 0.995) < 1e-9


def test_apply_noop_when_off():
    cfg = {"thresholds_by_markup": {"ae": {"fortifikatsii": 0.998215}}, "fusion": {"povr_min": 0.99}}
    out = M.apply(cfg)
    assert out["thresholds_by_markup"]["ae"]["fortifikatsii"] == 0.998215
    assert out["fusion"]["povr_min"] == 0.99


def test_apply_on_remaps():
    cfg = {
        "ae_recall_relax": {"enabled": True},
        "thresholds_by_markup": {
            "ae": {"fortifikatsii": 0.998215, "kurgany_povrezhdennye": 0.99},
            "li_ae": {"kurgany_povrezhdennye": 0.993, "kurgany_tselye": 1.000001},
        },
        "fusion": {"povr_min": 0.99, "tselye_min": 0.85},
    }
    out = M.apply(cfg)
    assert abs(out["thresholds_by_markup"]["ae"]["fortifikatsii"] - 0.995) < 1e-9
    assert abs(out["thresholds_by_markup"]["li_ae"]["kurgany_povrezhdennye"] - 0.95) < 1e-9
    assert abs(out["fusion"]["povr_min"] - 0.9) < 1e-9


def test_toml_section_present_and_off():
    import tomllib
    from pathlib import Path
    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert (cfg.get("ae_recall_relax") or {}).get("enabled") is False
    assert (cfg.get("dfine_aeor") or {}).get("enabled") is False
