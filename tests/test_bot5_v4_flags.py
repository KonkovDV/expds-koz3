"""bot-5 v4: D-FINE SpOr ([sp_dfine]) отделён от Ae/Or ([dfine_aeor]); оба OFF по умолчанию."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def test_sp_dfine_and_dfine_aeor_default_off_independent():
    cfg = _cfg()
    assert cfg["sp_dfine"] == {"enabled": False}
    assert cfg["dfine_aeor"] == {"enabled": False}
    # Включение одного в оверлее не трогает другой ключ конфига.
    only_sp = {**cfg, "sp_dfine": {"enabled": True, "weights": ["models/dfine_f0"]}}
    assert only_sp["dfine_aeor"]["enabled"] is False
    only_ae = {**cfg, "dfine_aeor": {"enabled": True}}
    assert only_ae["sp_dfine"]["enabled"] is False


def test_v4_optional_blocks_default_off():
    cfg = _cfg()
    assert cfg["gor_dense"]["enabled"] is False
    assert cfg["gor_gate"]["enabled"] is False
    assert cfg["orgate"]["enabled"] is False
