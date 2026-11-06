"""проверка 29.10 (требование): при [sp_dfine] enabled = true контейнер без DFine* в transformers,
без весов или порогов — жёсткая ошибка, а не тихий v2. Синтетика: без весов, без сети, transformers — заглушка."""
import sys
import types

import pytest

from expds import sp_dfine

THR = {c: 0.5 for c in sp_dfine.CLASSES}


def _cfg(tmp_path, **own):
    w = tmp_path / "models" / "dfine"
    w.mkdir(parents=True)
    (w / "config.json").write_text("{}")
    (w / "model.safetensors").write_bytes(b"")
    st = {"enabled": True, "weights": ["models/dfine"], **own}
    return {"sp_dfine": st, "thresholds_by_markup": {"sp_dfine": dict(THR)}}


def _fake_transformers(monkeypatch, version="5.21.0", with_dfine=True):
    m = types.ModuleType("transformers")
    m.__version__ = version
    if with_dfine:
        m.DFineConfig = type("DFineConfig", (), {})
        m.DFineForObjectDetection = type("DFineForObjectDetection", (), {})
    monkeypatch.setitem(sys.modules, "transformers", m)


def test_strict_is_default():
    assert sp_dfine.DEFAULTS["strict"] is True
    assert sp_dfine.DEFAULTS["enabled"] is False


def test_preflight_ok(tmp_path, monkeypatch):
    _fake_transformers(monkeypatch)
    info = sp_dfine.preflight(_cfg(tmp_path), tmp_path)
    assert info["transformers"] == "5.21.0" and info["problems"] == []


def test_missing_dfine_class_is_hard_fail(tmp_path, monkeypatch):
    _fake_transformers(monkeypatch, with_dfine=False)
    with pytest.raises(sp_dfine.DFineUnavailable, match="DFine"):
        sp_dfine.preflight(_cfg(tmp_path), tmp_path)


def test_version_pin_mismatch_is_hard_fail(tmp_path, monkeypatch):
    _fake_transformers(monkeypatch, version="4.57.0")
    with pytest.raises(sp_dfine.DFineUnavailable, match="пин"):
        sp_dfine.preflight(_cfg(tmp_path), tmp_path)


def test_missing_weights_and_thresholds(tmp_path, monkeypatch):
    _fake_transformers(monkeypatch)
    cfg = _cfg(tmp_path, weights=[])
    cfg["thresholds_by_markup"] = {}
    with pytest.raises(sp_dfine.DFineUnavailable) as e:
        sp_dfine.preflight(cfg, tmp_path)
    assert "weights пуст" in str(e.value) and "порогов" in str(e.value)


def test_non_strict_keeps_soft_behaviour(tmp_path, monkeypatch):
    _fake_transformers(monkeypatch, with_dfine=False)
    info = sp_dfine.preflight(_cfg(tmp_path, strict=False), tmp_path)
    assert info["problems"] and info["strict"] is False


def test_region_feats_model_load_failure_raises_when_strict(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ImportError("cannot import name 'DFineConfig'")

    monkeypatch.setattr(sp_dfine, "load_models", boom)
    monkeypatch.setattr(sp_dfine, "_device", lambda st: "cpu")
    reg = types.SimpleNamespace(name="r", rasters={"sp_or": [tmp_path / "a.tif"]})
    with pytest.raises(sp_dfine.DFineUnavailable):
        sp_dfine.region_feats(reg, _cfg(tmp_path), None, {}, tmp_path)
    # strict = false: старое мягкое поведение — пусто, без исключения
    assert sp_dfine.region_feats(reg, _cfg(tmp_path / "x", strict=False), None, {}, tmp_path / "x") == ([], False)
