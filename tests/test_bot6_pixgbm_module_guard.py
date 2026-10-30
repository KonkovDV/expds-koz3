"""bot-6 29.10: [geophys.pixgbm] enabled=true без expds/geo_pixgbm.py (модуль вне git, docs/PIXGBM_DECISION.md)
не должен проходить молча. e2e r1 @b0add08: REF=NEW 0.312509 — ложное равенство, pix не работал ни в одном прогоне.
Проверяем: ERROR в логе; EXPDS_PIXGBM_STRICT=1 — исключение; check_submission — FAIL без модуля в архиве."""
import importlib.util
import logging
import sys
from pathlib import Path

import pytest

import expds
from expds import pipeline
from expds.catalog import Region

ROOT = Path(__file__).resolve().parents[1]
CFG = {"geophys": {"pixgbm": {"enabled": True, "replace": True}}}


def _hide_module(monkeypatch):
    monkeypatch.delattr(expds, "geo_pixgbm", raising=False)
    monkeypatch.setitem(sys.modules, "expds.geo_pixgbm", None)   # → ImportError на from . import geo_pixgbm


def _reg(tmp_path):
    return Region(name="r_test", path=tmp_path)


def test_missing_module_logs_error(tmp_path, monkeypatch, caplog):
    _hide_module(monkeypatch)
    monkeypatch.delenv("EXPDS_PIXGBM_STRICT", raising=False)
    with caplog.at_level(logging.WARNING, logger="expds.pipeline"):
        out = pipeline.geophys_feats(_reg(tmp_path), CFG, pipeline.TimeGuard(1000))
    assert out == []
    hits = [r for r in caplog.records if "модуль прежней реализации в этой сборке отсутствует" in r.getMessage()]
    assert hits and all(r.levelno == logging.ERROR for r in hits)


def test_missing_module_strict_raises(tmp_path, monkeypatch):
    _hide_module(monkeypatch)
    monkeypatch.setenv("EXPDS_PIXGBM_STRICT", "1")
    with pytest.raises(pipeline.PixgbmModuleMissing):
        pipeline.geophys_feats(_reg(tmp_path), CFG, pipeline.TimeGuard(1000))


def test_disabled_pixgbm_never_complains(tmp_path, monkeypatch, caplog):
    _hide_module(monkeypatch)
    monkeypatch.setenv("EXPDS_PIXGBM_STRICT", "1")
    with caplog.at_level(logging.WARNING, logger="expds.pipeline"):
        out = pipeline.geophys_feats(_reg(tmp_path), {"geophys": {"pixgbm": {"enabled": False}}},
                                     pipeline.TimeGuard(1000))
    assert out == [] and not [r for r in caplog.records if "geo_pixgbm" in r.getMessage()]


def _cs():
    spec = importlib.util.spec_from_file_location("check_submission_b6", ROOT / "scripts" / "check_submission.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_check_submission_fails_without_module():
    cs = _cs()
    cs.RESULTS.clear()
    cs.check_pixgbm_module({"enabled": True}, {"inference/main.py", "expds/pipeline.py"})
    assert any(s == "FAIL" and "geo_pixgbm.py" in m for s, _c, m in cs.RESULTS)
    cs.RESULTS.clear()
    cs.check_pixgbm_module({"enabled": True}, {"expds/geo_pixgbm.py"})
    cs.check_pixgbm_module({"enabled": False}, set())
    assert not [r for r in cs.RESULTS if r[0] == "FAIL"]
