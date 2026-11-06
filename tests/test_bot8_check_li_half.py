""": проверки [li_half] в scripts/check_submission.py (check_li_half, ключи смоука)."""
import importlib.util
import re
from pathlib import Path

import pytest

from expds import li_half

ROOT = Path(__file__).resolve().parents[1]
THR = {"kurgany_povrezhdennye": 0.5, "kurgany_tselye": 0.5}


# ---------- check_submission ----------

@pytest.fixture(scope="module")
def cs():
    spec = importlib.util.spec_from_file_location("cs_lihalf", ROOT / "scripts" / "check_submission.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(cs, cfg):
    cs.RESULTS.clear()
    cs.check_li_half(cfg)
    return [(s, m) for s, c, m in cs.RESULTS if c == "li_half"]


def test_check_li_half(cs):
    assert _run(cs, {}) == [] and _run(cs, {"li_half": {"enabled": False}}) == []
    r = _run(cs, {"li_half": {"enabled": "true"}, "thresholds_by_markup": {"li_half": THR}})
    assert [s for s, _ in r] == ["WARN"] and "ВЫКЛЮЧЕННЫМ" in r[0][1]
    r = _run(cs, {"li_half": {"enabled": True}})
    assert [s for s, _ in r] == ["WARN"] and "НЕ будет" in r[0][1]           # только выгрузка, как у
    r = _run(cs, {"li_half": {"enabled": True, "group": "li"}, "thresholds_by_markup": {"li": THR}})
    assert [s for s, _ in r] == ["FAIL"]
    r = _run(cs, {"li_half": {"enabled": True}, "thresholds_by_markup": {"li_half": {"kurgany_tselye": 0.5}}})
    assert [s for s, _ in r] == ["FAIL"] and "kurgany_povrezhdennye" in r[0][1]
    r = _run(cs, {"li_half": {"enabled": True, "tta": "1"}, "thresholds_by_markup": {"li_half": THR}})
    assert [s for s, _ in r] == ["FAIL"]
    r = _run(cs, {"li_half": {"enabled": True}, "thresholds_by_markup": {"li_half": THR}})
    assert [s for s, _ in r] == ["PASS"]
    cs.RESULTS.clear()


def test_check_li_half_mirrors_pipeline(cs):
    assert cs.LI_HALF_CLASSES == li_half.DEFAULTS["classes"]
    assert cs.LI_HALF_FORBIDDEN_GROUPS == set(li_half.FORBIDDEN_GROUPS)
    for cfg in ({"li_half": {"enabled": True}}, {"li_half": {"enabled": "true"}}, {"li_half": {"enabled": 1}}, {}):
        assert cs.li_half_on(cfg) == li_half.enabled(cfg)


def test_li_half_smoke_keys_match_pipeline_logs(cs):
    src = (ROOT / "expds" / "pipeline.py").read_text(encoding="utf-8")
    for k in cs.LI_HALF_SMOKE_WARN:
        assert k in src, k
    for ln in src.splitlines():
        if re.search(r"log\.(warning|exception|error)\(", ln) and "li_half" in ln:
            assert any(k in ln for k in cs.LI_HALF_SMOKE_WARN), ln
