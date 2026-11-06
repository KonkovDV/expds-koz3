""" (v4): тихий отказ li→ae виден — маркер LI_TO_AE_FAILED в логе даёт FAIL check_submission."""
import importlib.util, io, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cs():
    spec = importlib.util.spec_from_file_location("cs_li", ROOT / "scripts" / "check_submission.py")
    m = importlib.util.module_from_spec(spec); sys.modules["cs_li"] = m; spec.loader.exec_module(m)
    return m


def _zip():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("configs/default.toml", "[sp_dfine]\nenabled = false\n")
    return zipfile.ZipFile(io.BytesIO(b.getvalue()))


def test_marker_fails():
    cs = _cs(); cs.RESULTS.clear()
    cs.dfine_lines(_zip(), "2026 ERROR expds.pipeline: LI_TO_AE_FAILED: 103: копирование ... (ValueError: x)\n")
    assert any(r[0] == "FAIL" and "LI_TO_AE_FAILED" in r[2] for r in cs.RESULTS), cs.RESULTS


def test_clean_log_no_fail():
    cs = _cs(); cs.RESULTS.clear()
    cs.dfine_lines(_zip(), "2026 INFO all fine\n")
    assert not any(r[0] == "FAIL" and r[1] == "li_to_ae" for r in cs.RESULTS)


def test_pipeline_has_marker():
    assert "LI_TO_AE_FAILED" in (ROOT / "expds" / "pipeline.py").read_text()
