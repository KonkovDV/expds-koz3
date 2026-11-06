"""check_submission не валит архив за необязательные пакеты PIXGBM (psutil, tl2cgen, lleaves)
и за список типов в [geophys.pixgbm] maps (это не режим карт gray/all)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cs():
    spec = importlib.util.spec_from_file_location("check_submission_t", ROOT / "scripts" / "check_submission.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_optional_third_not_required(tmp_path, capsys):
    (tmp_path / "inference").mkdir()
    (tmp_path / "expds").mkdir()
    (tmp_path / "inference" / "main.py").write_text("import expds.a\n", encoding="utf-8")
    (tmp_path / "expds" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "expds" / "a.py").write_text(
        "import numpy\ntry:\n    import psutil\nexcept ImportError:\n    psutil = None\n"
        "def f():\n    import tl2cgen\n", encoding="utf-8")
    cs = _cs()
    _reach, third = cs.check_imports_static(tmp_path)
    assert "numpy" in third
    assert not third & {"psutil", "tl2cgen", "lleaves"}
    assert "необязательные" in capsys.readouterr().out


def test_probe_skips_pixgbm_maps_list():
    src = (ROOT / "scripts" / "check_submission.py").read_text(encoding="utf-8")
    assert 'if name == "pixgbm" and isinstance(sub, dict):' in src
