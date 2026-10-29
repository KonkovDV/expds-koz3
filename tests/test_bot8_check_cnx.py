"""bot-8 bot8-cnx: проверки [li_cnx] в scripts/check_submission.py (check_cnx), без сборки архива."""
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CX = ROOT / "inference" / "cnx"


@pytest.fixture(scope="module")
def cs():
    spec = importlib.util.spec_from_file_location("cs_cnx", ROOT / "scripts" / "check_submission.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _zip(mut=lambda n, d: d):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for p in sorted(CX.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                n = p.relative_to(ROOT).as_posix()
                d = mut(n, p.read_bytes())
                if d is not None:
                    z.writestr(n, d)
    return zipfile.ZipFile(io.BytesIO(b.getvalue()))


ON = {"li_cnx": {"enabled": True, "weights": ["models/cnx/w.pt"]}}


def _run(cs, z, cfg):
    cs.RESULTS.clear()
    third = cs.check_cnx(z, cfg)
    return third, [(s, msg) for s, c, msg in cs.RESULTS if c == "cnx"]


def test_ok(cs):
    third, res = _run(cs, _zip(), ON)
    assert [s for s, _ in res] == ["PASS", "PASS"]
    assert {"numpy", "scipy", "laspy"} <= third
    assert not {"matplotlib", "osgeo"} & third      # rvt.blend/tile не достижимы из cnx_prep


def test_disabled(cs):
    third, res = _run(cs, _zip(), {"li_cnx": {"enabled": False}})
    assert third == set() and [s for s, _ in res] == ["WARN"]
    third, res = _run(cs, zipfile.ZipFile(io.BytesIO(_empty())), {})
    assert third == set() and res == []


def _empty():
    b = io.BytesIO()
    zipfile.ZipFile(b, "w").close()
    return b.getvalue()


@pytest.mark.parametrize("mut, cfg, needle", [
    (lambda n, d: d + b"\n" if n.endswith("relief_ext.py") else d, ON, "MANIFEST"),
    (lambda n, d: None if n.endswith(".whl") else d, ON, "rvt_py-*.whl"),
    (lambda n, d: None if n.endswith("expds_core.abi3.so") else d, ON, "expds_core.abi3.so"),
    (lambda n, d: d.replace(b"import numpy as np", b"import numpy as np\nimport requests", 1)
     if n.endswith("cnx_prep.py") else d, ON, "сетевые"),
    (lambda n, d: d, {"li_cnx": {"enabled": True, "weights": []}}, "weights пуст"),
])
def test_fail(cs, mut, cfg, needle):
    _, res = _run(cs, _zip(mut), cfg)
    assert any(s == "FAIL" and needle in msg for s, msg in res), res


def test_layout_ignores_cnx_py(cs):
    cs.RESULTS.clear()
    cs.check_layout(_zip())
    assert not any("лишние .py" in msg for _, _, msg in cs.RESULTS)
