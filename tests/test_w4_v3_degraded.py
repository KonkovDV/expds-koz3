import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def test_marker_written_and_reset(tmp_path, capsys):
    from expds import degraded
    out = tmp_path / "o" / "result.geojson"
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / degraded.MARKER).write_text("stale")
    degraded.set_output(out)
    assert not (tmp_path / "o" / degraded.MARKER).exists()
    degraded.mark("sp_dfine", "SP_DFINE_UNAVAILABLE: test")
    d = json.loads((tmp_path / "o" / degraded.MARKER).read_text())
    assert d["degraded"] and d["reasons"][0]["component"] == "sp_dfine"
    assert "V3_DEGRADED" in capsys.readouterr().err


def test_check_submission_fails_on_marker(tmp_path, monkeypatch):
    import check_submission as C
    got = []
    monkeypatch.setattr(C, "report", lambda st, sec, msg: got.append((st, sec)))
    C.degraded_lines("ok\n", tmp_path)
    assert got == [("PASS", "v3")]
    got.clear()
    (tmp_path / "V3_DEGRADED.json").write_text(json.dumps({"reasons": [{"component": "sp_dfine", "reason": "x"}]}))
    C.degraded_lines("WARNING !!! V3_DEGRADED: sp_dfine\n", tmp_path)
    assert [g for g in got if g[0] == "FAIL"] == [("FAIL", "v3"), ("FAIL", "v3")]


def test_vendor_abi_check(tmp_path):
    import importlib.machinery

    from expds import sp_dfine
    (tmp_path / "regex").mkdir()
    assert not sp_dfine._vendor_abi_ok(tmp_path)
    (tmp_path / "regex" / ("_regex" + importlib.machinery.EXTENSION_SUFFIXES[0])).write_bytes(b"")
    assert sp_dfine._vendor_abi_ok(tmp_path)
