"""Release v5: sha256 fail-loud for PIXGBM weights (WEIGHTS_MANIFEST_v5)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

gp = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (module is not in this tree)")


def _write(txt: Path, body: str = "feature_names=a\n") -> str:
    txt.write_text(body, encoding="utf-8")
    return hashlib.sha256(txt.read_bytes()).hexdigest()


def test_verify_empty_fails(tmp_path: Path):
    p = tmp_path / "poly.txt"
    p.write_bytes(b"")
    with pytest.raises(RuntimeError, match="EMPTY"):
        gp.verify_weight_sha256(p, "deadbeef", label="poly")


def test_verify_mismatch_fails(tmp_path: Path):
    p = tmp_path / "poly.txt"
    _write(p)
    with pytest.raises(RuntimeError, match="MISMATCH"):
        gp.verify_weight_sha256(p, "0" * 64, label="poly")


def test_verify_match_ok(tmp_path: Path):
    p = tmp_path / "poly.txt"
    dig = _write(p)
    assert gp.verify_weight_sha256(p, dig, label="poly") == dig


def test_load_models_strict_mismatch(tmp_path: Path, monkeypatch):
    dig = _write(tmp_path / "poly.txt")
    _write(tmp_path / "finds.txt", "feature_names=b\n")
    monkeypatch.setenv("EXPDS_PIXGBM_SHA_STRICT", "1")
    with pytest.raises(RuntimeError, match="MISMATCH"):
        gp.load_models(
            tmp_path,
            backend="lightgbm",
            names=("poly",),
            expected_sha256={"poly": "0" * 64},
            environ={"EXPDS_PIXGBM_SHA_STRICT": "1"},
        )


def test_load_models_strict_ok_skips_lgbm_without_booster(tmp_path: Path, monkeypatch):
    """sha passes; Booster may fail without lightgbm model format — use stub expected=actual."""
    dig = _write(tmp_path / "poly.txt", "tree\nfeature_names=a\n")
    # Without a real LightGBM model file LGBMModel() may fail — that's fine if sha ran first.
    # Use a fake LGBMModel via monkeypatch after verify.
    calls = []

    def fake_verify(path, expected, *, label=""):
        calls.append((Path(path).name, expected))
        return expected

    monkeypatch.setattr(gp, "verify_weight_sha256", fake_verify)

    class Dummy:
        backend = "lightgbm"
        def feature_name(self):
            return ["a"]
        def predict(self, X, num_threads=1):
            return [0.0] * len(X)

    monkeypatch.setattr(gp, "LGBMModel", lambda txt: Dummy())
    out = gp.load_models(
        tmp_path,
        backend="lightgbm",
        names=("poly",),
        expected_sha256={"poly": dig},
        environ={"EXPDS_PIXGBM_SHA_STRICT": "1"},
    )
    assert "poly" in out
    assert calls and calls[0][0] == "poly.txt"


def test_strict_off_skips_mismatch(tmp_path: Path, monkeypatch):
    _write(tmp_path / "poly.txt")

    class Dummy:
        backend = "lightgbm"
        def feature_name(self):
            return ["a"]
        def predict(self, X, num_threads=1):
            return []

    monkeypatch.setattr(gp, "LGBMModel", lambda txt: Dummy())
    out = gp.load_models(
        tmp_path,
        backend="lightgbm",
        names=("poly",),
        expected_sha256={"poly": "0" * 64},
        environ={"EXPDS_PIXGBM_SHA_STRICT": "0"},
    )
    assert "poly" in out


def test_default_expected_matches_manifest():
    exp = gp.expected_pixgbm_sha256(cfg={}, environ={})
    assert exp["poly"] == gp.DEFAULT_EXPECTED_SHA256["poly"]
    assert exp["finds"] == gp.DEFAULT_EXPECTED_SHA256["finds"]
    # docs JSON should agree
    import json
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "docs" / "weights_expected_sha256.json").read_text(encoding="utf-8"))
    assert data["pixgbm"]["poly"] == exp["poly"]
