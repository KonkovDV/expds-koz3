"""Tests for release assembler: profiles, inject sha, compliance negatives, OFF≠v2."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


apply_mod = _load("apply_release_profile", "scripts/apply_release_profile.py")
kit_mod = _load("kit_denylist", "scripts/kit_denylist.py")
assemble_mod = _load("assemble_release", "scripts/assemble_release.py")
compare_mod = _load("compare_submission_bytes", "scripts/compare_submission_bytes.py")


def test_profiles_exist():
    d = ROOT / "configs" / "release" / "profiles"
    for name in ("off", "fr1h", "sat2", "dfine_ae", "fr1h_sat2_dfine_ae"):
        assert (d / f"{name}.toml").is_file()


def test_off_profile_no_features(tmp_path: Path):
    profile, cfg, out = apply_mod.materialize("off", root=ROOT, out_toml=tmp_path / "off.toml")
    assert profile["features"] == {"fr1h": False, "sat2": False, "dfine_ae": False}
    assert "frame_prior" not in (cfg.get("post") or {})
    assert not (cfg.get("dfine_aeor") or {}).get("enabled")
    assert out.is_file()


def test_fr1h_profile_enables_frame_prior(tmp_path: Path):
    profile, cfg, _ = apply_mod.materialize("fr1h", root=ROOT, out_toml=tmp_path / "fr1h.toml")
    assert profile["features"]["fr1h"] is True
    fp = (cfg.get("post") or {}).get("frame_prior")
    assert fp is not None
    assert "sp_or" in fp.get("groups", [])


def test_dfine_ae_profile_enables_flag(tmp_path: Path):
    profile, cfg, _ = apply_mod.materialize("dfine_ae", root=ROOT, out_toml=tmp_path / "df.toml")
    assert profile["features"]["dfine_ae"] is True
    assert (cfg.get("dfine_aeor") or {}).get("enabled") is True


def test_unknown_profile():
    with pytest.raises(FileNotFoundError):
        apply_mod.load_profile("no_such_profile_xyz")


def test_inject_bad_sha_fails_loudly(tmp_path: Path, monkeypatch):
    """Negative: corrupted inject target → v2_release_artifacts --check fails."""
    # Point ROOT inject at a temp tree is hard; instead call compare path +
    # simulate MANIFEST check by writing wrong bytes next to a copied check.
    # Use assemble step_inject with a monkeypatched script that exits 1.
    fake = tmp_path / "bad_inject.py"
    fake.write_text("import sys; print('FAIL  models/pixgbm/all/poly.txt: sha256 deadbeef ≠ d1fe3efa'); sys.exit(1)\n")
    calls = []

    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        import subprocess

        if "v2_release_artifacts" in " ".join(cmd) or str(fake) in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="FAIL bad sha\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    # Direct unit: compare_mod already covers mismatch; here ensure assemble surfaces inject rc
    monkeypatch.setattr(assemble_mod, "_run", lambda cmd, cwd=None: __import__("subprocess").CompletedProcess(
        cmd, 1, stdout="FAIL  models/pixgbm/all/poly.txt: sha256 deadbeef\n", stderr=""
    ))
    rc = assemble_mod.step_inject(skip=False)
    assert rc == 1


def test_sat2_without_pack_fails(tmp_path: Path):
    profile = {"features": {"sat2": True, "fr1h": False, "dfine_ae": False}, "artifacts": {}}
    assert assemble_mod.step_sat2_artifact(profile, tmp_path, None) == 1


def test_sat2_with_pack_copies(tmp_path: Path):
    pack = tmp_path / "src.zip"
    pack.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty zip EOCD
    profile = {
        "features": {"sat2": True, "fr1h": False, "dfine_ae": False},
        "artifacts": {},
    }
    assert assemble_mod.step_sat2_artifact(profile, tmp_path / "out", pack) == 0
    assert (tmp_path / "out" / "sat2_replace_pack.zip").is_file()


def _poly_fc():
    ring = [[4_200_000.0, 7_500_000.0], [4_200_050.0, 7_500_000.0],
            [4_200_050.0, 7_500_050.0], [4_200_000.0, 7_500_050.0], [4_200_000.0, 7_500_000.0]]
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "fid": 0, "class_name": "kurgany_tselye", "confidence": 0.9,
                "region_name": "t", "crs": "EPSG:3857", "markup_type": "sp_or",
            },
        }],
    }


def test_oversize_scored_zip_fails(tmp_path: Path):
    # Build a zip > 100 MB is slow; instead unit-test the size branch via direct check
    # by writing a small zip and monkeypatching MAX / using kit_members + size logic.
    p = tmp_path / "big.zip"
    # Write ~1 MB then claim limit is tiny via monkeypatch
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.geojson", json.dumps(_poly_fc()))
    p.write_bytes(buf.getvalue())
    old = assemble_mod.MAX_SUBMISSION_MB
    assemble_mod.MAX_SUBMISSION_MB = 0.000001  # essentially anything fails
    try:
        rc = assemble_mod.step_submission_compliance(p)
    finally:
        assemble_mod.MAX_SUBMISSION_MB = old
    assert rc == 1


def test_kit_file_in_scored_zip_fails(tmp_path: Path):
    p = tmp_path / "kitish.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.geojson", json.dumps(_poly_fc()))
        zf.writestr("models/li_r18.pt", b"FAKEWEIGHT")
    p.write_bytes(buf.getvalue())
    rc = assemble_mod.step_submission_compliance(p)
    assert rc == 1


def test_off_ne_v2_byte_mismatch(tmp_path: Path):
    ref = tmp_path / "v2.bin"
    cand = tmp_path / "off.bin"
    ref.write_bytes(b"v2-reference-bytes")
    cand.write_bytes(b"off-but-different")
    assert compare_mod.main([str(ref), str(cand)]) == 1
    rc = assemble_mod.step_byte_compare(ref, cand, required_for_off=True, is_off=True)
    assert rc == 1


def test_off_eq_v2_byte_pass(tmp_path: Path):
    ref = tmp_path / "v2.bin"
    cand = tmp_path / "off.bin"
    ref.write_bytes(b"same-bytes")
    cand.write_bytes(b"same-bytes")
    assert assemble_mod.step_byte_compare(ref, cand, required_for_off=True, is_off=True) == 0


def test_kit_denylist_flags_weights():
    bad = kit_mod.kit_members(["result.geojson", "models/foo.pt", "expds/geo_pixgbm.py"])
    assert bad
    assert any("foo.pt" in x or "weight" in x for x in bad)


def test_assemble_skip_build_off_profile(tmp_path: Path):
    """End-to-end-ish on box: inject + profile + budget, no kit build."""
    # inject may PUT files into the real worktree — that's OK (gitignored).
    rc = assemble_mod.main([
        "--profile", "off",
        "--out-dir", str(tmp_path / "dist"),
        "--skip-build",
    ])
    assert rc == 0
    assert (tmp_path / "dist" / "effective_off.toml").is_file()
    assert (tmp_path / "dist" / "RELEASE_MANIFEST.json").is_file()
    man = json.loads((tmp_path / "dist" / "RELEASE_MANIFEST.json").read_text())
    assert man["profile"] == "off"
    assert man["ok"] is True


def test_budget_guard_soft_ceiling():
    bg = _load("budget_guard", "scripts/budget_guard.py")
    assert bg.check(1200.0, None, 1200.0) == []
    assert bg.check(1500.0, None, 1200.0)
    assert bg.check(None, 1300.0, 1200.0)
