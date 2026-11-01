"""Tests for release assembler: profiles, inject sha, compliance, OFF==v2, dfine ckpt."""
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


def _poly_fc(tag: str = ""):
    ring = [
        [4_200_000.0, 7_500_000.0],
        [4_200_050.0, 7_500_000.0],
        [4_200_050.0, 7_500_050.0],
        [4_200_000.0, 7_500_050.0],
        [4_200_000.0, 7_500_000.0],
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "fid": 0,
                    "class_name": "kurgany_tselye",
                    "confidence": 0.9,
                    "region_name": "t" + tag,
                    "crs": "EPSG:3857",
                    "markup_type": "sp_or",
                },
            }
        ],
    }


def _write_geojson(path: Path, fc: dict) -> Path:
    path.write_text(json.dumps(fc, separators=(",", ":")), encoding="utf-8")
    return path


def _write_submission_zip(path: Path, fc: dict) -> Path:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.geojson", json.dumps(fc, separators=(",", ":")))
    path.write_bytes(buf.getvalue())
    return path


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
    monkeypatch.setattr(
        assemble_mod,
        "_run",
        lambda cmd, cwd=None: __import__("subprocess").CompletedProcess(
            cmd, 1, stdout="FAIL  models/pixgbm/all/poly.txt: sha256 deadbeef\n", stderr=""
        ),
    )
    rc = assemble_mod.step_inject(skip=False)
    assert rc == 1


def test_sat2_without_pack_fails(tmp_path: Path):
    profile = {"features": {"sat2": True, "fr1h": False, "dfine_ae": False}, "artifacts": {}}
    assert assemble_mod.step_sat2_artifact(profile, tmp_path, None) == 1


def test_sat2_with_pack_copies(tmp_path: Path):
    pack = tmp_path / "src.zip"
    pack.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    profile = {
        "features": {"sat2": True, "fr1h": False, "dfine_ae": False},
        "artifacts": {},
    }
    assert assemble_mod.step_sat2_artifact(profile, tmp_path / "out", pack) == 0
    assert (tmp_path / "out" / "sat2_replace_pack.zip").is_file()


def test_dfine_ae_missing_ckpt_sha_fails():
    profile = {"features": {"dfine_ae": True, "fr1h": False, "sat2": False}}
    rc, meta = assemble_mod.step_dfine_ckpt(profile, None, None)
    assert rc == 1
    assert meta is None
    rc2, _ = assemble_mod.step_dfine_ckpt(profile, None, "abc")
    assert rc2 == 1


def test_dfine_ae_sha_mismatch_fails(tmp_path: Path):
    profile = {"features": {"dfine_ae": True, "fr1h": False, "sat2": False}}
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"WEIGHTS")
    rc, _ = assemble_mod.step_dfine_ckpt(profile, ckpt, "0" * 64)
    assert rc == 1


def test_dfine_ae_ckpt_sha_ok(tmp_path: Path):
    profile = {"features": {"dfine_ae": True, "fr1h": False, "sat2": False}}
    ckpt = tmp_path / "model.safetensors"
    data = b"WEIGHTS-OK"
    ckpt.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    rc, meta = assemble_mod.step_dfine_ckpt(profile, ckpt, digest)
    assert rc == 0
    assert meta["sha256"] == digest


def test_dfine_ae_assemble_main_requires_ckpt(tmp_path: Path):
    rc = assemble_mod.main(
        [
            "--profile",
            "dfine_ae",
            "--out-dir",
            str(tmp_path / "dist"),
            "--skip-build",
            "--skip-inject",
        ]
    )
    assert rc == 1


def test_oversize_submission_zip_fails(tmp_path: Path):
    """100 MB limit applies to SUBMISSION zip only."""
    p = tmp_path / "big.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.geojson", json.dumps(_poly_fc()))
    p.write_bytes(buf.getvalue())
    old = assemble_mod.MAX_SUBMISSION_MB
    assemble_mod.MAX_SUBMISSION_MB = 0.000001
    try:
        rc, _ = assemble_mod.step_submission_compliance(p)
    finally:
        assemble_mod.MAX_SUBMISSION_MB = old
    assert rc == 1


def test_kit_size_not_subject_to_100mb(tmp_path: Path):
    """Large kit alone must not trip the 100 MB submission rule."""
    kit = tmp_path / "huge_kit.zip"
    # ~2 MB fake kit — under any reasonable disk, but we assert info-only path
    kit.write_bytes(b"0" * (2 * 1_000_000))
    old = assemble_mod.MAX_SUBMISSION_MB
    assemble_mod.MAX_SUBMISSION_MB = 0.5  # would fail if applied to kit
    try:
        rc, info = assemble_mod.step_submission_compliance(None, kit)
    finally:
        assemble_mod.MAX_SUBMISSION_MB = old
    assert rc == 0
    assert info.get("kit_size_mb", 0) >= 1.0


def test_kit_file_in_scored_zip_fails(tmp_path: Path):
    p = tmp_path / "kitish.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.geojson", json.dumps(_poly_fc()))
        zf.writestr("models/li_r18.pt", b"FAKEWEIGHT")
    p.write_bytes(buf.getvalue())
    rc, _ = assemble_mod.step_submission_compliance(p)
    assert rc == 1


def test_off_ne_v2_byte_mismatch_geojson(tmp_path: Path):
    ref = _write_geojson(tmp_path / "v2.geojson", _poly_fc())
    cand = _write_geojson(tmp_path / "off.geojson", _poly_fc(tag="DIFF"))
    assert compare_mod.main([str(ref), str(cand)]) == 1
    rc, meta = assemble_mod.step_byte_compare(ref, cand, is_off=True, skip_inject=True)
    assert rc == 1
    assert meta["v2_ref_file_sha256"]


def test_off_eq_v2_geojson_and_zip(tmp_path: Path):
    fc = _poly_fc()
    ref_gj = _write_geojson(tmp_path / "v2.geojson", fc)
    cand_zip = _write_submission_zip(tmp_path / "off.zip", fc)
    rc, meta = assemble_mod.step_byte_compare(ref_gj, cand_zip, is_off=True, skip_inject=True)
    assert rc == 0
    assert meta["v2_ref_file_sha256"] == assemble_mod.sha256_file(ref_gj)
    assert meta["v2_ref_kind"] == "geojson"

    ref_zip = _write_submission_zip(tmp_path / "v2.zip", fc)
    cand_gj = _write_geojson(tmp_path / "off2.geojson", fc)
    rc2, meta2 = assemble_mod.step_byte_compare(ref_zip, cand_gj, is_off=True, skip_inject=True)
    assert rc2 == 0
    assert meta2["v2_ref_kind"].startswith("zip:")


def test_kit_denylist_flags_weights():
    bad = kit_mod.kit_members(["result.geojson", "models/foo.pt", "expds/geo_pixgbm.py"])
    assert bad
    assert any("foo.pt" in x or "weight" in x for x in bad)


def test_assemble_skip_build_off_profile(tmp_path: Path):
    """End-to-end-ish on box: inject + profile + budget + OFF stand-in, no kit build."""
    rc = assemble_mod.main(
        [
            "--profile",
            "off",
            "--out-dir",
            str(tmp_path / "dist"),
            "--skip-build",
        ]
    )
    assert rc == 0
    assert (tmp_path / "dist" / "effective_off.toml").is_file()
    man = json.loads((tmp_path / "dist" / "RELEASE_MANIFEST.json").read_text())
    assert man["profile"] == "off"
    assert man["ok"] is True
    assert man.get("off_eq_v2_mode") == "inject_check_standin"


def test_assemble_records_v2_ref_sha(tmp_path: Path):
    fc = _poly_fc()
    ref = _write_submission_zip(tmp_path / "submission_bot7cand_v2_eb268d6.zip", fc)
    cand = _write_geojson(tmp_path / "cand.geojson", fc)
    rc = assemble_mod.main(
        [
            "--profile",
            "off",
            "--out-dir",
            str(tmp_path / "dist"),
            "--skip-build",
            "--skip-inject",
            "--v2-ref",
            str(ref),
            "--result",
            str(cand),
        ]
    )
    assert rc == 0
    man = json.loads((tmp_path / "dist" / "RELEASE_MANIFEST.json").read_text())
    assert man["v2_ref_file_sha256"] == assemble_mod.sha256_file(ref)
    assert man["off_eq_v2_mode"] == "byte_compare"


def test_budget_guard_soft_ceiling():
    bg = _load("budget_guard", "scripts/budget_guard.py")
    assert bg.check(1200.0, None, 1200.0) == []
    assert bg.check(1500.0, None, 1200.0)
    assert bg.check(None, 1300.0, 1200.0)
