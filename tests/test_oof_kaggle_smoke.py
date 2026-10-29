"""CPU smoke для GPU-5 OOF Kaggle (без torch/S3/GPU)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts" / "oof_kaggle_jobs.py"
KERNEL = ROOT / "kaggle" / "2026-10-29" / "oof_honest10" / "koz3-bot8-oof-honest10.py"


def _load():
    spec = importlib.util.spec_from_file_location("oof_kaggle_jobs", LIB)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["oof_kaggle_jobs"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lib():
    return _load()


def test_job_catalog(lib):
    jobs = lib.select_jobs()
    assert len(jobs) >= 8
    ids = {j["id"] for j in jobs}
    assert "spor_focal" in ids and "li_gor_mask" in ids and "ae_fort_mask" in ids
    outs = {j["out"] for j in jobs}
    assert "spor_focal_oof.geojson" in outs
    assert "li_gor_mask_oof.geojson" in outs


def test_select_jobs_filter(lib):
    js = lib.select_jobs(["spor_gor", "ae_fort_mask"])
    assert [j["id"] for j in js] == ["spor_gor", "ae_fort_mask"]
    with pytest.raises(KeyError):
        lib.select_jobs(["nope"])


def test_smoke_writes_status_and_sha(lib, tmp_path: Path):
    r1 = lib.run_smoke(tmp_path)
    assert r1["skipped"] is False
    assert (tmp_path / "STATUS.txt").read_text(encoding="utf-8").startswith("DONE")
    assert (tmp_path / "SHA256SUMS").is_file()
    out = tmp_path / "oof" / "spor_focal_oof.geojson"
    assert out.is_file()
    assert r1["sha256"] == lib.sha256_file(out)
    assert r1["n_features"] == 2
    # idempotent rerun
    r2 = lib.run_smoke(tmp_path)
    assert r2["skipped"] is True
    assert r2["sha256"] == r1["sha256"]


def test_is_job_complete_requires_matching_sha(lib, tmp_path: Path):
    job = lib.select_jobs(["spor_focal"])[0]
    out = lib.job_out_path(tmp_path, job["out"])
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(lib.synthetic_fc(1)), encoding="utf-8")
    lib.write_done(tmp_path, job, "deadbeef", 1, 0.1)
    assert not lib.is_job_complete(tmp_path, job)  # sha mismatch
    sha = lib.sha256_file(out)
    lib.write_done(tmp_path, job, sha, 1, 0.1)
    assert lib.is_job_complete(tmp_path, job)


def test_kernel_smoke_env(tmp_path: Path, monkeypatch):
    """Ядро с OOF_SMOKE=1 — без GPU/S3, exit 0, STATUS DONE."""
    if not KERNEL.is_file() or not LIB.is_file():
        pytest.skip("kernel files missing")
    out = tmp_path / "oof_honest10"
    env = {
        **dict(**{k: v for k, v in __import__("os").environ.items()}),
        "OOF_SMOKE": "1",
        "OOF_OUT": str(out),
    }
    # run from repo root so scripts/ is findable; also lib copied beside kernel
    r = subprocess.run(
        [sys.executable, str(KERNEL)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    status = (out / "STATUS.txt").read_text(encoding="utf-8")
    assert status.startswith("DONE")
    assert (out / "oof" / "spor_focal_oof.geojson").is_file()
