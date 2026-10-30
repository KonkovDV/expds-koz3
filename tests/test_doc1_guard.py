"""W4 DOC-1 (30.10): жёсткий страж архива — имена EN, sha256-список, шаблон в содержимом, ссылки конфига."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_archive_clean as G  # noqa: E402

CFG_OK = '[geophys.pixgbm]\nenabled = true\nimpl = "clean"\nmodel_dir = "models/pix_clean"\n'


def make_zip(tmp_path: Path, files: dict[str, bytes]) -> Path:
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        for n, b in files.items():
            f.writestr(n, b)
    return z


@pytest.fixture
def deny(tmp_path):
    p = tmp_path / "deny.sha256"
    p.write_text("# test\n" + hashlib.sha256(b"fake EN-derived bytes\n").hexdigest() + "  test:fake\n")
    return p


BASE = {"metadata.json": b"{}", "inference/main.py": b"print(1)\n", "configs/default.toml": CFG_OK.encode()}


def test_clean_passes(tmp_path, deny):
    assert G.check(make_zip(tmp_path, BASE), deny) == []


@pytest.mark.parametrize("name", ["expds/geo_pixgbm.py", "inference/lib/geo_pixgbm.py", "expds/geo_finds_gbm.py",
                                  "models/pixgbm/all/poly.txt", "analysis/x.py", "w4pix/k/run.py",
                                  "scripts/pixgbm_compile.py", "expds/pix_legacy.py"])
def test_banned_names(tmp_path, deny, name):
    bad = G.check(make_zip(tmp_path, {**BASE, name: b"x = 1\n"}), deny)
    assert any(name in b for b in bad), bad


def test_denylisted_sha_under_innocent_name(tmp_path, deny):
    bad = G.check(make_zip(tmp_path, {**BASE, "expds/innocent.py": b"fake EN-derived bytes\n"}), deny)
    assert any("sha256" in b and "innocent" in b for b in bad), bad


@pytest.mark.parametrize("text", [b"from . import geo_pixgbm\n", b"# port of ehimennathan code\n",
                                  b'md = "models/PIXGBM/all"\n'])
def test_content_pattern(tmp_path, deny, text):
    bad = G.check(make_zip(tmp_path, {**BASE, "expds/x.py": text}), deny)
    assert any("шаблон" in b for b in bad), bad


@pytest.mark.parametrize("cfg", ['[geophys.pixgbm]\nenabled = true\nimpl = "legacy"\n',
                                 '[kurgan]\nmodel = "runs/kurgan_pix_s3/model.pt"\n',
                                 '[x]\nsrc = "w4pix/k/out"\n',
                                 '[x]\nsrc = "koz3-w2-hillshade-parity-s3"\n'])
def test_config_refs(tmp_path, deny, cfg):
    assert G.check(make_zip(tmp_path, {**BASE, "configs/default.toml": cfg.encode()}), deny)


def test_missing_denylist_fails(tmp_path):
    assert G.check(make_zip(tmp_path, BASE), tmp_path / "nope.sha256")


def test_repo_denylist_loads_and_excludes_empty():
    d = G.load_denylist()
    assert len(d) >= 10 and hashlib.sha256(b"").hexdigest() not in d


def test_build_with_planted_geo_pixgbm_fails(tmp_path):
    """Сквозной негатив: сборка из git с подложенным файлом geo_pixgbm.py (через --so) — exit ≠ 0, архива нет."""
    fake = tmp_path / "geo_pixgbm.py"
    fake.write_text("# planted fake\n")
    out = tmp_path / "dist"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_submission_git.py"), "--repo", str(ROOT),
                        "--ref", "HEAD", "--label", "doc1neg", "--out-dir", str(out), "--so", str(fake)],
                       capture_output=True, text=True, timeout=1200)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "DOC-1" in r.stderr and "geo_pixgbm.py" in r.stderr
    assert not list(out.glob("*.zip")) and not list(out.glob("*.part"))
