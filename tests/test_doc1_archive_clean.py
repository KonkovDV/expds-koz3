"""W4 DOC-1: проверка архива на файлы, производные от EN, и на включённый legacy PIXGBM."""
import importlib.util
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _zip(tmp_path, files):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        for n, t in files.items():
            f.writestr(n, t)
    return z


OFF = '[geophys.pixgbm]\nenabled = false\n'


@pytest.mark.parametrize("bad", ["expds/geo_pixgbm.py", "expds/geo_finds_gbm.py", "models/pixgbm/all/poly.txt",
                                 "models/pixgbm/clean/finds.txt", "analysis/x.py", "scripts/pixgbm_train.py"])
def test_banned_file_fails(tmp_path, bad):
    ck = _load("check_archive_clean")
    assert ck.check(_zip(tmp_path, {"configs/default.toml": OFF, "expds/pipeline.py": "", bad: "x"}))


def test_clean_archive_passes_and_legacy_flag_fails(tmp_path):
    ck = _load("check_archive_clean")
    ok = {"configs/default.toml": OFF, "expds/pipeline.py": "", "models/pix_clean/poly.txt": "x"}
    assert ck.check(_zip(tmp_path, ok)) == []
    on = dict(ok, **{"configs/default.toml": '[geophys.pixgbm]\nenabled = true\n'})
    assert ck.check(_zip(tmp_path, on))
    clean_on = dict(ok, **{"configs/default.toml": '[geophys.pixgbm]\nenabled = true\nimpl = "clean"\n'})
    assert ck.check(_zip(tmp_path, clean_on)) == []


def test_repo_default_config_and_collect_are_clean():
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert cfg["geophys"]["pixgbm"]["enabled"] is False
    ck, bs = _load("check_archive_clean"), _load("build_submission")
    try:
        names = [p.relative_to(ROOT).as_posix() for p in bs.collect()]
    except SystemExit as e:   # ветка без весов: сборка падает раньше по другой причине
        pytest.skip(f"collect(): {e}")
    assert ck.name_violations(names) == []
    assert "check_archive_clean" in (ROOT / "scripts" / "build_submission.py").read_text(encoding="utf-8")
