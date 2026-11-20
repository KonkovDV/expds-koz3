"""Манифест сданного архива (docs/final_archive.sha256) и всё, что из него собирается.

178 членов архива финальной версии делятся на четыре источника: 62 файла из git, 51 файл весов
(models/MANIFEST.sha256), 64 файла vendor/ и колесо RVT (публичные пакеты, scripts/fetch_thirdparty.py).
Тесты ниже идут и без весов: файлы из git должны совпадать с архивом побайтно. Иначе правка кода, который
лежит в архиве, тихо ломает воспроизводимость сданной версии, а полный тест сборки
(test_final_config.py) без весов пропускается.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

from conftest import ROOT

MANIFEST = ROOT / "docs" / "final_archive.sha256"
MEMBERS = 178


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_fa_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest() -> dict[str, str]:
    return _load("verify_archive").load_manifest(MANIFEST)


def _tracked() -> set[str]:
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("нет git: рабочая копия без истории")
    return {p for p in out.decode("utf-8").split("\0") if p}


def test_manifest_is_well_formed():
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    names = [line.split(maxsplit=1)[1] for line in lines]
    assert len(lines) == MEMBERS
    assert names == sorted(names) and len(set(names)) == MEMBERS
    for line in lines:
        sha, name = line.split("  ", 1)
        assert len(sha) == 64 and int(sha, 16) >= 0 and not name.startswith(("/", "..")), line


def test_git_members_are_byte_identical_to_the_submitted_archive():
    """Код и конфиг в архиве не меняются после сдачи: каждый член архива из git совпадает с манифестом."""
    man = _manifest()
    in_git = sorted(set(man) & _tracked())
    assert len(in_git) == 62, len(in_git)
    changed = [n for n in in_git if hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != man[n]]
    assert not changed, f"изменён код сданного архива: {changed}"


def test_every_member_has_a_known_source():
    """Каждый член архива — из git, из весов, из vendor/ или колесо RVT."""
    man = _manifest()
    tracked = _tracked()
    weights = {line.split()[1] for line in (ROOT / "models" / "MANIFEST.sha256").read_text().splitlines() if line}
    rest = [n for n in man if n not in tracked and n not in weights
            and not n.startswith(("vendor/", "inference/cnx/wheels/"))]
    assert not rest, rest
    assert sum(n.startswith("vendor/") for n in man) == 64
    assert sum(n in weights for n in man) == 51


def test_weight_manifest_agrees_with_the_archive_manifest():
    man = _manifest()
    for line in (ROOT / "models" / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split()
            assert man.get(name) == sha, name


def test_cnx_manifest_agrees_with_the_archive_manifest():
    man = _manifest()
    cnx = json.loads((ROOT / "inference" / "cnx" / "MANIFEST.json").read_text(encoding="utf-8"))["sha256"]
    for rel, sha in cnx.items():
        assert man.get(f"inference/cnx/{rel}") == sha, rel


def test_builder_lays_out_exactly_the_manifest_members(monkeypatch):
    """scripts/build_submission.py кладёт в архив ровно те пути, что в манифесте (веса могут отсутствовать)."""
    builder = _load("build_submission")
    man = _manifest()
    real_is_file = Path.is_file
    # Весов в git нет: отсутствующий файл под models/ считается на месте, каталоги остаются каталогами.
    monkeypatch.setattr(Path, "is_file", lambda p: real_is_file(p) or (ROOT / "models" in p.parents and not p.exists()))
    cfg_path = ROOT / "configs" / "default.toml"
    names = {n for n, _ in builder.collect(tomllib.loads(cfg_path.read_text(encoding="utf-8")), cfg_path.resolve())}
    assert names <= set(man), sorted(names - set(man))
    must = {n for n in man if not n.startswith(("vendor/", "inference/cnx/wheels/", "models/pix_clean/"))}
    assert must <= names, sorted(must - names)


def test_scripts_agree_on_the_final_archive():
    """sha256 архива и колеса RVT одинаковы в тестах, сверке архива и загрузке сторонних файлов."""
    from test_final_config import ARCHIVE_SHA256

    assert _load("verify_archive").FINAL_SHA256 == ARCHIVE_SHA256
    fetch = _load("fetch_thirdparty")
    cnx = json.loads((ROOT / "inference" / "cnx" / "MANIFEST.json").read_text(encoding="utf-8"))["sha256"]
    assert fetch.WHEELS["rvt_py-2.2.1-py3-none-any.whl"][1] == cnx["wheels/rvt_py-2.2.1-py3-none-any.whl"]
    assert len(fetch.expected()) == 65


def _zip(path: Path, members: dict[str, bytes], date=(2026, 11, 17, 0, 0, 0), mode=0o644) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=date)
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_STORED if Path(name).suffix in (".pt", ".so", ".o", ".whl") else zipfile.ZIP_DEFLATED
            z.writestr(info, members[name])
    return path


def test_verify_archive_reports_each_kind_of_difference(tmp_path):
    va = _load("verify_archive")
    files = {"a.py": b"print(1)\n", "models/w.pt": b"\x00weights"}
    man = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    assert va.check_zip(_zip(tmp_path / "ok.zip", files), man) == []
    bad = va.check_zip(_zip(tmp_path / "content.zip", {**files, "a.py": b"print(2)\n"}), man)
    assert bad == ["другое содержимое: a.py [git]"]
    assert any("дата" in p for p in va.check_zip(_zip(tmp_path / "date.zip", files, date=(2026, 11, 18, 0, 0, 0)), man))
    assert any("права" in p for p in va.check_zip(_zip(tmp_path / "mode.zip", files, mode=0o755), man))
    missing = va.check_zip(_zip(tmp_path / "missing.zip", {"a.py": files["a.py"]}), man)
    assert missing == ["нет члена models/w.pt [веса (models/MANIFEST.sha256)]"]
    extra = va.check_zip(_zip(tmp_path / "extra.zip", {**files, "b.py": b""}), man)
    assert extra == ["лишний член b.py"]


def test_fetch_thirdparty_unpacks_into_vendor_and_rejects_escapes(tmp_path):
    fetch = _load("fetch_thirdparty")
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("pkg/", b"")
        z.writestr("pkg/__init__.py", b"x = 1\n")
        z.writestr("pkg-1.0.dist-info/LICENSE", b"MIT\n")
    assert sorted(fetch.unpack(wheel, tmp_path / "root")) == ["vendor/pkg-1.0.dist-info/LICENSE", "vendor/pkg/__init__.py"]
    assert (tmp_path / "root" / "vendor" / "pkg" / "__init__.py").read_bytes() == b"x = 1\n"
    evil = tmp_path / "evil-1.0-py3-none-any.whl"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.py", b"")
    with pytest.raises(SystemExit):
        fetch.unpack(evil, tmp_path / "root")
    assert not (tmp_path / "escape.py").exists()
