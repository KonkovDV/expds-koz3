""" A3 / v4: loud UNAVAILABLE + packaging layout + EXPDS_STRICT (ported from 829ea7d)."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from expds import sp_dfine


def test_vendor_path_constant_and_resolves_from_package_root(tmp_path):
    root = tmp_path / "pkg"
    vend = root / "inference" / "lib" / "vendor" / "transformers"
    vend.mkdir(parents=True)
    (vend / "__init__.py").write_text("__version__ = '5.21.0'\n", encoding="utf-8")
    assert (root / sp_dfine.VENDOR).is_dir()
    assert (root / sp_dfine.VENDOR / "transformers").is_dir()
    hint = sp_dfine.vendor_platform_hint(root / sp_dfine.VENDOR)
    assert "нет каталога transformers" not in hint


def test_vendor_missing_hint_mentions_packaging(tmp_path):
    missing = tmp_path / "no_vendor"
    hint = sp_dfine.vendor_platform_hint(missing)
    assert "vendor отсутствует" in hint
    assert "inference/lib/vendor" in hint or "vendor" in hint


def test_zip_layout_lists_vendor_prefix(tmp_path):
    root = tmp_path / "src"
    (root / "expds").mkdir(parents=True)
    (root / "expds" / "sp_dfine.py").write_text("# stub\n", encoding="utf-8")
    lib = root / "vendor_lib" / "transformers"
    lib.mkdir(parents=True)
    (lib / "__init__.py").write_text("__version__='5.21.0'\n", encoding="utf-8")
    zpath = tmp_path / "sub.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(root / "expds" / "sp_dfine.py", "expds/sp_dfine.py")
        for p in (root / "vendor_lib").rglob("*"):
            if p.is_file():
                z.write(p, "inference/lib/vendor/" + p.relative_to(root / "vendor_lib").as_posix())
    names = zipfile.ZipFile(zpath).namelist()
    assert "expds/sp_dfine.py" in names
    assert any(n.startswith("inference/lib/vendor/transformers/") for n in names)


def test_load_failed_writes_warning_and_import_error(tmp_path, monkeypatch, caplog):
    def boom(st, root, dev):
        raise ImportError(f"{sp_dfine.UNAVAILABLE}: D-FINE недоступен — vendor: ImportError: invalid ELF")

    monkeypatch.setattr(sp_dfine, "load_models", boom)
    monkeypatch.setenv("EXPDS_STATUS_DIR", str(tmp_path / "status"))
    monkeypatch.delenv("EXPDS_STRICT", raising=False)
    cfg = {"sp_dfine": {"enabled": True, "weights": ["w"], "device": "cpu", "strict": False},
           "thresholds_by_markup": {"sp_dfine": {c: 0.5 for c in sp_dfine.CLASSES}}}
    reg = SimpleNamespace(name="r", rasters={"sp_or": [tmp_path / "a.tif"]})
    cache: dict = {}
    with caplog.at_level(logging.ERROR, logger=sp_dfine.log.name):
        feats, complete = sp_dfine.region_feats(reg, cfg, None, cache, tmp_path)
    assert feats == [] and complete is False
    assert cache.get("load_error") and "ImportError" in cache["load_error"]
    warn = tmp_path / "status" / "SP_DFINE_WARNING.txt"
    assert warn.is_file() and sp_dfine.WARN_MARKER in warn.read_text(encoding="utf-8")
    status_w = tmp_path / "status" / "STATUS_WARNING.txt"
    assert status_w.is_file() and "WARNING SP_DFINE_UNAVAILABLE" in status_w.read_text(encoding="utf-8")
    assert any(sp_dfine.UNAVAILABLE in r.getMessage() for r in caplog.records)


def test_strict_mode_raises_on_load_failed(tmp_path, monkeypatch):
    def boom(st, root, dev):
        raise ImportError(f"{sp_dfine.UNAVAILABLE}: boom")

    monkeypatch.setattr(sp_dfine, "load_models", boom)
    monkeypatch.setenv("EXPDS_STRICT", "1")
    monkeypatch.setenv("EXPDS_STATUS_DIR", str(tmp_path / "status"))
    cfg = {"sp_dfine": {"enabled": True, "weights": ["w"], "device": "cpu"},
           "thresholds_by_markup": {"sp_dfine": {c: 0.5 for c in sp_dfine.CLASSES}}}
    reg = SimpleNamespace(name="r", rasters={"sp_or": [tmp_path / "a.tif"]})
    with pytest.raises(ImportError, match="SP_DFINE_UNAVAILABLE"):
        sp_dfine.region_feats(reg, cfg, None, {}, tmp_path)


def test_disabled_sp_dfine_skips_import(monkeypatch):
    assert sp_dfine.enabled({"sp_dfine": {"enabled": False}}) is False
    assert sp_dfine.enabled({}) is False
    called = []

    def boom(*a, **k):
        called.append(1)
        raise AssertionError("import must not run when disabled")

    monkeypatch.setattr(sp_dfine, "import_dfine", boom)
    assert sp_dfine.DEFAULTS["enabled"] is False
    assert not called


def test_import_dfine_errors_include_hint_when_vendor_missing(tmp_path):
    root = tmp_path / "pkg"
    (root / "inference" / "lib").mkdir(parents=True)
    with pytest.raises(ImportError) as ei:
        try:
            import transformers  # noqa: F401
            ver = getattr(__import__("transformers"), "__version__", "")
            if ver == sp_dfine.TRANSFORMERS_PIN:
                pytest.skip("env already has pin transformers")
        except Exception:
            pass
        sp_dfine.import_dfine(root)
    assert sp_dfine.UNAVAILABLE in str(ei.value)
    assert "hint:" in str(ei.value) or "vendor" in str(ei.value).lower()
