"""Byte-compare helper for OFF==v2 gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_submission_bytes.py"
spec = importlib.util.spec_from_file_location("compare_submission_bytes", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_identical(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello-v2")
    b.write_bytes(b"hello-v2")
    assert mod.compare(a, b) == []
    assert mod.main([str(a), str(b)]) == 0


def test_mismatch_summary(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"AAAA")
    b.write_bytes(b"AAAB")
    errs = mod.compare(a, b)
    assert errs
    assert any("sha256" in e for e in errs)
    assert any("first_diff_offset=3" in e for e in errs)
    assert mod.main([str(a), str(b)]) == 1


def test_missing_ref(tmp_path: Path):
    b = tmp_path / "b.bin"
    b.write_bytes(b"x")
    assert mod.main([str(tmp_path / "nope"), str(b)]) == 1
