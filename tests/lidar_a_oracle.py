"""Test-time loader of the reference implementation (read-only local copy; never vendored into this repo).

EXPDS_LIDAR_A_ORACLE points at the reference checkout (default: the bot workspace copy). Tests that need it call
`oracle(name)` and are skipped when the copy is absent.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ORACLE_ROOT = Path(os.environ.get("EXPDS_LIDAR_A_ORACLE", "/workspace/koz3_shared/competitors/EhimenNathan-EXpedition"))
_cache = {}


def oracle_available():
    return all((ORACLE_ROOT / "koz3" / f"{n}.py").is_file() for n in ("dtm", "relief", "hm"))


def oracle(name):
    """Reference module koz3/<name>.py loaded by file path under a private name (skip if missing)."""
    if not oracle_available():
        pytest.skip(f"reference copy not found at {ORACLE_ROOT}")
    if name not in _cache:
        spec = importlib.util.spec_from_file_location(f"_lidar_a_oracle_{name}", ORACLE_ROOT / "koz3" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cache[name] = mod
    return _cache[name]


def oracle_relief(rel, extra, unclip=False):
    """Context-free switch of the reference relief flags (module-level constants there)."""
    rel.EXTRA, rel.N_CH, rel.UNCLIP = extra, 12 if extra else 9, unclip
    return rel
