"""Release v5: ≤20 min soft_ceiling / budget_guard."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "budget_guard.py"
_spec = importlib.util.spec_from_file_location("budget_guard_v5", SCRIPT)
assert _spec and _spec.loader
bg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bg)


def test_soft_ceiling_default_toml_is_1200():
    cfg = ROOT / "configs" / "default.toml"
    soft = bg.soft_ceiling_from_toml(cfg)
    assert soft == 1200.0


def test_check_pass_under_budget():
    assert bg.check(1200.0, 228.0, 1200.0) == []


def test_check_fail_wall():
    errs = bg.check(1200.0, 1300.0, 1200.0)
    assert any("wall_sec" in e for e in errs)


def test_check_fail_soft():
    errs = bg.check(1800.0, None, 1200.0)
    assert any("soft_ceiling" in e for e in errs)


def test_main_pass_config():
    assert bg.main(["--config", str(ROOT / "configs" / "default.toml"), "--require-soft-ceiling"]) == 0


def test_main_fail_wall():
    assert bg.main(["--wall-sec", "2000", "--max-sec", "1200"]) == 1
