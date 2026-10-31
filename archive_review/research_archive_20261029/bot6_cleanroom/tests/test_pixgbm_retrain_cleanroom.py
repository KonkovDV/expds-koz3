"""bot-6: flag-default-OFF and synthetic LightGBM determinism (no contest data)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pixgbm_retrain_cleanroom.py"


def _load():
    spec = importlib.util.spec_from_file_location("pixgbm_retrain_cleanroom", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_enable_retrain_defaults_off():
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 2
    assert "disabled_default_off" in r.stdout


def test_parse_expected_hashes_fallback():
    mod = _load()
    h = mod.parse_expected_hashes(None)
    assert h["poly.txt"].startswith("d1fe3efa")
    assert h["finds.txt"].startswith("5fa8c09d")
    md = ROOT / "models" / "pixgbm" / "all" / "MODELS_NOT_IN_GIT.md"
    if md.is_file():
        h2 = mod.parse_expected_hashes(md)
        assert h2["poly.txt"] == h["poly.txt"]


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm missing")
def test_synthetic_deterministic_lgbm_repeatable():
    """Same params + seed + num_threads=1 → identical model string twice (synthetic X,y)."""
    import lightgbm as lgb

    mod = _load()
    params = mod.make_deterministic_lgbm(1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    texts = []
    for _ in range(2):
        b = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=20)
        texts.append(b.model_to_string())
    assert texts[0] == texts[1]
