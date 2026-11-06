"""flag-default-OFF and synthetic LightGBM determinism (no contest data)."""
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


def test_smoke_cli_two_runs_identical_sha256():
    r = subprocess.run([sys.executable, str(SCRIPT), "--smoke"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "SMOKE_IDENTICAL" in r.stdout
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("smoke sha256")]
    assert len(lines) == 2
    assert lines[0].split()[-1] == lines[1].split()[-1]


def test_roc_auc_perfect_and_ties():
    mod = _load()
    y = np.array([0, 0, 1, 1], np.float32)
    assert mod.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert abs(mod.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9
    assert np.isnan(mod.roc_auc(np.ones(3), np.array([0.2, 0.3, 0.4])))


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


def test_compare_models_missing_not_vacuous_ok(tmp_path):
    """Missing poly/finds must NOT yield metric_fallback.status=ok (Kaggle v6 bug)."""
    mod = _load()
    expected = {"poly.txt": "aa", "finds.txt": "bb"}
    empty = tmp_path / "all"
    empty.mkdir()
    r = mod.compare_models(empty, expected, 1e-4)
    assert r["all_sha_ok"] is False
    assert r["metric_fallback"]["status"] != "ok"
    assert r["metric_fallback"]["checks"]["poly.txt"]["error"] == "missing new"


def test_rss_mb_importable_without_resource(monkeypatch):
    """scripts/pixgbm.rss_mb must not hard-require Unix `resource` (Windows PC)."""
    import importlib.util

    # Simulate missing resource by shadowing before load is heavy; just call helper
    # via a tiny exec of the function body pattern used in pixgbm.py.
    spec = importlib.util.spec_from_file_location("pixgbm_cli_win", ROOT / "scripts" / "pixgbm.py")
    # Loading pixgbm.py pulls geo_pixgbm — skip if unavailable.
    if not (ROOT / "expds" / "geo_pixgbm.py").is_file():
        pytest.skip("geo_pixgbm.py absent")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Ensure top-level import does not require `resource`
    src = (ROOT / "scripts" / "pixgbm.py").read_text(encoding="utf-8")
    assert "\nimport resource\n" not in src
    assert "def rss_mb(" in src


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm missing")
def test_save_booster_non_ascii_path(tmp_path):
    """LightGBM save/load via ASCII tempfile when destination path has non-ASCII (Cyrillic)."""
    import lightgbm as lgb

    pix_path = ROOT / "scripts" / "pixgbm.py"
    src = pix_path.read_text(encoding="utf-8")
    assert "def save_booster(" in src
    assert "def load_booster(" in src
    start = src.index("def _path_has_non_ascii")
    end = src.index("\ndef rss_mb")
    ns: dict = {}
    exec(
        "import os\nimport shutil\nimport tempfile\nfrom pathlib import Path\n"
        "from expds.ascii_tmp import ascii_tmpdir\n" + src[start:end],
        ns,
    )
    save_booster = ns["save_booster"]
    load_booster = ns["load_booster"]
    assert ns["_path_has_non_ascii"](tmp_path / "Гнёздово" / "poly.txt") is True
    assert ns["_path_has_non_ascii"](tmp_path / "all" / "poly.txt") is False

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    bst = lgb.train(
        dict(objective="binary", verbose=-1, num_threads=1, seed=0),
        lgb.Dataset(X, label=y),
        num_boost_round=5,
    )
    dest = tmp_path / "семья_Гнёздово" / "poly.txt"
    save_booster(bst, dest)
    assert dest.is_file() and dest.stat().st_size > 100
    loaded = load_booster(dest)
    assert loaded.num_trees() == bst.num_trees()
    pred_a = bst.predict(X[:10], num_threads=1)
    pred_b = loaded.predict(X[:10], num_threads=1)
    assert np.allclose(pred_a, pred_b)


def test_make_reference_lgbm_omits_forced_seeds():
    """REF-matching params must not force bagging_seed=0 / deterministic."""
    mod = _load()
    ref = mod.make_reference_lgbm(2)
    assert ref["num_threads"] == 2
    assert ref["seed"] == 0
    assert "bagging_seed" not in ref
    assert "feature_fraction_seed" not in ref
    assert "data_random_seed" not in ref
    assert "deterministic" not in ref
    assert "force_col_wise" not in ref
    det = mod.make_deterministic_lgbm(12)
    assert det["deterministic"] is True
    assert det["bagging_seed"] == 0
    assert det["num_threads"] == 12


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm missing")
def test_reference_seed_derivation_matches_ref_model():
    """LightGBM 4.7 + seed=0 alone → bagging_seed=7719 etc. (same as REF poly.txt)."""
    import lightgbm as lgb

    mod = _load()
    params = mod.make_reference_lgbm(2)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    b = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=3)
    text = b.model_to_string()
    assert "[bagging_seed: 7719]" in text
    assert "[feature_fraction_seed: 2437]" in text
    assert "[data_random_seed: 38]" in text
    assert "[deterministic: 0]" in text
    assert "[force_col_wise: 0]" in text
    assert "[num_threads: 2]" in text
