#!/usr/bin/env python3
"""Clean-room-adjacent PIXGBM retrain wrapper (bot-6).

Reproduces / verifies models/pixgbm/all/{poly,finds}.txt against
models/pixgbm/all/MODELS_NOT_IN_GIT.md without changing inference.

Default behaviour is OFF: the script refuses to train unless
--enable-retrain is passed. It never imports or patches the
inference path used by the submission pipeline except through the
existing scripts/pixgbm.py + expds.geo_pixgbm training entry points.

Requires:
  - lightgbm==4.7.0 (pinned; refuses other versions unless --allow-other-lgbm)
  - expds/geo_pixgbm.py present in the working tree (gitignored on main;
    place the build-time copy or a branch that has it before running)
  - contest train root (e.g. F:/2/train-expds-3) — never commit data

Example (PC, after checking out bots/bot6-pixgbm-cleanroom + placing geo_pixgbm.py):

  python scripts/pixgbm_retrain_cleanroom.py \\
    --enable-retrain \\
    --root F:/2/train-expds-3 \\
    --out C:/Sloi/pc_jobs/pixgbm_retrain_bot6 \\
    --expected-hashes models/pixgbm/all/MODELS_NOT_IN_GIT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Expected sha256 from MODELS_NOT_IN_GIT.md (hardcoded fallback if file absent).
DEFAULT_EXPECTED = {
    "poly.txt": "d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800",
    "finds.txt": "5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a",
}

# Deterministic overrides applied ONLY when --enable-retrain is set.
# Original LGBM dict (geo_pixgbm.LGBM) has bagging_fraction=0.7, num_threads=2,
# no deterministic / force_col_wise — bit-exact model text is unlikely across machines.
def make_deterministic_lgbm(num_threads: int = 1) -> dict:
    """Deterministic LightGBM params. num_threads=1 → best bit-exact chance;
    higher values (TPU-VM CPU host) keep deterministic/force_col_wise/seeds but
    may still diverge across hosts — then metric fallback applies."""
    return dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        lambda_l2=2.0,
        verbose=-1,
        num_threads=max(1, int(num_threads)),
        seed=0,
        bagging_seed=0,
        feature_fraction_seed=0,
        data_random_seed=0,
        deterministic=True,
        force_col_wise=True,
    )


DETERMINISTIC_LGBM = make_deterministic_lgbm(1)

PINNED_LGBM = "4.7.0"
# Relative metric tolerance when sha256 mismatches (AUC / keep-style surrogate on OOF).
DEFAULT_METRIC_TOL = 1e-4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_expected_hashes(md_path: Path | None) -> dict[str, str]:
    if md_path is None or not md_path.is_file():
        return dict(DEFAULT_EXPECTED)
    text = md_path.read_text(encoding="utf-8")
    found = {}
    for name in ("poly.txt", "finds.txt"):
        m = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*([0-9a-f]{{64}})\s*\|", text)
        if m:
            found[name] = m.group(1)
    for k, v in DEFAULT_EXPECTED.items():
        found.setdefault(k, v)
    return found


def check_lightgbm(allow_other: bool) -> str:
    import lightgbm as lgb

    ver = getattr(lgb, "__version__", "unknown")
    if ver != PINNED_LGBM and not allow_other:
        raise SystemExit(
            f"lightgbm=={PINNED_LGBM} required (got {ver}); "
            "pip install 'lightgbm==4.7.0' or pass --allow-other-lgbm"
        )
    return ver


def apply_deterministic_params(num_threads: int = 1) -> dict:
    """Monkey-patch geo_pixgbm.LGBM for this process only. Does not write the module."""
    from expds import geo_pixgbm as G

    old = dict(G.LGBM)
    params = make_deterministic_lgbm(num_threads)
    G.LGBM.clear()
    G.LGBM.update(params)
    DETERMINISTIC_LGBM.clear()
    DETERMINISTIC_LGBM.update(params)
    return old


def run_extract_train(root: Path, out: Path, rounds: int) -> dict:
    """Call existing scripts/pixgbm.py extract + train as subprocess for auditability."""
    pix = ROOT / "scripts" / "pixgbm.py"
    if not pix.is_file():
        raise SystemExit(f"missing {pix}")
    feats = out / "feats"
    models = out / "models"
    feats.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r1 = subprocess.run(
        [sys.executable, str(pix), "extract", "--root", str(root), "--out", str(feats)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r1.returncode != 0:
        raise SystemExit(f"extract failed:\n{r1.stderr[-4000:]}")
    r2 = subprocess.run(
        [
            sys.executable,
            str(pix),
            "train",
            "--feats",
            str(feats),
            "--models",
            str(models),
            "--rounds",
            str(rounds),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r2.returncode != 0:
        raise SystemExit(f"train failed:\n{r2.stderr[-4000:]}")
    return {
        "extract_sec": None,
        "train_rounds": rounds,
        "wall_sec": round(time.time() - t0, 1),
        "extract_tail": r1.stdout[-2000:],
        "train_tail": r2.stdout[-2000:],
        "models_dir": str(models),
    }


def train_inprocess(root: Path, out: Path, rounds: int, num_threads: int = 1) -> dict:
    """In-process train with DETERMINISTIC_LGBM applied to G.LGBM before fit."""
    # Import after path setup; patch before calling train helpers.
    apply_deterministic_params(num_threads)
    # Reuse scripts/pixgbm helpers by importing the module as a library.
    import importlib.util

    spec = importlib.util.spec_from_file_location("pixgbm_cli", ROOT / "scripts" / "pixgbm.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    feats = out / "feats"
    models = out / "models"
    feats.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    class A:
        pass

    ae = A()
    ae.root = str(root)
    ae.out = str(feats)
    ae.config = str(ROOT / "configs" / "default.toml")
    ae.regions = None
    mod.cmd_extract(ae)

    at = A()
    at.feats = str(feats)
    at.models = str(models)
    at.rounds = rounds
    at.config = ae.config
    at.regions = None
    mod.cmd_train(at)

    return {
        "mode": "inprocess_deterministic",
        "train_rounds": rounds,
        "wall_sec": round(time.time() - t0, 1),
        "models_dir": str(models),
        "lgbm_params": DETERMINISTIC_LGBM,
    }


def compare_models(models_all: Path, expected: dict[str, str], metric_tol: float) -> dict:
    result = {"sha256": {}, "match": {}, "all_sha_ok": True, "metric_fallback": None}
    for name, exp in expected.items():
        p = models_all / name
        if not p.is_file():
            result["sha256"][name] = None
            result["match"][name] = False
            result["all_sha_ok"] = False
            continue
        got = sha256_file(p)
        result["sha256"][name] = got
        ok = got == exp
        result["match"][name] = ok
        if not ok:
            result["all_sha_ok"] = False
    if not result["all_sha_ok"]:
        # Documented fallback: compare leaf counts / feature names / a cheap self-predict
        # agreement rate between old and new boosters when reference models are provided.
        result["metric_fallback"] = {
            "note": (
                "sha256 mismatch is expected across hosts when original training used "
                "num_threads=2 without deterministic=True. Fallback: load both boosters "
                "and compare predict() on a fixed random feature matrix; require "
                f"max|Δp| ≤ {metric_tol} OR identical feature_name() and num_trees()."
            ),
            "tolerance": metric_tol,
            "status": "pending_manual_or_reference",
        }
        ref_dir = ROOT / "models" / "pixgbm" / "all"
        try:
            import lightgbm as lgb
            import numpy as np

            checks = {}
            for name in expected:
                new_p = models_all / name
                ref_p = ref_dir / name
                if not new_p.is_file():
                    checks[name] = {"error": "missing new"}
                    continue
                bn = lgb.Booster(model_file=str(new_p))
                entry = {
                    "num_trees": bn.num_trees(),
                    "feature_name": bn.feature_name(),
                }
                if ref_p.is_file():
                    br = lgb.Booster(model_file=str(ref_p))
                    rng = np.random.default_rng(0)
                    X = rng.normal(size=(512, len(bn.feature_name()))).astype(np.float32)
                    pn = bn.predict(X, num_threads=1)
                    pr = br.predict(X, num_threads=1)
                    mad = float(np.max(np.abs(pn - pr)))
                    entry["ref_num_trees"] = br.num_trees()
                    entry["ref_feature_name_equal"] = br.feature_name() == bn.feature_name()
                    entry["max_abs_predict_delta"] = mad
                    entry["within_tol"] = mad <= metric_tol
                checks[name] = entry
            result["metric_fallback"]["checks"] = checks
            # Vacuous True when every check is "missing new" (no within_tol keys) must NOT pass.
            tol_vals = [c.get("within_tol") for c in checks.values() if "within_tol" in c]
            missing = any(c.get("error") == "missing new" for c in checks.values())
            if missing or not tol_vals:
                result["metric_fallback"]["status"] = "mismatch_or_no_ref"
            elif all(tol_vals):
                result["metric_fallback"]["status"] = "ok"
            else:
                result["metric_fallback"]["status"] = "mismatch_or_no_ref"
        except Exception as e:  # noqa: BLE001 — log and continue
            result["metric_fallback"]["error"] = repr(e)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--enable-retrain",
        action="store_true",
        default=False,
        help="REQUIRED to actually train. Default OFF — script only validates env and exits 2.",
    )
    ap.add_argument("--root", type=Path, help="Contest train data root (e.g. F:/2/train-expds-3)")
    ap.add_argument("--out", type=Path, help="Output directory for feats/, models/, retrain_log.json")
    ap.add_argument("--rounds", type=int, default=300, help="Boosting rounds (default 300 = geo_pixgbm.ROUNDS)")
    ap.add_argument(
        "--expected-hashes",
        type=Path,
        default=ROOT / "models" / "pixgbm" / "all" / "MODELS_NOT_IN_GIT.md",
        help="Markdown table with expected sha256",
    )
    ap.add_argument("--metric-tol", type=float, default=DEFAULT_METRIC_TOL)
    ap.add_argument("--allow-other-lgbm", action="store_true")
    ap.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="LightGBM num_threads; 0/default = os.cpu_count() (set 1 for max bit-exact chance).",
    )
    ap.add_argument(
        "--subprocess",
        action="store_true",
        help="Run extract/train as subprocess without deterministic patch "
        "(legacy reproduction of original train path).",
    )
    ap.add_argument(
        "--models-subdir",
        default="all",
        help="Subdir under --out/models to compare (default: all)",
    )
    a = ap.parse_args()
    if a.num_threads is None or a.num_threads <= 0:
        import os as _os
        env_nt = _os.environ.get("PIXGBM_NUM_THREADS", "").strip()
        if env_nt:
            a.num_threads = max(1, int(env_nt))
        else:
            a.num_threads = max(1, int(_os.cpu_count() or 1))

    log: dict = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pinned_lightgbm": PINNED_LGBM,
        "enable_retrain": bool(a.enable_retrain),
        "argv": sys.argv,
        "repo_root": str(ROOT),
    }

    try:
        log["lightgbm_version"] = check_lightgbm(a.allow_other_lgbm)
    except ModuleNotFoundError as e:
        log["lightgbm_version"] = None
        log["lightgbm_import_error"] = repr(e)
        if a.enable_retrain:
            _write_log(a.out, log)
            raise SystemExit("lightgbm not installed; pip install 'lightgbm==4.7.0'") from e
    except SystemExit as e:
        log["error"] = str(e)
        _write_log(a.out, log)
        raise

    geo = ROOT / "expds" / "geo_pixgbm.py"
    log["geo_pixgbm_present"] = geo.is_file()
    if geo.is_file():
        log["geo_pixgbm_sha256"] = sha256_file(geo)

    if not a.enable_retrain:
        log["status"] = "disabled_default_off"
        log["hint"] = "Re-run with --enable-retrain --root DATA --out OUT"
        _write_log(a.out, log)
        print(json.dumps(log, ensure_ascii=False, indent=2))
        sys.exit(2)

    if a.root is None or a.out is None:
        raise SystemExit("--root and --out required when --enable-retrain is set")
    if not geo.is_file():
        raise SystemExit(
            "expds/geo_pixgbm.py missing (gitignored on main). "
            "Place the build-time module (sha256 35e4331b… per docs/PIXGBM_DECISION.md) before retraining."
        )

    a.out.mkdir(parents=True, exist_ok=True)
    expected = parse_expected_hashes(a.expected_hashes)
    log["expected_sha256"] = expected

    if a.subprocess:
        train_info = run_extract_train(a.root, a.out, a.rounds)
    else:
        train_info = train_inprocess(a.root, a.out, a.rounds, num_threads=a.num_threads)
    log["train"] = train_info

    models_all = Path(train_info["models_dir"]) / a.models_subdir
    missing_models = [n for n in expected if not (models_all / n).is_file()]
    if missing_models:
        log["compare"] = {
            "sha256": {n: None for n in expected},
            "match": {n: False for n in expected},
            "all_sha_ok": False,
            "metric_fallback": {
                "status": "mismatch_or_no_ref",
                "error": f"missing models after train: {missing_models}",
            },
        }
        log["status"] = "train_produced_no_models"
        log["hint"] = (
            "extract/train wrote no models/all/{poly,finds}.txt — check --root has "
            "geo 200_–217_ rasters + labels (ObjectPoly/FindsPoints)."
        )
        _write_log(a.out, log)
        print(json.dumps(log, ensure_ascii=False, indent=2))
        sys.exit(1)

    log["compare"] = compare_models(models_all, expected, a.metric_tol)
    log["status"] = "ok" if log["compare"]["all_sha_ok"] else "sha_mismatch_see_metric_fallback"
    _write_log(a.out, log)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    sys.exit(0 if log["compare"]["all_sha_ok"] or (log["compare"].get("metric_fallback") or {}).get("status") == "ok" else 1)


def _write_log(out: Path | None, log: dict) -> None:
    if out is None:
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrain_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
