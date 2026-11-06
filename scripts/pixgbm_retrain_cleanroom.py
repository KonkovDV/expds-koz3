#!/usr/bin/env python3
"""Clean-room-adjacent PIXGBM retrain wrapper.

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

Example (PC). The script materializes expds/geo_pixgbm.py from commit eb268d6
when it is missing (the file is gitignored). Historical params are the default
(seed=0, num_threads=2, deterministic left off — as in the saved models).

  python scripts/pixgbm_retrain_cleanroom.py --enable-retrain ^
    --root F:\\2\\train-expds-3 --out C:\\Sloi\\pc_jobs\\pixgbm_retrain

Smoke (no contest data, two deterministic runs, identical sha256):

  python scripts/pixgbm_retrain_cleanroom.py --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Pin threads before NumPy/LightGBM load. Historical reproduction uses 2,
# matching the saved models. --deterministic / --smoke force 1.
if "--smoke" in sys.argv or "--deterministic" in sys.argv:
    os.environ["OMP_NUM_THREADS"] = "1"
else:
    os.environ["OMP_NUM_THREADS"] = os.environ.get("PIXGBM_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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


def make_reference_lgbm(num_threads: int = 2) -> dict:
    """Params matching original train (geo_pixgbm.LGBM + seed=0).

    Evidence from REF models/pixgbm/all/*.txt (LightGBM 4.7.0):
      num_threads=2, deterministic=0, force_col_wise=0, seed=0,
      bagging_seed=7719, feature_fraction_seed=2437, data_random_seed=38
    Those bagging/feature/data seeds are LightGBM's derivation from seed=0 when
    they are NOT passed explicitly. Passing bagging_seed=0 (deterministic path)
    forces zeros and diverges from REF — that was the d173a75 PC fail cause #1.
    """
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
        # intentionally omit bagging_seed / feature_fraction_seed / data_random_seed
        # intentionally omit deterministic / force_col_wise
    )


REFERENCE_LGBM = make_reference_lgbm(2)

PINNED_LGBM = "4.7.0"
# Pass when sha256 misses: functional agreement on one site family.
# Holdout = lexicographically last family name in the extracted sample.
# The shipped model is `all` (trained on every family, including this slice):
# the slice checks that the retrained function matches the eb268d6 reference,
# it is not an unseen-generalization score.
AUC_TOL = 0.01
PROB_TOL = 0.02
# Deprecated name kept so older callers (`--metric-tol 1e-4`) still parse.
# Values below 0.001 were a random-matrix max|Δp| check and are not the pass line.
DEFAULT_METRIC_TOL = AUC_TOL

MODULE_COMMIT = "eb268d6"
MODULE_SHA256 = "35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b"
REFERENCE_COMMIT = "eb268d6"


def git_show(spec: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), "show", spec])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def apply_reference_params(num_threads: int = 2) -> dict:
    """Patch G.LGBM to original REF-matching params (no deterministic / forced seeds)."""
    from expds import geo_pixgbm as G

    old = dict(G.LGBM)
    params = make_reference_lgbm(num_threads)
    G.LGBM.clear()
    G.LGBM.update(params)
    REFERENCE_LGBM.clear()
    REFERENCE_LGBM.update(params)
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


def fit_all(mod, feats: Path, models: Path, rounds: int):
    """Train only the shipped `all` models (same rows as scripts/pixgbm.py held='all')."""
    parts = mod.load_feats(str(feats))
    dest = Path(models) / "all"
    dest.mkdir(parents=True, exist_ok=True)
    saved = {}
    for target, name in (("yp", "poly"), ("yf", "finds")):
        sub = [p for p in parts if (p[target] >= 0).any()]
        if not sub:
            continue
        booster, info = mod.fit(sub, target, rounds)
        mod.save_booster(booster, dest / f"{name}.txt")
        saved[name] = {
            "n": int(info.get("n", 0)),
            "pos": int(info.get("pos", 0)),
            "families": [str(f) for f in info.get("families", [])],
        }
    return parts, saved


def roc_auc(y, p) -> float:
    """ROC-AUC with average ranks for ties. NaN if a class is missing."""
    import numpy as np

    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    m = y >= 0
    y = y[m]
    p = p[m]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    n = len(p)
    while i < n:
        j = i
        while j + 1 < n and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def smoke_two_runs(rounds: int = 20) -> dict:
    """Two deterministic trainings on synthetic data. Identical sha256 is the proof."""
    import lightgbm as lgb
    import numpy as np

    params = make_deterministic_lgbm(1)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 8)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)
    names = [f"f{i}" for i in range(8)]
    w = np.ones(len(y), np.float32)
    hashes = []
    for _ in range(2):
        booster = lgb.train(
            params,
            lgb.Dataset(x, label=y, weight=w, feature_name=names),
            num_boost_round=rounds,
        )
        hashes.append(sha256_bytes(booster.model_to_string().encode()))
    return {
        "ok": hashes[0] == hashes[1],
        "sha256": hashes,
        "rounds": rounds,
        "num_threads": 1,
        "deterministic": True,
        "lightgbm": __import__("lightgbm").__version__,
    }


def ensure_geo_module(allow_mismatch: bool) -> dict:
    """Use the training-era module (eb268d6). Materialize it if the gitignored file is absent."""
    dest = ROOT / "expds" / "geo_pixgbm.py"
    info = {"path": str(dest), "pinned_sha256": MODULE_SHA256, "commit": MODULE_COMMIT}
    if dest.is_file():
        got = sha256_file(dest)
        info["sha256"] = got
        info["matched_pin"] = got == MODULE_SHA256
        if got != MODULE_SHA256 and not allow_mismatch:
            raise SystemExit(
                f"expds/geo_pixgbm.py sha256 {got} != pinned {MODULE_SHA256} ({MODULE_COMMIT}). "
                "Move that file aside or pass --allow-module-mismatch."
            )
        return info
    try:
        data = git_show(f"{MODULE_COMMIT}:expds/geo_pixgbm.py")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"cannot read {MODULE_COMMIT}:expds/geo_pixgbm.py ({exc}). "
            "The commit is an ancestor of main; a shallow clone may not have it."
        ) from exc
    got = sha256_bytes(data)
    if got != MODULE_SHA256:
        raise SystemExit(f"object sha {got} != pin {MODULE_SHA256}")
    dest.write_bytes(data)
    info["sha256"] = got
    info["matched_pin"] = True
    info["materialized"] = True
    return info


def materialize_reference(dest: Path, expected: dict[str, str]) -> dict:
    """Write eb268d6 poly.txt/finds.txt under the output dir. Do not put them in the repo."""
    dest.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, exp in expected.items():
        data = git_show(f"{REFERENCE_COMMIT}:models/pixgbm/all/{name}")
        got = sha256_bytes(data)
        (dest / name).write_bytes(data)
        info[name] = {"sha256": got, "matches_expected": got == exp}
        if got != exp:
            raise SystemExit(f"reference {name} sha {got} != {exp}")
    return info


def evaluate_holdout(parts, new_dir: Path, ref_dir: Path) -> dict:
    """Compare retrained `all` vs eb268d6 on the last family name (sorted)."""
    import lightgbm as lgb
    import numpy as np

    families = sorted({str(p["family"]) for p in parts})
    out: dict = {
        "holdout_rule": "sorted(families)[-1]; model `all` was trained on this family too",
        "auc_tol": AUC_TOL,
        "prob_tol": PROB_TOL,
        "families": families,
        "targets": {},
    }
    if not families:
        out["status"] = "no_rows"
        return out
    holdout = families[-1]
    out["holdout_family"] = holdout
    slice_parts = [p for p in parts if str(p["family"]) == holdout]
    oks = []
    for target, name in (("yp", "poly.txt"), ("yf", "finds.txt")):
        xs, ys = [], []
        for p in slice_parts:
            yy = np.asarray(p[target])
            m = yy >= 0
            if not np.any(m):
                continue
            xs.append(np.asarray(p["X"])[m])
            ys.append(yy[m])
        entry: dict = {}
        if not xs:
            entry["status"] = "no_rows"
            out["targets"][name] = entry
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys).astype(np.float32)
        entry["n"] = int(len(y))
        entry["n_pos"] = int((y == 1).sum())
        new_p, ref_p = new_dir / name, ref_dir / name
        if not new_p.is_file() or not ref_p.is_file():
            entry["status"] = "missing_model"
            oks.append(False)
            out["targets"][name] = entry
            continue
        new_b = lgb.Booster(model_file=str(new_p))
        ref_b = lgb.Booster(model_file=str(ref_p))
        if list(new_b.feature_name()) != list(ref_b.feature_name()):
            entry["status"] = "feature_mismatch"
            oks.append(False)
            out["targets"][name] = entry
            continue
        pred_new = new_b.predict(x, num_threads=1)
        pred_ref = ref_b.predict(x, num_threads=1)
        auc_new = roc_auc(y, pred_new)
        auc_ref = roc_auc(y, pred_ref)
        entry["auc_new"] = auc_new
        entry["auc_ref"] = auc_ref
        if auc_new == auc_new and auc_ref == auc_ref:
            entry["auc_abs_diff"] = abs(auc_new - auc_ref)
        else:
            entry["auc_abs_diff"] = None
        entry["mean_abs_prob"] = float(np.mean(np.abs(pred_new - pred_ref)))
        ok = (
            entry["auc_abs_diff"] is not None
            and entry["auc_abs_diff"] <= AUC_TOL
            and entry["mean_abs_prob"] <= PROB_TOL
        )
        entry["status"] = "ok" if ok else "fail"
        oks.append(ok)
        out["targets"][name] = entry
    if oks and all(oks):
        out["status"] = "ok"
    elif not oks:
        out["status"] = "no_rows"
    else:
        out["status"] = "fail"
    return out


def train_inprocess(
    root: Path, out: Path, rounds: int, num_threads: int = 1, *, match_reference: bool = False,
    only_all: bool = True,
) -> tuple[dict, list]:
    """In-process train; deterministic patch by default, or REF-matching params."""
    # Import after path setup; patch before calling train helpers.
    if match_reference:
        apply_reference_params(num_threads)
        mode = "inprocess_match_reference"
        params_snapshot = dict(REFERENCE_LGBM)
    else:
        apply_deterministic_params(num_threads)
        mode = "inprocess_deterministic"
        params_snapshot = dict(DETERMINISTIC_LGBM)
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

    parts: list = []
    if only_all:
        parts, saved = fit_all(mod, feats, models, rounds)
    else:
        at = A()
        at.feats = str(feats)
        at.models = str(models)
        at.rounds = rounds
        at.config = ae.config
        at.regions = None
        mod.cmd_train(at)
        parts = mod.load_feats(str(feats))
        saved = {}

    return {
        "mode": mode,
        "train_rounds": rounds,
        "only_all": bool(only_all),
        "wall_sec": round(time.time() - t0, 1),
        "models_dir": str(models),
        "lgbm_params": params_snapshot,
        "match_reference": bool(match_reference),
        "fit_info": saved,
    }, parts


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
        checks = {}
        for name in expected:
            if not (models_all / name).is_file():
                checks[name] = {"error": "missing new"}
        result["metric_fallback"] = {
            "note": (
                "sha256 mismatch is not decided here. The pass line is log['holdout']: "
                f"last site family, |ΔAUC| ≤ {AUC_TOL}, mean|Δp| ≤ {PROB_TOL}. "
                f"(argument metric_tol={metric_tol} is ignored.)"
            ),
            "status": "mismatch_or_no_ref" if checks else "see_holdout",
            "checks": checks,
        }
    return result


def write_reports(out: Path | None, log: dict) -> None:
    _write_log(out, log)
    if out is None:
        return
    out = Path(out)
    comp = log.get("compare") or {}
    hold = log.get("holdout") or {}
    lines = [
        "# PIXGBM retrain",
        "",
        f"- utc: {log.get('utc')}",
        f"- status: {log.get('status')}",
        f"- verdict: {log.get('verdict')}",
        f"- lightgbm: {log.get('lightgbm_version')} (pin {PINNED_LGBM})",
        f"- mode: {log.get('train_mode')}",
        "",
        "## sha256",
        "",
    ]
    for name, digest in (comp.get("sha256") or {}).items():
        lines.append(f"- {name}: `{digest}` match={comp.get('match', {}).get(name)}")
    lines += [
        "",
        "## Допуск, если sha256 не совпал",
        "",
        f"- Срез: последняя по имени семья участков (`sorted(families)[-1]`). "
        f"Модель `all` училась и на ней: это совпадение функции с эталоном `{REFERENCE_COMMIT}`, не честный unseen.",
        f"- Метрика: ROC-AUC и средний |Δp|.",
        f"- Порог: |ΔAUC| ≤ {AUC_TOL} и mean|Δp| ≤ {PROB_TOL} для poly и finds.",
        f"- Эталон: `models/pixgbm/all/{{poly,finds}}.txt` из коммита `{REFERENCE_COMMIT}` "
        "(скрипт кладёт копию только в каталог `--out`, не в git).",
        f"- Почему sha256 может не сойтись: в эталоне `[deterministic: 0]`, `[num_threads: 2]`, "
        "bagging/feature_fraction 0.7. LightGBM это не обещает побитово между CPU.",
        "",
        f"- holdout status: {hold.get('status')}",
        f"- holdout family: {hold.get('holdout_family')}",
    ]
    for name, ent in (hold.get("targets") or {}).items():
        lines.append(f"- {name}: {json.dumps(ent, ensure_ascii=False)}")
    lines.append("")
    (out / "retrain_log.md").write_text("\n".join(lines), encoding="utf-8")


def _print_verdict(log: dict) -> None:
    comp = log.get("compare") or {}
    for name, digest in (comp.get("sha256") or {}).items():
        flag = "MATCH" if comp.get("match", {}).get(name) else "DIFF"
        print(f"{name} {digest} {flag}", flush=True)
    print(f"verdict {log.get('verdict')} status {log.get('status')}", flush=True)
    hold = log.get("holdout") or {}
    if hold:
        print(
            f"holdout {hold.get('holdout_family')} status {hold.get('status')} "
            f"|dAUC|<={AUC_TOL} mean|dp|<={PROB_TOL}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--enable-retrain",
        action="store_true",
        default=False,
        help="REQUIRED to actually train. Default OFF — script only validates env and exits 2.",
    )
    ap.add_argument("--smoke", action="store_true", help="Two deterministic synthetic trains; exit 0 if sha256 matches.")
    ap.add_argument("--root", type=Path, help="Contest train data root (e.g. F:/2/train-expds-3)")
    ap.add_argument("--out", type=Path, help="Output directory for feats/, models/, retrain_log.json")
    ap.add_argument("--rounds", type=int, default=300, help="Boosting rounds (default 300 = geo_pixgbm.ROUNDS)")
    ap.add_argument(
        "--expected-hashes",
        type=Path,
        default=ROOT / "models" / "pixgbm" / "all" / "MODELS_NOT_IN_GIT.md",
        help="Markdown table with expected sha256",
    )
    ap.add_argument(
        "--metric-tol",
        type=float,
        default=DEFAULT_METRIC_TOL,
        help="Deprecated. Pass line is |ΔAUC|≤0.01 and mean|Δp|≤0.02 (AUC_TOL, PROB_TOL).",
    )
    ap.add_argument("--allow-other-lgbm", action="store_true")
    ap.add_argument("--allow-module-mismatch", action="store_true")
    ap.add_argument("--lofo", action="store_true", help="Also train leave-family-out models (slower).")
    ap.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="LightGBM num_threads. 0 → 2 in historical mode, 1 with --deterministic.",
    )
    ap.add_argument(
        "--subprocess",
        action="store_true",
        help="Run extract/train as a subprocess (unpatched G.LGBM, all LOFO folds).",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="deterministic=True, num_threads=1, zero seeds. Re-runs match; historical sha256 will not.",
    )
    ap.add_argument(
        "--match-reference",
        action="store_true",
        help="Historical params (default unless --deterministic): seed=0, num_threads=2, "
        "no deterministic flag. This is the sha256 attempt.",
    )
    ap.add_argument(
        "--models-subdir",
        default="all",
        help="Subdir under --out/models to compare (default: all)",
    )
    a = ap.parse_args()
    historical = not a.deterministic
    if a.num_threads is None or a.num_threads <= 0:
        env_nt = os.environ.get("PIXGBM_NUM_THREADS", "").strip()
        if env_nt and not a.smoke:
            a.num_threads = max(1, int(env_nt))
        else:
            a.num_threads = 2 if historical else 1
    os.environ["OMP_NUM_THREADS"] = str(a.num_threads)

    log: dict = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pinned_lightgbm": PINNED_LGBM,
        "enable_retrain": bool(a.enable_retrain),
        "train_mode": "deterministic" if a.deterministic else "historical",
        "match_reference": historical,
        "num_threads": int(a.num_threads) if a.num_threads else None,
        "auc_tol": AUC_TOL,
        "prob_tol": PROB_TOL,
        "argv": sys.argv,
        "repo_root": str(ROOT),
        "why_sha256_may_miss": (
            "Reference poly.txt/finds.txt were saved with deterministic=0, num_threads=2, "
            "feature_fraction=0.7, bagging_fraction=0.7. LightGBM does not promise bit-stable "
            "trees across CPUs in that mode. Adding deterministic=true changes the parameter "
            "dump and the trees, so that path cannot match the recorded sha256."
        ),
    }

    if a.smoke:
        try:
            log["lightgbm_version"] = check_lightgbm(a.allow_other_lgbm)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            sys.exit(2)
        report = smoke_two_runs()
        log["smoke"] = report
        log["status"] = "smoke_ok" if report["ok"] else "smoke_mismatch"
        log["verdict"] = "SMOKE_IDENTICAL" if report["ok"] else "SMOKE_DIFF"
        print(f"smoke sha256 run1 {report['sha256'][0]}", flush=True)
        print(f"smoke sha256 run2 {report['sha256'][1]}", flush=True)
        print(f"verdict {log['verdict']}", flush=True)
        write_reports(a.out, log)
        sys.exit(0 if report["ok"] else 1)

    try:
        log["lightgbm_version"] = check_lightgbm(a.allow_other_lgbm)
    except ModuleNotFoundError as e:
        log["lightgbm_version"] = None
        log["lightgbm_import_error"] = repr(e)
        if a.enable_retrain:
            write_reports(a.out, log)
            raise SystemExit("lightgbm not installed; pip install 'lightgbm==4.7.0'") from e
    except SystemExit as e:
        log["error"] = str(e)
        write_reports(a.out, log)
        raise

    if not a.enable_retrain:
        log["status"] = "disabled_default_off"
        log["hint"] = "Re-run with --enable-retrain --root DATA --out OUT"
        write_reports(a.out, log)
        print(json.dumps({k: log[k] for k in ("status", "hint", "lightgbm_version") if k in log}, ensure_ascii=False))
        sys.exit(2)

    if a.root is None or a.out is None:
        raise SystemExit("--root and --out required when --enable-retrain is set")

    try:
        log["geo_module"] = ensure_geo_module(a.allow_module_mismatch)
    except SystemExit as exc:
        log["error"] = str(exc)
        log["status"] = "no_module"
        write_reports(a.out, log)
        raise

    a.out.mkdir(parents=True, exist_ok=True)
    expected = parse_expected_hashes(a.expected_hashes)
    log["expected_sha256"] = expected

    parts: list = []
    if a.subprocess:
        train_info = run_extract_train(a.root, a.out, a.rounds)
        import importlib.util

        spec = importlib.util.spec_from_file_location("pixgbm_cli", ROOT / "scripts" / "pixgbm.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        parts = mod.load_feats(str(a.out / "feats"))
    else:
        train_info, parts = train_inprocess(
            a.root,
            a.out,
            a.rounds,
            num_threads=a.num_threads,
            match_reference=historical,
            only_all=not a.lofo,
        )
    log["train"] = train_info

    models_all = Path(train_info["models_dir"]) / a.models_subdir
    missing_models = [n for n in expected if not (models_all / n).is_file()]
    if missing_models:
        log["compare"] = {
            "sha256": {n: None for n in expected},
            "match": {n: False for n in expected},
            "all_sha_ok": False,
        }
        log["status"] = "train_produced_no_models"
        log["verdict"] = "FAIL"
        log["hint"] = (
            "no models/all/{poly,finds}.txt — check --root has regions 200–217 "
            "with MagGr/EtGr/TerFl rasters and ObjectPoly/FindsPoints labels."
        )
        write_reports(a.out, log)
        _print_verdict(log)
        sys.exit(1)

    log["compare"] = compare_models(models_all, expected, a.metric_tol)
    try:
        ref_info = materialize_reference(a.out / "reference", expected)
        log["reference"] = ref_info
        log["holdout"] = evaluate_holdout(parts, models_all, a.out / "reference")
    except (subprocess.CalledProcessError, SystemExit, OSError) as exc:
        log["holdout"] = {"status": "no_reference", "error": str(exc)}

    sha_ok = bool(log["compare"]["all_sha_ok"])
    hold_status = (log.get("holdout") or {}).get("status")
    if sha_ok:
        log["verdict"] = "SHA256_MATCH"
        log["status"] = "ok"
        code = 0
    elif hold_status == "ok":
        log["verdict"] = "TOLERANCE_PASS"
        log["status"] = "tolerance_pass"
        code = 0
    elif hold_status in (None, "no_reference", "no_rows"):
        log["verdict"] = "INCOMPLETE"
        log["status"] = "sha_mismatch_no_tolerance"
        code = 2
    else:
        log["verdict"] = "FAIL"
        log["status"] = "sha_mismatch_tolerance_fail"
        code = 1
    write_reports(a.out, log)
    _print_verdict(log)
    sys.exit(code)


def _write_log(out: Path | None, log: dict) -> None:
    if out is None:
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrain_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
