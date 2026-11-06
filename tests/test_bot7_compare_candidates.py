"""scripts/compare_candidates.py — rule PASS/FAIL, Δ, li↔ae, md/csv."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRIC = os.environ.get("EXPDS_METRIC") or str(ROOT / "materials" / "compute_metrics.py")
pytestmark = pytest.mark.skipif(not Path(METRIC).exists(), reason="нет compute_metrics.py")
SCRIPT = ROOT / "scripts" / "compare_candidates.py"
X0, Y0 = 3_545_000.0, 7_310_000.0
K = "kurgany_tselye"


def _load_cc():
    spec = importlib.util.spec_from_file_location("compare_candidates", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    from expds import metric

    monkeypatch.setattr(metric, "_SHM_CACHE_DIR", tmp_path / "shm")
    monkeypatch.delenv("EXPDS_MATCH_SCOPE", raising=False)
    metric._UTM_GEOM_CACHE.clear()
    metric._SCOPE_WARNED.clear()
    yield


def sq(x, y, s=10.0):
    return {"type": "Polygon", "coordinates": [[[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]]}


def feat(geom, cls, layer=None, conf=None, region="R"):
    p = {"class_name": cls, "region_name": region}
    if layer is not None:
        p["markup_type"] = layer
    if conf is not None:
        p["confidence"] = conf
    return {"type": "Feature", "geometry": geom, "properties": p}


def fc(fs):
    return {"type": "FeatureCollection", "features": fs}


def _write_cand(folder: Path, features: list, sub: str | None = "out_B") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if sub:
        dest = folder / sub
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "result.geojson"
    else:
        path = folder / "result.geojson"
    path.write_text(json.dumps(fc(features), ensure_ascii=False), encoding="utf-8")
    return path


def test_find_geojson_prefers_out_B(tmp_path):
    cc = _load_cc()
    root = tmp_path / "cand"
    _write_cand(root, [feat(sq(X0, Y0), K, "li", 0.9)], sub="out_A")
    _write_cand(root, [feat(sq(X0, Y0), K, "li", 0.5)], sub="out_B")
    got = cc.find_geojson(root)
    assert got.as_posix().endswith("out_B/result.geojson")


def test_rule_pass_fail_delta_and_liae(tmp_path):
    cc = _load_cc()
    # GT: один li kurgany + один ae (не-kurgany для контроля) + один sp_or
    gt = fc([
        feat(sq(X0, Y0), K, "li"),
        feat(sq(X0 + 80, Y0), "object_poly", "ae"),
        feat(sq(X0 + 160, Y0), "fortifikatsii", "sp_or"),
    ])
    gt_path = tmp_path / "gt.geojson"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")

    # good: exact match same layers → high F1, no cross li↔ae kurgany
    good = tmp_path / "good"
    _write_cand(good, [
        feat(sq(X0, Y0), K, "li", 0.9),
        feat(sq(X0 + 80, Y0), "object_poly", "ae", 0.9),
        feat(sq(X0 + 160, Y0), "fortifikatsii", "sp_or", 0.9),
    ])

    # bad: ae steals li kurgany GT (cross li↔ae) + missing sp_or → layer drop
    bad = tmp_path / "bad"
    _write_cand(bad, [
        feat(sq(X0, Y0), K, "ae", 0.95),  # cross-layer TP li↔ae kurgany
        feat(sq(X0 + 80, Y0), "object_poly", "ae", 0.5),
        # no sp_or → sp_or F1 drops
    ], sub="out_A")

    # baseline = good candidate scores (so good PASS, bad may FAIL on sp_or)
    good_pred = json.loads((good / "out_B" / "result.geojson").read_text(encoding="utf-8"))
    base_scores = cc.layer_scores(good_pred, gt, METRIC)["global"]
    baseline = {k: base_scores[k] for k in ("ALL", "li", "sp_or", "ae")}

    r_good = cc.evaluate_candidate(good, gt, baseline, METRIC, label="good")
    r_bad = cc.evaluate_candidate(bad, gt, baseline, METRIC, label="bad")

    assert r_good["rule"] == "PASS"
    assert abs(r_good["delta_vs_v2"]["ALL"]) < 1e-9
    assert r_good["liae_kurgany_cross_tp"] == 0

    assert r_bad["liae_kurgany_cross_tp"] == 1
    assert r_bad["delta_vs_v2"]["sp_or"] < -0.01
    assert r_bad["delta_vs_v2"]["ALL"] <= 0
    # sp_or must drop vs baseline (no preds) → FAIL
    assert r_bad["rule"] == "FAIL"
    assert "sp_or" in r_bad["rule_failed_layers"]


def test_cli_md_csv_experiments(tmp_path):
    cc = _load_cc()
    gt = fc([
        feat(sq(X0, Y0), K, "li"),
        feat(sq(X0 + 80, Y0), K, "ae"),
    ])
    gt_path = tmp_path / "gt.geojson"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")

    v2 = tmp_path / "v2"
    _write_cand(v2, [
        feat(sq(X0, Y0), K, "li", 0.9),
        feat(sq(X0 + 80, Y0), K, "ae", 0.9),
    ], sub=None)

    cand = tmp_path / "cand"
    # cross-layer: ae hits li GT
    _write_cand(cand, [
        feat(sq(X0, Y0), K, "ae", 0.95),
        feat(sq(X0 + 80, Y0), K, "ae", 0.5),
    ], sub="out_B")

    out = tmp_path / "out"
    exp_file = tmp_path / "EXPERIMENTS.csv"
    exp_file.write_text(
        "date_msk,bot,branch,sha,experiment,eval,ALL_F1,li,sp_or,ae,delta_vs_base,verdict,notes\n",
        encoding="utf-8",
    )
    cmd = [
        sys.executable, str(SCRIPT),
        "--gt", str(gt_path),
        "--v2", str(v2),
        "--out", str(out),
        "--workers", "2",
        "--labels", "candX",
        "--branch", "bots/bot7-compare-candidates",
        "--sha", "deadbeef",
        "--experiments-row",
        "--experiments-file", str(exp_file),
        str(cand),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out / "compare.md").is_file()
    assert (out / "compare.csv").is_file()
    assert (out / "summary.json").is_file()
    md = (out / "compare.md").read_text(encoding="utf-8")
    assert "candX" in md and "global" in md and "layer" in md
    assert "| env |" in md or "| candidate | env |" in md
    with (out / "compare.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # global + layer
    assert rows[0]["name"] == "candX"
    assert "env" in rows[0] and rows[0]["env"] == "unknown"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["results"][0]["liae_kurgany_cross_tp"] >= 1
    assert summary["results"][0]["env"] == "unknown"
    exp_lines = exp_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(exp_lines) == 2
    assert "compare_candidates:candX" in exp_lines[1]
    assert "deadbeef" in exp_lines[1]
    assert "env=unknown" in exp_lines[1]


def test_evaluate_rule_math():
    cc = _load_cc()
    baseline = {"li": 0.5, "sp_or": 0.4, "ae": 0.3}
    # drop exactly 0.01 → still PASS (threshold is "more than 0.01")
    rule, failed = cc.evaluate_rule({"li": 0.49, "sp_or": 0.4, "ae": 0.3}, baseline, drop_max=0.01)
    assert rule == "PASS" and failed == []
    rule, failed = cc.evaluate_rule({"li": 0.489, "sp_or": 0.4, "ae": 0.3}, baseline, drop_max=0.01)
    assert rule == "FAIL" and failed == ["li"]
    # Δ math via evaluate_candidate covered above; direct delta
    scores = {"ALL": 0.30, "li": 0.49, "sp_or": 0.4, "ae": 0.3}
    base = {"ALL": 0.3125, "li": 0.5, "sp_or": 0.4, "ae": 0.3}
    assert abs((scores["ALL"] - base["ALL"]) - (-0.0125)) < 1e-12

def test_resolve_out_default_koz3_and_fallback(tmp_path, monkeypatch):
    cc = _load_cc()
    # explicit --out wins
    got = cc.resolve_out(str(tmp_path / "explicit"))
    assert got == (tmp_path / "explicit").resolve()

    # KOZ3_OUT set → KOZ3_OUT/compare
    koz = tmp_path / "koz3out"
    monkeypatch.setenv("KOZ3_OUT", str(koz))
    got = cc.resolve_out(None)
    assert got == (koz / "compare").resolve()

    # KOZ3_OUT unset → ./compare_out (cwd-relative resolve)
    monkeypatch.delenv("KOZ3_OUT", raising=False)
    monkeypatch.chdir(tmp_path)
    got = cc.resolve_out(None)
    assert got == (tmp_path / "compare_out").resolve()


def test_cli_default_out_and_env_column(tmp_path, monkeypatch):
    gt = fc([
        feat(sq(X0, Y0), K, "li"),
        feat(sq(X0 + 80, Y0), K, "ae"),
    ])
    gt_path = tmp_path / "gt.geojson"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")

    v2 = tmp_path / "v2"
    _write_cand(v2, [
        feat(sq(X0, Y0), K, "li", 0.9),
        feat(sq(X0 + 80, Y0), K, "ae", 0.9),
    ], sub=None)

    cand_d = tmp_path / "cand_d"
    _write_cand(cand_d, [
        feat(sq(X0, Y0), K, "li", 0.9),
        feat(sq(X0 + 80, Y0), K, "ae", 0.9),
    ], sub="out_B")
    cand_h = tmp_path / "cand_h"
    _write_cand(cand_h, [
        feat(sq(X0, Y0), K, "li", 0.9),
        feat(sq(X0 + 80, Y0), K, "ae", 0.9),
    ], sub="out_B")

    koz = tmp_path / "job"
    monkeypatch.setenv("KOZ3_OUT", str(koz))
    # no --out → should land in KOZ3_OUT/compare
    cmd = [
        sys.executable, str(SCRIPT),
        "--gt", str(gt_path),
        "--v2", str(v2),
        "--workers", "2",
        "--labels", "v2like,hostcand",
        "--env", "docker,host",
        "--branch", "bots/bot7-compare-candidates",
        "--sha", "abc1234",
        "--experiments-row",
        str(cand_d), str(cand_h),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "out=" in proc.stdout
    out = koz / "compare"
    assert (out / "compare.md").is_file()
    assert (out / "compare.csv").is_file()
    md = (out / "compare.md").read_text(encoding="utf-8")
    assert "| candidate | env |" in md
    assert "| v2like | docker |" in md
    assert "| hostcand | host |" in md
    with (out / "compare.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    envs = {(r["name"], r["env"]) for r in rows}
    assert ("v2like", "docker") in envs
    assert ("hostcand", "host") in envs
    # also accept path:env form
    out2 = tmp_path / "out2"
    cmd2 = [
        sys.executable, str(SCRIPT),
        "--gt", str(gt_path),
        "--v2", str(v2),
        "--out", str(out2),
        "--workers", "1",
        f"dockerA={cand_d}:docker",
        f"hostB={cand_h}:host",
    ]
    proc2 = subprocess.run(cmd2, cwd=str(ROOT), capture_output=True, text=True)
    assert proc2.returncode == 0, proc2.stderr + proc2.stdout
    md2 = (out2 / "compare.md").read_text(encoding="utf-8")
    assert "| dockerA | docker |" in md2
    assert "| hostB | host |" in md2
    summary = json.loads((out2 / "summary.json").read_text(encoding="utf-8"))
    assert summary["results"][0]["env"] == "docker"
    assert summary["results"][1]["env"] == "host"


def test_parse_candidate_spec_windows_safe():
    cc = _load_cc()
    lab, path, env = cc.parse_candidate_spec(r"C:\Sloi\pc_jobs\v3e2e\res_v2")
    assert lab is None and env == "unknown"
    assert "res_v2" in str(path)
    lab, path, env = cc.parse_candidate_spec("v2=C:/data/res_v2:docker")
    assert lab == "v2" and env == "docker"
    assert str(path).replace("\\", "/").endswith("res_v2") or "res_v2" in str(path)

