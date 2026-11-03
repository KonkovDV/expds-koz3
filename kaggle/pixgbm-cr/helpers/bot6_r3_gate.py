#!/usr/bin/env python3
"""bot-6 r3 gate: validity of REF/NEW pixgbm geo runs + equivalence |dF1|<=0.01.

Exit codes (distinct INVALID):
  0  PASS
  9  equivalence fail (|dF1|>0.01 on ALL or maggr/etgr/terfl)
 10  in-container sha mismatch (caller may also exit 10 earlier)
 11  pix not actually run (empty steps / no full / missing module / skipped)
 12  docker exit != 0 (recorded)
 13  geojson mtime before container start
 14  result geojsons byte-identical while model shas differ (suspect)
 19  scores.json missing/unparsed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Cyrillic patterns as unicode escapes (helpers stay ASCII-source friendly).
RE_MISSING_MODULE = re.compile(
    "\u043d\u0435\u0442 \u0432 \u044d\u0442\u043e\u0439 \u0441\u0431\u043e\u0440\u043a\u0435"  # net v etoj sborke
)
RE_PIX_SKIP_REGION = re.compile(
    r"pixgbm \u043f\u0440\u043e\u043f\u0443\u0449\u0435\u043d(?!\s*\u2014)"  # propuschen (not emdash classic)
)
# "pixgbm <region>: ... stupeni {...}; faza X.X s"
RE_PIX_LINE = re.compile(
    r"pixgbm\s+(\S+):\s+backend=(\S+)\s+[^\r\n]*?"
    r"\u0441\u0442\u0443\u043f\u0435\u043d\u0438\s+(\{[^\}]*\})"  # stupeni
    r"[^\r\n]*?"
    r"\u0444\u0430\u0437\u0430\s+([0-9.]+)\s*\u0441",  # faza N s
    re.IGNORECASE,
)
RE_STEPS_EMPTY = re.compile(r"\u0441\u0442\u0443\u043f\u0435\u043d\u0438\s+\{\}")
RE_STEPS_FULL = re.compile(r"['\"]full['\"]\s*:")

GEO_LAYERS = ("maggr", "etgr", "terfl")
TOL = 0.01


def _read(path: Path) -> str:
    if not path or not Path(path).is_file():
        return ""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def parse_pix_logs(log_text: str) -> dict:
    lines = RE_PIX_LINE.findall(log_text)
    empty = RE_STEPS_EMPTY.findall(log_text)
    full_hits = RE_STEPS_FULL.findall(log_text)
    missing = RE_MISSING_MODULE.findall(log_text)
    # region-level skip: "pixgbm propuschen" but not the per-raster "propuschen — classic"
    skipped = []
    for m in re.finditer(r"[^\r\n]*pixgbm[^\r\n]*", log_text):
        line = m.group(0)
        if RE_PIX_SKIP_REGION.search(line) and "\u2014" not in line and "classic" not in line.lower():
            # also allow ASCII 'classic' path; Cyrillic emdash form is per-raster
            if RE_PIX_SKIP_REGION.search(line):
                skipped.append(line.strip())
    # Broader: exact pipeline message "%s: pixgbm propuschen"
    for m in re.finditer(
        r"(?m)^.*?pixgbm \u043f\u0440\u043e\u043f\u0443\u0449\u0435\u043d\s*$", log_text
    ):
        skipped.append(m.group(0).strip())

    phase_s = 0.0
    steps_counts: dict[str, int] = {}
    per_region = []
    for reg, backend, steps_s, phase in lines:
        try:
            steps = eval(steps_s, {"__builtins__": {}}, {})  # steps dict from log repr
        except Exception:
            steps = {"_raw": steps_s}
        if isinstance(steps, dict):
            for k, v in steps.items():
                if isinstance(v, int):
                    steps_counts[k] = steps_counts.get(k, 0) + v
        try:
            phase_s = max(phase_s, float(phase))  # PHASE spent is cumulative; take max seen
        except ValueError:
            pass
        per_region.append({"region": reg, "backend": backend, "steps": steps, "phase_s": float(phase)})

    # Prefer last cumulative phase if present
    if per_region:
        phase_s = float(per_region[-1]["phase_s"])

    return {
        "n_pix_lines": len(lines),
        "n_empty_steps": len(empty),
        "n_full_hits": len(full_hits),
        "missing_module": len(missing),
        "region_skips": skipped[:20],
        "n_region_skips": len(skipped),
        "phase_seconds": phase_s,
        "steps_counts": steps_counts,
        "per_region": per_region,
    }


def validity_for_run(name: str, log_text: str, docker_ec: int | None,
                     geojson: Path | None, t_start: float | None,
                     incontainer_ok: bool) -> list[tuple[int, str]]:
    fails = []
    if not incontainer_ok:
        fails.append((10, "%s incontainer_sha_mismatch" % name))
    pix = parse_pix_logs(log_text)
    if pix["missing_module"] > 0:
        fails.append((11, "%s missing_module_in_log" % name))
    if pix["n_region_skips"] > 0:
        fails.append((11, "%s pixgbm_region_skipped" % name))
    if pix["n_empty_steps"] > 0:
        fails.append((11, "%s empty_steps_{}" % name))
    if pix["n_full_hits"] == 0:
        fails.append((11, "%s no_full_step" % name))
    if pix["phase_seconds"] <= 0:
        fails.append((11, "%s phase_seconds==0" % name))
    if docker_ec is not None and docker_ec != 0:
        fails.append((12, "%s docker_exit=%s" % (name, docker_ec)))
    if geojson is not None and geojson.is_file() and t_start is not None:
        mtime = geojson.stat().st_mtime
        if mtime < t_start:
            fails.append((13, "%s geojson_mtime_before_start" % name))
    return fails, pix


def equivalence(scores: dict, tol: float = TOL) -> dict:
    rows = scores.get("rows") or []
    allrow = next((r for r in rows if r.get("region") == "ALL" and r.get("markup") == "ALL"), None)
    layer_rows = {
        r["markup"]: r
        for r in rows
        if r.get("region") == "ALL" and r.get("markup") in GEO_LAYERS
    }
    out = {
        "tol": tol,
        "delta_ALL": (allrow or {}).get("delta"),
        "f1_ALL": {k: v for k, v in (allrow or {}).items() if k.startswith("f1_")},
        "layers": {},
        "layer_fails": [],
        "pass": False,
        "reason": "",
    }
    for mt in GEO_LAYERS:
        if mt not in layer_rows:
            continue
        d = layer_rows[mt].get("delta")
        out["layers"][mt] = {
            "delta": d,
            **{k: v for k, v in layer_rows[mt].items() if k.startswith("f1_")},
        }
        if d is None or abs(float(d)) > tol:
            out["layer_fails"].append((mt, d))

    reasons = []
    if allrow is None or allrow.get("delta") is None:
        reasons.append("no ALL/ALL delta")
    elif abs(float(allrow["delta"])) > tol:
        reasons.append("ALL |delta|=%s > %s" % (allrow["delta"], tol))
    if out["layer_fails"]:
        reasons.append(
            "layer: " + ",".join("%s=%s" % (a, b) for a, b in out["layer_fails"])
        )
    # Require that at least one geo layer row exists (else not a geo test)
    if not out["layers"]:
        reasons.append("no maggr/etgr/terfl ALL rows in scores")
    if not reasons:
        out["pass"] = True
        out["reason"] = "ok"
    else:
        out["reason"] = "; ".join(reasons)

    # Optional per-class note from counts_* if present (not F1)
    for key, val in scores.items():
        if key.startswith("counts_"):
            out[key] = dict(val) if isinstance(val, dict) else val
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None, help="JobDir")
    ap.add_argument("--scores", default=None)
    ap.add_argument("--tol", type=float, default=TOL)
    # optional explicit inputs for unit self-check without a full JobDir layout
    ap.add_argument("--log-ref", default=None)
    ap.add_argument("--log-new", default=None)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        job = Path(args.job) if args.job else Path("/tmp/bot6_r3_gate_self")
        job.mkdir(parents=True, exist_ok=True)
        return _self_check(job)
    if not args.job:
        ap.error("--job is required unless --self-check")
    job = Path(args.job)
    job.mkdir(parents=True, exist_ok=True)

    notes = job / "notes"
    scores_path = Path(args.scores) if args.scores else job / "scores.json"
    gate = {
        "pass": False,
        "tol": args.tol,
        "validity_fails": [],
        "runs": {},
    }

    meta_path = notes / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    for name in ("ref", "new"):
        log_override = None
        if name == "ref" and args.log_ref:
            log_override = args.log_ref
        elif name == "new" and args.log_new:
            log_override = args.log_new
        if log_override:
            log_txt = _read(Path(log_override))
        else:
            log_txt = _read(job / ("out_%s" % name) / "main.log") + _read(
                job / ("out_%s" % name) / "main.err")
        docker_ec = (meta.get("docker_ec") or {}).get(name)
        t_start = (meta.get("t_start_epoch") or {}).get(name)
        geo = job / ("out_%s" % name) / "result.geojson"
        in_ok = bool((meta.get("incontainer_ok") or {}).get(name, True))
        fails, pix = validity_for_run(name, log_txt, docker_ec, geo, t_start, in_ok)
        gate["runs"][name] = pix
        gate["validity_fails"].extend([{"code": c, "msg": m} for c, m in fails])

    # identical geojson + different models -> INVALID
    ident = meta.get("result_identical")
    model_differ = meta.get("model_sha_differ", True)
    if ident is True and model_differ:
        gate["validity_fails"].append({
            "code": 14,
            "msg": "result_geojson_identical_but_models_differ",
        })
    gate["result_identical"] = ident
    gate["result_sha"] = meta.get("result_sha")

    if gate["validity_fails"]:
        # prefer smallest distinct code
        code = min(f["code"] for f in gate["validity_fails"])
        gate["reason"] = "INVALID: " + "; ".join(f["msg"] for f in gate["validity_fails"])
        (job / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(gate, ensure_ascii=False))
        return code

    if not scores_path.is_file():
        gate["reason"] = "no scores.json"
        (job / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(gate, ensure_ascii=False))
        return 19

    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    eq = equivalence(scores, args.tol)
    gate["equivalence"] = eq
    gate["pass"] = bool(eq["pass"])
    gate["reason"] = eq["reason"]
    (job / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False))
    return 0 if gate["pass"] else 9


def _self_check(job: Path) -> int:
    """Synthetic fixtures: empty steps fail; full steps + tiny |delta| pass."""
    # Case A: empty steps -> exit 11
    log_bad = (
        "INFO pixgbm 200_X: backend=lightgbm potokov=2 "
        "\u0441\u0442\u0443\u043f\u0435\u043d\u0438 {}"
        "; \u0444\u0430\u0437\u0430 0.0 \u0441 (cpu 0.0 \u0441), uroven classic\n"
    )
    log_good = (
        "INFO pixgbm 200_X: backend=lightgbm potokov=2 "
        "\u0441\u0442\u0443\u043f\u0435\u043d\u0438 {'full': 3}"
        "; \u0444\u0430\u0437\u0430 12.5 \u0441 (cpu 10.0 \u0441), uroven full\n"
        "INFO pixgbm 201_Y: backend=lightgbm potokov=2 "
        "\u0441\u0442\u0443\u043f\u0435\u043d\u0438 {'full': 2}"
        "; \u0444\u0430\u0437\u0430 25.0 \u0441 (cpu 20.0 \u0441), uroven full\n"
    )
    scores_ok = {
        "rows": [
            {"region": "ALL", "markup": "ALL", "delta": 0.002,
             "f1_pix_ref": 0.50, "f1_pix_new": 0.502},
            {"region": "ALL", "markup": "maggr", "delta": -0.005,
             "f1_pix_ref": 0.40, "f1_pix_new": 0.395},
            {"region": "ALL", "markup": "etgr", "delta": 0.0,
             "f1_pix_ref": 0.30, "f1_pix_new": 0.30},
            {"region": "ALL", "markup": "terfl", "delta": 0.008,
             "f1_pix_ref": 0.20, "f1_pix_new": 0.208},
        ],
        "counts_pix_ref": {"maggr/object_poly": 10, "maggr/finds_points": 5},
        "counts_pix_new": {"maggr/object_poly": 11, "maggr/finds_points": 5},
    }
    scores_bad = {
        "rows": [
            {"region": "ALL", "markup": "ALL", "delta": 0.05,
             "f1_pix_ref": 0.50, "f1_pix_new": 0.55},
            {"region": "ALL", "markup": "maggr", "delta": 0.05,
             "f1_pix_ref": 0.40, "f1_pix_new": 0.45},
        ]
    }

    results = {}
    # A: bad log
    dA = job / "_selfA"
    dA.mkdir(parents=True, exist_ok=True)
    (dA / "notes").mkdir(exist_ok=True)
    (dA / "out_ref").mkdir(exist_ok=True)
    (dA / "out_new").mkdir(exist_ok=True)
    (dA / "out_ref" / "main.log").write_text(log_bad, encoding="utf-8")
    (dA / "out_new" / "main.log").write_text(log_good, encoding="utf-8")
    (dA / "notes" / "run_meta.json").write_text(json.dumps({
        "docker_ec": {"ref": 0, "new": 0},
        "incontainer_ok": {"ref": True, "new": True},
        "result_identical": False,
        "model_sha_differ": True,
    }), encoding="utf-8")
    (dA / "scores.json").write_text(json.dumps(scores_ok), encoding="utf-8")
    # touch geojsons with recent mtime
    for v in ("ref", "new"):
        (dA / ("out_%s" % v) / "result.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    import subprocess
    r = subprocess.run([sys.executable, __file__, "--job", str(dA)], capture_output=True, text=True)
    results["empty_steps"] = {"ec": r.returncode, "out": r.stdout.strip()[:200]}

    # B: good logs + pass scores
    dB = job / "_selfB"
    dB.mkdir(parents=True, exist_ok=True)
    (dB / "notes").mkdir(exist_ok=True)
    for v in ("ref", "new"):
        (dB / ("out_%s" % v)).mkdir(exist_ok=True)
        (dB / ("out_%s" % v) / "main.log").write_text(log_good, encoding="utf-8")
        (dB / ("out_%s" % v) / "result.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    (dB / "notes" / "run_meta.json").write_text(json.dumps({
        "docker_ec": {"ref": 0, "new": 0},
        "incontainer_ok": {"ref": True, "new": True},
        "result_identical": False,
        "model_sha_differ": True,
    }), encoding="utf-8")
    (dB / "scores.json").write_text(json.dumps(scores_ok), encoding="utf-8")
    r = subprocess.run([sys.executable, __file__, "--job", str(dB)], capture_output=True, text=True)
    results["pass_eq"] = {"ec": r.returncode, "out": r.stdout.strip()[:300]}

    # C: good logs + fail equivalence
    dC = job / "_selfC"
    dC.mkdir(parents=True, exist_ok=True)
    (dC / "notes").mkdir(exist_ok=True)
    for v in ("ref", "new"):
        (dC / ("out_%s" % v)).mkdir(exist_ok=True)
        (dC / ("out_%s" % v) / "main.log").write_text(log_good, encoding="utf-8")
        (dC / ("out_%s" % v) / "result.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    (dC / "notes" / "run_meta.json").write_text(json.dumps({
        "docker_ec": {"ref": 0, "new": 0},
        "incontainer_ok": {"ref": True, "new": True},
        "result_identical": False,
        "model_sha_differ": True,
    }), encoding="utf-8")
    (dC / "scores.json").write_text(json.dumps(scores_bad), encoding="utf-8")
    r = subprocess.run([sys.executable, __file__, "--job", str(dC)], capture_output=True, text=True)
    results["fail_eq"] = {"ec": r.returncode, "out": r.stdout.strip()[:300]}

    # D: identical results + differing models
    dD = job / "_selfD"
    dD.mkdir(parents=True, exist_ok=True)
    (dD / "notes").mkdir(exist_ok=True)
    for v in ("ref", "new"):
        (dD / ("out_%s" % v)).mkdir(exist_ok=True)
        (dD / ("out_%s" % v) / "main.log").write_text(log_good, encoding="utf-8")
        (dD / ("out_%s" % v) / "result.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    (dD / "notes" / "run_meta.json").write_text(json.dumps({
        "docker_ec": {"ref": 0, "new": 0},
        "incontainer_ok": {"ref": True, "new": True},
        "result_identical": True,
        "model_sha_differ": True,
    }), encoding="utf-8")
    (dD / "scores.json").write_text(json.dumps(scores_ok), encoding="utf-8")
    r = subprocess.run([sys.executable, __file__, "--job", str(dD)], capture_output=True, text=True)
    results["identical_suspect"] = {"ec": r.returncode, "out": r.stdout.strip()[:200]}

    ok = (
        results["empty_steps"]["ec"] == 11
        and results["pass_eq"]["ec"] == 0
        and results["fail_eq"]["ec"] == 9
        and results["identical_suspect"]["ec"] == 14
    )
    summary = {"self_check_pass": ok, "cases": results}
    (job / "self_check_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
