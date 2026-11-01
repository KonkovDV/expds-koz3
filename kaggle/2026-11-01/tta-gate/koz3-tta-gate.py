#!/usr/bin/env python3
"""KOZ3 bot-8 · TTA/MS gate on D-FINE Ae + satellite U-Net (honest-10).

Slug: dmitrijkv/koz3-ae-dfine-gate
Resource: NvidiaTeslaT4, internet, dataset dmitrijkv/koz3-s3-cred,
          kernel_sources dmitrijkv/koz3-dfine-ae (same as koz3-dfine-ae-ft).

Variants (env TTA_GATE_VARIANTS=comma list, default all):
  bprime      — B′ flags-off (orsp_tta off, rot90_tta false, unet scales [1.0])
  tta_ae      — + D-FINE Ae rot90_tta=true
  tta_ms_sat  — + [orsp_tta] unet_tta=4, unet_scales=[0.75,1.0,1.25] on sp_or/sp_focal2
  both        — Ae rot90 + sat TTA/MS

Resumable under OUT/:
  STATUS.txt | STATUS.json | SHA256SUMS | logs/ | variants/<id>/{done.json,score.json,wall.json,overlay.toml}

DRY-RUN (box CPU, low RAM, no GPU/S3/weights):
  TTA_GATE_DRY=1 TTA_GATE_OUT=/tmp/tta-gate-dry python koz3-tta-gate.py
Synthetic scores + wall; same STATUS/sha256/resume contract. No LB submit.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

T0 = time.time()
OUT = Path(os.environ.get("TTA_GATE_OUT", "/kaggle/working/tta_gate"))
DRY = os.environ.get("TTA_GATE_DRY", "").strip().lower() in ("1", "true", "yes")
BPRIME = float(os.environ.get("TTA_GATE_BPRIME", "0.315800"))
DELTA_MIN = float(os.environ.get("TTA_GATE_DELTA_MIN", "0.003"))
DROP_MAX = float(os.environ.get("TTA_GATE_DROP_MAX", "0.01"))
# LB wall budget (seconds); report must keep estimate ≤ 20 min
LB_WALL_BUDGET_S = float(os.environ.get("TTA_GATE_LB_BUDGET_S", str(20 * 60)))

# Overlay specs — code lives on bots/bot8-tta-gate (orsp_tta @ f3aafa4 + rot90 in dfine_aeor)
VARIANTS: dict[str, dict] = {
    "bprime": {
        "label": "B′ flags-off",
        "orsp_tta": {"enabled": False},
        "dfine_aeor": {"rot90_tta": False, "scales": [1.0]},
        "tta": {},
        "ms": {},
    },
    "tta_ae": {
        "label": "+TTA Ae (rot90)",
        "orsp_tta": {"enabled": False},
        "dfine_aeor": {"rot90_tta": True, "scales": [1.0], "flip_tta": True},
        "tta": {"ae": 4},
        "ms": {},
    },
    "tta_ms_sat": {
        "label": "+TTA/MS sat U-Net",
        "orsp_tta": {
            "enabled": True,
            "unet_tta": 4,
            "unet_scales": [0.75, 1.0, 1.25],
            "unet_markups": ["sp_or", "sp_focal2"],
            "dfine_views": "off",
            "dfine_gsd": [1.0],
        },
        "dfine_aeor": {"rot90_tta": False, "scales": [1.0]},
        "tta": {},
        "ms": {},
    },
    "both": {
        "label": "+TTA Ae + TTA/MS sat",
        "orsp_tta": {
            "enabled": True,
            "unet_tta": 4,
            "unet_scales": [0.75, 1.0, 1.25],
            "unet_markups": ["sp_or", "sp_focal2"],
            "dfine_views": "off",
            "dfine_gsd": [1.0],
        },
        "dfine_aeor": {"rot90_tta": True, "scales": [0.75, 1.0, 1.25], "flip_tta": True},
        "tta": {"ae": 4},
        "ms": {"ae": [0.75, 1.0, 1.25]},
    },
}

# Synthetic honest-10 deltas for DRY only (illustrative; not real scores)
_DRY_SCORES = {
    "bprime": {"honest10": round(BPRIME, 6), "layers": {"ae": 0.020, "or": 0.180, "sp_or": 0.310}},
    "tta_ae": {"honest10": round(BPRIME + 0.0012, 6), "layers": {"ae": 0.022, "or": 0.180, "sp_or": 0.310}},
    "tta_ms_sat": {"honest10": round(BPRIME + 0.0021, 6), "layers": {"ae": 0.020, "or": 0.180, "sp_or": 0.314}},
    "both": {"honest10": round(BPRIME + 0.0034, 6), "layers": {"ae": 0.022, "or": 0.180, "sp_or": 0.315}},
}
_DRY_WALL_S = {"bprime": 0.05, "tta_ae": 0.08, "tta_ms_sat": 0.10, "both": 0.12}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_status(state: str, **extra) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {"status": state, "ts": ts, "dry": DRY, "bprime": BPRIME, **extra}
    (OUT / "STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [state, f"ts={ts}", f"dry={int(DRY)}", f"bprime={BPRIME}"]
    for k, v in extra.items():
        lines.append(f"{k}={v}")
    (OUT / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sha256sums(paths: list[Path]) -> Path:
    lines = []
    for p in paths:
        if p.is_file():
            rel = p.relative_to(OUT).as_posix()
            lines.append(f"{sha256_file(p)}  {rel}")
    out = OUT / "SHA256SUMS"
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def selected_variants() -> list[str]:
    raw = os.environ.get("TTA_GATE_VARIANTS", "").strip()
    if not raw:
        return list(VARIANTS.keys())
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in ids if x not in VARIANTS]
    if bad:
        raise KeyError(f"unknown variant id(s): {bad}; known={list(VARIANTS)}")
    return ids


def variant_dir(vid: str) -> Path:
    return OUT / "variants" / vid


def is_complete(vid: str) -> bool:
    d = variant_dir(vid)
    done = d / "done.json"
    score = d / "score.json"
    wall = d / "wall.json"
    if not (done.is_file() and score.is_file() and wall.is_file()):
        return False
    try:
        meta = json.loads(done.read_text(encoding="utf-8"))
        return bool(meta.get("sha256")) and meta["sha256"] == sha256_file(score)
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def dump_overlay(vid: str, spec: dict) -> Path:
    """Write a minimal TOML-ish overlay note (JSON twin) for audit; real run merges into cfg."""
    d = variant_dir(vid)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "overlay.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # human-readable toml snippet for operators
    lines = [f"# variant {vid}: {spec.get('label', '')}"]
    ot = spec.get("orsp_tta") or {}
    lines.append("[orsp_tta]")
    for k, v in ot.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, list):
            lines.append(f"{k} = {json.dumps(v)}")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    da = spec.get("dfine_aeor") or {}
    lines.append("[dfine_aeor]")
    for k, v in da.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, list):
            lines.append(f"{k} = {json.dumps(v)}")
        else:
            lines.append(f"{k} = {v}")
    (d / "overlay.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def gate_check(vid: str, score: dict, baseline_layers: dict[str, float]) -> dict:
    h = float(score["honest10"])
    delta = h - BPRIME
    drops = {}
    for layer, base_v in baseline_layers.items():
        cur = float(score.get("layers", {}).get(layer, base_v))
        drops[layer] = cur - base_v
    layer_fail = {k: v for k, v in drops.items() if v < -DROP_MAX}
    ok_delta = delta >= DELTA_MIN
    ok_layers = not layer_fail
    # bprime itself is the reference — always "pass" structurally
    if vid == "bprime":
        decision = "BASELINE"
    elif ok_delta and ok_layers:
        decision = "PASS"
    elif (not ok_delta) and ok_layers:
        decision = "FAIL_DELTA"
    elif ok_delta and not ok_layers:
        decision = "FAIL_LAYER_DROP"
    else:
        decision = "FAIL"
    return {
        "decision": decision,
        "delta_vs_bprime": round(delta, 6),
        "delta_min": DELTA_MIN,
        "layer_deltas": {k: round(v, 6) for k, v in drops.items()},
        "layer_fail": layer_fail,
        "drop_max": DROP_MAX,
    }


def run_dry_variant(vid: str) -> dict:
    spec = VARIANTS[vid]
    d = variant_dir(vid)
    d.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    dump_overlay(vid, spec)
    t0 = time.perf_counter()
    # tiny synthetic work (keep RAM low)
    blob = (vid + json.dumps(spec, sort_keys=True)).encode()
    for _ in range(200):
        blob = hashlib.sha256(blob).digest()
    time.sleep(_DRY_WALL_S[vid])
    wall_s = time.perf_counter() - t0
    score = {
        "variant": vid,
        "label": spec["label"],
        "honest10": _DRY_SCORES[vid]["honest10"],
        "layers": dict(_DRY_SCORES[vid]["layers"]),
        "synthetic": True,
        "note": "DRY-RUN synthetic — not a real honest-10 measurement",
    }
    (d / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wall = {
        "variant": vid,
        "wall_s": round(wall_s, 4),
        "lb_budget_s": LB_WALL_BUDGET_S,
        "synthetic": True,
        # rough LB extrapolation placeholder (dry): scale mock wall by fictional factor
        "est_lb_wall_s": round(wall_s * 50.0, 2),
    }
    (d / "wall.json").write_text(json.dumps(wall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = sha256_file(d / "score.json")
    done = {
        "id": vid,
        "sha256": sha,
        "seconds": round(wall_s, 4),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry": True,
    }
    (d / "done.json").write_text(json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "logs" / f"{vid}.log").write_text(
        f"dry variant={vid} wall_s={wall_s:.4f} honest10={score['honest10']}\n", encoding="utf-8"
    )
    return {"score": score, "wall": wall, "sha256": sha}


def run_real_variant(vid: str) -> dict:
    """Full path: requires attached repo + data + weights (bot 1 launch after Ae gate)."""
    raise NotImplementedError(
        "Real honest-10 path not executed in this environment. "
        "Attach repo snapshot (bots/bot8-tta-gate), mock/honest-10 work, Ae f4pack weights; "
        "then set TTA_GATE_DRY=0 and wire inference/main.py overlays from overlay.json. "
        f"variant={vid}"
    )


def summarize(results: dict[str, dict]) -> dict:
    base_layers = results.get("bprime", {}).get("score", {}).get("layers") or {}
    rows = []
    for vid, r in results.items():
        score = r["score"]
        wall = r["wall"]
        gate = gate_check(vid, score, base_layers if vid != "bprime" else score.get("layers", {}))
        rows.append(
            {
                "variant": vid,
                "honest10": score["honest10"],
                "delta": gate["delta_vs_bprime"],
                "decision": gate["decision"],
                "wall_s": wall["wall_s"],
                "est_lb_wall_s": wall.get("est_lb_wall_s"),
                "layer_deltas": gate["layer_deltas"],
            }
        )
        (variant_dir(vid) / "gate.json").write_text(
            json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "bprime": BPRIME,
        "delta_min": DELTA_MIN,
        "drop_max": DROP_MAX,
        "lb_budget_s": LB_WALL_BUDGET_S,
        "dry": DRY,
        "rows": rows,
        "protocol": "honest-10 vs B′; PASS if Δ≥0.003 and no layer drop >0.01",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", phase="start")
    log("OUT=", OUT, "DRY=", DRY, "B′=", BPRIME)
    ids = selected_variants()
    results: dict[str, dict] = {}
    try:
        for vid in ids:
            if is_complete(vid):
                log("skip complete", vid)
                score = json.loads((variant_dir(vid) / "score.json").read_text(encoding="utf-8"))
                wall = json.loads((variant_dir(vid) / "wall.json").read_text(encoding="utf-8"))
                results[vid] = {"score": score, "wall": wall, "sha256": sha256_file(variant_dir(vid) / "score.json")}
                continue
            write_status("RUNNING", phase=f"variant:{vid}")
            log("run", vid, VARIANTS[vid]["label"])
            if DRY:
                results[vid] = run_dry_variant(vid)
            else:
                results[vid] = run_real_variant(vid)
        summary = summarize(results)
        # collect sha256 of all score/wall/done/summary
        paths = [OUT / "summary.json"]
        for vid in ids:
            d = variant_dir(vid)
            paths += [d / "score.json", d / "wall.json", d / "done.json", d / "gate.json", d / "overlay.json"]
        write_sha256sums([p for p in paths if p.is_file()])
        write_status(
            "DONE",
            phase="complete",
            n_variants=len(ids),
            decisions=",".join(f"{r['variant']}:{r['decision']}" for r in summary["rows"]),
        )
        log("DONE", json.dumps(summary["rows"], ensure_ascii=False))
        return 0
    except Exception as e:
        write_status("FAILED", phase="error", error=type(e).__name__, msg=str(e)[:200])
        (OUT / "logs" / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log("FAILED", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
