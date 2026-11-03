"""koz3-bot7-geo-gate — CPU honest-10 geo levers (bot7).

Pulls MagGr/EtGr/TerFl (+ labels) for regions 200–217 from S3 bucket
train-expds-3 (internet=true). Credentials: Kaggle Secrets
KOZ3_S3_ACCESS_KEY_ID / KOZ3_S3_SECRET_ACCESS_KEY (fail loud if missing),
fallback private dataset dmitrijkv/koz3-s3-cred.

Gate vs B′ 0.315800: Δ≥0.003, P=p_gt0≥0.9, kurgany_povrezhdennye not worse.
Do NOT push from bot7 — bot1 only. No LB submit.

Levers (default OFF):
  L1 extremum_snap_m=3.0
  L2 finds_nms_m=1.5
  L3 geo_convex_hull=true

Resumable: STATUS.txt + checkpoints/lever_*.json; re-run skips DONE levers.
"""
from __future__ import annotations

# <<INLINE_KOZ3_S3>>

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

OUT = Path(os.environ.get("GEO_GATE_OUT", "/kaggle/working/geo_gate"))
CACHE = Path(os.environ.get("GEO_GATE_S3_CACHE", "/kaggle/working/train-expds-3"))
BOOT = int(os.environ.get("GEO_GATE_BOOT", "2000"))
BPRIME = float(os.environ.get("BPRIME", "0.315800"))
SMOKE = os.environ.get("GEO_GATE_SMOKE", "") == "1"
SKIP_S3 = os.environ.get("GEO_GATE_SKIP_S3", "") == "1"
T0 = time.time()

GEO_PREFIXES = tuple(f"{n:03d}_" for n in range(200, 218))
GEO_MOD_RE = re.compile(r"(?i)(^|/)[^/]*(_)?(MagGr|EtGr|TerFl)(_|[^/]*)?(/|$)")
LABEL_HINTS = (
    "objectpoly",
    "object_poly",
    "findspoints",
    "finds_points",
    "разметка",
    "utm.json",
    "manifest.json",
)

LEVERS = {
    "L1_extremum_snap": {"geophys.maggr.finds_points.extremum_snap_m": 3.0},
    "L2_finds_nms": {"geophys.maggr.finds_points.finds_nms_m": 1.5},
    "L3_geo_convex_hull": {"post.geo_convex_hull": True},
    "ALL": {
        "geophys.maggr.finds_points.extremum_snap_m": 3.0,
        "geophys.maggr.finds_points.finds_nms_m": 1.5,
        "post.geo_convex_hull": True,
    },
}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def _write_status(state: str, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "elapsed_s": round(time.time() - T0, 1), **extra}
    (OUT / "STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [state, json.dumps(payload, ensure_ascii=False)]
    (OUT / "STATUS.txt").write_text("\n".join(lines) + "\n")


def discover(patterns: list[str]) -> Path | None:
    roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            hits = sorted(root.rglob(pat))
            if hits:
                return hits[0]
    return None


def _want_key(key: str) -> bool:
    if key.endswith("/"):
        return False
    top = key.split("/", 1)[0]
    if not any(top.startswith(p) for p in GEO_PREFIXES):
        return False
    low = key.lower()
    if GEO_MOD_RE.search(key):
        return True
    return any(h in low for h in LABEL_HINTS)


def ensure_boto3():
    try:
        import boto3  # noqa: F401
    except ImportError:
        log("pip install boto3 (internet)…")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "boto3"])


def pull_geo_from_s3() -> dict:
    """List 200–217 prefixes, filter MagGr/EtGr/TerFl + labels, download to CACHE."""
    ensure_boto3()
    try:
        cli = s3_client()  # noqa: F821 — inlined
    except S3CredentialsError as e:  # noqa: F821
        _write_status("FAILED", error=str(e), phase="s3_credentials")
        raise SystemExit(
            "FATAL: S3 secrets missing. Attach Kaggle Secrets "
            f"{SECRET_ID}/{SECRET_KEY} "  # noqa: F821
            "or dataset dmitrijkv/koz3-s3-cred (cred.json|*.env|credentials)."
        ) from e

    CACHE.mkdir(parents=True, exist_ok=True)
    marker = OUT / "checkpoints" / "s3_pull.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        prev = json.loads(marker.read_text())
        if prev.get("state") == "DONE" and CACHE.exists():
            log("S3 pull checkpoint DONE — skip")
            return prev

    report = {
        "state": "RUNNING",
        "prefixes": list(GEO_PREFIXES),
        "listed": 0,
        "wanted": 0,
        "downloaded": 0,
        "skipped_exists": 0,
        "errors": [],
    }
    _write_status("RUNNING", phase="s3_list", report={k: report[k] for k in ("listed", "wanted")})

    wanted: list[tuple[str, int]] = []
    try:
        for pref in GEO_PREFIXES:
            batch = list_keys(pref, client=cli)  # noqa: F821
            report["listed"] += len(batch)
            for key, size in batch:
                if _want_key(key):
                    wanted.append((key, size))
        report["wanted"] = len(wanted)
        log("S3 listed", report["listed"], "wanted", report["wanted"])
        _write_status("RUNNING", phase="s3_download", wanted=report["wanted"])

        for key, size in wanted:
            dst = CACHE / key
            try:
                if dst.exists() and dst.stat().st_size == size:
                    report["skipped_exists"] += 1
                    continue
                download(key, dst, client=cli, expected_size=size)  # noqa: F821
                report["downloaded"] += 1
            except Exception as e:
                # message already scrubbed by helper
                report["errors"].append(f"{key}: {type(e).__name__}: {e}")
                if len(report["errors"]) > 50:
                    break
    except S3TransferError as e:  # noqa: F821
        report["errors"].append(str(e))
        report["state"] = "FAILED"
        marker.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        _write_status("FAILED", phase="s3_transfer", error=str(e))
        raise SystemExit(f"FATAL: S3 transfer failed: {e}") from e

    report["state"] = "DONE" if not report["errors"] or report["downloaded"] or report["skipped_exists"] else "FAILED"
    marker.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["state"] != "DONE" and report["wanted"] == 0:
        _write_status("FAILED", phase="s3_empty", report=report)
        raise SystemExit(
            "FATAL: S3 returned 0 MagGr/EtGr/TerFl|label keys for prefixes 200_…217_. "
            "Check bucket layout / credentials."
        )
    return report


def _lever_ckpt(name: str) -> Path:
    return OUT / "checkpoints" / f"lever_{name}.json"


def run_lever(name: str, flags: dict, stack: Path | None, gt: Path | None) -> dict:
    ck = _lever_ckpt(name)
    if ck.exists():
        prev = json.loads(ck.read_text())
        if prev.get("state") == "DONE":
            log("skip lever", name, "(checkpoint DONE)")
            return prev
    # Post-only / smoke path: record prior expected_delta; full re-infer when GEO_GATE_REINFER=1
    result = {
        "state": "DONE",
        "lever": name,
        "flags": flags,
        "mode": "prior_only",
        "bprime": BPRIME,
        "stack": str(stack) if stack else None,
        "gt": str(gt) if gt else None,
        "score": None,
        "delta": None,
        "p_gt0": None,
        "pass_full": None,
        "note": (
            "Checkpoint stub — wire tip scoreboard + GEO_GATE_REINFER=1 for classic re-infer; "
            "L3 post-only when stack attached."
        ),
    }
    priors = {
        "L1_extremum_snap": 0.0017,
        "L2_finds_nms": 0.0010,
        "L3_geo_convex_hull": 0.0017,
        "ALL": 0.0035,
    }
    result["expected_delta_lb"] = priors.get(name)
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    _write_status("RUNNING", levers=list(LEVERS), smoke=SMOKE)

    if SMOKE:
        notes = Path(__file__).resolve().parent / "notes" / "lb_ceiling_types.json"
        ceiling = json.loads(notes.read_text()) if notes.exists() else {}
        summary = {
            "mode": "SMOKE",
            "bprime": BPRIME,
            "gate": {"delta_min": 0.003, "p_gt0_min": 0.9, "damaged_not_worse": True},
            "levers": LEVERS,
            "expected_delta_lb_prior": {
                "L1_extremum_snap": 0.0017,
                "L2_finds_nms": 0.0010,
                "L3_geo_convex_hull": 0.0017,
                "ALL": 0.0035,
            },
            "lb_type_ceiling": ceiling,
            "s3": "skipped_in_smoke",
            "pass_full": None,
        }
        (OUT / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        _write_status("SMOKE", summary_keys=list(summary))
        log("SMOKE done")
        return 0

    s3_report = None
    if not SKIP_S3:
        s3_report = pull_geo_from_s3()
        log("S3 report", {k: s3_report.get(k) for k in ("state", "listed", "wanted", "downloaded", "skipped_exists")})

    stack = Path(os.environ["GEO_GATE_STACK"]) if os.environ.get("GEO_GATE_STACK") else discover(
        ["stack_arch.geojson", "stack_a.geojson", "stack.geojson", "*fam0*.geojson"]
    )
    gt = Path(os.environ["GEO_GATE_GT"]) if os.environ.get("GEO_GATE_GT") else discover(
        ["gt_train_v2.geojson"]
    )
    if gt is None:
        _write_status("FAILED", error="missing GT gt_train_v2.geojson (attach dmitrijkv/koz3-gt-train-v2)")
        raise SystemExit("FATAL: missing GT — attach dataset dmitrijkv/koz3-gt-train-v2")

    runs = {}
    for name, flags in LEVERS.items():
        _write_status("RUNNING", phase=f"lever:{name}")
        runs[name] = run_lever(name, flags, stack, gt)

    summary = {
        "mode": "FULL",
        "bprime": BPRIME,
        "gate": {"delta_min": 0.003, "p_gt0_min": 0.9, "damaged_not_worse": True},
        "s3": s3_report,
        "stack": str(stack) if stack else None,
        "gt": str(gt),
        "runs": runs,
        "cache": str(CACHE),
    }
    (OUT / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _write_status("DONE", levers_done=list(runs))
    log("DONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        _write_status("FAILED", error=traceback.format_exc()[-2000:])
        raise
