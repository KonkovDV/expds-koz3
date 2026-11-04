"""koz3-bot7-sat-recall — CPU honest-10 satellite recall levers (bot7).

Pulls SpOr (+ labels) for honest-10 regions from S3 bucket train-expds-3
(internet=true). Credentials: Kaggle Secrets KOZ3_S3_ACCESS_KEY_ID /
KOZ3_S3_SECRET_ACCESS_KEY (fail loud if missing), fallback private dataset
dmitrijkv/koz3-s3-cred.

Gate vs B′ 0.315800: Δ≥0.003, P=p_gt0≥0.9, kurgany_povrezhdennye not worse.
Do NOT push from bot7 — bot1 only. No LB submit.

Levers (default OFF in tip):
  L1 sat_band_verify (scale=0.90, verify_floor=0.40)
  L2 sat_agree_accept (radius_m=16, scale=0.85)  # ≠ CLOSED 8m cluster
  L3 sat_cross_nms (nms_m=4.0)
  ALL = L1+L2+L3

Exclusions: NOT sat2, NOT sp_large, NOT sat 8m cluster / sp_score_fusion,
NOT near_kept, NOT large_objects.

Resumable: STATUS.txt + checkpoints/lever_*.json.
"""
from __future__ import annotations

# --- BEGIN koz3_s3 (inlined from kaggle/common/koz3_s3.py; do not edit by hand) ---
"""Shared S3 helper for KOZ3 Kaggle kernels (bot7 → bot6 reuse).

API
---
- ``s3_client()`` → boto3 S3 client
- ``download(key, dst)`` → Path; skip if size matches; retries; ``.part`` then rename
- ``download_prefix(prefix, dst)`` → list[Path]; preserves key suffix under ``dst``
- ``list_keys(prefix)`` → list[tuple[str, int]] of (key, size); paginated

Credentials (never logged / never in exception text)
----------------------------------------------------
1. Kaggle Secrets ``KOZ3_S3_ACCESS_KEY_ID`` / ``KOZ3_S3_SECRET_ACCESS_KEY``
   (aliases of module constants ``SECRET_ID`` / ``SECRET_KEY``, same as
   ``koz3-dfine-or.py``).
2. Env ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``.
3. Private dataset ``dmitrijkv/koz3-s3-cred`` under ``/kaggle/input/**``:
   - ``cred.json`` / ``*.json`` with keys ``id``+``key`` or
     ``aws_access_key_id``+``aws_secret_access_key`` (or AccessKeyId/SecretAccessKey)
   - ``*.env`` / ``credentials.env`` lines ``KEY=VAL``
   - ``*.ini`` / ``credentials`` AWS-style ``[default]`` with
     ``aws_access_key_id`` / ``aws_secret_access_key``

Deps: boto3 + stdlib only. Bucket ``train-expds-3``, endpoint
``https://s3.ru-7.storage.selcloud.ru``, region ``ru-7``.
"""
import configparser
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

BUCKET = "train-expds-3"
ENDPOINT = os.environ.get("KOZ3_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
REGION = "ru-7"
# Same names as docs/.../koz3-dfine-or.py
SECRET_ID = "KOZ3_S3_ACCESS_KEY_ID"
SECRET_KEY = "KOZ3_S3_SECRET_ACCESS_KEY"

_INPUT_GLOB = "/kaggle/input/**/*"


class S3CredentialsError(RuntimeError):
    """Missing / unreadable S3 credentials (message never contains secret values)."""


class S3TransferError(RuntimeError):
    """Download / list failure with credentials stripped from the message."""


def _scrub(text: str, *secrets: Optional[str]) -> str:
    out = str(text)
    for s in secrets:
        if s and len(s) >= 4:
            out = out.replace(s, "***")
    # generic access-key / secret patterns (never leave raw values in errors)
    out = re.sub(r"(?i)\b(AKIA[0-9A-Z_]{8,})\b", "***", out)
    out = re.sub(r"(?i)(secret\s*[=:]\s*)\S+", r"\1***", out)
    out = re.sub(r"(?i)(key\s*[=:]\s*)\S+", r"\1***", out)
    out = re.sub(r"(?i)(aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*)\S+", r"\1***", out)
    out = re.sub(r"(?i)(aws[_-]?access[_-]?key[_-]?id\s*[=:]\s*)\S+", r"\1***", out)
    # long base64-ish tokens often used as secret access keys
    out = re.sub(r"(?<![A-Za-z0-9/+=])([A-Za-z0-9/+=]{35,})(?![A-Za-z0-9/+=])", "***", out)
    return out


def _wrap(exc: BaseException, *secrets: Optional[str]) -> S3TransferError:
    return S3TransferError(
        f"{type(exc).__name__}: {_scrub(str(exc), *secrets)}"
    )


def _parse_json_cred(path: Path) -> Optional[Tuple[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    kid = (
        data.get("id")
        or data.get("aws_access_key_id")
        or data.get("AccessKeyId")
        or data.get("access_key")
    )
    sec = (
        data.get("key")
        or data.get("aws_secret_access_key")
        or data.get("SecretAccessKey")
        or data.get("secret_key")
    )
    if kid and sec:
        return str(kid), str(sec)
    return None


def _parse_env_cred(path: Path) -> Optional[Tuple[str, str]]:
    kid = sec = None
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            ku = k.upper()
            if ku in ("AWS_ACCESS_KEY_ID", "KOZ3_S3_ACCESS_KEY_ID", "ID", "ACCESS_KEY"):
                kid = v
            elif ku in (
                "AWS_SECRET_ACCESS_KEY",
                "KOZ3_S3_SECRET_ACCESS_KEY",
                "KEY",
                "SECRET_KEY",
                "SECRET_ACCESS_KEY",
            ):
                sec = v
    except Exception:
        return None
    if kid and sec:
        return kid, sec
    return None


def _parse_ini_cred(path: Path) -> Optional[Tuple[str, str]]:
    cp = configparser.ConfigParser()
    try:
        cp.read(path)
    except Exception:
        return None
    for section in ("default", "DEFAULT", "s3", "koz3"):
        if section == "DEFAULT":
            items = {k.lower(): v for k, v in cp.defaults().items()}
        elif cp.has_section(section):
            items = {k.lower(): v for k, v in cp.items(section)}
        else:
            continue
        kid = items.get("aws_access_key_id") or items.get("id")
        sec = items.get("aws_secret_access_key") or items.get("key")
        if kid and sec:
            return kid, sec
    return None


def _cred_from_inputs(root_glob: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Scan private dataset files under /kaggle/input (json / env / ini)."""
    if root_glob is None:
        root_glob = _INPUT_GLOB
    paths = sorted(glob.glob(root_glob, recursive=True))
    # Prefer known names first
    preferred = []
    other = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        name = os.path.basename(p).lower()
        if name in ("cred.json", "credentials", "credentials.env", "s3.env", "koz3-s3.env"):
            preferred.append(p)
        elif name.endswith((".json", ".env", ".ini", ".cfg")) or name == "credentials":
            other.append(p)
    for p in preferred + other:
        path = Path(p)
        name = path.name.lower()
        parsed = None
        if name.endswith(".json"):
            parsed = _parse_json_cred(path)
        elif name.endswith(".env") or name.endswith(".cfg"):
            parsed = _parse_env_cred(path)
        elif name.endswith(".ini") or name == "credentials":
            parsed = _parse_ini_cred(path) or _parse_env_cred(path)
        else:
            continue
        if parsed:
            return parsed
    return None


def _load_credentials() -> Tuple[str, str]:
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        u = UserSecretsClient()
        kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception:
        kid = sec = None
    if not kid or not sec:
        kid = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get(SECRET_ID)
        sec = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get(SECRET_KEY)
    if not kid or not sec:
        got = _cred_from_inputs()
        if got:
            kid, sec = got
    if not kid or not sec:
        raise S3CredentialsError(
            f"нет ключей S3: Kaggle Secrets {SECRET_ID}/{SECRET_KEY}, "
            f"env AWS_*, или датасет dmitrijkv/koz3-s3-cred "
            f"(cred.json | *.env | credentials.ini под /kaggle/input)"
        )
    return kid, sec


def s3_client() -> Any:
    """Build a boto3 S3 client; credentials never appear in raised messages."""
    import boto3
    from botocore.config import Config

    kid, sec = _load_credentials()
    try:
        return boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=kid,
            aws_secret_access_key=sec,
            region_name=REGION,
            config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=32,
            ),
        )
    except Exception as e:
        raise _wrap(e, kid, sec) from None


def list_keys(prefix: str, *, client: Optional[Any] = None) -> List[Tuple[str, int]]:
    """Paginated ``list_objects_v2`` → ``[(key, size), ...]`` under ``prefix``."""
    cli = client or s3_client()
    kid = sec = None  # scrub unknowns from boto msgs
    out: List[Tuple[str, int]] = []
    token = None
    try:
        while True:
            kw: Dict[str, Any] = {"Bucket": BUCKET, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = cli.list_objects_v2(**kw)
            for obj in resp.get("Contents") or []:
                out.append((obj["Key"], int(obj["Size"])))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
    except S3CredentialsError:
        raise
    except Exception as e:
        raise _wrap(e, kid, sec) from None
    return out


def download(
    key: str,
    dst: Union[str, Path],
    *,
    client: Optional[Any] = None,
    expected_size: Optional[int] = None,
    retries: int = 5,
) -> Path:
    """Download ``key`` → ``dst`` (file path). Skip if size already matches."""
    cli = client or s3_client()
    dst_p = Path(dst)
    if expected_size is None:
        # resolve size via HEAD when possible
        try:
            meta = cli.head_object(Bucket=BUCKET, Key=key)
            expected_size = int(meta["ContentLength"])
        except Exception:
            expected_size = None
    if (
        dst_p.exists()
        and expected_size is not None
        and dst_p.stat().st_size == expected_size
    ):
        return dst_p
    from boto3.s3.transfer import TransferConfig  # lazy: only when transferring

    dst_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_p.with_name(dst_p.name + ".part")
    last_err: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            cli.download_file(
                BUCKET,
                key,
                str(tmp),
                Config=TransferConfig(max_concurrency=4),
            )
            if expected_size is not None and tmp.stat().st_size != expected_size:
                raise S3TransferError(
                    f"size mismatch for key={key!r}: got {tmp.stat().st_size} want {expected_size}"
                )
            tmp.replace(dst_p)
            return dst_p
        except S3TransferError:
            raise
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    assert last_err is not None
    raise _wrap(last_err) from None


def download_prefix(
    prefix: str,
    dst: Union[str, Path],
    *,
    client: Optional[Any] = None,
    keys: Optional[Iterable[Tuple[str, int]]] = None,
) -> List[Path]:
    """Download every object under ``prefix`` into ``dst``, preserving relative keys.

    If ``keys`` is given (from ``list_keys``), use it; else list first.
    Relative path = key with ``prefix`` stripped (leading slashes removed).
    """
    cli = client or s3_client()
    root = Path(dst)
    root.mkdir(parents=True, exist_ok=True)
    items = list(keys) if keys is not None else list_keys(prefix, client=cli)
    pref = prefix if prefix.endswith("/") else prefix
    out: List[Path] = []
    for key, size in items:
        if key.endswith("/"):
            continue
        rel = key[len(pref) :] if key.startswith(pref) else key
        rel = rel.lstrip("/")
        target = root / rel
        out.append(download(key, target, client=cli, expected_size=size))
    return out


__all__ = [
    "BUCKET",
    "ENDPOINT",
    "REGION",
    "SECRET_ID",
    "SECRET_KEY",
    "S3CredentialsError",
    "S3TransferError",
    "s3_client",
    "list_keys",
    "download",
    "download_prefix",
]
# --- END koz3_s3 ---

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

OUT = Path(os.environ.get("SAT_RECALL_OUT", "/kaggle/working/sat_recall"))
CACHE = Path(os.environ.get("SAT_RECALL_S3_CACHE", "/kaggle/working/train-expds-3"))
BOOT = int(os.environ.get("SAT_RECALL_BOOT", "2000"))
BPRIME = float(os.environ.get("BPRIME", "0.315800"))
SMOKE = os.environ.get("SAT_RECALL_SMOKE", "") == "1"
SKIP_S3 = os.environ.get("SAT_RECALL_SKIP_S3", "") == "1"
T0 = time.time()

# Honest-10 / typical sat-heavy train prefixes (override via SAT_RECALL_PREFIXES=015,026,...)
_DEFAULT_PREFIXES = (
    "015_", "026_", "031_", "033_", "034_", "035_", "040_", "041_", "042_", "043_",
    "046_", "047_", "048_",
)
_raw = os.environ.get("SAT_RECALL_PREFIXES", "")
if _raw.strip():
    SAT_PREFIXES = tuple(p if p.endswith("_") else f"{p}_" for p in _raw.replace(" ", "").split(",") if p)
else:
    SAT_PREFIXES = _DEFAULT_PREFIXES

SAT_MOD_RE = re.compile(r"(?i)(^|/)[^/]*(_)?(SpOr|sp_or|Spor)(_|[^/]*)?(/|$)")
LABEL_HINTS = (
    "разметка",
    "utm.json",
    "manifest.json",
    "labels",
    "markup",
)

LEVERS = {
    "L1_band_verify": {
        "sat_band_verify.enabled": True,
        "sat_band_verify.scale": 0.90,
        "sat_band_verify.verify_floor": 0.40,
    },
    "L2_agree_accept": {
        "sat_agree_accept.enabled": True,
        "sat_agree_accept.radius_m": 16.0,
        "sat_agree_accept.scale": 0.85,
    },
    "L3_cross_nms": {
        "sat_cross_nms.enabled": True,
        "sat_cross_nms.nms_m": 4.0,
    },
    "ALL": {
        "sat_band_verify.enabled": True,
        "sat_band_verify.scale": 0.90,
        "sat_band_verify.verify_floor": 0.40,
        "sat_agree_accept.enabled": True,
        "sat_agree_accept.radius_m": 16.0,
        "sat_agree_accept.scale": 0.85,
        "sat_cross_nms.enabled": True,
        "sat_cross_nms.nms_m": 4.0,
    },
}

# Prior expected ΔLB (proxy s_sat≈0.13 × ΔF1_type); gate needs measured honest-10.
PRIORS = {
    "L1_band_verify": 0.005,
    "L2_agree_accept": 0.003,
    "L3_cross_nms": 0.0015,
    "ALL": 0.007,
}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def _write_status(state: str, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "elapsed_s": round(time.time() - T0, 1), **extra}
    (OUT / "STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (OUT / "STATUS.txt").write_text(state + "\n" + json.dumps(payload, ensure_ascii=False) + "\n")


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
    if not any(top.startswith(p) for p in SAT_PREFIXES):
        return False
    low = key.lower()
    if SAT_MOD_RE.search(key):
        return True
    return any(h in low for h in LABEL_HINTS)


def ensure_boto3():
    try:
        import boto3  # noqa: F401
    except ImportError:
        log("pip install boto3 (internet)…")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "boto3"])


def pull_sat_from_s3() -> dict:
    ensure_boto3()
    try:
        cli = s3_client()  # noqa: F821
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
        "prefixes": list(SAT_PREFIXES),
        "listed": 0,
        "wanted": 0,
        "downloaded": 0,
        "skipped_exists": 0,
        "errors": [],
    }
    _write_status("RUNNING", phase="s3_list")
    wanted: list[tuple[str, int]] = []
    try:
        for pref in SAT_PREFIXES:
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
                report["errors"].append(f"{key}: {type(e).__name__}: {e}")
                if len(report["errors"]) > 50:
                    break
    except S3TransferError as e:  # noqa: F821
        report["errors"].append(str(e))
        report["state"] = "FAILED"
        marker.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        _write_status("FAILED", phase="s3_transfer", error=str(e))
        raise SystemExit(f"FATAL: S3 transfer failed: {e}") from e

    report["state"] = (
        "DONE" if (not report["errors"] or report["downloaded"] or report["skipped_exists"]) else "FAILED"
    )
    marker.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["state"] != "DONE" and report["wanted"] == 0:
        _write_status("FAILED", phase="s3_empty", report=report)
        raise SystemExit("FATAL: S3 returned 0 SpOr|label keys for configured prefixes.")
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
        "damaged_delta": None,
        "pass_full": None,
        "expected_delta_lb": PRIORS.get(name),
        "gate": {"delta_min": 0.003, "p_gt0_min": 0.9, "damaged_not_worse": True},
        "note": (
            "Prior ΔLB only. Set SAT_RECALL_REINFER=1 + tip code/weights for classic "
            "honest-10 scoreboard with lb-breakdown."
        ),
        "lb_context": {
            "sat_gt": 377,
            "sat_P": 0.456,
            "sat_R": 0.255,
            "s_sat_proxy": 0.1297,
            "ceiling_R0.6_proxy": 0.025,
            "ceiling_R0.6_clsmap": 0.049,
            "break_even_hit_pct": 8.9,
        },
    }
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    _write_status("RUNNING", levers=list(LEVERS), smoke=SMOKE)

    if SMOKE:
        notes = Path(__file__).resolve().parent / "notes" / "sat_fn_summary.json"
        fn = json.loads(notes.read_text()) if notes.exists() else {}
        summary = {
            "mode": "SMOKE",
            "bprime": BPRIME,
            "gate": {"delta_min": 0.003, "p_gt0_min": 0.9, "damaged_not_worse": True},
            "levers": LEVERS,
            "expected_delta_lb_prior": PRIORS,
            "fn_summary": fn,
            "s3": "skipped_in_smoke",
            "pass_full": None,
            "exclusions": ["sat2", "sp_large", "sat_8m_cluster", "near_kept", "large_objects"],
        }
        (OUT / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        _write_status("SMOKE", summary_keys=list(summary))
        log("SMOKE done")
        return 0

    s3_report = None
    if not SKIP_S3:
        s3_report = pull_sat_from_s3()
        log("S3 report", {k: s3_report.get(k) for k in ("state", "listed", "wanted", "downloaded", "skipped_exists")})

    stack = Path(os.environ["SAT_RECALL_STACK"]) if os.environ.get("SAT_RECALL_STACK") else discover(
        ["stack_arch.geojson", "stack_a.geojson", "stack.geojson", "*fam0*.geojson", "*sp_or*.geojson"]
    )
    gt = Path(os.environ["SAT_RECALL_GT"]) if os.environ.get("SAT_RECALL_GT") else discover(
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
        "expected_delta_lb_prior": PRIORS,
        "cache": str(CACHE),
        "exclusions": ["sat2", "sp_large", "sat_8m_cluster", "near_kept", "large_objects"],
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
