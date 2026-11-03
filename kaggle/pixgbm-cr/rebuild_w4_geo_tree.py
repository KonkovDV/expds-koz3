#!/usr/bin/env python3
"""Rebuild KOZ-3 geo-200-217 train tree for pixgbm-cr gate (bot-6).

Primary source: Selcloud S3 bucket train-expds-3 (MagGr/EtGr/TerFl rasters +
GT labels for regions 200–217). Cache under /kaggle/working with skip-if-exists
and size check. Offline w4-geo shards are NOT used (do not exist).

S3 client is inlined VERBATIM from bots/bot7-geo-gate@11a47d2a
kaggle/common/koz3_s3.py (git blob 7e1d0ab119f2d2c678f753582b22a4344104d5bc).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

GEO_PREFIXES = tuple(f"{n:03d}_" for n in range(200, 218))
GEO_MOD_TOKENS = ("_MagGr_", "_EtGr_", "_TerFl_", "_maggr_", "_etgr_", "_terfl_")
# Labels / meta kept alongside rasters (catalog/gt loaders need these).
LABEL_SUFFIXES = (
    ".geojson",
    ".json",
    ".gpkg",
    ".shp",
    ".dbf",
    ".shx",
    ".prj",
    ".cpg",
)
LABEL_NAME_HINTS = (
    "objectpoly",
    "findspoints",
    "разметка",
    "utm.json",
    "object_poly",
    "finds_points",
)

# --- BEGIN koz3_s3 (source: bots/bot7-geo-gate@11a47d2a kaggle/common/koz3_s3.py blob 7e1d0ab119f2d2c678f753582b22a4344104d5bc; mid-file omit: from __future__ import annotations) ---
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
from typing import Any, Iterable

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


def _scrub(text: str, *secrets: str | None) -> str:
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


def _wrap(exc: BaseException, *secrets: str | None) -> S3TransferError:
    return S3TransferError(
        f"{type(exc).__name__}: {_scrub(str(exc), *secrets)}"
    )


def _parse_json_cred(path: Path) -> tuple[str, str] | None:
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


def _parse_env_cred(path: Path) -> tuple[str, str] | None:
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


def _parse_ini_cred(path: Path) -> tuple[str, str] | None:
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


def _cred_from_inputs(root_glob: str | None = None) -> tuple[str, str] | None:
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


def _load_credentials() -> tuple[str, str]:
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


def list_keys(prefix: str, *, client: Any | None = None) -> list[tuple[str, int]]:
    """Paginated ``list_objects_v2`` → ``[(key, size), ...]`` under ``prefix``."""
    cli = client or s3_client()
    kid = sec = None  # scrub unknowns from boto msgs
    out: list[tuple[str, int]] = []
    token = None
    try:
        while True:
            kw: dict[str, Any] = {"Bucket": BUCKET, "Prefix": prefix}
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
    dst: str | Path,
    *,
    client: Any | None = None,
    expected_size: int | None = None,
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
    last_err: BaseException | None = None
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
    dst: str | Path,
    *,
    client: Any | None = None,
    keys: Iterable[tuple[str, int]] | None = None,
) -> list[Path]:
    """Download every object under ``prefix`` into ``dst``, preserving relative keys.

    If ``keys`` is given (from ``list_keys``), use it; else list first.
    Relative path = key with ``prefix`` stripped (leading slashes removed).
    """
    cli = client or s3_client()
    root = Path(dst)
    root.mkdir(parents=True, exist_ok=True)
    items = list(keys) if keys is not None else list_keys(prefix, client=cli)
    pref = prefix if prefix.endswith("/") else prefix
    out: list[Path] = []
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


def _want_key(key: str) -> bool:
    """Keep MagGr/EtGr/TerFl rasters + GT/meta for regions 200–217 only."""
    key_u = key.replace("\\", "/")
    top = key_u.split("/", 1)[0]
    if not any(top.startswith(p) for p in GEO_PREFIXES):
        return False
    name = key_u.rsplit("/", 1)[-1]
    low = name.lower()
    # modality folder or filename
    if any(tok in key_u for tok in GEO_MOD_TOKENS):
        return True
    if any(h in low for h in LABEL_NAME_HINTS):
        return True
    if low == "utm.json":
        return True
    # geojson/shp sitting next to geo modalities in region root
    if any(low.endswith(suf) for suf in LABEL_SUFFIXES):
        # drop obvious non-geo modalities in path
        if any(x in key_u for x in ("_SpOr_", "_Ae_", "_Or_", "_La_", "_Li_", "_LiDAR_")):
            # still allow if filename itself is ObjectPoly/FindsPoints style
            if any(h in low for h in ("objectpoly", "findspoints", "разметка", "utm")):
                return True
            return False
        return True
    return False


def find_inputs(root: Path | None = None) -> dict[str, Path]:
    root = root or Path("/kaggle/input")
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if p.is_dir():
            out[p.name.lower()] = p
    return out


def list_geo_region_prefixes() -> list[str]:
    """List bucket top-level region folders whose names start with 200_…217_."""
    cli = s3_client()
    tok = None
    found: list[str] = []
    while True:
        kw = {"Bucket": BUCKET, "Prefix": "", "Delimiter": "/"}
        if tok:
            kw["ContinuationToken"] = tok
        r = cli.list_objects_v2(**kw)
        for p in r.get("CommonPrefixes", []) or []:
            pref = p["Prefix"]
            name = pref.rstrip("/")
            if any(name.startswith(gp) for gp in GEO_PREFIXES):
                found.append(pref)
        if not r.get("IsTruncated"):
            break
        tok = r["NextContinuationToken"]
    return sorted(found)


def fetch_geo_tree(dest: Path) -> dict:
    """List S3 prefixes for 200–217, filter MagGr/EtGr/TerFl + labels, cache under dest."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "bucket": BUCKET,
        "endpoint": ENDPOINT,
        "region": REGION,
        "root": str(dest),
        "prefixes": [],
        "downloaded": 0,
        "skipped_exists": 0,
        "filtered_out": 0,
        "errors": [],
        "keys": [],
    }
    try:
        prefixes = list_geo_region_prefixes()
    except (SystemExit, S3CredentialsError, S3TransferError) as e:
        report["errors"].append(f"{type(e).__name__}:{e}")
        report["hollow"] = True
        return report
    except Exception as e:
        report["errors"].append(f"list_prefixes:{type(e).__name__}")
        report["hollow"] = True
        return report

    report["prefixes"] = prefixes
    cli = s3_client()
    # Probe once (no key material in message)
    try:
        cli.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
        report["s3_ok"] = True
    except Exception as e:
        report["s3_ok"] = False
        report["errors"].append(f"S3 недоступен ({type(e).__name__})")
        report["hollow"] = True
        return report

    for pref in prefixes:
        for key, size in list_keys(pref, client=cli):
            if key.endswith("/"):
                continue
            if not _want_key(key):
                report["filtered_out"] += 1
                continue
            local = dest / key
            try:
                if local.exists() and local.is_file() and local.stat().st_size == size:
                    report["skipped_exists"] += 1
                else:
                    download(key, local, expected_size=size, client=cli)
                    report["downloaded"] += 1
                report["keys"].append({"key": key, "size": size, "path": str(local)})
            except Exception as e:
                report["errors"].append(f"{key}:{type(e).__name__}")

    geo_dirs = sorted(
        p.name
        for p in dest.iterdir()
        if p.is_dir() and any(p.name.startswith(gp) for gp in GEO_PREFIXES)
    )
    report["geo_region_dirs"] = geo_dirs
    report["n_geo_dirs"] = len(geo_dirs)
    n_tif = 0
    for d in geo_dirs:
        for p in (dest / d).rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".tif", ".tiff", ".img"}:
                continue
            path_s = str(p).replace("\\", "/")
            if any(tok.strip("_").lower() in path_s.lower() for tok in ("MagGr", "EtGr", "TerFl")):
                n_tif += 1
    report["n_geo_raster_files_guess"] = n_tif
    report["hollow"] = len(geo_dirs) == 0 or n_tif == 0 or bool(
        report["errors"] and report["downloaded"] == 0 and report["skipped_exists"] == 0
    )
    # Drop per-key list from STATUS-sized dumps if huge; keep count
    report["n_keys_kept"] = len(report["keys"])
    if len(report["keys"]) > 50:
        report["keys"] = report["keys"][:50] + [{"truncated": True, "n": report["n_keys_kept"]}]
    return report


def rebuild(dest: Path | None = None, input_root: Path | None = None) -> dict:
    """Public entry used by pixgbm_cr_gate.py — S3-first geo tree for 200–217."""
    dest = dest or Path(os.environ.get("PIXGBM_TREE", "/kaggle/working/train-expds-3"))
    dest.mkdir(parents=True, exist_ok=True)
    inputs = find_inputs(input_root)
    report = fetch_geo_tree(dest)
    report["inputs"] = sorted(inputs)
    report["source"] = "s3"
    return report


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rep = rebuild(dest=dest)
    print(json.dumps({k: v for k, v in rep.items() if k != "keys"}, ensure_ascii=False, indent=2))
    raise SystemExit(2 if rep.get("hollow") else 0)
