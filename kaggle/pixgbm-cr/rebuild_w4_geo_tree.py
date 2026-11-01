#!/usr/bin/env python3
"""Rebuild KOZ-3 geo-200-217 train tree for pixgbm-cr gate (bot-6).

Primary source: Selcloud S3 bucket train-expds-3 (MagGr/EtGr/TerFl rasters +
GT labels for regions 200–217). Cache under /kaggle/working with skip-if-exists
and size check. Offline w4-geo shards are NOT used (do not exist).

S3 client is an inline copy of the shared API (bots/bot7-geo-gate
kaggle/common/koz3_s3.py) so the block can be swapped verbatim later.
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

BUCKET = os.environ.get("PIXGBM_S3_BUCKET", "train-expds-3")
ENDPOINT = os.environ.get("PIXGBM_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
REGION = os.environ.get("PIXGBM_S3_REGION", "ru-7")
SECRET_ID = "KOZ3_S3_ACCESS_KEY_ID"
SECRET_KEY = "KOZ3_S3_SECRET_ACCESS_KEY"


# --- BEGIN koz3_s3 (inline; replace with bots/bot7-geo-gate kaggle/common/koz3_s3.py) ---
def s3_client():
    """boto3 S3 client for Selcloud. Keys never logged or written to working."""
    import boto3
    from botocore.config import Config

    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient

        u = UserSecretsClient()
        kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception:
        kid = sec = None
    if not kid or not sec:
        kid = os.environ.get("AWS_ACCESS_KEY_ID")
        sec = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not kid or not sec:
        import glob
        import json as _j

        for _p in sorted(glob.glob("/kaggle/input/**/cred.json", recursive=True)):
            try:
                with open(_p, encoding="utf-8") as _fh:
                    _c = _j.load(_fh)
            except Exception:
                continue
            if not isinstance(_c, dict):
                continue
            _k = _c.get("id") or _c.get("aws_access_key_id")
            _s = _c.get("key") or _c.get("aws_secret_access_key")
            if _k and _s:
                kid, sec = _k, _s
                break
    if not kid or not sec:
        raise SystemExit(
            f"нет ключей S3: Kaggle Secrets {SECRET_ID}/{SECRET_KEY} "
            "или датасет dmitrijkv/koz3-s3-cred (cred.json)"
        )
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


def list_keys(prefix: str):
    """Yield (key, size) under prefix (paginated)."""
    cli = s3_client()
    tok = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if tok:
            kw["ContinuationToken"] = tok
        r = cli.list_objects_v2(**kw)
        for o in r.get("Contents", []) or []:
            yield o["Key"], int(o["Size"])
        if not r.get("IsTruncated"):
            return
        tok = r["NextContinuationToken"]


def download(key: str, dst, size: int | None = None):
    """Download one key to dst. Skip if exists with matching size. Retries."""
    from boto3.s3.transfer import TransferConfig

    dst = Path(dst)
    cli = s3_client()
    if size is None:
        head = cli.head_object(Bucket=BUCKET, Key=key)
        size = int(head["ContentLength"])
    if dst.exists() and dst.is_file() and dst.stat().st_size == size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    last_err = None
    for a in range(5):
        try:
            cli.download_file(
                BUCKET, key, str(tmp), Config=TransferConfig(max_concurrency=4)
            )
            if tmp.stat().st_size != size:
                raise RuntimeError(
                    f"size mismatch {tmp.stat().st_size} != {size} for {key}"
                )
            tmp.replace(dst)
            return dst
        except Exception as e:
            last_err = e
            time.sleep(5 * (a + 1))
    raise RuntimeError(f"не скачан {key}: {type(last_err).__name__}")


def download_prefix(prefix: str, dst):
    """Download all objects under prefix into dst, preserving key paths under prefix."""
    dst = Path(dst)
    out = []
    for key, size in list_keys(prefix):
        if key.endswith("/"):
            continue
        if key.startswith(prefix):
            rel = key[len(prefix):].lstrip("/")
        else:
            rel = key
        local = dst / rel
        download(key, local, size=size)
        out.append((key, size, str(local)))
    return out


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
    except SystemExit as e:
        report["errors"].append(str(e))
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
        for key, size in list_keys(pref):
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
                    download(key, local, size=size)
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
