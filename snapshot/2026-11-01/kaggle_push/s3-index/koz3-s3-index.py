#!/usr/bin/env python3
"""KOZ3: индекс объектов S3 bucket train-expds-3 (Selectel ru-7).

Пишет в /kaggle/working:
  - s3_index.txt          — key, size, last_modified (UTC ISO) по одной строке
  - s3_prefix_summary.txt — count/total_size по префиксам глубины ≤3
  - s3_hits_fam0.txt      — ключи с fam0|stack_A|stack_arch|family_clusters|oof|jsonl
  - STATUS.txt            — DONE / FAILED
Секреты никогда не печатаются.
"""
from __future__ import annotations

import os
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(os.environ.get("S3_INDEX_OUT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)

BUCKET, ENDPOINT = "train-expds-3", os.environ.get(
    "DFINE_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru"
)
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"

HIT_RE = re.compile(
    r"(fam0|stack_A|stack_arch|family_clusters|oof|jsonl)", re.IGNORECASE
)
PREFIX_DEPTH = 3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def write_status(state: str) -> None:
    (OUT / "STATUS.txt").write_text(state + "\n", encoding="utf-8")


def _cred_from_inputs():
    # приватный датасет с cred.json (поля id/key или aws_access_key_id/aws_secret_access_key); значения не печатаются
    import glob
    import json as _j

    for _p in sorted(glob.glob("/kaggle/input/**/cred.json", recursive=True)):
        try:
            with open(_p, encoding="utf-8") as _fh:
                _c = _j.load(_fh)
        except Exception as _e:
            log("cred.json не читается:", type(_e).__name__)
            continue
        if not isinstance(_c, dict):
            continue
        _k = _c.get("id") or _c.get("aws_access_key_id")
        _s = _c.get("key") or _c.get("aws_secret_access_key")
        if _k and _s:
            log("ключи S3 из приватного датасета")
            return _k, _s
    return None, None


def s3_client():
    import boto3
    from botocore.config import Config

    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient

        u = UserSecretsClient()
        kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception as e:
        log(
            "Kaggle Secrets недоступны:",
            type(e).__name__,
            "— пробую env / приватный датасет",
        )
    if not kid or not sec:
        kid, sec = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        )
    if not kid or not sec:
        kid, sec = _cred_from_inputs()
    if not kid or not sec:
        raise SystemExit(
            f"нет ключей S3: Kaggle Secrets {SECRET_ID}/{SECRET_KEY} или датасет dmitrijkv/koz3-s3-cred (cred.json)"
        )
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=kid,
        aws_secret_access_key=sec,
        region_name="ru-7",
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=32
        ),
    )


def _iso_utc(lm) -> str:
    if lm is None:
        return ""
    if isinstance(lm, datetime):
        if lm.tzinfo is None:
            lm = lm.replace(tzinfo=timezone.utc)
        return lm.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(lm)


def _prefixes_up_to(key: str, depth: int = PREFIX_DEPTH):
    parts = [p for p in key.split("/") if p != ""]
    # object under a/b/c/file → prefixes a/, a/b/, a/b/c/ (depth≤3); skip bare filename at root
    n = len(parts)
    if n <= 1:
        return []
    # last component is object name; prefixes are first 1..min(depth, n-1) segments
    limit = min(depth, n - 1)
    out = []
    for i in range(1, limit + 1):
        out.append("/".join(parts[:i]) + "/")
    return out


def paginate_objects(cli, *, prefix: str = "", delimiter: str | None = None):
    """Yield (contents_list, common_prefixes_list) pages from list_objects_v2."""
    token = None
    while True:
        kw = dict(Bucket=BUCKET, MaxKeys=1000)
        if prefix:
            kw["Prefix"] = prefix
        if delimiter:
            kw["Delimiter"] = delimiter
        if token:
            kw["ContinuationToken"] = token
        resp = cli.list_objects_v2(**kw)
        yield resp.get("Contents") or [], resp.get("CommonPrefixes") or []
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break


def list_top_level_prefixes(cli):
    prefixes = []
    root_objs = []
    for contents, commons in paginate_objects(cli, delimiter="/"):
        root_objs.extend(contents)
        for cp in commons:
            p = cp.get("Prefix")
            if p:
                prefixes.append(p)
    return sorted(set(prefixes)), root_objs


def list_all_objects(cli, top_prefixes):
    """Full bucket via no-delimiter pagination; also walk each top-level prefix."""
    seen = {}
    # whole-bucket pagination
    for contents, _ in paginate_objects(cli):
        for obj in contents:
            k = obj.get("Key")
            if k is not None:
                seen[k] = obj
    # per top-level prefix (Delimiter path already discovered; re-list for coverage)
    for pref in top_prefixes:
        for contents, _ in paginate_objects(cli, prefix=pref):
            for obj in contents:
                k = obj.get("Key")
                if k is not None:
                    seen[k] = obj
    return [seen[k] for k in sorted(seen)]


def build_outputs(objects):
    index_lines = []
    hits = []
    prefix_stats = defaultdict(lambda: {"count": 0, "size": 0})

    for obj in objects:
        key = obj.get("Key") or ""
        size = int(obj.get("Size") or 0)
        lm = _iso_utc(obj.get("LastModified"))
        index_lines.append(f"{key}\t{size}\t{lm}")
        if HIT_RE.search(key):
            hits.append(f"{key}\t{size}\t{lm}")
        for pref in _prefixes_up_to(key, PREFIX_DEPTH):
            prefix_stats[pref]["count"] += 1
            prefix_stats[pref]["size"] += size

    # also aggregate "" (bucket root) totals
    root_count = len(objects)
    root_size = sum(int(o.get("Size") or 0) for o in objects)

    summary_lines = [
        f"# bucket={BUCKET} objects={root_count} total_bytes={root_size}",
        "# prefix\tcount\ttotal_bytes",
        f"/\t{root_count}\t{root_size}",
    ]
    for pref in sorted(prefix_stats):
        st = prefix_stats[pref]
        summary_lines.append(f"{pref}\t{st['count']}\t{st['size']}")

    return index_lines, summary_lines, hits


def main() -> int:
    write_status("RUNNING")
    try:
        cli = s3_client()
        try:
            cli.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
        except Exception as e:
            raise SystemExit(
                f"S3 недоступен ({type(e).__name__}): проверьте секреты/интернет (детали скрыты)."
            )
        log("S3 OK", BUCKET, ENDPOINT)

        top_prefixes, root_objs = list_top_level_prefixes(cli)
        log("top-level prefixes:", len(top_prefixes), "root objects:", len(root_objs))
        for p in top_prefixes:
            log("  prefix:", p)

        objects = list_all_objects(cli, top_prefixes)
        log("objects total:", len(objects))

        index_lines, summary_lines, hits = build_outputs(objects)

        (OUT / "s3_index.txt").write_text(
            "\n".join(index_lines) + ("\n" if index_lines else ""), encoding="utf-8"
        )
        (OUT / "s3_prefix_summary.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )
        (OUT / "s3_hits_fam0.txt").write_text(
            "\n".join(hits) + ("\n" if hits else ""), encoding="utf-8"
        )
        log(
            "wrote",
            f"index={len(index_lines)}",
            f"prefixes={len(summary_lines) - 3}",
            f"hits={len(hits)}",
        )
        write_status("DONE")
        return 0
    except SystemExit as e:
        log("FAILED:", e)
        write_status("FAILED")
        # still exit non-zero for Kaggle visibility
        print(str(e), file=sys.stderr, flush=True)
        return 1
    except Exception as e:
        log("FAILED:", type(e).__name__)
        traceback.print_exc()
        write_status("FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
