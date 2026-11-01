#!/usr/bin/env python3
"""Локальный dry-run: mock boto3, без сети к S3 и без реальных секретов."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "koz3-s3-index.py"


class FakePaginatorState:
    def __init__(self, pages):
        self.pages = pages
        self.i = 0


class FakeS3:
    """Минимальный list_objects_v2 с пагинацией и Delimiter."""

    def __init__(self):
        # synthetic tree
        self.objects = [
            {
                "Key": "fam0/stack_A/part-000.jsonl",
                "Size": 100,
                "LastModified": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            },
            {
                "Key": "fam0/family_clusters/meta.json",
                "Size": 20,
                "LastModified": datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
            },
            {
                "Key": "stack_arch/oof/fold0.geojson",
                "Size": 50,
                "LastModified": datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
            },
            {
                "Key": "other/deep/a/b/c.bin",
                "Size": 7,
                "LastModified": datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc),
            },
            {
                "Key": "readme.txt",
                "Size": 3,
                "LastModified": datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc),
            },
        ]

    def list_objects_v2(self, **kw):
        bucket = kw.get("Bucket")
        assert bucket == "train-expds-3"
        max_keys = int(kw.get("MaxKeys") or 1000)
        prefix = kw.get("Prefix") or ""
        delimiter = kw.get("Delimiter")
        token = kw.get("ContinuationToken")

        matched = [o for o in self.objects if o["Key"].startswith(prefix)]

        if delimiter == "/":
            # only immediate children under prefix
            contents = []
            commons = set()
            for o in matched:
                rest = o["Key"][len(prefix) :]
                if not rest:
                    continue
                if "/" in rest:
                    commons.add(prefix + rest.split("/", 1)[0] + "/")
                else:
                    contents.append(o)
            return {
                "Contents": contents[:max_keys],
                "CommonPrefixes": [{"Prefix": p} for p in sorted(commons)],
                "IsTruncated": False,
            }

        # force multi-page when MaxKeys small or when token used — always page by 2 for dry-run
        page_size = 2 if max_keys >= 2 else max_keys
        start = int(token) if token else 0
        chunk = matched[start : start + page_size]
        nxt = start + page_size
        truncated = nxt < len(matched)
        out = {
            "Contents": chunk,
            "CommonPrefixes": [],
            "IsTruncated": truncated,
        }
        if truncated:
            out["NextContinuationToken"] = str(nxt)
        return out


def load_module(out_dir: Path):
    os.environ["S3_INDEX_OUT"] = str(out_dir)
    # fake env creds so s3_client() does not need kaggle_secrets
    os.environ["AWS_ACCESS_KEY_ID"] = "DRYRUN_KEY_ID"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "DRYRUN_SECRET"
    # ensure secrets path fails quietly
    sys.modules.pop("kaggle_secrets", None)

    fake = FakeS3()

    def fake_client(*a, **k):
        # never touch network; ignore endpoint/keys
        return fake

    boto3_mod = mock.MagicMock()
    boto3_mod.client = fake_client
    botocore_config = mock.MagicMock()
    config_mod = mock.MagicMock()
    config_mod.Config = mock.MagicMock(return_value=object())

    with mock.patch.dict(
        sys.modules,
        {
            "boto3": boto3_mod,
            "botocore": mock.MagicMock(),
            "botocore.config": config_mod,
        },
    ):
        spec = importlib.util.spec_from_file_location("koz3_s3_index", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        rc = mod.main()
    return rc, out_dir


def main():
    with tempfile.TemporaryDirectory(prefix="s3-index-dryrun-") as td:
        out = Path(td)
        rc, out_dir = load_module(out)
        status = (out_dir / "STATUS.txt").read_text(encoding="utf-8").strip()
        index = (out_dir / "s3_index.txt").read_text(encoding="utf-8").splitlines()
        summary = (out_dir / "s3_prefix_summary.txt").read_text(encoding="utf-8")
        hits = (out_dir / "s3_hits_fam0.txt").read_text(encoding="utf-8").splitlines()
        print("DRYRUN_RC", rc)
        print("STATUS", status)
        print("INDEX_LINES", len(index))
        for line in index:
            print("  INDEX", line)
        print("HITS", len(hits))
        for line in hits:
            print("  HIT", line)
        print("SUMMARY>>>")
        print(summary, end="")
        print("<<<SUMMARY")
        ok = (
            rc == 0
            and status == "DONE"
            and len(index) == 5
            and len(hits) >= 3
            and "fam0/" in summary
            and "stack_arch/" in summary
        )
        # ensure secret values never appear in outputs
        blob = "\n".join(index) + summary + "\n".join(hits) + status
        if "DRYRUN_SECRET" in blob or "DRYRUN_KEY_ID" in blob:
            print("DRYRUN_FAIL secrets leaked into outputs")
            sys.exit(2)
        if not ok:
            print("DRYRUN_FAIL checks")
            sys.exit(1)
        print("DRYRUN_OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
