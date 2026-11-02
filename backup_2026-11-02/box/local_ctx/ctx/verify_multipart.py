"""Проверка крупных файлов составным ETag из head_object, не усечённым списком."""
import hashlib
import importlib.util
import math
import sys
import urllib3
from pathlib import Path

urllib3.disable_warnings()

ROOT = Path(r"F:\2\train-expds-3")
MANIFEST = Path(r"F:\2\etag_manifest.tsv")
TERMINAL = Path(r"C:\Users\Пользователь\.cursor\projects\c-Sloi\terminals\390042.txt")
JOBS = Path(r"F:\2\multipart_jobs.tsv")
PROGRESS = Path(r"F:\2\multipart_verify_progress.txt")
BAD = Path(r"F:\2\multipart_verify_bad.txt")
PART_MIB = (10, 8, 5, 16, 32, 64, 4, 20, 25, 50, 100, 128, 256, 2, 1, 3, 6, 12, 24, 48, 96)


def bad_keys() -> set[str]:
    found = set()
    for line in TERMINAL.read_text(encoding="utf-8").splitlines():
        if line.startswith("md5\t"):
            found.add(line.split("\t", 1)[1])
    return found


def client_and_args():
    spec = importlib.util.spec_from_file_location(
        "s3sync", r"F:\expds3\organizers\materials\s3_script\s3_sync_train-expds-3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s3sync"] = mod
    spec.loader.exec_module(mod)
    from botocore.config import Config
    import boto3
    args = mod.parse_args(["--mode", "verify", "--local-dir", str(ROOT), "--log-file", r"F:\2\s3_multipart_quiet.log"])
    client = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        verify=False,
        config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=30,
                      retries={"max_attempts": 3, "mode": "standard"}),
    )
    return client, args


def part_candidates(size: int, parts: int) -> list[int]:
    if parts < 2:
        return []
    lo = math.ceil(size / parts)
    hi = (size - 1) // (parts - 1)
    found = []
    for mib in PART_MIB:
        part = mib * 1024 * 1024
        if lo <= part <= hi and math.ceil(size / part) == parts:
            found.append(part)
    return found


def multipart_md5(path: Path, part_size: int) -> str:
    outer = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            block = fh.read(part_size)
            if not block:
                break
            outer.update(hashlib.md5(block).digest())
    return outer.hexdigest()


def head_jobs() -> None:
    wanted = bad_keys()
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        size_s, _etag, key = line.split("\t", 2)
        if key in wanted:
            rows.append((int(size_s), key))
    client, args = client_and_args()
    lines = []
    simple = multipart = head_fail = ambiguous = no_part = 0
    for index, (size, key) in enumerate(rows, 1):
        try:
            remote = client.head_object(Bucket=args.bucket, Key=key)["ETag"].strip('"')
        except Exception as exc:  # noqa: BLE001
            head_fail += 1
            lines.append(f"head_fail\t{size}\t\t\t{key}\t{exc}")
            print(f"head {index}/{len(rows)} fail", flush=True)
            continue
        local = ROOT / key
        if not local.is_file() or local.stat().st_size != size:
            head_fail += 1
            lines.append(f"local\t{size}\t\t\t{key}")
            continue
        if "-" not in remote:
            simple += 1
            lines.append(f"simple\t{size}\t{remote.lower()}\t\t{key}")
        else:
            hexpart, _, parts_txt = remote.rpartition("-")
            parts = int(parts_txt)
            candidates = part_candidates(size, parts)
            multipart += 1
            if len(candidates) != 1:
                ambiguous += 1 if len(candidates) > 1 else 0
                no_part += 1 if not candidates else 0
            joined = ",".join(str(item) for item in candidates)
            lines.append(f"multipart\t{size}\t{hexpart.lower()}\t{parts}\t{joined}\t{key}")
        if index % 25 == 0 or index == len(rows):
            print(f"head {index}/{len(rows)} multipart {multipart} simple {simple} fail {head_fail}", flush=True)
    JOBS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(
        f"HEAD_DONE files {len(rows)} multipart {multipart} simple {simple} "
        f"head_fail {head_fail} ambiguous {ambiguous} no_part {no_part}",
        flush=True,
    )


def hash_jobs() -> None:
    jobs = []
    for line in JOBS.read_text(encoding="utf-8").splitlines():
        if line.startswith("multipart\t"):
            kind, size_s, expected, parts_s, joined, key = line.split("\t", 5)
            jobs.append((int(size_s), expected, int(parts_s), [int(x) for x in joined.split(",") if x], key))
    done = set()
    if PROGRESS.exists():
        done = {line.split("\t", 1)[-1] for line in PROGRESS.read_text(encoding="utf-8").splitlines() if line.startswith("ok\t")}
    ok = len(done)
    bad = 0
    hashed = 0
    pending = [(size, expected, parts, candidates, key) for size, expected, parts, candidates, key in jobs if key not in done]
    print(f"hash_pending {len(pending)} already_ok {ok}", flush=True)
    for index, (size, expected, parts, candidates, key) in enumerate(pending, 1):
        path = ROOT / key
        matched = False
        for part_size in candidates:
            digest = multipart_md5(path, part_size)
            hashed += size
            if digest == expected:
                matched = True
                break
        if matched:
            ok += 1
            with PROGRESS.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(f"ok\t{key}\n")
            print(f"ok {ok}/{len(jobs)} {hashed / 2**30:.1f} GiB", flush=True)
        else:
            bad += 1
            with BAD.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(f"{size}\t{parts}\t{key}\n")
            print(f"FAIL size {size} parts {parts}", flush=True)
    print(f"HASH_DONE ok {ok} bad {bad} of {len(jobs)}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "head"
    if mode == "head":
        head_jobs()
    elif mode == "hash":
        hash_jobs()
    else:
        raise SystemExit("mode is head or hash")
