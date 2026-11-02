"""Сверка F:\\2\\train-expds-3 с бакетом: размер и ETag. Служебные остатки уезжают в F:\\2\\Trash."""
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(r"F:\2\train-expds-3")
TRASH = Path(r"F:\2\Trash")
JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}

spec = importlib.util.spec_from_file_location(
    "s3sync", r"F:\expds3\organizers\materials\s3_script\s3_sync_train-expds-3.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["s3sync"] = mod
spec.loader.exec_module(mod)
from botocore.config import Config
import boto3

args = mod.parse_args(["--mode", "verify", "--local-dir", str(ROOT), "--log-file", r"F:\2\s3_academic.log"])
client = boto3.client(
    "s3",
    endpoint_url=args.endpoint,
    aws_access_key_id=args.access_key,
    aws_secret_access_key=args.secret_key,
    verify=False,
    config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=25,
                  retries={"max_attempts": 2, "mode": "standard"}),
)
PROGRESS = Path(r"F:\2\verify_progress.txt")


def is_junk(path: Path) -> bool:
    rel = path.as_posix().lower()
    return (
        "cyberducksegment" in rel
        or path.name.lower() in JUNK_NAMES
        or path.name.endswith(".part")
        or bool(mod.PART_RE.search(path.name))
    )


def move_tree(src: Path) -> None:
    dest = TRASH / src.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    shutil.move(str(src), str(dest))


def move_leftovers(remote_keys: set[str]) -> tuple[int, int, list[str]]:
    moved = 0
    moved_bytes = 0
    kept_extra: list[str] = []
    segment_dirs = [p for p in ROOT.rglob("*") if p.is_dir() and p.name.endswith(".cyberducksegment")]
    for folder in segment_dirs:
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        move_tree(folder)
        moved += 1
        moved_bytes += size
    for path in list(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in remote_keys:
            continue
        if is_junk(path):
            moved_bytes += path.stat().st_size
            move_tree(path)
            moved += 1
        else:
            kept_extra.append(rel)
    return moved, moved_bytes, kept_extra


def main() -> None:
    objects = mod.list_remote(client, args)
    remote_keys = {obj.key for obj in objects}
    print(f"remote {len(objects)}", flush=True)
    moved, moved_bytes, kept_extra = move_leftovers(remote_keys)
    print(f"trash_items {moved} trash_GiB {moved_bytes / 2**30:.2f} other_extra {len(kept_extra)}", flush=True)
    for rel in kept_extra:
        print(f"extra {rel}", flush=True)

    done = set()
    if PROGRESS.exists():
        done = {line.split("\t", 1)[-1] for line in PROGRESS.read_text(encoding="utf-8").splitlines() if line.startswith("ok\t")}
    ok = len(done)
    missing = size_bad = sum_bad = unver = 0
    problems: list[str] = []
    hashed = 0
    for index, obj in enumerate(objects, 1):
        if obj.key in done:
            continue
        dest = ROOT / obj.key
        if not dest.is_file():
            missing += 1
            problems.append(f"missing {obj.key}")
            continue
        size = dest.stat().st_size
        if size != obj.size:
            size_bad += 1
            problems.append(f"size {obj.key} local {size} remote {obj.size}")
            continue
        result = mod.verify_object_checksum(client, args, obj, dest)
        hashed += size
        if result is True:
            ok += 1
            with PROGRESS.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(f"ok\t{obj.key}\n")
        elif result is False:
            sum_bad += 1
            problems.append(f"etag {obj.key}")
        else:
            unver += 1
            problems.append(f"unverifiable {obj.key}")
        if index % 50 == 0 or index == len(objects):
            print(
                f"checked {index}/{len(objects)} ok {ok} size_bad {size_bad} "
                f"etag_bad {sum_bad} unverifiable {unver} hashed_GiB {hashed / 2**30:.1f}",
                flush=True,
            )
    print(
        f"RESULT ok {ok} missing {missing} size_bad {size_bad} etag_bad {sum_bad} unverifiable {unver}",
        flush=True,
    )
    for line in problems:
        print(line, flush=True)
    if missing or size_bad or sum_bad or unver or kept_extra:
        raise SystemExit(1)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
