"""Дописывает хвост двух коротких ортофото одним потоком, с повтором после обрыва."""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(r"F:\2\train-expds-3")
# Последний байт, до которого Герите точно записан подряд. Дальше параллельная
# докачка оборвалась и могла оставить дыры, поэтому хвост пишется заново.
SOLID = {
    "01_Герите_Or_ортофото.tif": 9489067721,
}
CHUNK = 8 * 1024 * 1024

spec = importlib.util.spec_from_file_location(
    "s3sync", r"F:\expds3\organizers\materials\s3_script\s3_sync_train-expds-3.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["s3sync"] = mod
spec.loader.exec_module(mod)
args = mod.parse_args(["--mode", "verify", "--local-dir", str(ROOT), "--log-file", r"F:\2\s3_tail.log"])
client = mod.make_client(args)


def resume(path: Path) -> None:
    key = path.relative_to(ROOT).as_posix()
    remote = client.head_object(Bucket=args.bucket, Key=key)["ContentLength"]
    solid = SOLID.get(path.name)
    if solid is not None and path.stat().st_size != solid:
        with open(path, "r+b") as fh:
            fh.truncate(solid)
        print(path.name, "trimmed_to", solid, flush=True)
    while path.stat().st_size < remote:
        have = path.stat().st_size
        print(path.name, "from", have, "left", remote - have, flush=True)
        try:
            resp = client.get_object(Bucket=args.bucket, Key=key, Range=f"bytes={have}-")
            code = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code != 206:
                raise RuntimeError(f"range {code}")
            body = resp["Body"]
            try:
                with open(path, "ab", buffering=0) as fh:
                    while True:
                        chunk = body.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
            finally:
                body.close()
        except Exception as exc:
            print(path.name, "retry", type(exc).__name__, "at", path.stat().st_size, flush=True)
            time.sleep(5)
    final = path.stat().st_size
    print(path.name, "final", final, "match", final == remote, flush=True)
    if final != remote:
        raise RuntimeError(f"{path.name} {final} != {remote}")


def main() -> None:
    names = ("01_Герите_Or_ортофото.tif", "01_Лешки_Or_ортофото.tif")
    files = [next(ROOT.rglob(name)) for name in names]
    for path in files:
        resume(path)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
