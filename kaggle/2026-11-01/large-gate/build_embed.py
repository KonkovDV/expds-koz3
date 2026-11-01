#!/usr/bin/env python3
"""Rebuild koz3-w4-large-gate.py with embedded vendor/w5_rerank/scripts/data tarball."""
from __future__ import annotations
import base64, pathlib, subprocess, sys, textwrap, tempfile, gzip, tarfile, io, shutil
ROOT = pathlib.Path(__file__).resolve().parent

def main():
    body = (ROOT / "_body.py").read_text()
    if "<<<BUILD_EMBED>>>" not in body:
        # already embedded script — strip and rebuild from marker
        raise SystemExit("expected _body.py with <<<BUILD_EMBED>>> placeholder")
    tar_path = pathlib.Path(tempfile.mkstemp(suffix=".tar.gz")[1])
    # exclude bulky ijson.libs if needed? keep for C accel
    subprocess.check_call(
        ["tar", "-czf", str(tar_path),
         "--exclude=__pycache__",
         "vendor", "w5_rerank", "scripts", "data"],
        cwd=str(ROOT),
    )
    blob = tar_path.read_bytes()
    b64 = "\n".join(textwrap.wrap(base64.b64encode(blob).decode(), 76))
    script_text = body.replace("<<<BUILD_EMBED>>>", "\n" + b64 + "\n")
    out = ROOT / "koz3-w4-large-gate.py"
    out.write_text(script_text)
    tar_path.unlink(missing_ok=True)
    raw = gzip.decompress(base64.b64decode(b64))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        names = tf.getnames()
        assert "vendor/materials/compute_metrics.py" in names
        assert "vendor/expds/large_objects.py" in names
        assert "scripts/stack_apply_large_objects.py" in names
    print("OK", out, "bytes", out.stat().st_size, "bundle_gz", len(blob))

if __name__ == "__main__":
    main()
