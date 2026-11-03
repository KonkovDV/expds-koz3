#!/usr/bin/env python3
"""Rebuild koz3-w1-sat2-gate.py with embedded vendor/w5_rerank/scripts/data tarball.
Run from package root after editing sidecars. Does NOT push to Kaggle."""
from __future__ import annotations
import base64, pathlib, subprocess, sys, textwrap, tempfile, gzip, tarfile, io
ROOT = pathlib.Path(__file__).resolve().parent
BODY = ROOT / "_body.py"  # optional; else regenerate from current script stripping blob

def main():
    tar_path = pathlib.Path(tempfile.mkstemp(suffix=".tar.gz")[1])
    subprocess.check_call(
        ["tar", "-czf", str(tar_path), "vendor", "w5_rerank", "scripts", "data"],
        cwd=str(ROOT),
    )
    blob = tar_path.read_bytes()
    b64 = "\n".join(textwrap.wrap(base64.b64encode(blob).decode(), 76))
    # take current script and replace BUNDLE_GZ_B64 contents
    script = ROOT / "koz3-w1-sat2-gate.py"
    text = script.read_text()
    start = text.index('BUNDLE_GZ_B64 = r"""') + len('BUNDLE_GZ_B64 = r"""')
    end = text.index('"""', start)
    text = text[:start] + "\n" + b64 + "\n" + text[end:]
    script.write_text(text)
    tar_path.unlink(missing_ok=True)
    # verify
    raw = gzip.decompress(base64.b64decode(b64))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        assert "vendor/materials/compute_metrics.py" in tf.getnames()
    print("OK", script, "bytes", script.stat().st_size, "bundle_gz", len(blob))

if __name__ == "__main__":
    main()
