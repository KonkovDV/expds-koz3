"""bot-9: kaggle kernels push uploads only code_file, so the sidecar src tar must be embedded.
usage: python3 embed_src.py <kernel.py> <src.tar>  (in-place; replaces sidecar loader block)"""
import sys, gzip, base64, hashlib
from pathlib import Path
py, tar = Path(sys.argv[1]), Path(sys.argv[2])
raw = tar.read_bytes()
sha = hashlib.sha256(raw).hexdigest()
b64 = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode()
lines = py.read_text(encoding="utf-8").splitlines(True)
i0 = next(i for i, l in enumerate(lines) if l.startswith("_here = Path(__file__)"))
i1 = next(i for i, l in enumerate(lines) if l.startswith("_raw = _tar.read_bytes()"))
new = ["import base64\n", "SRC_EMB = " + repr(b64) + "\n",
       "_raw = gzip.decompress(base64.b64decode(SRC_EMB))\n"]
lines[i0:i1 + 1] = new
txt = "".join(lines)
assert ("SRC_TAR_SHA256 = \"%s\"" % sha) in txt, "tar sha constant mismatch"
compile(txt, str(py), "exec")
py.write_text(txt, encoding="utf-8")
print(py, "bytes", py.stat().st_size, "sha", hashlib.sha256(py.read_bytes()).hexdigest(), "tar", sha)
