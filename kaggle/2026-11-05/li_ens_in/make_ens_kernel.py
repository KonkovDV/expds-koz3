#!/usr/bin/env python3
"""Ядро Kaggle для OOF ансамбля c5-in + c12-in прод-кодом на T4 (PREREG kaggle/2026-11-05/li_ens_in/PREREG.md).

    python make_ens_kernel.py --out DIR [--template KERNEL_METADATA.json]

Создаёт два каталога:
- DIR/code — приватный датасет `dmitrijkv/koz3-ens-code`: `koz3_ens_code.tar` (git archive HEAD: expds/, centerhead/,
  ens_oof.py) и dataset-metadata.json;
- DIR/koz3-li-ens-oof — ядро: распаковывает код, ставит smp 0.6.0, как ядра обучения, и запускает ens_oof.py --device cuda.
  Вход: рельеф c12 (`dmitrijkv/koz3-cnx-li-prep12`) и веса из выходов ядер c5-in a/b/c и c12-in a/b/c.
Выход ядра — /kaggle/working/ens: oof_cnx_e512-in_f{0..4}.geojson и region_fold.json.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OWNER = "dmitrijkv"
CODE_DS = f"{OWNER}/koz3-ens-code"
PREP12 = f"{OWNER}/koz3-cnx-li-prep12"
KERNELS = [f"{OWNER}/koz3-li-cnx-in-{k}" for k in "abc"] + [f"{OWNER}/koz3-li-c12-in-{k}" for k in "abc"]
SLUG = "koz3-li-ens-oof"
PATHS = ["expds", "centerhead", "kaggle/2026-11-05/li_ens_in/ens_oof.py"]

SCRIPT = '''# OOF ансамбля лидара c5-in + c12-in прод-кодом (PREREG kaggle/2026-11-05/li_ens_in/PREREG.md), commit {sha}
import glob, json, os, subprocess, sys, tarfile, time
from pathlib import Path
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.6.0"], check=False)
code = Path("/kaggle/working/code")
tar = sorted(glob.glob("/kaggle/input/**/koz3_ens_code.tar", recursive=True))
if len(tar) == 1:
    with tarfile.open(tar[0]) as t:
        t.extractall(code)
else:
    # Kaggle распаковывает tar датасета сам: ищем каталог, где лежат expds и ens_oof.py
    hits = sorted(glob.glob("/kaggle/input/**/kaggle/2026-11-05/li_ens_in/ens_oof.py", recursive=True))
    assert len(hits) == 1, (tar, hits)
    src = next(up for up in Path(hits[0]).parents
               if (up / "expds").is_dir() and (up / "kaggle/2026-11-05/li_ens_in/ens_oof.py").is_file())
    code.symlink_to(src)
relief = None
for p in sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True)):
    try:
        m = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    if isinstance(m, dict) and len(m.get("tags") or []) == 12 and m.get("regions"):
        relief = Path(p).parent
assert relief is not None, "нет рельефа c12 (manifest с 12 тегами)"
print("рельеф c12:", relief, flush=True)
for v in ("c5-in", "c12-in"):
    w = sorted(glob.glob(f"/kaggle/input/**/weights_cnx_{{v}}_f*.pt", recursive=True))
    print(v, len(w), "весов", flush=True)
    assert len(w) == 5, w
cmd = [sys.executable, str(code / "kaggle/2026-11-05/li_ens_in/ens_oof.py"), str(relief), "/kaggle/working/ens",
       "--c5", "/kaggle/input", "--c12", "/kaggle/input", "--device", "cuda"]
r = subprocess.run(cmd, env={{**os.environ, "PYTHONPATH": str(code)}})
print("ИТОГ:", "OK" if r.returncode == 0 else f"код {{r.returncode}}", f"за {{time.time() - t0:.0f}} с", flush=True)
sys.exit(r.returncode)
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=Path("/workspace/c12/BUILD/koz3-li-c12-in-a/kernel-metadata.json"))
    a = ap.parse_args()
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    arch = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD", *PATHS], capture_output=True,
                          check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(arch)) as t:
        names = t.getnames()
    assert any(n.startswith("centerhead/") for n in names) and "kaggle/2026-11-05/li_ens_in/ens_oof.py" in names
    cd = a.out / "code"
    cd.mkdir(parents=True, exist_ok=True)
    (cd / "koz3_ens_code.tar").write_bytes(arch)
    (cd / "dataset-metadata.json").write_text(json.dumps({"title": "koz3-ens-code", "id": CODE_DS,
                                                          "licenses": [{"name": "other"}], "isPrivate": True},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    kd = a.out / SLUG
    kd.mkdir(parents=True, exist_ok=True)
    (kd / f"{SLUG}.py").write_text(SCRIPT.format(sha=sha), encoding="utf-8")
    meta = json.loads(a.template.read_text(encoding="utf-8"))
    meta.update(id=f"{OWNER}/{SLUG}", title=SLUG, code_file=f"{SLUG}.py", is_private=True, enable_gpu=True,
                enable_internet=True, dataset_sources=[CODE_DS, PREP12], kernel_sources=KERNELS)
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    compile((kd / f"{SLUG}.py").read_text(encoding="utf-8"), SLUG, "exec")
    print(json.dumps({"code_tar": str(cd / "koz3_ens_code.tar"), "files": len(names), "kernel": str(kd), "sha": sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
