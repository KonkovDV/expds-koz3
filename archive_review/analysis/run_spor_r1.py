"""Спутник на шаге 1 м: тайл 128 м, курган ~36 м виден вместе с полем вокруг.

На шаге 0,5 м тайл был 64 м и насыпь занимала почти всё окно: пиксельный F1
вне обучения оставался 0. Пятна радиуса 8–28 м при этом попадают в курганы,
значит контраст есть на масштабе больше тайла.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
W = Path(r"F:\expds3\work")
LOG = W / "spor_r1.log"
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], out: Path) -> int:
    with out.open("w", encoding="utf-8") as fh:
        return subprocess.run([PY, "-X", "utf8", *args], cwd=ROOT, env=ENV,
                              stdout=fh, stderr=subprocess.STDOUT).returncode


LOG.write_text("", encoding="utf-8")
cache = W / "cache" / "spor_r1_t128_s25"
rc = run(["-m", "expds.tiles", "--root", r"F:/expds3/train-expds-3",
          "--gt", str(W / "gt_train.geojson"), "--markup", "sp_or", "--out", str(cache),
          "--res", "1.0", "--tile", "128", "--shrink", "0.25",
          "--classes", "kurgany_povrezhdennye", "kurgany_tselye"],
         W / "spor_r1_cache.log")
log(f"cache rc={rc}")
if rc != 0:
    raise SystemExit(rc)
rc = run(["scripts/nn_oof.py", "--markup", "sp_or", "--root", r"F:\expds3\train-expds-3",
          "--cache", str(cache), "--work", str(W / "spor_r1_oof"),
          "--out", str(W / "spor_r1_oof.geojson"), "--folds", "5", "--epochs", "30",
          "--batch", "32", "--config", str(ROOT / "configs" / "default.toml")],
         W / "spor_r1_oof.log")
log(f"oof rc={rc}")
