"""После проверок каналов: заново OOF модели спутника 0,5 м (курганы, архитектуры, фортификации)."""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
W = Path(r"F:\expds3\work")
LOG = W / "spor_all.log"


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


LOG.write_text("", encoding="utf-8")
while "channel oofs done" not in (W / "channel_oofs.log").read_text(encoding="utf-8", errors="replace"):
    time.sleep(60)
log("видеокарта свободна")
with (W / "oof_spor_all.log").open("w", encoding="utf-8") as fh:
    rc = subprocess.run([PY, "-X", "utf8", "scripts/nn_oof.py", "--markup", "sp_or", "--root", r"F:\expds3\train-expds-3",
                         "--cache", str(W / "cache" / "spor_r05_t128_s04"), "--work", str(W / "spor_all_oof"),
                         "--out", str(W / "spor_all_oof.geojson"), "--folds", "5", "--epochs", "30", "--batch", "32",
                         "--config", str(ROOT / "configs" / "default.toml")],
                        cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
log(f"oof sp_or rc={rc}")
log("spor done")
