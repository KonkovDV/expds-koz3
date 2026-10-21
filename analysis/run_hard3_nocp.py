"""Тот же кэш с новыми LAS Гнёздова, но без copy-paste: что из двух изменений уронило балл."""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
W = Path(r"F:\expds3\work")
S = Path(r"C:\Users\Пользователь\AppData\Local\Temp\expds_materials")
LOG = W / "hard3_nocp.log"
CACHE = W / "cache" / "li_tin_hard3_t128"
PRED = W / "li_hard3_nocp_oof.geojson"
FUSED = W / "fusion" / "li_hard3_nocp_fused.geojson"


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


env = {**os.environ, "PYTHONUNBUFFERED": "1"}
LOG.write_text("", encoding="utf-8")
with (W / "li_hard3_nocp_train.log").open("w", encoding="utf-8") as out:
    rc = subprocess.run([PY, "-X", "utf8", "scripts/nn_oof.py", "--markup", "li", "--root", r"F:\expds3\train-expds-3",
                         "--cache", str(CACHE), "--work", str(W / "li_hard3_nocp_oof"), "--out", str(PRED),
                         "--folds", "5", "--epochs", "30", "--batch", "32",
                         "--config", str(ROOT / "configs" / "default.toml")],
                        cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
log(f"oof rc={rc}")
if rc != 0 or not PRED.exists():
    raise SystemExit(rc or 1)
rc = subprocess.run([PY, "-X", "utf8", str(S / "fusion_apply.py"), r"F:\expds3\train-expds-3", str(PRED), str(FUSED)],
                    cwd=ROOT, env=env).returncode
log(f"fusion rc={rc}")
for name, regions in (("las15", W / "regions_las15.txt"), ("new", W / "regions_las_new.txt"),
                      ("las49", W / "regions_las49.txt")):
    r = subprocess.run([PY, "-X", "utf8", str(S / "sweep_many.py"), str(W / "gt_train.geojson"), str(regions), "1",
                        str(FUSED)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log(f"sweep {name} {r.stdout.strip()}")
log("nocp done")
