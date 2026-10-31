"""Ансамбль из двух моделей на фолд на кэше hard2: та же разбивка, вторая модель с другим зерном."""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
W = Path(r"F:\expds3\work")
S = Path(r"C:\Users\Пользователь\AppData\Local\Temp\expds_materials")
LOG = W / "ens2.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


LOG.write_text("", encoding="utf-8")
pred = W / "li_ens2_oof.geojson"
env = {**os.environ, "PYTHONUNBUFFERED": "1"}
rc = subprocess.run([PY, "-X", "utf8", "scripts/nn_oof.py", "--markup", "li", "--root", r"F:\expds3\train-expds-3",
                     "--cache", str(W / "cache" / "li_tin_hard2_t128"), "--work", str(W / "li_hard2_oof"),
                     "--out", str(pred), "--folds", "5", "--epochs", "30", "--batch", "32", "--seeds", "2",
                     "--resume", "--config", str(ROOT / "configs" / "default.toml")],
                    cwd=ROOT, env=env).returncode
log(f"oof rc={rc}")
if rc != 0 or not pred.exists():
    raise SystemExit(rc or 1)
fused = W / "fusion" / "li_ens2_fused.geojson"
rc = subprocess.run([PY, "-X", "utf8", str(S / "fusion_apply.py"), r"F:\expds3\train-expds-3", str(pred), str(fused)],
                    cwd=ROOT, env=env).returncode
log(f"fusion rc={rc}")
for name, regions in (("las15", W / "regions_las15.txt"), ("new", W / "regions_las_new.txt"),
                      ("las49", W / "regions_las49.txt")):
    r = subprocess.run([PY, "-X", "utf8", str(S / "sweep_many.py"), str(W / "gt_train.geojson"), str(regions), "1",
                        str(fused)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log(f"sweep {name} {r.stdout.strip()}")
log("ens2 done")
