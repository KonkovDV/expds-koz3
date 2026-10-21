"""После ансамбля: кэши текущим кодом (3 и 5 каналов) и две проверки вне обучения.

cur3 — те же каналы, что у hard2, но окна пересчитаны кодом инференса: вклад согласования.
opn5 — плюс положительная открытость и SVF.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
W = Path(r"F:\expds3\work")
S = Path(r"C:\Users\Пользователь\AppData\Local\Temp\expds_materials")
LOG = W / "channel_oofs.log"
SRC = W / "cache" / "li_tin_hard2_t128"
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], out: Path | None = None) -> int:
    fh = out.open("w", encoding="utf-8") if out else None
    try:
        return subprocess.run([PY, "-X", "utf8", *args], cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT).returncode
    finally:
        if fh:
            fh.close()


LOG.write_text("", encoding="utf-8")
variants = {"cur3": ["lrm_s", "lrm_l", "slope"], "opn5": ["lrm_s", "lrm_l", "slope", "opns", "svf"]}
for name, tags in variants.items():
    dst = W / "cache" / f"li_{name}_hard2_t128"
    rc = run([str(W / "recut_channels.py"), str(SRC), str(dst), *tags], W / f"recut_{name}.log")
    log(f"recut {name} rc={rc}")
while "ens2 done" not in (W / "ens2.log").read_text(encoding="utf-8", errors="replace"):
    time.sleep(60)
log("ens2 закончил, видеокарта свободна")
for name in variants:
    cache = W / "cache" / f"li_{name}_hard2_t128"
    pred = W / f"li_{name}_oof.geojson"
    rc = run(["scripts/nn_oof.py", "--markup", "li", "--root", r"F:\expds3\train-expds-3", "--cache", str(cache),
              "--work", str(W / f"li_{name}_oof"), "--out", str(pred), "--folds", "5", "--epochs", "30",
              "--batch", "32", "--config", str(ROOT / "configs" / "default.toml")], W / f"oof_{name}.log")
    log(f"oof {name} rc={rc}")
    if rc != 0 or not pred.exists():
        continue
    fused = W / "fusion" / f"li_{name}_fused.geojson"
    run([str(S / "fusion_apply.py"), r"F:\expds3\train-expds-3", str(pred), str(fused)])
    for part, regions in (("las15", W / "regions_las15.txt"), ("new", W / "regions_las_new.txt"),
                          ("las49", W / "regions_las49.txt")):
        r = subprocess.run([PY, "-X", "utf8", str(S / "sweep_many.py"), str(W / "gt_train.geojson"), str(regions), "1",
                            str(fused)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log(f"sweep {name} {part} {r.stdout.strip()}")
log("channel oofs done")
