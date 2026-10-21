"""Сквозной прогон pipeline.run на нескольких участках с LAS: с фоновым DEM и без."""
import logging
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(ROOT))
SRC = Path(r"F:\expds3\train-expds-3")
BENCH = Path(r"F:\expds3\work\bench_in")
NAMES = ["021_Новенькое", "101_Гнёздово_Днепровская", "179_Гнёздово_Лесная_1", "036_Тимерево"]

BENCH.mkdir(parents=True, exist_ok=True)
for n in NAMES:
    link = BENCH / n
    if not link.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(SRC / n)], check=True,
                       stdout=subprocess.DEVNULL)

base = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8")
cfgs = {}
for depth in (0, 1, 2, 3):
    text = base.replace("prefetch = 2", f"prefetch = {depth}")
    assert tomllib.loads(text)["lidar"]["prefetch"] == depth
    p = Path(r"F:\expds3\work") / f"bench_p{depth}.toml"
    p.write_text(text, encoding="utf-8")
    cfgs[depth] = p

logging.basicConfig(level=logging.WARNING)
from expds import pipeline  # noqa: E402

runs = [("warm", cfgs[0])] + [(f"p{d}", cfgs[d]) for d in (0, 1, 2, 3, 0, 2)]
for label, cfg in runs:
    t = time.perf_counter()
    s = pipeline.run(BENCH, Path(r"F:\expds3\work") / f"bench_{label}.geojson", cfg, ROOT)
    print(f"{label:3s} {time.perf_counter() - t:6.1f}s written {s['written']} raw {s['raw']}", flush=True)
