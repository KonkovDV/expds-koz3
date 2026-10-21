"""cProfile сквозного прогона на смеси модальностей, как в публичной валидации."""
import cProfile
import pstats
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(ROOT))
from expds import pipeline  # noqa: E402
from expds.catalog import scan_root  # noqa: E402

SRC = Path(r"F:\expds3\train-expds-3")
BENCH = Path(r"F:\expds3\work\bench_mix")
regions = scan_root(SRC)
pick = []
for mt in ("sp_or", "ae", "maggr", "etgr", "terfl"):
    have = [r for r in regions if r.rasters.get(mt) and not r.las]
    have.sort(key=lambda r: -sum(p.stat().st_size for p in r.rasters[mt]))
    pick += [r.name for r in have[:2] if r.name not in pick]
BENCH.mkdir(parents=True, exist_ok=True)
for n in pick:
    link = BENCH / n
    if not link.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(SRC / n)], check=True, stdout=subprocess.DEVNULL)
print("участки:", pick, flush=True)
prof = cProfile.Profile()
t = time.perf_counter()
prof.enable()
s = pipeline.run(BENCH, Path(r"F:\expds3\work\bench_mix.geojson"), ROOT / "configs" / "default.toml", ROOT)
prof.disable()
print(f"всего {time.perf_counter() - t:.1f} с, записано {s['written']}, сырых {s['raw']}", flush=True)
st = pstats.Stats(prof)
st.sort_stats("cumulative").print_stats(35)
st.sort_stats("tottime").print_stats(25)
