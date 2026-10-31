"""Сквозной прогон стенда и сверка с прежним выходом: те же объекты, то же время?"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(ROOT))
from expds import pipeline  # noqa: E402

bench, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
t = time.perf_counter()
s = pipeline.run(bench, out, ROOT / "configs" / "default.toml", ROOT)
dt = time.perf_counter() - t


def canon(path: Path) -> list:
    fc = json.loads(path.read_text(encoding="utf-8"))
    return sorted((f["properties"]["class_name"], round(f["properties"]["confidence"], 9),
                   json.dumps(f["geometry"], sort_keys=True)) for f in fc["features"])


a, b = canon(ref), canon(out)
print(f"{bench.name}: {dt:.1f} с, записано {s['written']} (было {len(a)}), совпадает: {a == b}")
