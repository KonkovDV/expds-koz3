"""Тот же стенд без отсечки до полигонизации: выход должен совпасть с обычным прогоном."""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(ROOT))
from expds import pipeline  # noqa: E402

pipeline.output_floors = lambda cfg, markup: None
bench, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pipeline.run(bench, out, ROOT / "configs" / "default.toml", ROOT)


def canon(path: Path) -> list:
    fc = json.loads(path.read_text(encoding="utf-8"))
    return sorted((f["properties"]["class_name"], round(f["properties"]["confidence"], 9),
                   json.dumps(f["geometry"], sort_keys=True)) for f in fc["features"])


print("без отсечки == с отсечкой:", canon(ref) == canon(out), len(canon(out)))
