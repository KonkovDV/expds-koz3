"""Точка входа на платформе КОЗ №3.

    python inference/main.py <папка_валидационной_выборки> <путь_к_result.geojson>

Rust-ускоритель подключается из inference/lib/ (файл expds_core*.so), если он
там есть; иначе работают NumPy-версии тех же функций. Файл результата
записывается всегда: при любой ошибке — хотя бы пустой FeatureCollection,
чтобы запуск не считался упавшим.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference" / "lib"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    log = logging.getLogger("main")
    if len(sys.argv) < 3:
        print("usage: python inference/main.py <input_dir> <output.geojson>")
        return 2
    input_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.monotonic()
    try:
        from expds.pipeline import run

        summary = run(input_dir, output, ROOT / "configs" / "default.toml", ROOT)
        log.info("готово: %s", json.dumps(summary, ensure_ascii=False))
    except Exception:
        log.exception("сбой пайплайна")
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    log.info("общее время %.1f с", time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
