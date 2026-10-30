"""Точка входа на платформе КОЗ №3.

    python inference/main.py <папка_валидационной_выборки> <путь_к_result.geojson>

Второй аргумент по ТР (Прил. 2) — путь к файлу GeoJSON; примеры организаторов
(simple_solution, бейзлайн) передают папку. Если это папка или путь без
.geojson, результат пишется в <путь>/result.geojson.

Rust-ускорители подключаются из inference/lib/ (файлы expds_core*.so и
expds_fast*.so — сетка LAS 0,25 м, expds/las_grid_fast.py), если они там есть; иначе работают NumPy-версии тех же функций. Файл результата
записывается всегда: при любой ошибке — хотя бы пустой FeatureCollection,
чтобы запуск не считался упавшим.
"""
from __future__ import annotations

import faulthandler
import os

faulthandler.enable()
os.environ["GDAL_PAM_ENABLED"] = "NO"
os.environ["OMP_NUM_THREADS"] = "8"
# Декодирование блоков GeoTIFF (zstd/deflate) во все потоки; пиксели те же (проверено sha).
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ.setdefault("USER", "solution")
os.environ.setdefault("LOGNAME", "solution")

import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference" / "lib"))
sys.path.insert(0, str(ROOT))
import expds.watchdog  # старт отсчёта 30 минут до импорта моделей


def _paths(argv: list[str]) -> tuple[Path, Path] | None:
    """Аргументы, иначе EXPDS_INPUT/EXPDS_OUTPUT, иначе /data на платформе."""
    if len(argv) >= 3:
        input_dir, output = Path(argv[1]), Path(argv[2])
    elif os.environ.get("EXPDS_INPUT") and os.environ.get("EXPDS_OUTPUT"):
        input_dir, output = Path(os.environ["EXPDS_INPUT"]), Path(os.environ["EXPDS_OUTPUT"])
    elif Path("/data/input").is_dir():
        input_dir, output = Path("/data/input"), Path("/data/output/result.geojson")
    else:
        return None
    if output.is_dir() or output.suffix.lower() != ".geojson":
        output = output / "result.geojson"
    return input_dir, output


def _caches(output: Path) -> None:
    """Кэши библиотек — в папку результата, не в домашний каталог. GeoJSON туда не пишется."""
    tmp = output.parent / ".expds_tmp"
    try:
        tmp.mkdir(parents=True, exist_ok=True)
    except OSError:
        tmp = Path(tempfile.gettempdir())
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(tmp)
    tempfile.tempdir = None
    for key, sub in (("XDG_CACHE_HOME", "cache"), ("TORCH_HOME", "torch"), ("MPLCONFIGDIR", "mpl"),
                     ("NUMBA_CACHE_DIR", "numba"), ("CUDA_CACHE_PATH", "nv")):
        os.environ.setdefault(key, str(tmp / sub))
    home = os.environ.get("HOME")
    if not home or not os.access(home, os.W_OK):
        os.environ["HOME"] = str(tmp)
    os.environ["EXPDS_TMP"] = str(tmp)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    log = logging.getLogger("main")
    found = _paths(sys.argv)
    if found is None:
        print("usage: python inference/main.py <input_dir> <output.geojson>")
        return 2
    input_dir, output = found
    t0 = time.monotonic()
    _caches(output)
    try:
        from expds.pipeline import run

        summary = run(input_dir, output, ROOT / "configs" / "default.toml", ROOT)
        log.info("готово: %s", json.dumps(summary, ensure_ascii=False))
    except Exception:
        log.exception("сбой пайплайна")
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    tmp = os.environ.get("EXPDS_TMP")
    if tmp and Path(tmp).name == ".expds_tmp":
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("общее время %.1f с", time.monotonic() - t0)
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
