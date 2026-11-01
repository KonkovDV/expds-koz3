"""Необязательные ускорения CPU-пути — харвест ветки speedup-2026-10-28 (6602bb2), 29.10.2026.

Все флаги по умолчанию ВЫКЛЮЧЕНЫ: путь сдачи остаётся тем же, что в v2. Каждое ускорение задумано как
побайтно то же (тесты: tests/test_speed_flags.py, test_circles_batch.py, test_writer_fast.py,
test_meta_init.py, test_predict_slots.py), но на реальных данных и на 4090 не перемерено — поэтому выкл.

Включение (приоритет сверху вниз):
  * EXPDS_SPEED_<ИМЯ>=1|0       — один флаг (например EXPDS_SPEED_FAST_WRITER=1);
  * EXPDS_SPEEDUP=1|0           — все флаги из UMBRELLA разом;
  * configs/*.toml: [speed] enabled = true (все из UMBRELLA) и/или <имя> = true|false (один флаг).

Флаги:
  meta_init      — Segmenter строит сеть на устройстве meta и подставляет веса (assign) — без случайной
                   инициализации (~0,2 с на модель);
  no_pad_copy    — Segmenter.predict не копирует снимок через np.pad, если паддинг не нужен;
  read_ahead     — _raster_with_model читает следующее окно растра в фоновом потоке;
  batch_circles  — круги точек одним вызовом GEOS/PROJ (raster.circles_3857) в centers и geophys;
  fast_writer    — writer.to_feature_collection берёт координаты всех полигонов одним вызовом GEOS;
  fast_channels  — tiles.fix_channels без промежуточного repeat + concatenate;
  gdal_threads   — GDAL_NUM_THREADS=ALL_CPUS (декодирование блоков GeoTIFF во все потоки), если не задан;
  slot_accum     — накопление окон на CUDA «слотами» (Segmenter._predict_on_device_slots). НЕ входит в
                   UMBRELLA: на CPU медленнее цикла, на CUDA не замерено — только явно.

Не взято из ветки (см. docs/BRANCHES_ARCHIVE.md): las_grid_fast + rust/expds_fast (написаны под lidar_a —
линия EN), cpu_threads (нет потребителя в main), колесо .whl (бинарь в git).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

UMBRELLA = ("meta_init", "no_pad_copy", "read_ahead", "batch_circles", "fast_writer", "fast_channels",
            "gdal_threads")
FLAGS = UMBRELLA + ("slot_accum",)

_cfg: dict = {}


def _bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip() in ("0", "1"):
        return v.strip() == "1"
    return None


def configure(section: dict | None) -> list[str]:
    """[speed] из конфига. Нет секции — всё выкл. Возвращает список включённых флагов (для лога)."""
    global _cfg
    _cfg = {}
    for k, v in (section or {}).items():
        if k != "enabled" and k not in FLAGS:
            log.warning("[speed] %s: неизвестный флаг — пропущен", k)
            continue
        b = _bool(v)
        if b is None:
            log.warning("[speed] %s = %r: нужен true/false — пропущен (выкл.)", k, v)
            continue
        _cfg[k] = b
    if on("gdal_threads"):
        os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    return active()


def on(name: str) -> bool:
    """Включён ли флаг name. По умолчанию — нет."""
    if name not in FLAGS:
        raise KeyError(name)
    b = _bool(os.environ.get(f"EXPDS_SPEED_{name.upper()}"))
    if b is not None:
        return b
    if name in UMBRELLA:
        b = _bool(os.environ.get("EXPDS_SPEEDUP"))
        if b is not None:
            return b
    if name in _cfg:
        return _cfg[name]
    return bool(_cfg.get("enabled", False)) and name in UMBRELLA


def active() -> list[str]:
    return [f for f in FLAGS if on(f)]
