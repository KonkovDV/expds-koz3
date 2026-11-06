"""Необязательные ускорения CPU-пути — харвест ветки speedup-2026-10-28 (6602bb2), 29.10.2026.

Флаги [speed] по-прежнему выключены в конфиге: так собран измеренный архив. Пути из PROVEN
уже сверены побайтно (круги — WKB, запись — JSON, каналы — массив, загрузка весов — state_dict)
и идут в обычном прогоне. Выключить один: EXPDS_SPEED_<ИМЯ>=0. Не входят в PROVEN и остаются
выключенным: slot_accum (на CPU медленнее, на CUDA не перемерен). read_ahead читает
следующее окно своим дескриптором GDAL и входит в PROVEN.

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

Не взято из старой ветки: las_grid_fast и rust/expds_fast, cpu_threads (нет потребителя в main), колесо .whl (бинарь в git).

Дополнительно, вне общего набора ускорений, только если включить явно:
  dem_disk_cache — дисковый кэш DEM (expds/demcache.py); выставляет
                   EXPDS_DEM_DISK_CACHE на каталог по умолчанию, если ещё не задан.
                   Либо задать EXPDS_DEM_DISK_CACHE=/path без флага (тоже включает кэш).
  rust_match     — префильтр пар гейтов через Rust (EXPDS_RUST_MATCH=1); hit побайтно тот же.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

UMBRELLA = ("meta_init", "no_pad_copy", "read_ahead", "batch_circles", "fast_writer", "fast_channels",
            "gdal_threads")
FLAGS = UMBRELLA + ("slot_accum", "dem_disk_cache", "rust_match")
# Сверены с медленным путём до бита. slot_accum сюда не входит.
PROVEN = ("meta_init", "no_pad_copy", "read_ahead", "batch_circles", "fast_writer", "fast_channels",
          "gdal_threads")

_cfg: dict = {}


def _bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip() in ("0", "1"):
        return v.strip() == "1"
    return None


def use_fast(name: str) -> bool:
    """Быстрый путь с тем же выходом. Включён, пока его явно не выключили.

    Явное выключение: EXPDS_SPEED_<ИМЯ>=0, EXPDS_SPEEDUP=0 или <имя> = false в секции [speed].
    [speed] enabled = false само по себе эти пути не выключает.
    """
    if name not in PROVEN:
        return on(name)
    own = _bool(os.environ.get(f"EXPDS_SPEED_{name.upper()}"))
    if own is not None:
        return own
    umbrella = _bool(os.environ.get("EXPDS_SPEEDUP"))
    if umbrella is not None:
        return umbrella
    if name in _cfg:
        return bool(_cfg[name])
    return True


def configure(section: dict | None) -> list[str]:
    """[speed] из конфига. Нет секции — флаги on() выкл. Пути PROVEN при этом всё равно быстрые.
    Возвращает список флагов, которые on() считает включёнными (для лога)."""
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
    if section is not None and use_fast("gdal_threads"):
        os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    if on("dem_disk_cache"):
        from . import demcache
        os.environ.setdefault("EXPDS_DEM_DISK_CACHE", str(demcache.default_dir()))
    if on("rust_match"):
        os.environ.setdefault("EXPDS_RUST_MATCH", "1")
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
