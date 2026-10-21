"""Сторож времени платформы и контрольные записи результата.

Контейнер снимают на 30-й минуте без файла. Сторож за три минуты до этого
прогоняет уже найденные объекты через ту же постобработку и завершает процесс
через os._exit: обычный sys.exit ждал бы фоновую триангуляцию DEM.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .constants import PLATFORM_LIMIT_S
from . import post, writer

log = logging.getLogger(__name__)

#: Старт процесса. main.py выставляет его сразу, до тяжёлых импортов.
PROCESS_START = time.monotonic()
#: Запас на запись файла до снятия контейнера.
GUARD_MARGIN_S = 180.0


@dataclass
class RunState:
    """Накопленные объекты прогона. Пишет и основной поток, и сторож."""

    output: Path
    cfg: dict
    feats: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, found: list[dict]) -> None:
        if not found:
            return
        with self.lock:
            self.feats.extend(found)

    def snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.feats)


def finish(feats: list[dict], cfg: dict) -> list[dict]:
    """Та же цепочка, что в конце pipeline.run."""
    out = post.only_known_classes(feats)
    out = post.apply_thresholds(out, cfg.get("thresholds", {}), cfg.get("default_threshold", 0.15),
                                cfg.get("thresholds_by_markup", {}))
    out = post.merge_within_markup(out)
    out = post.finalize(out)
    return post.cap_features(out)


_write_lock = threading.Lock()


def checkpoint(state: RunState, reason: str) -> int:
    """Атомарно записывает текущие объекты. Возвращает, сколько их осталось после порогов."""
    feats = state.snapshot()
    with _write_lock:
        kept = finish(feats, state.cfg)
        writer.write_geojson(kept, state.output)
    log.info("контрольная запись (%s): %d объектов", reason, len(kept))
    return len(kept)


def arm(state: RunState, delay_s: float, exit_fn=os._exit) -> threading.Thread:
    """Через delay_s записывает результат и завершает процесс."""

    def _fire() -> None:
        time.sleep(max(0.0, delay_s))
        log.warning("сторож времени: записываю результат и выхожу")
        try:
            checkpoint(state, "watchdog")
        finally:
            exit_fn(0)

    thread = threading.Thread(target=_fire, name="expds-watchdog", daemon=True)
    thread.start()
    return thread


def arm_platform(state: RunState, exit_fn=os._exit) -> threading.Thread:
    """Дедлайн — 30 минут платформы минус три минуты, от старта процесса."""
    left = PLATFORM_LIMIT_S - GUARD_MARGIN_S - (time.monotonic() - PROCESS_START)
    log.info("сторож сработает через %.0f с", max(0.0, left))
    return arm(state, left, exit_fn)
