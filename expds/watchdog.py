"""Сторож времени платформы и контрольные записи результата.

Контейнер снимают на 30-й минуте. Файл результата пишется один раз в конце
и ещё раз сторожем за три минуты до снятия. Промежуточная запись в тот же
путь опасна: площадка может счесть первый файл готовым ответом и остановить
контейнер.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .constants import PLATFORM_LIMIT_S
from . import post, writer

log = logging.getLogger(__name__)


def _fmt_counts(feats: list[dict], key: str) -> str:
    """«li 12, sp_or 3» по полю объекта. Пустой список — «пусто»."""
    counts: dict[str, int] = {}
    for feat in feats:
        name = feat.get(key) or "нет"
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "пусто"
    return ", ".join(f"{name} {counts[name]}" for name in sorted(counts))

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
    note: dict = field(default_factory=dict)

    def add(self, found: list[dict]) -> None:
        if not found:
            return
        with self.lock:
            self.feats.extend(found)

    def snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.feats)

    def replace(self, feats: list[dict]) -> None:
        with self.lock:
            self.feats = list(feats)


def finish(feats: list[dict], cfg: dict) -> list[dict]:
    """Та же цепочка, что в конце pipeline.run."""
    out = post.only_known_classes(feats)
    out = post.apply_thresholds(out, cfg.get("thresholds", {}), cfg.get("default_threshold", 0.15),
                                cfg.get("thresholds_by_markup", {}))
    out = post.merge_within_markup(out)
    post_cfg = cfg.get("post") or {}
    if post_cfg.get("multipole_object_poly"):
        out = post.multipole_object_poly(out, post_cfg.get("multipole_object_poly") if isinstance(post_cfg.get("multipole_object_poly"), dict) else None)
    if post_cfg.get("filter_finds_points"):
        out = post.filter_finds_points_regimes(out, post_cfg.get("filter_finds_points") if isinstance(post_cfg.get("filter_finds_points"), dict) else None)
    if post_cfg.get("kurgan_twins"):
        out = post.kurgan_cross_modal_twins(out, post_cfg.get("kurgan_twins") if isinstance(post_cfg.get("kurgan_twins"), dict) else None)
    if post_cfg.get("micro_circles"):
        out = post.micro_circles(out, post_cfg.get("micro_circles") if isinstance(post_cfg.get("micro_circles"), dict) else None)
    if post_cfg.get("fill_holes"):
        out = post.fill_holes(out, post_cfg["fill_holes"])
    if post_cfg.get("core"):
        out = post.shape_cores(out, post_cfg["core"])
    if post_cfg.get("split"):
        out = post.split_large(out, post_cfg["split"])
    if post_cfg.get("ellipse_classes") or post_cfg.get("fort_medial"):
        from . import open_ideas
        if post_cfg.get("ellipse_classes"):
            names = set(post_cfg["ellipse_classes"])
            out = [dict(f, geom=open_ideas.ellipse_of(f["geom"])) if f.get("class_name") in names else f
                   for f in out]
        if post_cfg.get("fort_medial"):
            out = open_ideas.medial_fort_circles(out, float(post_cfg.get("fort_medial_m", 2.0)),
                                                 float(post_cfg.get("fort_medial_spacing_m", 25.0)))
    out = post.finalize(out)
    out = post.cap_features(out)
    # Перепись одного класса. Без ключа список тот же, что у полной сдачи.
    only = post_cfg.get("only_classes")
    if only is not None:
        out = post.keep_classes(out, only)
    return out


def safe_finish(feats: list[dict], cfg: dict) -> list[dict]:
    """finish(); если постобработка упала — урезанная цепочка, чтобы ответ не остался пустым (bot-8)."""
    try:
        return finish(feats, cfg)
    except Exception:
        log.exception("постобработка упала, пишу ответ без склейки и доп. шагов")
    out = post.only_known_classes(feats)
    reranked: list[dict] = []
    rr = cfg.get("rerank_arch") or {}
    if rr.get("enabled"):
        # bot-7 (bot7-candidate): реранк архитектуры со своими порогами — и в запасной цепочке
        try:
            from . import rerank_arch

            out, reranked = rerank_arch.apply(out, rr)
        except Exception:
            log.exception("rerank_arch в запасной цепочке упал, архитектура — по базовым порогам")
            reranked = []
    out = post.apply_thresholds(out, cfg.get("thresholds", {}), cfg.get("default_threshold", 0.15),
                                cfg.get("thresholds_by_markup", {}))
    out = out + reranked
    mc = (cfg.get("post") or {}).get("micro_circles")
    if mc:
        # bot-7: форма под метрику (микрокруги) — и в запасной цепочке, иначе пороги/геометрия разойдутся
        try:
            out = post.micro_circles(out, mc if isinstance(mc, dict) else None)
        except Exception:
            log.exception("micro_circles в запасной цепочке упал, круги оставлены как есть")
    kept: list[dict] = []
    for f in out:
        try:
            kept += post.finalize([f])
        except Exception:
            continue
    kept = post.cap_features(kept)
    only = (cfg.get("post") or {}).get("only_classes")
    return post.keep_classes(kept, only) if only is not None else kept


_write_lock = threading.Lock()


def _emit_protocol(state: RunState, raw: list[dict], kept: list[dict], reason: str) -> None:
    """Одна запись протокола. Сырые объекты — до порога; записанные — строки файла.

    Площадка показывает stdout. Блок самодостаточен: по нему видно, увидел ли
    код LAS и снимки и отработал ли порог, без списка участков.
    """
    note = state.note
    phases = note.get("phases") or {}
    phase_text = ", ".join(
        f"{name}: сырых {info.get('raw', 0)}, {info.get('seconds', 0):.0f} с"
        for name, info in phases.items()) or "нет"
    size = state.output.stat().st_size if state.output.is_file() else 0
    log.info("===== ПРОТОКОЛ ПРОГОНА (%s) =====", reason)
    log.info("определения: сырые — до порога, склейки и отсечения; записано — строки файла")
    log.info("устройство: %s; torch %s; cuda %s; ускоритель %s",
             note.get("device", "не задано"), note.get("torch", "нет"),
             note.get("cuda", "нет"), note.get("backend", "не задано"))
    log.info("конфиг SHA256 %s; бюджет %s с; запас %s с",
             note.get("config_sha256", "нет"), note.get("budget_s", "нет"),
             note.get("reserve_s", "нет"))
    log.info("вход: регионов %s, из них с LAS %s; файлов LAS %s; растров %s; пропущено файлов %s",
             note.get("regions", "нет"), note.get("regions_with_las", "нет"),
             note.get("las_files", "нет"), note.get("rasters", "нет"),
             note.get("skipped", "нет"))
    log.info("фазы: %s", phase_text)
    log.info("до порога %d, по слоям: %s", len(raw), _fmt_counts(raw, "markup_type"))
    log.info("до порога по классам: %s", _fmt_counts(raw, "class_name"))
    log.info("записано %d, файл %d байт", len(kept), size)
    log.info("после порога по классам: %s", _fmt_counts(kept, "class_name"))
    log.info("после порога по слоям: %s", _fmt_counts(kept, "markup_type"))
    log.info("===== КОНЕЦ ПРОТОКОЛА =====")
    for handler in logging.getLogger().handlers:
        handler.flush()


def checkpoint(state: RunState, reason: str) -> int:
    """Атомарно записывает текущие объекты и печатает протокол. Возвращает, сколько их осталось после порогов."""
    feats = state.snapshot()
    with _write_lock:
        kept = safe_finish(feats, state.cfg)
        writer.write_geojson(kept, state.output)
        _emit_protocol(state, feats, kept, reason)
        return len(kept)


def cleanup_tmp() -> None:
    """Удаляет .expds_tmp, если main.py завёл туда кэши. Файл результата не трогает."""
    import shutil

    tmp = os.environ.get("EXPDS_TMP")
    if tmp and Path(tmp).name == ".expds_tmp":
        shutil.rmtree(tmp, ignore_errors=True)


def arm(state: RunState, delay_s: float, exit_fn=os._exit) -> threading.Thread:
    """Через delay_s записывает результат и завершает процесс."""

    def _fire() -> None:
        time.sleep(max(0.0, delay_s))
        log.warning("сторож времени: записываю результат и выхожу")
        try:
            checkpoint(state, "watchdog")
        finally:
            for handler in logging.getLogger().handlers:
                handler.flush()
            cleanup_tmp()
            exit_fn(0)

    thread = threading.Thread(target=_fire, name="expds-watchdog", daemon=True)
    thread.start()
    return thread


def arm_signal(state: RunState, exit_fn=os._exit):
    """По SIGTERM записывает результат и выходит. Если запись уже идёт, не перебивает её."""

    def _on_term(signum, _frame) -> None:
        log.warning("сигнал %s: записываю результат", signum)
        if not _write_lock.acquire(blocking=False):
            return
        try:
            feats = state.snapshot()
            kept = finish(feats, state.cfg)
            writer.write_geojson(kept, state.output)
            _emit_protocol(state, feats, kept, "signal")
        finally:
            _write_lock.release()
            for handler in logging.getLogger().handlers:
                handler.flush()
            cleanup_tmp()
            exit_fn(0)

    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _on_term)
    return previous


def arm_platform(state: RunState, exit_fn=os._exit, margin_s: float | None = None) -> threading.Thread:
    """Дедлайн — 30 минут платформы минус запас, от старта процесса."""
    margin = GUARD_MARGIN_S if margin_s is None else float(margin_s)
    left = PLATFORM_LIMIT_S - margin - (time.monotonic() - PROCESS_START)
    log.info("сторож сработает через %.0f с", max(0.0, left))
    return arm(state, left, exit_fn)
