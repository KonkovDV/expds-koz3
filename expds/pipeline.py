"""Сквозной инференс: папка валидационной выборки → result.geojson.

Порядок по всему набору, а не внутри участка. Иначе облака LAS съедают
30 минут, и геофизика (вес 20) с оптикой не запускаются:

1. геофизика по всем участкам — растры маленькие;
2. лидар: LAS → DEM → модель, иначе CPU-кандидаты; участки с LAS первыми;
3. оптика (ae, sp_or, or) на оставшемся времени, только при наличии модели.

Файл результата записывается в конце и сторожем времени.
"""
from __future__ import annotations

import hashlib
import logging
import os
import signal
import sys
import time
import tomllib
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import accel, candidates, fusion, geophys, lidar_cnx, post, writer
from .catalog import Region, read_utm_json, scan_root
from .constants import DEFAULT_BUDGET_S

log = logging.getLogger(__name__)


class PixgbmModuleMissing(RuntimeError):
    """bot-6: [geophys.pixgbm] enabled=true, а expds/geo_pixgbm.py нет и EXPDS_PIXGBM_STRICT=1."""


class TimeGuard:
    """Учёт времени: сколько прошло и хватает ли запаса на следующий шаг.

    bot-8: дедлайн внутри цикла окон снимков (window_ok). Окно не начинается, если до конца
    бюджета [time] budget_s или до сторожа (watchdog.deadline_left: 1800 − guard_margin_s = 1620 с
    от старта процесса) осталось не больше window_reserve_s. None — проверка выключена.
    clock — для тестов (поддельные часы); по умолчанию time.monotonic.
    """

    def __init__(self, budget_s: float, window_reserve_s: float | None = None, hard_left=None, clock=None):
        self.clock = clock or time.monotonic
        #: bot-8: старт run() — от него, как и раньше, считается бюджет [time] budget_s (left/ok).
        self.run_t0 = self.clock()
        #: bot-8: самый ранний известный старт в базе self.clock (по умолчанию time.monotonic, НЕ time.time).
        #: До set_soft() — run_t0; в run() set_soft(watchdog.PROCESS_START) даёт тот же отсчёт, что у сторожа
        #: 1620 с. Абсолютные дедлайны (PIXGBM: t0 + PIXGBM_BUDGET_S) сравнивать с guard.clock() / elapsed().
        self.t0 = self.run_t0
        self.budget = budget_s
        self.window_reserve_s = window_reserve_s
        self.hard_left = hard_left
        self.partial: list[dict] = []
        #: bot-8: мягкий потолок (с от t0). None — выключен (все прямые вызовы вне run()).
        self.soft_ceiling: float | None = None
        self.soft_need = dict(SOFT_NEED_DEFAULT)
        self.soft_skipped: list[dict] = []

    def set_soft(self, process_start: float | None = None, soft_ceiling_s: float | None = None,
                 soft_need: dict | None = None) -> "TimeGuard":
        """bot-8: мягкий потолок и ранний старт. process_start — в базе time.monotonic; учитывается только
        при часах по умолчанию (поддельные часы тестов живут в своей базе): t0 = min(process_start, run_t0)."""
        if process_start is not None and self.clock is time.monotonic:
            self.t0 = min(float(process_start), self.run_t0)
        self.soft_ceiling = None if soft_ceiling_s is None else float(soft_ceiling_s)
        self.soft_need = dict(SOFT_NEED_DEFAULT, **(soft_need or {}))
        return self

    def elapsed(self) -> float:
        """Секунд от t0 (старт процесса в run(), иначе создание guard)."""
        return self.clock() - self.t0

    def run_elapsed(self) -> float:
        """Секунд от старта run() — основа бюджета budget_s (поведение 1500 с не меняется)."""
        return self.clock() - self.run_t0

    def left(self) -> float:
        return self.budget - self.run_elapsed()

    def soft_left(self) -> float:
        """SOFT_CEILING − elapsed(). Потолок выключен — +inf."""
        if self.soft_ceiling is None:
            return float("inf")
        return self.soft_ceiling - self.elapsed()

    def soft_ok(self, need_s: float = 0.0, phase: str | None = None, what: str = "") -> bool:
        """Хватает ли до мягкого потолка need_s секунд. phase — для журнала: при отказе одна строка
        «мягкий потолок: <phase> …» и запись в soft_skipped (в протокол прогона)."""
        sl = self.soft_left()
        if sl > need_s:
            return True
        if phase is not None:
            self.soft_skipped.append({"phase": phase, "what": what, "soft_left_s": round(sl, 1),
                                      "need_s": need_s, "elapsed_s": round(self.elapsed(), 1)})
            log.warning("мягкий потолок: %s пропущено%s — soft_left %.0f с ≤ %.0f с (потолок %.0f с, прошло %.0f с)",
                        phase, f" ({what})" if what else "", sl, need_s, self.soft_ceiling, self.elapsed())
        return False

    def soft_need_s(self, phase: str) -> float:
        return float(self.soft_need.get(phase, 0.0))

    def ok(self, need_s: float = 0.0) -> bool:
        return self.left() > need_s

    def window_left(self) -> float:
        """Секунд до ближайшего из: конец бюджета, срабатывание сторожа."""
        left = self.left()
        if self.hard_left is not None:
            left = min(left, float(self.hard_left()))
        return left

    def window_ok(self) -> bool:
        return self.window_reserve_s is None or self.window_left() > float(self.window_reserve_s)


#: [time] window_reserve_s по умолчанию (bot-8): запас до конца бюджета, при котором окно снимка уже не начинается.
DEFAULT_WINDOW_RESERVE_S = 30.0

#: bot-8: мягкий потолок прогона, с от старта процесса (BUDGET.md §7.13/§7.14). Рабочий потолок пакета,
#: чтобы пережить частный набор до 2× публичного; лимит площадки 1800, TimeGuard 1500 и сторож 1620 — без изменений.
#: EXPDS_SOFT_CEILING_S (env) > [time] soft_ceiling_s > 1200. Неверное значение или ≤ 0 — потолок выключен.
SOFT_CEILING_DEFAULT_S = 1200.0
#: bot-8: сколько секунд до мягкого потолка нужно, чтобы НАЧАТЬ необязательный шаг (порог soft_left).
#: Оценки по замерам GTX 1650 (BUDGET.md §1, docs/PLAN.md:147): σ 57 с на весь mock_val, самый тяжёлый
#: снимок 106 с; кроп — малая сеть (тайл 96) по кандидатам sp_focal/sp_focal2, отдельно не мерен.
#: Переопределение: [time.soft] <ключ> = с или env EXPDS_SOFT_NEED_<КЛЮЧ>_S.
#:   wide_window — окно 256 м (sp_focal2) на участке без LAS, на каждый SpOr-растр: 120 (> 106 с снимка);
#:   sigma       — канал σ (li_dev), на каждый участок с LAS: 60 (σ всего 57 с) + резерв кропа, если он включён;
#:   verify      — проверка кропа курганов спутника после всех фаз: 60.
#: Порядок отключения при росте времени: окно 256 м (в оптике) → σ (раньше кропа: держит его резерв) → кроп.
#: Обязательное (геофизика, лидар, основные сети оптики, finish/safe_finish, запись GeoJSON) потолок не трогает.
SOFT_NEED_DEFAULT = {"wide_window": 120.0, "sigma": 60.0, "verify": 60.0}


def _warn(msg: str) -> None:
    sys.stderr.write(f"expds: {msg}\n")
    log.warning("%s", msg)


def soft_ceiling_from(cfg: dict, environ=None) -> float | None:
    """Мягкий потолок в секундах или None (выключен). Неверное значение/≤ 0 — None с предупреждением в stderr."""
    env = os.environ if environ is None else environ
    raw = env.get("EXPDS_SOFT_CEILING_S")
    src = "EXPDS_SOFT_CEILING_S"
    if raw is None or str(raw).strip() == "":
        raw = (cfg.get("time") or {}).get("soft_ceiling_s", SOFT_CEILING_DEFAULT_S)
        src = "[time] soft_ceiling_s"
    if isinstance(raw, bool):
        _warn(f"мягкий потолок выключен: {src}={raw!r} (нужно число секунд > 0)")
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        _warn(f"мягкий потолок выключен: {src}={raw!r} — не число")
        return None
    if not np.isfinite(val) or val <= 0:
        _warn(f"мягкий потолок выключен: {src}={raw!r} (нужно > 0)")
        return None
    return val


def soft_need_from(cfg: dict, environ=None) -> dict:
    """Пороги soft_left по шагам: SOFT_NEED_DEFAULT, [time.soft], env EXPDS_SOFT_NEED_<KEY>_S."""
    env = os.environ if environ is None else environ
    need = dict(SOFT_NEED_DEFAULT)
    table = ((cfg.get("time") or {}).get("soft") or {})
    for key in need:
        for src, raw in ((f"[time.soft] {key}", table.get(key)),
                         (f"EXPDS_SOFT_NEED_{key.upper()}_S", env.get(f"EXPDS_SOFT_NEED_{key.upper()}_S"))):
            if raw is None or str(raw).strip() == "":
                continue
            try:
                val = float(raw)
                if not np.isfinite(val) or val < 0:
                    raise ValueError
                need[key] = val
            except (TypeError, ValueError):
                _warn(f"порог мягкого потолка {src}={raw!r} не принят, остаётся {need[key]:.0f} с")
    return need


#: bot-8: TimeGuard текущего run() для дедлайна в цикле окон (_raster_with_model). Вне run() — None:
#: скрипты (geo_predict, nn_oof) и прямые вызовы работают как раньше; подписи вызывающих не меняются.
_WINDOW_GUARD: "TimeGuard | None" = None


def window_reserve(cfg: dict) -> float | None:
    """[time] window_deadline (true) / window_reserve_s (30). window_deadline = false — без проверки в цикле окон."""
    t = cfg.get("time") or {}
    if not t.get("window_deadline", True):
        return None
    return float(t.get("window_reserve_s", DEFAULT_WINDOW_RESERVE_S))


def load_config(path: str | Path | None) -> dict:
    """Конфиг TOML (configs/default.toml); отсутствующий файл → пустой словарь."""
    if path and Path(path).exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def _params(cls, cfg: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (cfg or {}).items() if k in names})


def model_specs(models: dict, markup: str) -> list[tuple[str | list[str], list[str] | None]]:
    """Модели модальности из [models]: путь, таблица {path, classes} или их список.

    classes ограничивает выход модели, чтобы две модели одной модальности
    (например, мелкая и крупная для городищ) не выдавали один класс дважды.
    path может быть списком: это ансамбль, вероятности моделей усредняются.
    """
    v = models.get(markup)
    out = []
    for it in v if isinstance(v, list) else ([v] if v else []):
        if isinstance(it, str):
            out.append((it, None))
        elif isinstance(it, dict) and it.get("path"):
            out.append((it["path"], list(it.get("classes") or []) or None))
    return out


def model_ensemble_only(models: dict, markup: str) -> list[list[str] | None]:
    """Классы, по которым усредняется ансамбль. Нет ключа — среднее по всем классам."""
    v = models.get(markup)
    out = []
    for it in v if isinstance(v, list) else ([v] if v else []):
        out.append(list(it["ensemble_classes"]) if isinstance(it, dict) and it.get("ensemble_classes") else None)
    return out


def model_files(path: str | list[str]) -> list[str]:
    """Файлы весов одной записи model_specs."""
    return list(path) if isinstance(path, list) else [path]


def rooted_verify(cfg: dict, root: Path) -> dict:
    """[verify] weights относительно корня архива, как у [models] (bot-8: раньше — от текущей папки)."""
    opts = dict(cfg.get("verify") or {})
    w = opts.get("weights")
    if w and not Path(w).is_absolute():
        opts["weights"] = str(Path(root) / w)
    return {**cfg, "verify": opts}


def _segmenters(models: dict, markup: str, root: Path, tta: int = 1) -> list:
    from .infer_nn import Ensemble, Segmenter

    segs = []
    for (path, classes), only in zip(model_specs(models, markup), model_ensemble_only(models, markup)):
        files = [Path(f) if Path(f).is_absolute() else root / f for f in model_files(path)]
        missing = [f for f in files if not f.exists()]
        if missing:
            log.warning("модель %s не найдена: %s", markup, ", ".join(map(str, missing)))
            continue
        members = [Segmenter(f) for f in files]
        seg = members[0] if len(members) == 1 else Ensemble(members, only=only)
        seg.keep = set(classes) if classes else None
        seg.tta = tta
        segs.append(seg)
    return segs


def _cached_segmenters(seg_cache: dict, cfg: dict, markup: str, root: Path) -> list:
    """Модели модальности загружаются один раз за прогон."""
    if markup not in seg_cache:
        seg_cache[markup] = _segmenters(cfg.get("models", {}), markup, root,
                                        int(cfg.get("tta", {}).get(markup, 1)))
    return seg_cache[markup]


def _kept(seg, feats: list[dict]) -> list[dict]:
    keep = getattr(seg, "keep", None)
    return feats if keep is None else [f for f in feats if f["class_name"] in keep]


def region_crs(reg: Region, cfg: dict | None = None) -> str | None:
    """CRS участка. Без [crs] policy это UTM.json, как в замере 0,353793."""
    from .crs import resolve_region_crs

    policy = ((cfg or {}).get("crs") or {}).get("policy", "utm_json")
    utm = read_utm_json(getattr(reg, "utm_json", None))
    if policy == "utm_json":
        return utm
    rasters = [p for files in reg.rasters.values() for p in files]
    return resolve_region_crs(utm, list(reg.las), rasters, policy)


def _with_copies(feats: list[dict], reg: Region, cfg: dict) -> list[dict]:
    rules = (cfg.get("fusion") or {}).get("copy") or []
    if not rules:
        return feats
    try:
        return feats + fusion.copy_layers(feats, reg.rasters, rules, fallback_crs=region_crs(reg, cfg))
    except Exception:
        log.exception("%s: копии слоёв пропущены", reg.name)
        return feats


def geophys_feats(reg: Region, cfg: dict, guard: TimeGuard) -> list[dict]:
    """Классические finds_points и object_poly. Маски extra/ применяются после детектора."""
    feats: list[dict] = []
    crs_region = region_crs(reg, cfg)
    geo_cfg = cfg.get("geophys", {})
    if "ground_metres" in geo_cfg:                 # opt-in: шаг EPSG:3857 × cos φ (crs.ground_scale)
        from . import crs as _crs
        _crs.GROUND_METRES = bool(geo_cfg["ground_metres"])
    for mt in ("maggr", "etgr", "terfl"):
        maps = (geo_cfg.get(mt) or {}).get("maps", "gray")
        files = geophys.select_rasters(reg.rasters.get(mt, []), mt, maps)
        if not files:
            continue
        if (geo_cfg.get(mt) or {}).get("classic", True) is False:
            # Классика выключена конфигом: эту модальность считает сеть (geo_model_feats).
            continue
        classes = ("finds_points", "object_poly") if mt == "maggr" else ("object_poly",)
        opts = {c: geophys.params_for(geo_cfg, mt, c) for c in classes}
        boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
        noise = geophys.load_mask(reg.noise, mt, crs_region)
        mine: list[dict] = []
        for path in files:
            if not guard.ok(5):
                log.warning("время на исходе: геофизика %s пропущена", path.name)
                break
            try:
                b_map, n_map = geophys.sane_masks(path, boundary, noise, geo_cfg, crs_region)
                for cls in classes:
                    p, o = opts[cls]
                    found = (geophys.finds_points(path, reg.name, mt, p, fallback_crs=crs_region)
                             if cls == "finds_points"
                             else geophys.object_polys(path, reg.name, mt, p, fallback_crs=crs_region))
                    if o["rank"]:
                        found = geophys.rank_within_source(found)
                    found = geophys.apply_masks(found, b_map, n_map if o["use_noise"] else None)
                    mine += [f for f in found if f["confidence"] >= o["min_confidence"]]
            except Exception as exc:
                log.exception("%s: ошибка геофизики (%s)", path.name, exc)
        mt_cfg = geo_cfg.get(mt) or {}
        if (maps == "all" and "finds_points" in classes and len(files) > 1
                and mt_cfg.get("dedupe_maps", True)):
            # Одна находка на серой и цветной карте — один объект ([geophys.<модальность>] maps = "all").
            # dedupe_maps = false: эталон размечен по каждой карте отдельно (211: серая и цвет_2),
            # метрика сводит класс по всем картам — дубль на другой карте может быть попаданием.
            mine = geophys.dedupe_points(mine, 2.0, crs_region)
        if mt_cfg.get("layer_copy", False):
            try:
                layers = {f.name: geophys.source_crs(f, crs_region) for f in files}
                mine += geophys.layer_copies(mine, layers, float(mt_cfg.get("layer_copy_decay", 1.0)), f"{mt}_copy")
            except Exception:
                log.exception("%s: копии по слоям %s пропущены", reg.name, mt)
        feats += mine
    pix = geo_cfg.get("pixgbm") or {}
    if pix.get("enabled", False):
        # bot-3 PIXGBM: пиксельный LightGBM — доп. источник кандидатов (группы <метод>_pix). По умолчанию выключен.
        # W4: impl="clean" — чистая перепись по docs/PIXGBM_SPEC_CLEANROOM.md (expds/pix_clean.py),
        # тот же контракт region_feats/status/dedupe_union; по умолчанию прежняя реализация.
        impl = str(pix.get("impl", "legacy"))
        try:
            if impl == "clean":
                from . import pix_clean as pixmod
                md_default = "models/pix_clean"
            else:  # DOC-1: прежняя реализация грузится через expds/pix_legacy.py — в архив решения не входит
                from . import pix_legacy
                try:
                    pixmod, md_default = pix_legacy.load(), pix_legacy.MODEL_DIR
                except ImportError:
                    # expds/geo_pixgbm.py не хранится в main (docs/MERGE_2026-10-29.md) — классика.
                    # bot-6: ERROR, не WARNING — иначе прогон тихо отличается от v2.
                    # EXPDS_PIXGBM_STRICT=1 — исключение вместо классики (прогоны на ПК и в CI).
                    log.error("%s: pixgbm impl=legacy, но expds/geo_pixgbm.py нет в этой сборке — классика "
                              "(сборка ≠ v2 LB; см. docs/PIXGBM_DECISION.md)", reg.name)
                    if os.environ.get("EXPDS_PIXGBM_STRICT", "") == "1":
                        raise PixgbmModuleMissing("expds/geo_pixgbm.py нет, а [geophys.pixgbm] impl=legacy")
                    return feats
            cls = set(pix.get("classes", ["finds_points", "object_poly"]))
            # bot-7: относительный model_dir — от корня репо/архива, а не от текущей папки процесса
            md = Path(pix.get("model_dir", md_default))
            if not md.is_absolute():
                md = Path(__file__).resolve().parents[1] / md
            # срок фазы ([geophys.pixgbm] budget_s, абсолютный от старта) и лестница — внутри region_feats (bot-3 5deb94a)
            pst: dict = {}
            new = [f for f in pixmod.region_feats(reg, cfg, md, guard=guard, status=pst)
                   if f["class_name"] in cls]
            feats = pix_merge(feats, new, pst, pix, reg.name, pixmod)
            # W4 GEO-ORG1 (флаг org_maggr, по умолчанию выключен): кандидаты MagGr по бейзлайну организаторов 01
            # (expds/pix_org.py) заменяют кандидатов того же класса maggr; группа порога maggr_org.
            if impl == "clean" and pix.get("org_maggr", False) and (guard is None or guard.left() > 60):
                from . import pix_org
                ost: dict = {}
                onew = pix_org.region_feats(reg, cfg, md / "org", status=ost)
                if onew:
                    feats = pix_merge(feats, onew, ost, {"replace": True}, reg.name, pixmod)
        except PixgbmModuleMissing:
            raise
        except Exception:
            if impl == "clean" and os.environ.get("EXPDS_PIXGBM_STRICT") == "1":
                raise
            log.exception("%s: pixgbm пропущен", reg.name)
    if pix.get("org_etgr", False) and (guard is None or guard.left() > 60):
        # W4 GEO-ORG3 (флаг org_etgr, по умолчанию выключен): EtGr object_poly по бейзлайну организаторов 02
        # (expds/pix_org_et.py) заменяют кандидатов object_poly/etgr (классику или pix); группа порога etgr_org.
        try:
            from . import pix_clean, pix_org_et
            # DOC-1: model_dir берётся только у impl="clean" (у legacy — свой каталог моделей, в архив не идёт)
            md = Path(pix.get("model_dir", "models/pix_clean") if str(pix.get("impl", "legacy")) == "clean"
                      else "models/pix_clean")
            if not md.is_absolute():
                md = Path(__file__).resolve().parents[1] / md
            est: dict = {}
            enew = pix_org_et.region_feats(reg, cfg, md / "org", status=est)
            if enew:
                feats = pix_merge(feats, enew, est, {"replace": True}, reg.name, pix_clean)
        except Exception:
            if os.environ.get("EXPDS_PIXGBM_STRICT") == "1":
                raise
            log.exception("%s: org_etgr пропущен", reg.name)
    return feats


def pix_merge(feats: list[dict], new: list[dict], pst: dict, pix: dict, reg_name: str, mod) -> list[dict]:
    """Классика участка + кандидаты PIXGBM по [geophys.pixgbm] union_name_re / replace (вынесено из geophys_feats
    без изменения поведения, W4: тот же код собирает OOF из кэша классики, scripts/pix_clean_oof.py)."""
    import re
    ure = pix.get("union_name_re", "")
    if ure and re.search(ure, reg_name):
        # вариант H/U на участках, чьё имя подходит под union_name_re (H: «Гнёздово»): классика остаётся,
        # pix-кандидат выбрасывается, если его центр внутри классического полигона (object_poly)
        # или ≤1 м от классической находки (finds_points)
        new = mod.dedupe_union(feats, new)
    elif pix.get("replace", False):
        # вариант S: классика того же класса и метода на этом участке заменяется кандидатами pix
        # Растры, где pix не отработал (бюджет/ошибка, status["fell"]), сохраняют классику.
        have = {(f["class_name"], f["markup_type"]) for f in new}
        done, fell = pst.get("done", {}), pst.get("fell", {})

        def _drop(f):
            key = (f.get("class_name"), f.get("markup_type"))
            if key not in have:
                return False
            return not fell.get(key) or f.get("source") in done.get(key, set())
        feats = [f for f in feats if not _drop(f)]
    return feats + new


def cpu_kurgans(d, region: str, source: str, cfg: dict, root: Path, cand_p) -> list[dict]:
    """Пики рельефа и LightGBM. Включается, когда модели нет или она упала на этом LAS."""
    kurgan_model = cfg.get("candidates", {}).get("lgbm_model")
    found = candidates.detect(d, cand_p)
    scores = candidates.score(found["features"], root / kurgan_model if kurgan_model else None)
    return [dict(f, threshold_group="li_cpu")
            for f in candidates.to_features(found, scores, d.crs, region, cand_p, source=source)]


def li_mask_feats(reg: Region, las: Path, cfg: dict, seg_cache: dict, root: Path, dem_for) -> list[dict]:
    """Городища отдельной модели на своём DEM. Группа порога li_mask, не li и не li_coarse.

    Модель видит шаг и каналы из своего чекпойнта. Курганы она не выдаёт.
    Нет файла весов — пустой список, основной лидар не трогается.
    """
    from .lidar import hillshade_set
    from .lidar import hillshade_for
    hillshade_set = hillshade_for(cfg)  # lidar.hillshade_style (CI2-8), по умолчанию vendor

    segs = _cached_segmenters(seg_cache, cfg, "li_mask", root)
    out: list[dict] = []
    for seg in segs:
        d = dem_for(seg.meta)
        if d.crs is None:
            continue
        found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set, None, None,
                                          output_floors(cfg, "li_mask")))
        for feat in found:
            if feat.get("class_name") != "gorodishcha":
                continue
            feat["threshold_group"] = "li_mask"
            out.append(feat)
    return out


def li_fort_feats(reg: Region, las: Path, cfg: dict, seg_cache: dict, root: Path, dem_for) -> list[dict]:
    """Фортификации отдельной маской на DEM 1 м. Группа порога li_fort, не li и не li_coarse.

    Модель видит шаг и каналы из своего чекпойнта. Курганы она не выдаёт.
    Нет файла весов — пустой список, основной лидар не трогается.
    """
    from .lidar import hillshade_set
    from .lidar import hillshade_for
    hillshade_set = hillshade_for(cfg)  # lidar.hillshade_style (CI2-8), по умолчанию vendor

    segs = _cached_segmenters(seg_cache, cfg, "li_fort", root)
    out: list[dict] = []
    for seg in segs:
        d = dem_for(seg.meta)
        if d.crs is None:
            continue
        found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set, None, None,
                                          output_floors(cfg, "li_fort")))
        for feat in found:
            if feat.get("class_name") != "fortifikatsii":
                continue
            feat["threshold_group"] = "li_fort"
            out.append(feat)
    return out


def lidar_feats(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
                prefetch=None) -> list[dict]:
    """LAS → объекты лидара и, где есть аэрофото, копии курганов в слой ae.

    Сбой загрузки или инференса модели не оставляет участок без курганов:
    для этого LAS считается запасной детектор.
    """
    from .lidar import build_dem, dem_params, hillshade_set
    from .lidar import hillshade_for
    hillshade_set = hillshade_for(cfg)  # lidar.hillshade_style (CI2-8), по умолчанию vendor

    feats: list[dict] = []
    mosaic_kurgans: list[dict] = []
    crs_region = region_crs(reg, cfg)
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
    for las in reg.las:
        if not guard.ok(cfg.get("time", {}).get("reserve_s", 120)):
            log.warning("время на исходе: пропускаю %s", las.name)
            break
        try:
            segs = _cached_segmenters(seg_cache, cfg, "li", root)
        except Exception:
            log.exception("модель li не загрузилась, дальше запасной детектор")
            seg_cache["li"] = []
            segs = []
        dems: dict[tuple, object] = {}

        def dem_for(meta, _las=las):
            dp = dem_params(meta, cfg)
            key = (dp["res"], dp["fill"])
            if key not in dems:
                dems[key] = (prefetch.get(_las, dp["res"], dp["fill"], crs_region) if prefetch is not None
                             else build_dem(_las, res=dp["res"], fill=dp["fill"], crs=crs_region))
            return dems[key]

        produced = False
        if segs:
            try:
                from .lidar import DEFAULTS, build_dem_mosaic, would_coarsen

                dp0 = dem_params(segs[0].meta, cfg)
                split = (bool(cfg.get("lidar", {}).get("mosaic", False))
                         and would_coarsen(las, dp0["res"], DEFAULTS["max_pixels"]))
                if split:
                    d = (prefetch.get(las, dp0["res"], dp0["fill"], crs_region) if prefetch is not None
                         else build_dem(las, res=dp0["res"], fill=dp0["fill"], crs=crs_region))
                else:
                    d = dem_for(segs[0].meta)
                if d.crs is None:
                    log.warning("%s: нет CRS (ни UTM.json, ни WKT в LAS) — пропуск", las.name)
                    continue
                rescore = cfg.get("li", {}).get("mode", "rescore") == "rescore"
                group = lidar_threshold_group(d.info)
                floors = output_floors(cfg, group or "li")
                kur = {"kurgany_povrezhdennye", "kurgany_tselye"}
                for seg in segs:
                    found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set,
                                                      cand_p if rescore else None, cfg.get("li"), floors))
                    if split:
                        found = [f for f in found if f.get("class_name") not in kur]
                    if group:
                        for f in found:
                            if not f.get("threshold_group"):
                                f["threshold_group"] = group
                    feats += found
                if split:
                    try:
                        mosaic = build_dem_mosaic(las, res=dp0["res"], fill=dp0["fill"], crs=crs_region)
                    except Exception:
                        log.exception("%s: мозаика не собралась, курганы с грубой сетки", las.name)
                        mosaic = None
                    fresh: list[dict] = []
                    if mosaic is not None and mosaic.crs is not None:
                        for seg in segs:
                            found = _li_with_model(seg, mosaic, reg.name, las.name, hillshade_set,
                                                   cand_p if rescore else None, cfg.get("li"), None)
                            fresh += [f for f in found if f.get("class_name") in kur]
                    else:
                        for seg in segs:
                            found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set,
                                                              cand_p if rescore else None, cfg.get("li"), floors))
                            found = [f for f in found if f.get("class_name") in kur]
                            if group:
                                for f in found:
                                    if not f.get("threshold_group"):
                                        f["threshold_group"] = group
                            fresh = found
                    mosaic_kurgans += [f for f in fresh if not f.get("threshold_group")]
                    feats += fresh
                hf = cfg.get("hillfort") or {}
                if hf.get("enabled", False):
                    # Своя ЦМР, как в scripts/hillfort_oof.py. Параметры, включая
                    # relief_radius_l_m, берутся из конфига: принятое табло — один проход.
                    try:
                        from . import hillfort
                        hp = hillfort.params_from(hf)
                        feats += hillfort.candidates(hillfort.build_dem(las, crs_region, hp), reg.name, las.name, hp)
                    except Exception:
                        log.exception("%s: кандидаты городищ по рельефу пропущены", las.name)
                produced = True
            except Exception:
                log.exception("%s: инференс li упал, запасной детектор", las.name)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        if produced:
            if guard.ok(cfg.get("time", {}).get("reserve_s", 120)):
                try:
                    feats += li_mask_feats(reg, las, cfg, seg_cache, root, dem_for)
                except Exception:
                    log.exception("%s: маска городищ пропущена", las.name)
                try:
                    feats += li_fort_feats(reg, las, cfg, seg_cache, root, dem_for)
                except Exception:
                    log.exception("%s: маска фортификаций пропущена", las.name)
            continue
        try:
            d = dem_for(None)
        except Exception:
            log.exception("%s: DEM для запасного детектора не построен", las.name)
            continue
        if d.crs is None:
            log.warning("%s: нет CRS — пропуск", las.name)
            continue
        try:
            feats += cpu_kurgans(d, reg.name, las.name, cfg, root, cand_p)
        except Exception:
            log.exception("%s: запасной детектор упал", las.name)

    # Группа мозаики ставится до копий: копия мозаики — li_ae, копия мелкой сетки — ae, как в табло.
    for f in mosaic_kurgans:
        if f.get("markup_type") == "li" and not f.get("threshold_group"):
            f["threshold_group"] = "li_mosaic"
    fp = _params(fusion.FusionParams, cfg.get("fusion"))
    if fp.li_to_ae and reg.rasters.get("ae"):
        try:
            feats += fusion.li_to_ae(feats, reg.rasters["ae"], fp, fallback_crs=crs_region)
        except Exception as exc:
            # bot-7 (v4, REPORT13_lbv2 бота 10): громкий отказ — копии аэро-курганов с лидара потеряны, маркер ловит check_submission
            log.exception("LI_TO_AE_FAILED: %s: копирование курганов лидара в ae не выполнено, ae без копий (%s: %s)",
                          reg.name, type(exc).__name__, str(exc)[:300])
    return _with_copies(feats, reg, cfg)


# Круги, которые принятый пакет уже оставляет. Новый круг ближе 12 м к такому
# кругу того же участка и класса выбрасывается. Числа — совместный оптимум
# замера 0,353793 (notes/pack_wide_dev_2026-10-23.json). sp_or и sp_focal —
# одна карта центров: в сдаче группа читается из markup_type, в табло она
# подписана sp_focal. В конфиге порог лидара 0,06415 / 0,194220.
# 0,05333 — более ранняя точка, в конфиг её не возвращать.
KEPT_KURGAN_CONF = {
    "ae": {"kurgany_povrezhdennye": 0.99053, "kurgany_tselye": 0.802255},
    "li": {"kurgany_povrezhdennye": 0.06415, "kurgany_tselye": 0.19422},
    "li_ae": {"kurgany_povrezhdennye": 0.993061, "kurgany_tselye": 1.000001},
    "li_mosaic": {"kurgany_povrezhdennye": 0.959005, "kurgany_tselye": 0.17042},
    "sp_focal": {"kurgany_povrezhdennye": 0.364475, "kurgany_tselye": 0.372525},
    "sp_or": {"kurgany_povrezhdennye": 0.364475, "kurgany_tselye": 0.372525},
}


def ground_point(geom):
    """Внутренняя точка в метрах.

    Мелкие координаты уже метры. Крупные — веб-меркатор: вся фигура переводится
    в зону UTM по центроиду, и только потом берётся внутренняя точка. Так же
    устроен отбор 12 м в принятом пакете.
    """
    from shapely import get_coordinates

    xs = get_coordinates(geom)[:, 0]
    if xs.size == 0 or float(np.max(np.abs(xs))) <= 1_000_000:
        return geom.representative_point()
    from .crs import WEB_MERCATOR, to_crs, transformer, utm_for_lonlat

    lon, lat = transformer(WEB_MERCATOR, "EPSG:4326").transform(geom.centroid.x, geom.centroid.y)
    return to_crs(geom, WEB_MERCATOR, utm_for_lonlat(float(lon), float(lat))).representative_point()


def kept_kurgan_anchors(feats: list[dict]) -> list[dict]:
    """Круги курганов, которые сданный порог своей группы уже оставляет."""
    kept = []
    for feat in feats:
        cls = feat.get("class_name")
        group = feat.get("threshold_group") or feat.get("markup_type")
        thr = KEPT_KURGAN_CONF.get(group, {}).get(cls)
        if thr is None or float(feat.get("confidence") or 0) < thr:
            continue
        kept.append(feat)
    return kept


def drop_near_kurgans(cands: list[dict], anchors: list[dict], dist_m: float = 12.0) -> list[dict]:
    """Убирает круг, если ближе dist_m к уже оставленному кругу того же участка и класса."""
    from shapely.strtree import STRtree

    buckets: dict[tuple, list] = {}
    for feat in anchors:
        cls = feat.get("class_name")
        if cls not in ("kurgany_povrezhdennye", "kurgany_tselye") or feat.get("geom") is None:
            continue
        buckets.setdefault((feat.get("region_name"), cls), []).append(ground_point(feat["geom"]))
    trees = {key: STRtree(pts) for key, pts in buckets.items() if pts}
    kept = []
    for feat in cands:
        pts = buckets.get((feat.get("region_name"), feat.get("class_name")))
        if not pts or feat.get("geom") is None:
            kept.append(feat)
            continue
        point = ground_point(feat["geom"])
        key = (feat.get("region_name"), feat.get("class_name"))
        near = any(pts[int(i)].distance(point) <= dist_m for i in trees[key].query(point.buffer(dist_m)))
        if not near:
            kept.append(feat)
    return kept


def optical_feats(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path) -> list[dict]:
    """Обученные модели ae, sp_or и or. Без файла весов модальность пропускается."""
    feats: list[dict] = []
    for mt in ("ae", "sp_or", "or"):
        if not reg.rasters.get(mt):
            continue
        try:
            segs = _cached_segmenters(seg_cache, cfg, mt, root)
        except Exception:
            log.exception("модель %s не загрузилась, модальность пропущена", mt)
            seg_cache[mt] = []
            continue
        for seg in segs:
            for path in reg.rasters.get(mt, []):
                if not guard.ok(60):
                    break
                try:
                    feats += _kept(seg, _raster_with_model(seg, path, reg.name, mt,
                                                           min_conf=_sp_floors(cfg, mt),
                                                           fallback_crs=region_crs(reg, cfg)))
                except Exception as exc:
                    log.exception("%s: ошибка модели %s (%s)", path.name, mt, exc)
    feats = _wide_without_lidar(reg, cfg, guard, seg_cache, root, feats)
    if (cfg.get("post") or {}).get("frame_prior"):
        feats = _with_scene_drel(reg, cfg, feats)
    return _with_copies(feats, reg, cfg) + geo_model_feats(reg, cfg, guard, seg_cache, root)


def optical_bytes(reg: Region) -> int:
    """Размер снимков ae / sp_or / or участка в байтах (0, если файл не читается)."""
    total = 0
    for mt in ("ae", "sp_or", "or"):
        for path in reg.rasters.get(mt, []):
            try:
                total += Path(path).stat().st_size
            except OSError:
                pass
    return total


def optics_order(regions: list[Region], cfg: dict) -> list[Region]:
    """Порядок участков в проходе оптики. [time] optics_order = "small_first" — от дешёвых снимков к дорогим.

    Проход оптики идёт третьим и обрывается по бюджету. Спутниковая сцена — это обычно один памятник
    в центре и курганы рядом, а время растёт с размером файла. Поэтому при нехватке времени выгоднее
    обойти больше участков: гигантское ортофото в начале списка иначе съедает оптику всех остальных
    (на ПК один участок с 25,5 ГБ ортофото шёл 1624 с). Если время есть, выход тот же. По умолчанию — прежний порядок.
    """
    if (cfg.get("time") or {}).get("optics_order", "name") != "small_first":
        return list(regions)
    return sorted(regions, key=lambda r: (optical_bytes(r), r.name))


def _restore_optics_order(state, regions: list[Region], n_before: int, by_reg: dict[str, list[dict]]) -> None:
    """bot-7 v3: оптика могла идти в порядке optics_order (small_first), но объекты в состоянии — в исходном
    порядке участков, как при optics_order = "name". Поэтому выход при запасе времени побайтно тот же, а при
    нехватке — те же объекты пройденных участков. Состояние сторож пишет и во время прохода (запись идёт сразу),
    перестановка — только в конце, и только если в состояние за это время ничего чужого не добавилось."""
    snap = state.snapshot()
    n_opt = sum(len(v) for v in by_reg.values())
    if len(snap) != n_before + n_opt:
        log.warning("порядок оптики не восстановлен: состояние изменилось во время прохода")
        return
    state.replace(snap[:n_before] + [f for r in regions for f in by_reg.get(r.name, [])])


def _with_scene_drel(reg: Region, cfg: dict, feats: list[dict]) -> list[dict]:
    """scene_drel для [post] frame_prior: положение объекта в кадре снимков своего типа (post.scene_drel)."""
    from .geophys import map_footprint

    boxes: dict[str, list] = {}
    for mt in ("ae", "sp_or", "or"):
        for path in reg.rasters.get(mt, []):
            try:
                fp = map_footprint(path, region_crs(reg, cfg))
            except Exception:
                log.exception("%s: рамка снимка не прочитана, приор кадра без неё", Path(path).name)
                fp = None
            if fp is not None:
                boxes.setdefault(mt, []).append(fp)
    return post.annotate_scene_drel(feats, boxes)


#: Модальности геофизики, для которых [models] может назвать сеть. Группа порога — «<модальность>_nn».
GEO_NN = ("maggr", "etgr", "terfl")


def geo_model_feats(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path) -> list[dict]:
    """Сеть на картах геофизики: [models] maggr / etgr / terfl. Без строки в конфиге — пусто.

    Растры те же, что у классики (select_rasters: без цветных копий и срезов EtGr глубже 2 м).
    Объекты идут в группу «<модальность>_nn», порог — [thresholds_by_markup.<модальность>_nn].
    Без этой секции действует [thresholds] класса (у геофизики 0,0), поэтому сеть
    включается только вместе с порогами из табло. Классику той же модальности выключает
    [geophys.<модальность>] classic = false, иначе выдаются обе ветки.
    Маски границ и шума здесь не применяются: так же предсказывает scripts/nn_oof.py.
    """
    out: list[dict] = []
    models = cfg.get("models", {})
    for mt in GEO_NN:
        if not models.get(mt):
            continue
        maps = ((cfg.get("geophys") or {}).get(mt) or {}).get("maps", "gray")
        files = geophys.select_rasters(reg.rasters.get(mt, []), mt, maps)
        if not files:
            continue
        try:
            segs = _cached_segmenters(seg_cache, cfg, mt, root)
        except Exception:
            log.exception("модель %s не загрузилась, сеть геофизики пропущена", mt)
            seg_cache[mt] = []
            continue
        group = f"{mt}_nn"
        floors = output_floors(cfg, group)
        added: list[dict] = []
        for seg in segs:
            for path in files:
                if not guard.ok(30):
                    log.warning("время на исходе: сеть %s пропущена на %s", mt, path.name)
                    break
                try:
                    found = _kept(seg, _raster_with_model(seg, path, reg.name, mt, min_conf=floors,
                                                          fallback_crs=region_crs(reg, cfg)))
                except Exception as exc:
                    log.exception("%s: ошибка сети %s (%s)", path.name, mt, exc)
                    continue
                for feat in found:
                    feat["threshold_group"] = group
                added += found
        if maps == "all" and len(files) > 1 and ((cfg.get("geophys") or {}).get(mt) or {}).get("dedupe_maps", True):
            # Только находки этой модальности: чужие классы в том же списке не склеиваются.
            added = geophys.dedupe_points(added, 2.0, region_crs(reg, cfg))
        out += added
    return out


def _wide_without_lidar(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
                        feats: list[dict]) -> list[dict]:
    """Окно 256 м только там, где нет LAS.

    Круг ближе 12 м к кругу карты центров, который сданный порог уже оставляет,
    выбрасывается. Иначе совместный перебор поднимает порог сданной карты.
    """
    if reg.las or not cfg.get("models", {}).get("sp_focal2") or not reg.rasters.get("sp_or"):
        return feats
    try:
        segs = _cached_segmenters(seg_cache, cfg, "sp_focal2", root)
    except Exception:
        log.exception("модель sp_focal2 не загрузилась")
        seg_cache["sp_focal2"] = []
        return feats
    floors = _sp_floors(cfg, "sp_focal2")
    added: list[dict] = []
    for seg in segs:
        for path in reg.rasters.get("sp_or", []):
            if not guard.ok(60):
                break
            if not _soft_ok(guard, "wide_window", f"окно 256 м sp_focal2, {reg.name}"):
                break
            try:
                found = _kept(seg, _raster_with_model(seg, path, reg.name, "sp_or", min_conf=floors,
                                                      fallback_crs=region_crs(reg, cfg)))
            except Exception as exc:
                log.exception("%s: ошибка окна 256 м (%s)", path.name, exc)
                continue
            for feat in found:
                feat["threshold_group"] = "sp_focal2"
            added += found
    kept = drop_near_kurgans(added, kept_kurgan_anchors(feats), 12.0)
    if _sp_fusion_spec(cfg) is not None:
        # Со слиянием отброшенные отбором 12 м остаются соседями для sp_score_fusion, но не выдаются.
        ids = {id(f) for f in kept}
        for f in added:
            if id(f) not in ids:
                f["sp_partner_only"] = True
        return feats + added
    return feats + kept


def _sp_fusion_spec(cfg: dict):
    """[post] sp_score_fusion: None, если выключено; иначе словарь настроек."""
    sf = (cfg.get("post") or {}).get("sp_score_fusion")
    if not sf:
        return None
    return sf if isinstance(sf, dict) else {}


def _sp_floors(cfg: dict, markup: str) -> dict[str, float]:
    """Пол генерации: порог выдачи; со слиянием курганы sp_or / sp_focal2 — сырые от gen_floor."""
    floors = output_floors(cfg, markup)
    spec = _sp_fusion_spec(cfg)
    if spec is None or markup not in ("sp_or", "sp_focal2"):
        return floors
    from .post import sp_fusion_gen_floors
    return sp_fusion_gen_floors(floors, spec)


def cnx_site_feats(reg: Region, cfg: dict, guard, seg_cache: dict, root: Path, prefetch=None) -> list[dict]:
    """bot-8 cnx: ConvNeXt-лидар bot-9 ([li_cnx]) по всему участку → объекты группы li_cnx (до отбора дублей).

    Добавочный канал: модель li уже отработала в lidar_feats, здесь ничего из её выдачи не меняется.
    Все LAS участка — одна подготовка (lidar_cnx.bind_las сливает их, контракт §6.1). Курганы — по
    кандидатам-пикам ЦМР каждого LAS (rescore, как у li), прочие классы из li_cnx.classes — масками
    один раз на участок. Любой отказ (нет порогов cnx, веса, мягкий потолок, подготовка, CRS, сеть) —
    пустой список с предупреждением: без cnx остаётся выдача как при enabled=false.
    """
    from .lidar import build_dem, dem_params, hillshade_set
    from .lidar import hillshade_for
    hillshade_set = hillshade_for(cfg)  # lidar.hillshade_style (CI2-8), по умолчанию vendor

    if not reg.las or not lidar_cnx.enabled(cfg):
        return []
    opts = lidar_cnx.cnx_cfg(cfg)
    want = list(opts.get("classes") or [])
    problem = lidar_cnx.threshold_problem(cfg, want or None)
    if problem:
        if not seg_cache.get("li_cnx_thr_warned"):
            seg_cache["li_cnx_thr_warned"] = True
            _warn(f"[li_cnx] включён, но {problem}: объекты cnx НЕ выдаются (пороги li к шкале cnx "
                  f"не применимы); выдача без cnx не меняется")
        return []
    if "li_cnx" not in seg_cache:
        try:
            seg_cache["li_cnx"] = lidar_cnx.load(cfg, root)
        except Exception as exc:
            _warn(f"cnx: веса [li_cnx] не загрузились ({str(exc)[-300:]}) — объектов cnx нет")
            seg_cache["li_cnx"] = None
    seg = seg_cache["li_cnx"]
    if seg is None:
        return []
    keep = [c for c in seg.classes if seg.keep is None or c in seg.keep]
    problem = lidar_cnx.threshold_problem(cfg, keep)
    if problem:
        _warn(f"[li_cnx]: {problem} — объекты cnx НЕ выдаются")
        return []
    if getattr(guard, "soft_ceiling", None) is not None and \
            not guard.soft_ok(float(opts["soft_need_s"]), phase="cnx", what=reg.name):
        log.warning("%s: cnx пропущен мягким потолком", reg.name)
        return []
    reserve = float(cfg.get("time", {}).get("reserve_s", 120))
    left = getattr(guard, "window_left", None) or guard.left

    def infer_budget() -> float:
        b = float(left()) - reserve
        if getattr(guard, "soft_ceiling", None) is not None:
            b = min(b, float(guard.soft_left()))
        return max(0.0, b)

    crs_region = region_crs(reg, cfg)
    try:
        bound = lidar_cnx.bind_las(seg, list(reg.las), crs_region, opts, deadline=lambda: not guard.ok(reserve),
                                   time_left=lambda: max(0.0, float(left()) - reserve), infer_budget=infer_budget)
    except Exception as exc:
        log.warning("%s: cnx не сработал (%s) — объектов cnx нет", reg.name, str(exc)[-400:])
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return []
    group = str(opts["threshold_group"])
    floors = output_floors(cfg, group)
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
    rescore = cfg.get("li", {}).get("mode", "rescore") == "rescore"
    kur = [c for c in keep if c in lidar_cnx.KURGANS] if rescore else []
    rest = [c for c in keep if c not in kur]
    out: list[dict] = []
    try:
        dp = dem_params(seg.meta, cfg)
        for i, las in enumerate(reg.las):
            if not guard.ok(reserve):
                log.warning("%s: cnx — время на исходе, остальные LAS без cnx", reg.name)
                break
            d = (prefetch.get(las, dp["res"], dp["fill"], crs_region) if prefetch is not None
                 else build_dem(las, res=dp["res"], fill=dp["fill"], crs=crs_region))
            if d.crs is None:
                continue
            if not lidar_cnx._crs_equal(bound.geo.get("crs"), d.crs):
                raise lidar_cnx.CnxError(f"CRS cnx {bound.geo.get('crs')} ≠ CRS ЦМР {d.crs}")
            if kur:
                out += _li_with_model(bound.view(kur), d, reg.name, las.name, hillshade_set, cand_p,
                                      cfg.get("li"), floors)
            if rest and i == 0:
                out += _li_with_model(bound.view(rest), d, reg.name, las.name, hillshade_set, None,
                                      cfg.get("li"), floors)
    except Exception as exc:
        log.warning("%s: cnx: постобработка упала (%s) — объектов cnx нет", reg.name, str(exc)[-400:])
        return []
    for f in out:
        f["threshold_group"] = group
    return out


def dedupe_cnx(fresh: list[dict], pool: list[dict], cfg: dict, dist_m: float = 12.0) -> list[dict]:
    """bot-8 cnx: убирает объекты cnx, которые дублируют уже найденные (pool — объекты участка до cnx).

    Курганы — тем же отбором, что σ и окно 256 м (drop_near_kurgans, 12 м): якоря — круги, которые
    оставляет порог своей группы (KEPT_KURGAN_CONF; для групп вне таблицы — порог группы из конфига,
    напр. li_coarse/li_mosaic/li_cpu). Маски (городища, фортификации, архитектура — только если их явно
    включили в li_cnx.classes) — убираются при пересечении с полигоном того же класса и участка,
    который проходит порог своей группы (li, li_mask, li_fort, ae, …).
    """
    if not fresh:
        return []
    kur = [f for f in fresh if f.get("class_name") in lidar_cnx.KURGANS]
    rest = [f for f in fresh if f.get("class_name") not in lidar_cnx.KURGANS]

    def passes(f) -> bool:
        group = f.get("threshold_group") or f.get("markup_type")
        thr = output_floors(cfg, group).get(f.get("class_name"))
        return thr is not None and float(f.get("confidence") or 0) >= thr

    anchors = kept_kurgan_anchors(pool) + [
        f for f in pool if f.get("class_name") in lidar_cnx.KURGANS
        and (f.get("threshold_group") or f.get("markup_type")) not in KEPT_KURGAN_CONF and passes(f)]
    kept = drop_near_kurgans(kur, anchors, dist_m)
    if rest:
        from shapely.strtree import STRtree

        polys: dict[tuple, list] = {}
        for f in pool:
            if f.get("geom") is not None and f.get("class_name") not in lidar_cnx.KURGANS and passes(f):
                polys.setdefault((f.get("region_name"), f.get("class_name")), []).append(f["geom"])
        trees = {k: STRtree(v) for k, v in polys.items()}
        for f in rest:
            key = (f.get("region_name"), f.get("class_name"))
            g = f.get("geom")
            if g is None or key not in trees or not any(polys[key][int(i)].intersects(g)
                                                        for i in trees[key].query(g)):
                kept.append(f)
    n = len(fresh) - len(kept)
    if n:
        log.info("cnx: %d дублей уже найденных объектов убрано (из %d)", n, len(fresh))
    return kept


def _soft_ok(guard, phase: str, what: str = "", extra_s: float = 0.0) -> bool:
    """bot-8: мягкий потолок для необязательного шага. Guard без потолка (тесты, скрипты) — всегда True."""
    fn = getattr(guard, "soft_ok", None)
    if fn is None or getattr(guard, "soft_ceiling", None) is None:
        return True
    return fn(guard.soft_need_s(phase) + extra_s, phase=phase, what=what)


def li_dev_feats(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
                 prefetch=None) -> list[dict]:
    """Круги канала σ на пиках рельефа. Группа li_dev.

    Отбор 12 м делает run: сюда попадают все круги выше пола выдачи.
    Нет весов — пустой список.
    """
    from .lidar import build_dem, dem_params, hillshade_set
    from .lidar import hillshade_for
    hillshade_set = hillshade_for(cfg)  # lidar.hillshade_style (CI2-8), по умолчанию vendor

    if not reg.las or not cfg.get("models", {}).get("li_dev"):
        return []
    try:
        segs = _cached_segmenters(seg_cache, cfg, "li_dev", root)
    except Exception:
        log.exception("модель li_dev не загрузилась")
        seg_cache["li_dev"] = []
        return []
    crs_region = region_crs(reg, cfg)
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
    floors = output_floors(cfg, "li_dev")
    out: list[dict] = []
    for las in reg.las:
        if not guard.ok(60):
            break
        for seg in segs:
            dp = dem_params(seg.meta, cfg)
            d = (prefetch.get(las, dp["res"], dp["fill"], crs_region) if prefetch is not None
                 else build_dem(las, res=dp["res"], fill=dp["fill"], crs=crs_region))
            if d.crs is None:
                continue
            found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set,
                                              cand_p, cfg.get("li"), floors))
            for feat in found:
                if feat.get("class_name") not in ("kurgany_povrezhdennye", "kurgany_tselye"):
                    continue
                feat["threshold_group"] = "li_dev"
                out.append(feat)
    return out


def run_region(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
               prefetch=None) -> list[dict]:
    """Один участок: лидар, геофизика, оптика. Канала σ здесь нет. Платформа идёт через run(): четыре прохода."""
    return (lidar_feats(reg, cfg, guard, seg_cache, root, prefetch)
            + geophys_feats(reg, cfg, guard) + optical_feats(reg, cfg, guard, seg_cache, root))


def lidar_threshold_group(info: dict | None) -> str | None:
    """Группа порога этого LAS: li_coarse, если шаг DEM увеличен (облако не влезло в 60 млн ячеек)."""
    try:
        factor = int((info or {}).get("factor") or 1)
    except (TypeError, ValueError):
        return None
    return "li_coarse" if factor > 1 else None


def output_floors(cfg: dict, markup: str) -> dict[str, float]:
    """Пороги выдачи классов для объектов этого markup_type — те же, что потом применит post.apply_thresholds."""
    from .constants import CLASSES

    thr = cfg.get("thresholds", {})
    own = cfg.get("thresholds_by_markup", {}).get(markup, {})
    default = cfg.get("default_threshold", 0.15)
    floors = {c: float(own.get(c, thr.get(c, default))) for c in CLASSES}
    # [post] kurgan_neighbour_prior поднимает уверенность до floor + slope·cap раз до порогов: курган
    # группы приора ниже порога должен дожить до finish, иначе множитель только понижает.
    prior = ((cfg.get("post") or {}).get("kurgan_neighbour_prior"))
    if prior:
        spec = prior if isinstance(prior, dict) else {}
        if markup in set(spec.get("groups") or ("li", "li_coarse", "li_mosaic")):
            top = float(spec.get("floor", 0.5)) + float(spec.get("slope", 0.5)) * float(spec.get("cap", 2.0))
            for c in ("kurgany_povrezhdennye", "kurgany_tselye"):
                floors[c] /= max(top, 1.0)
    # [post] site_calibration считает шкалу участка по всем объектам с s ≥ floor, как в OOF: ранний пол
    # калибруемых групп не выше этого floor, иначе распределение участка на сдаче будет другим.
    calib = (cfg.get("post") or {}).get("site_calibration")
    if calib and markup in set(calib.get("groups") or ()):
        for c in set(calib.get("classes") or ("kurgany_povrezhdennye", "kurgany_tselye")):
            floors[c] = min(floors[c], float(calib.get("floor", 0.05)))
    # [post] big_groups (bot-2 bigobj F1): крупные маски получают свой порог <markup>_big в finish; ранний
    # пол не выше него, иначе крупные объекты ниже порога родителя срезаются ещё при полигонизации.
    big = (cfg.get("post") or {}).get("big_groups")
    if big:
        from .post import big_group_thresholds

        spec = big if isinstance(big, dict) else {}
        if not spec.get("groups") or markup in set(spec["groups"]):
            suffix = str(spec.get("suffix", "_big"))
            own_big = big_group_thresholds(cfg.get("thresholds_by_markup", {}), spec).get(markup + suffix, {})
            for c in set(spec.get("classes") or ("gorodishcha", "fortifikatsii")):
                if c in own_big:
                    floors[c] = min(floors[c], float(own_big[c]))
    return floors


def _li_with_model(seg, d, region: str, source: str, hillshade_set, cand_p=None,
                   li_cfg: dict | None = None, min_conf: dict[str, float] | None = None) -> list[dict]:
    """Каналы из DEM на рабочем шаге модели → вероятности → объекты.

    С cand_p курганы выдаются по кандидатам-пикам рельефа с уверенностью модели
    в их центрах: пики разделяют соседние насыпи, которые маска сливает в один
    объект (а слитый полигон метрика не засчитывает). Остальные классы — масками.
    Чекпойнт с target="centers" отдаёт круги вокруг пиков, как оптика. У сданных
    весов лидара этого поля нет.
    """
    from rasterio.transform import from_origin

    from .candidates import downsample_mean
    from .infer_nn import probs_to_features

    if seg.meta.get("li_source") in ("las", "cnx"):
        if seg.meta.get("li_source") == "cnx":
            # bot-8 cnx: вероятности уже посчитаны на сетке подготовки bot-9 (lidar_cnx.bind_las).
            probs, tr = seg.probs, seg.transform
        else:
            from .lidar import relief_channels

            img, tr = relief_channels(d, seg.res, tuple(seg.meta["li_tags"]))
            probs = seg.predict(img)
        keep = getattr(seg, "keep", None)
        out_classes = [c for c in seg.classes if keep is None or c in keep]
        if seg.meta.get("target") == "centers":
            from .centers import centers_to_features

            idx = [seg.classes.index(c) for c in out_classes]
            return centers_to_features(
                probs[idx], out_classes, tr, d.crs, float(tr.a), region, "li", source,
                radius_m=float(seg.meta.get("center_radius_m") or 6.0),
                min_dist_m=float(12.0 if seg.meta.get("min_dist_m") is None else seg.meta["min_dist_m"]),
                min_conf=min_conf)
        kur = [c for c in ("kurgany_povrezhdennye", "kurgany_tselye") if c in out_classes] if cand_p else []
        rest = [c for c in out_classes if c not in kur]
        feats = probs_to_features(probs[[seg.classes.index(c) for c in rest]], rest, tr, d.crs, tr.a,
                                  region, "li", source, min_conf=min_conf) if rest else []
        if kur:
            c = candidates.detect(d, cand_p)
            kp = probs[[seg.classes.index(k) for k in kur]]
            lc = li_cfg or {}
            if lc.get("nn_peaks", False):
                c = candidates.add_prob_peaks(c, kp.max(axis=0), tr, lc.get("nn_peak_min", 0.3),
                                              lc.get("nn_peak_radius_m", 2.5), lc.get("nn_peak_dedupe_m", 2.0))
            s = candidates.sample_probs(c, kp, tr)
            scores = np.column_stack([np.zeros(len(s), np.float32), s])
            feats += candidates.to_features(c, scores, d.crs, region, cand_p, classes=tuple(kur), source=source)
        return feats
    tags = seg.meta.get("li_tags") or ["g"]
    images = hillshade_set(d, tags=tuple(tags))
    k = max(1, int(round(seg.res / d.res)))
    chans = []
    for t in tags:
        a = images[t] if images[t].ndim == 3 else images[t][None]
        chans += [downsample_mean(a[i].astype(np.float32), k) for i in range(a.shape[0])]
    img = np.clip(np.nan_to_num(np.stack(chans), nan=0), 0, 255).astype(np.uint8)
    probs = seg.predict(img)
    tr = from_origin(d.left, d.top, d.res * k, d.res * k)
    return probs_to_features(probs, seg.classes, tr, d.crs, d.res * k, region, "li", source, min_conf=min_conf)


def _read_ahead(jobs: list, read, depth: int = 1):
    """(job, read(job)) по порядку; следующие depth окон читаются в фоновом потоке (depth ≤ 0 — без потока).

    speedup-2026-10-28 (харвест 29.10); включается флагом [speed] read_ahead (по умолчанию ВЫКЛ.).
    """
    from concurrent.futures import ThreadPoolExecutor

    if depth <= 0 or len(jobs) < 2:
        for job in jobs:
            yield job, read(job)
        return
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expds-read")
    try:
        futs = [pool.submit(read, job) for job in jobs[:depth]]
        for i, job in enumerate(jobs):
            if i + depth < len(jobs):
                futs.append(pool.submit(read, jobs[i + depth]))
            yield job, futs[i].result()
            futs[i] = None
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _raster_with_model(seg, path: Path, region: str, markup: str, block: int = 4096,
                       overlap: int = 256, min_conf: dict[str, float] | None = None,
                       fallback_crs: str | None = None, guard: "TimeGuard | None" = None) -> list[dict]:
    """Растр на рабочем шаге модели окнами block×block с перекрытием → объекты (память — одно окно).

    CRS — через crs.usable_raster_crs: безымянная LOCAL_CS с координатами UTM получает fallback_crs.
    guard (bot-8): перед каждым окном guard.window_ok(); при близком дедлайне — частичный результат
    по файлу (объекты уже пройденных окон), запись в guard.partial. По умолчанию — guard текущего
    run() (_WINDOW_GUARD); вне run() проверки нет, как раньше.
    """
    if guard is None:
        guard = _WINDOW_GUARD
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from shapely.geometry import box

    from .crs import WEB_MERCATOR, to_crs, usable_raster_crs
    from .infer_nn import probs_to_features
    from .raster import RASTER_FIX, raster_fix_on, zero_field_mask
    from .tiles import MAX_UPSAMPLE, band_stats, fix_channels, to_uint8, working_res

    as_centers = seg.meta.get("target") == "centers"

    stats = band_stats(str(path), markup)
    half = overlap // 2
    feats: list[dict] = []
    from .raster import open_metric

    with open_metric(path) as ds:
        native = abs(ds.res[0])
        if seg.meta.get("pixel_units") == "ground":
            from .raster import ground_step, raster_kind_and_latitude

            kind, lat = raster_kind_and_latitude(ds.crs, ds.transform, ds.width, ds.height)
            r = ground_step(native, kind, lat, seg.res, seg.meta.get("max_upsample") or MAX_UPSAMPLE)
        else:
            r = working_res(native, seg.res, seg.meta.get("max_upsample") or MAX_UPSAMPLE)
        scale = r / native
        w, h = max(1, int(ds.width / scale)), max(1, int(ds.height / scale))
        n = min(ds.count, 3)
        crs = usable_raster_crs(ds.crs.to_string() if ds.crs else None, ds.bounds, fallback_crs)
        step = block - overlap
        xs = list(range(0, max(w - overlap, 1), step))
        ys = list(range(0, max(h - overlap, 1), step))
        def read_block(job):
            _, _, y0, x0 = job
            bw, bh = min(block, w - x0), min(block, h - y0)
            win = Window(x0 * scale, y0 * scale, bw * scale, bh * scale)
            return bw, bh, win, ds.read(list(range(1, n + 1)), window=win, out_shape=(n, bh, bw),
                                        resampling=Resampling.average if scale > 1 else Resampling.bilinear)

        jobs = [(iy, ix, y0, x0) for iy, y0 in enumerate(ys) for ix, x0 in enumerate(xs)]
        # speedup-2026-10-28 (харвест 29.10), флаг read_ahead ([speed], по умолчанию ВЫКЛ.): следующее окно
        # читается в фоновом потоке (GDAL отпускает GIL), пока текущее идёт через сеть и полигоны. Данные и
        # порядок те же. Выключено — окно читается здесь же, после проверки дедлайна, как раньше.
        from . import speed

        ahead = _read_ahead(jobs, read_block) if speed.on("read_ahead") else None
        try:
            for iy, ix, y0, x0 in jobs:
                if guard is not None and not guard.window_ok():
                    done, total = iy * len(xs) + ix, len(xs) * len(ys)
                    log.warning("%s: дедлайн в цикле окон (%s, осталось %.0f с ≤ %s с) — частичный результат: "
                                "окон %d из %d, объектов %d", path.name, markup, guard.window_left(),
                                guard.window_reserve_s, done, total, len(feats))
                    guard.partial.append({"file": path.name, "region": region, "markup": markup,
                                          "windows_done": done, "windows_total": total, "feats": len(feats)})
                    return feats
                bw, bh, win, a = next(ahead)[1] if ahead is not None else read_block((iy, ix, y0, x0))
                if not a.any():
                    continue
                tr = ds.window_transform(win) @ rasterio.Affine.scale(win.width / bw, win.height / bh)
                img = fix_channels(to_uint8(a, markup, stats), int(seg.meta["in_channels"]))
                probs = seg.predict(img)
                zmask = None
                if raster_fix_on("zero_nodata", markup):
                    zmask = zero_field_mask(a, ds.nodata, int(RASTER_FIX.get("zero_nodata_px", 5)))
                    if not zmask.any():
                        zmask = None
                if as_centers:
                    from .centers import centers_to_features

                    found = centers_to_features(
                        probs, seg.classes, tr, crs, r, region, markup, path.name,
                        radius_m=float(seg.meta.get("center_radius_m") or 6.0),
                        min_dist_m=float(12.0 if seg.meta.get("min_dist_m") is None else seg.meta["min_dist_m"]),
                        min_conf=min_conf)
                else:
                    found = probs_to_features(probs, seg.classes, tr, crs, r, region, markup, path.name,
                                              min_conf=min_conf)
                # середина окна: соседние окна делят перекрытие пополам, объект попадает ровно в одно
                c0 = (half if ix > 0 else 0, half if iy > 0 else 0)
                c1 = (bw - (half if ix < len(xs) - 1 else 0), bh - (half if iy < len(ys) - 1 else 0))
                (ax, ay), (bx, by) = tr @ c0, tr @ c1
                core = box(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
                core = to_crs(core, crs, WEB_MERCATOR) if crs else core
                found = [f for f in found if core.contains(f["geom"].representative_point())]
                if zmask is not None and found:
                    found = _outside_zero_field(found, zmask, tr, crs)
                feats += found
        finally:
            if ahead is not None:
                ahead.close()   # дождаться фонового чтения до закрытия ds
    return feats


def _sp_dfine_phase(regions, cfg: dict, guard, state, root: Path, cache: dict | None = None) -> int:
    """bot-7 (v3): D-FINE по участкам с SpOr; drop_groups (вариант R) — только на участках, пройденных полностью."""
    from . import sp_dfine

    st = sp_dfine.settings(cfg)
    cache = {} if cache is None else cache   # bot-6: общий с dfine_aeor (модели по каталогу весов)
    n = 0
    full: list[str] = []
    for reg in regions:
        if not reg.rasters.get("sp_or"):
            continue
        if not guard.ok(60):
            log.warning("sp_dfine: времени не осталось")
            break
        if not _soft_ok(guard, "sp_dfine", f"D-FINE {reg.name}", float(st["soft_need_s"])):
            break
        try:
            found, complete = sp_dfine.region_feats(
                reg, cfg, guard, cache, root, crs_for=lambda r: read_utm_json(getattr(r, "utm_json", None)),   # как в обучении bot-9 (Scene: UTM.json), не [crs] vote
                soft_ok=lambda what, need: _soft_ok(guard, "sp_dfine", what, need))
        except sp_dfine.DFineUnavailable:
            raise   # strict: v3 не должен молча стать v2 (bot-10)
        except Exception:
            log.exception("%s: sp_dfine пропущен", reg.name)
            found, complete = [], False
        state.add(found)
        n += len(found)
        if complete:
            full.append(reg.name)
        log.info("sp_dfine n_det=%d region=%s complete=%s, прошло %.0f с", len(found), reg.name, complete,
                 guard.elapsed())
        if not cache.get("models"):
            break
    note = getattr(state, "note", None)
    info = note.setdefault("sp_dfine", {"enabled": True}) if isinstance(note, dict) else {}
    info.update({"models": len(cache.get("models") or []), "raw": n, "regions_full": len(full)})
    log.info("sp_dfine n_det=%d total, участков полностью %d, моделей %d", n, len(full), len(cache.get("models") or []))
    if not (cache.get("models") or []) and cache.get("load_error"):
        log.error("%s: фаза завершилась без моделей (%s)", sp_dfine.UNAVAILABLE, str(cache["load_error"])[:300])
    if st["drop_groups"] and full:
        before = len(state.snapshot())
        state.replace(sp_dfine.drop_groups(state.snapshot(), st["drop_groups"], full))
        log.info("sp_dfine: drop_groups %s на %d участках: −%d объектов", list(st["drop_groups"]), len(full),
                 before - len(state.snapshot()))
    return n


def _aeor_dfine_phase(regions, cfg: dict, guard, state, root: Path) -> int:
    """bot-1 (кандидат v4): D-FINE всех классов на Ae/Or (группы ae_dfine/or_dfine). ВЫКЛ. по умолчанию."""
    from . import aeor_dfine

    st = aeor_dfine.settings(cfg)
    cache: dict = {}
    found_all: list[dict] = []
    for reg in regions:
        if not any((reg.rasters or {}).get(m) for m in st["markups"]):
            continue
        if not guard.ok(60):
            log.warning("aeor_dfine: времени не осталось")
            break
        if not _soft_ok(guard, "aeor_dfine", f"D-FINE Ae/Or {reg.name}", float(st["soft_need_s"])):
            break
        try:
            found = aeor_dfine.region_feats(
                reg, cfg, guard, cache, root, crs_for=lambda r: read_utm_json(getattr(r, "utm_json", None)),
                soft_ok=lambda what, need: _soft_ok(guard, "aeor_dfine", what, need))
        except aeor_dfine.AeOrUnavailable:
            raise   # strict: включённый флаг без весов/порогов — ошибка, не тихий v3
        except Exception:
            log.exception("%s: aeor_dfine пропущен", reg.name)
            found = []
        found_all += found
    found_all = aeor_dfine.cap(found_all, int(st["max_objects"]))
    state.add(found_all)
    note = getattr(state, "note", None)
    if isinstance(note, dict):
        note.setdefault("aeor_dfine", {"enabled": True}).update({"raw": len(found_all)})
    log.info("aeor_dfine: ИТОГ объектов %d", len(found_all))
    return len(found_all)


def _outside_zero_field(found: list[dict], zmask, tr, crs: str | None) -> list[dict]:
    """[raster_fix] zero_nodata: убирает объекты, чья внутренняя точка в поле нулей окна (zmask H×W)."""
    import math

    from .crs import WEB_MERCATOR, transformer

    inv = ~tr
    t = transformer(WEB_MERCATOR, crs) if crs and crs != WEB_MERCATOR else None
    h, w = zmask.shape
    out = []
    for f in found:
        p = f["geom"].representative_point()
        x, y = t.transform(p.x, p.y) if t else (p.x, p.y)
        col, row = inv @ (x, y)
        r, c = int(math.floor(row)), int(math.floor(col))
        if 0 <= r < h and 0 <= c < w and zmask[r, c]:
            continue
        out.append(f)
    return out


def _dfine_aeor_phase(regions, cfg: dict, guard, state, root: Path, cache: dict | None = None) -> int:
    """bot-6 (на v3): D-FINE всех классов на Ae/Or, группы dfine_ae/dfine_or, markup_type родной.

    Не готов (preflight: веса/пороги/импорт) → DFINE_AEOR_UNAVAILABLE в журнале, фаза пропущена. Участки — от
    дешёвых к дорогим; мягкий потолок на каждый участок; потолок фазы [dfine_aeor] budget_s; сторож с запасом.
    """
    from . import dfine_aeor

    st = dfine_aeor.settings(cfg)
    pre = state.note.get("dfine_aeor") or {}
    if not pre.get("ok"):
        log.info("dfine_aeor n_det=0 total (не готов: %s)", "; ".join(pre.get("problems") or ["нет preflight"]))
        return 0
    cache = {} if cache is None else cache
    cache["cfg"] = cfg
    t_start = time.monotonic()
    phase_left = lambda: float(st["budget_s"]) - (time.monotonic() - t_start)   # noqa: E731
    n = 0
    found_all: list[dict] = []
    for reg in dfine_aeor.order_regions(regions, st["markups"]):
        if not guard.ok(60) or phase_left() <= 0:
            log.warning("dfine_aeor: времени не осталось (%s)", reg.name)
            break
        if not _soft_ok(guard, "dfine_aeor", f"D-FINE Ae/Or {reg.name}", float(st["soft_need_s"])):
            break
        try:
            found = dfine_aeor.region_feats(
                reg, cfg, guard, cache, root, crs_for=lambda r: read_utm_json(getattr(r, "utm_json", None)),
                soft_ok=lambda what, need: _soft_ok(guard, "dfine_aeor", what, need), phase_left=phase_left)
        except Exception:
            log.exception("%s: dfine_aeor пропущен", reg.name)
            found = []
        found_all += found
        n += len(found)
        log.info("dfine_aeor n_det=%d region=%s, прошло %.0f с", len(found), reg.name, guard.elapsed())
    found_all = dfine_aeor.cap(found_all, int(st["max_objects"]))
    state.add(found_all)
    state.note.setdefault("dfine_aeor", {}).update({"n_det": n, "kept": len(found_all),
                                                    "t_view": cache.get("t_view", {})})
    log.info("dfine_aeor n_det=%d total, после cap %d, %.0f с", n, len(found_all), time.monotonic() - t_start)
    return len(found_all)


def run(input_dir: str | Path, output_path: str | Path, config_path: str | Path | None = None,
        root: str | Path | None = None) -> dict:
    """Полный прогон: геофизика, лидар, оптика, затем канал σ.

    Круг канала σ остаётся, только если он дальше 12 м от кургана, который
    сданный порог уже оставляет. Окно 256 м отбирается так же в optical_feats.
    Файл результата пишется в конце. Сторож пишет его же незадолго до снятия
    контейнера, не после геофизики.
    """
    from .watchdog import RunState, arm_platform, arm_signal, checkpoint, deadline_left

    root = Path(root or Path(__file__).resolve().parents[1])
    cfg_path = Path(config_path or root / "configs" / "default.toml")
    cfg = load_config(cfg_path)
    from .raster import configure_raster_fix

    configure_raster_fix(cfg.get("raster_fix"))  # bot-6: правки чтения снимков, без секции — выключено
    from . import speed

    if fast := speed.configure(cfg.get("speed")):   # харвест speedup-2026-10-28: без секции — всё выкл.
        log.info("[speed] включено: %s", ", ".join(fast))
    infer = cfg.get("infer", {})
    try:
        from .infer_nn import configure_runtime
        configure_runtime(bool(infer.get("amp", False)), bool(infer.get("tf32", False)),
                          bool(infer.get("deterministic", True)))
    except Exception:
        log.exception("точность видеокарты не настроена, дальше без моделей torch")
    t_cfg = cfg.get("time", {})
    margin_s = float(t_cfg.get("guard_margin_s", 180))
    # bot-8: тот же бюджет и тот же дедлайн сторожа, что у arm_platform, — без новых часов
    from . import watchdog as _wd

    guard = TimeGuard(t_cfg.get("budget_s", DEFAULT_BUDGET_S), window_reserve(cfg),
                      hard_left=(lambda: deadline_left(margin_s)) if t_cfg.get("watchdog", True) else None)
    guard.set_soft(process_start=_wd.PROCESS_START, soft_ceiling_s=soft_ceiling_from(cfg),
                   soft_need=soft_need_from(cfg))
    if guard.soft_ceiling is not None:
        log.info("мягкий потолок %.0f с от старта процесса (прошло %.0f с); пороги: %s", guard.soft_ceiling,
                 guard.elapsed(), ", ".join(f"{k} {v:.0f} с" for k, v in guard.soft_need.items()))
    global _WINDOW_GUARD
    _WINDOW_GUARD = guard
    log.info("ускоритель: %s", accel.backend())
    torch_version, cuda_version = "нет", "нет"
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = str(torch.version.cuda or "нет")
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "torch не загрузился"
    log.info("устройство: %s", device)
    state = RunState(Path(output_path), cfg)
    state.note.update({
        "device": device,
        "torch": torch_version,
        "cuda": cuda_version,
        "backend": accel.backend(),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest() if cfg_path.is_file() else "нет файла",
        "budget_s": cfg.get("time", {}).get("budget_s", DEFAULT_BUDGET_S),
        "reserve_s": cfg.get("time", {}).get("reserve_s", 120),
    })
    if (cfg.get("sp_dfine") or {}).get("enabled"):
        # bot-10 (Murder Board 29.10): до любой работы — нет DFine*/весов/порогов → ошибка (strict), не тихий v2
        from . import sp_dfine as _spd

        state.note["sp_dfine"] = _spd.preflight(cfg, root)
        log.info("sp_dfine: preflight ok, transformers %s", state.note["sp_dfine"]["transformers"])
    if (cfg.get("aeor_dfine") or {}).get("enabled"):
        from . import aeor_dfine as _aod   # bot-1: strict preflight до работы, как у sp_dfine

        state.note["aeor_dfine"] = _aod.preflight(cfg, root)
    if (cfg.get("dfine_aeor") or {}).get("enabled"):
        from . import dfine_aeor as _dfa   # bot-6: ленивый импорт только при флаге; пороги из thresholds_file → cfg

        try:
            state.note["dfine_aeor"] = _dfa.preflight(cfg, root)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: preflight упал (%s: %s) — фаза пропущена", _dfa.UNAVAILABLE, type(exc).__name__, exc)
            state.note["dfine_aeor"] = {"ok": False, "problems": [str(exc)[:300]]}
    if cfg.get("time", {}).get("watchdog", True):
        margin = cfg.get("time", {}).get("guard_margin_s", 180)
        arm_platform(state, margin_s=float(margin))
    seg_cache: dict = {}
    summary = {"regions": 0, "raw": 0}
    prefetch = None
    phases: list[tuple[str, int, float]] = []
    regions = []
    previous_term = arm_signal(state)
    try:
        from .lidar import DemPrefetch

        regions = scan_root(input_dir)
        summary["regions"] = len(regions)
        if (cfg.get("orgate") or {}).get("enabled"):
            # bot-6 ORGATE: правило LAS и r_las — по наличию LAS во входном каталоге участка (как las_boxes_59 в трейне)
            from .orgate import CATALOG_KEY, catalog_info

            cfg[CATALOG_KEY] = catalog_info(regions)
        n_las = sum(len(r.las) for r in regions)
        by_raster: dict[str, int] = {}
        n_skipped = 0
        for reg in regions:
            n_skipped += len(getattr(reg, "skipped", ()) or ())
            for markup, files in getattr(reg, "rasters", {}).items():
                by_raster[markup] = by_raster.get(markup, 0) + len(files)
        n_rasters = sum(by_raster.values())
        raster_text = ", ".join(f"{name} {by_raster[name]}" for name in sorted(by_raster)) or "нет"
        log.info("к разбору: регионов %d, LAS %d, растров %d (%s), пропущено файлов %d",
                 len(regions), n_las, n_rasters, raster_text, n_skipped)
        try:
            from .health import log_inventory, runtime_report

            log.info("среда: %s", runtime_report(Path(output_path)))
            log_inventory(regions, cfg, root)
        except Exception:
            log.exception("отчёт о среде пропущен")
        state.note.update({
            "regions": len(regions),
            "regions_with_las": sum(1 for reg in regions if reg.las),
            "las_files": n_las,
            "rasters": raster_text,
            "skipped": n_skipped,
        })
        regions.sort(key=lambda r: (not r.las, r.name))
        depth = int(cfg.get("lidar", {}).get("prefetch", 2))
        if depth > 0:
            prefetch = DemPrefetch([(las, region_crs(r, cfg)) for r in regions for las in r.las], depth)
        t_phase = time.monotonic()
        n_phase = 0
        for reg in regions:
            try:
                found = geophys_feats(reg, cfg, guard)
            except Exception:
                # bot-8: сбой одного региона (конфиг, битый растр) не отменяет лидар и оптику
                log.exception("%s: геофизика упала, регион пропущен", reg.name)
                continue
            state.add(found)
            n_phase += len(found)
            if found:
                log.info("%s: геофизика %+d, прошло %.0f с", reg.name, len(found), guard.elapsed())
        phases.append(("геофизика", n_phase, time.monotonic() - t_phase))
        reserve = cfg.get("time", {}).get("reserve_s", 120)
        t_phase = time.monotonic()
        n_phase = 0
        optics_gb = sum(optical_bytes(r) for r in regions) / 2 ** 30
        optics_need = float(cfg.get("time", {}).get("optics_s_per_gb", 0.0)) * optics_gb
        if optics_need > 0:
            log.info("лидар оставит оптике %.0f с (%.2f ГБ снимков)", optics_need, optics_gb)
        for reg in regions:
            if not guard.ok(reserve + optics_need):
                log.warning("бюджет времени исчерпан (запас оптике %.0f с), оставшийся лидар пропущен", optics_need)
                break
            t = time.monotonic()
            try:
                found = lidar_feats(reg, cfg, guard, seg_cache, root, prefetch)
            except Exception:
                log.exception("%s: лидар пропущен", reg.name)
                found = []
            state.add(found)
            n_phase += len(found)
            log.info("%s: лидар %+d, %.1f с, прошло %.0f с", reg.name, len(found),
                     time.monotonic() - t, guard.elapsed())
        phases.append(("лидар", n_phase, time.monotonic() - t_phase))
        t_phase = time.monotonic()
        n_phase = 0
        n_before_optics = len(state.snapshot())
        optics_by_reg: dict[str, list[dict]] = {}
        for reg in optics_order(regions, cfg):
            if not guard.ok(60):
                log.warning("на оптику времени не осталось")
                break
            try:
                found = optical_feats(reg, cfg, guard, seg_cache, root)
            except Exception:
                log.exception("%s: оптика пропущена", reg.name)
                found = []
            state.add(found)
            optics_by_reg[reg.name] = found
            n_phase += len(found)
            if found:
                log.info("%s: оптика %+d, прошло %.0f с", reg.name, len(found), guard.elapsed())
        _restore_optics_order(state, regions, n_before_optics, optics_by_reg)
        dfine_cache: dict = {}   # bot-6: общий кэш моделей D-FINE (sp_dfine, dfine_aeor)
        phases.append(("оптика", n_phase, time.monotonic() - t_phase))
        if (cfg.get("sp_dfine") or {}).get("enabled"):
            # bot-7 (v3): D-FINE bot-9 на SpOr, группа sp_dfine; по умолчанию выключен, без флага ветка не входит
            t_phase = time.monotonic()
            n_phase = _sp_dfine_phase(regions, cfg, guard, state, root, dfine_cache)
            phases.append(("sp_dfine", n_phase, time.monotonic() - t_phase))
        if (cfg.get("dfine_aeor") or {}).get("enabled"):
            # bot-6: D-FINE всех классов на Ae/Or (группы dfine_ae/dfine_or); без флага ветка не входит
            t_phase = time.monotonic()
            n_phase = _dfine_aeor_phase(regions, cfg, guard, state, root, dfine_cache)
            phases.append(("dfine_aeor", n_phase, time.monotonic() - t_phase))
        if (cfg.get("aeor_dfine") or {}).get("enabled"):
            # bot-1 (v4-кандидат): D-FINE всех классов на Ae/Or; без флага ветка не входит
            t_phase = time.monotonic()
            n_phase = _aeor_dfine_phase(regions, cfg, guard, state, root)
            phases.append(("aeor_dfine", n_phase, time.monotonic() - t_phase))
        if (cfg.get("gor_dense") or {}).get("enabled"):
            # bot-5 (PFnoD бота 10): точки городищ плотного локализатора DINOv2-S на SpOr после sp_dfine/drop_groups и ДО gor_gate,
            # группа gor_dense + noD (без флага ветка не входит: выдача байт-в-байт как v3)
            t_phase = time.monotonic()
            n_phase = 0
            try:
                from . import gor_dense

                snap, n_phase = gor_dense.apply(state.snapshot(), {r.name: r for r in regions}, cfg, root, guard=guard,
                                                soft_ok=lambda what, need: _soft_ok(guard, "gor_dense", what, need))
                state.replace(snap)
            except Exception:
                log.exception("GOR_DENSE_UNAVAILABLE: gor_dense пропущен (выдача без точек и без noD)")
            phases.append(("gor_dense", n_phase, time.monotonic() - t_phase))
        if (cfg.get("gor_gate") or {}).get("enabled"):
            try:
                from . import gor_gate

                state.replace(gor_gate.apply(state.snapshot(), {r.name: r for r in regions}, cfg, root, guard=guard,
                                             soft_ok=lambda what, need: _soft_ok(guard, "gor_gate", what, need)))
            except Exception:
                log.exception("gor_gate пропущен")
        if lidar_cnx.enabled(cfg):
            # bot-8 cnx: [li_cnx] — добавочная фаза ConvNeXt-лидара bot-9 после оптики; дубли уже
            # найденных объектов участка (li, li_mask, li_fort, оптика) убираются (dedupe_cnx).
            t_phase = time.monotonic()
            n_phase = 0
            dist = float(lidar_cnx.cnx_cfg(cfg).get("dedupe_m") or 12.0)
            for reg in regions:
                if not reg.las:
                    continue
                if not guard.ok(reserve):
                    log.warning("на cnx времени не осталось")
                    break
                try:
                    fresh = cnx_site_feats(reg, cfg, guard, seg_cache, root, prefetch)
                    pool = [f for f in state.snapshot() if f.get("region_name") == reg.name]
                    kept = dedupe_cnx(fresh, pool, cfg, dist)
                except Exception:
                    log.exception("%s: cnx пропущен", reg.name)
                    kept = []
                state.add(kept)
                n_phase += len(kept)
                if kept:
                    log.info("%s: cnx %+d, прошло %.0f с", reg.name, len(kept), guard.elapsed())
            phases.append(("cnx", n_phase, time.monotonic() - t_phase))
        if guard.partial:
            state.note["partial_files"] = list(guard.partial)
            log.warning("дедлайн в цикле окон: частичный результат по %d файлам: %s", len(guard.partial),
                        ", ".join(f"{p['file']} {p['windows_done']}/{p['windows_total']}" for p in guard.partial))
        t_phase = time.monotonic()
        n_phase = 0
        for reg in regions:
            if not reg.las:
                continue
            if not guard.ok(60):
                log.warning("на канал σ времени не осталось")
                break
            if not _soft_ok(guard, "sigma", f"канал σ с {reg.name}",
                            guard.soft_need_s("verify") if (cfg.get("verify") or {}).get("enabled") else 0.0):
                break
            try:
                fresh = li_dev_feats(reg, cfg, guard, seg_cache, root, prefetch)
                pool = [f for f in state.snapshot()
                        if f.get("region_name") == reg.name and f.get("threshold_group") != "li_dev"]
                kept = drop_near_kurgans(fresh, kept_kurgan_anchors(pool), 12.0)
            except Exception:
                log.exception("%s: канал σ пропущен", reg.name)
                kept = []
            state.add(kept)
            n_phase += len(kept)
            if kept:
                log.info("%s: канал σ %+d, прошло %.0f с", reg.name, len(kept), guard.elapsed())
        phases.append(("канал σ", n_phase, time.monotonic() - t_phase))
    finally:
        _WINDOW_GUARD = None
        signal.signal(signal.SIGTERM, previous_term)
        if prefetch is not None:
            prefetch.close()
        summary["raw"] = len(state.snapshot())
        if (cfg.get("verify") or {}).get("enabled") and guard.ok(120) and _soft_ok(guard, "verify", "проверка кропа"):
            # Ошибка проверки кропа не должна оставить площадку без файла ответа.
            try:
                from .crop_model import maybe_verify

                checked = maybe_verify(state.snapshot(), {r.name: r for r in regions}, rooted_verify(cfg, root))
                if checked is not None:
                    state.replace(checked)
            except Exception:
                log.exception("проверка кропа пропущена")
        summary["phases"] = {name: {"raw": n, "seconds": round(sec, 1)} for name, n, sec in phases}
        state.note["phases"] = summary["phases"]
        if guard.soft_skipped:
            summary["soft_skipped"] = list(guard.soft_skipped)
            state.note["soft_skipped"] = list(guard.soft_skipped)
        summary["written"] = checkpoint(state, "конец")
        summary["seconds"] = round(guard.run_elapsed(), 1)
        summary["partial_files"] = list(guard.partial)
        for handler in logging.getLogger().handlers:
            handler.flush()
    return summary
