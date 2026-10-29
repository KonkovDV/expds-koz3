"""Второй проход лидарной сети на половинном разрешении (1 м) — крупные курганы.

Зачем: обучающий zoom-aug сети li — ×0,7…1,45, а курганы Эрентуны (OSINT п. 4) —
медиана диаметра 12 м, p90 18 м. Те же веса на сетке 1,0 м вместо 0,5 м видят
насыпь вдвое меньшей в пикселях, то есть внутри обученного диапазона.

Как устроено (бот 8, bench/half_res_lidar_estimate.md, и решение пользователя 29.10):
- тот же DEM, что у основного прохода (reuse_dem): каналы relief_channels(d, res=1.0);
  радиусы LRM (8/30 м) фиксированы в метрах и не меняются;
- tta = 1 (TTA8 в ×10 медленнее);
- ВТОРОЙ candidates.detect на шаге 1 м с relief_radius ≈ 30 м: детектор основного
  прохода от seg.res не зависит, и без него кандидаты те же — проход только
  пересчитал бы уверенность у уже найденных пиков;
- только классы курганов; уверенность — максимум карты в окне 5×5 вокруг пика;
- группа порога li_half (thresholds_by_markup.li_half);
- выдаются только НОВЫЕ курганы: отбор 12 м против кругов основного li,
  которые принятый порог уже оставляет (как li_dev: drop_near_kurgans + KEPT_KURGAN_CONF);
- мозаика (LAS > 60 млн ячеек на 0,1 м): проход идёт по DEM мозаики 0,5 м, которая
  строится кусками 400 м с полем 40 м (lidar.build_dem_mosaic) — перекрытие ≥ 40 м.

Выгрузка карт: EXPDS_LI_HALF_DUMP=<папка> — на каждый LAS файл
<участок>__<las>.npz (карты курганов float16 на 1 м, привязка, CRS, кандидаты
и их оценки). Работает и в pipeline, и в scripts/nn_oof.py (--predict-only):
там проход идёт только в выгрузку, основной --out не меняется.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

from . import candidates

log = logging.getLogger(__name__)

KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
DUMP_ENV = "EXPDS_LI_HALF_DUMP"

DEFAULTS = {
    "enabled": False,
    "res": 1.0,                 # шаг входа сети, м (основной проход — 0,5)
    "tta": 1,
    "reuse_dem": True,          # тот же DEM, что у основного прохода (другого пути нет)
    "classes": list(KUR),
    "max_px": 30_000_000,       # ячеек на шаге res; больше — LAS пропускается
    "skip_mosaic": False,       # True — не идти по мозаике (Эрентуна — мозаика, см. REPORT2 §11)
    "skip_coarse": True,        # v1: огрублённый DEM li_coarse (без мозаики) пропускается
    "dedupe_m": 12.0,
    "group": "li_half",
    # bot-8: время — мягким потолком pipeline (_soft_ok "li_half": [time.soft] li_half /
    # EXPDS_SOFT_NEED_LI_HALF_S, по умолчанию 30 с) и оценкой на LAS est_s(); ключ soft_need_s здесь не читается.
    # второй детектор кандидатов: шаг 1 м, окно рельефа 30 м; прочее — ×2 к [candidates]
    "cand": {"work_res": 1.0, "relief_radius_m": 30.0, "peak_radius_m": 5.0,
             "min_relief_m": 0.08, "r_in_m": 5.0, "r_out_m": 16.0, "circle_m": 2.0},
}


#: bot-8: значения enabled не-bool, о которых уже предупредили (одна строка на значение, как lidar_cnx).
_ENABLED_WARNED: set = set()


def _enabled_raw(own: dict) -> bool:
    """bot-8: enabled — только TOML-bool true (то же правило, что [li_cnx] enabled и check_submission).
    Строка "true", 1 и т. п. — выключено с предупреждением."""
    en = own.get("enabled", False)
    if isinstance(en, bool):
        return en
    key = repr(en)
    if key not in _ENABLED_WARNED:
        _ENABLED_WARNED.add(key)
        log.warning("[li_half] enabled=%s — не bool (в TOML true/false без кавычек): li_half ВЫКЛЮЧЕН", key)
    return False


def params(cfg: dict | None) -> dict:
    """[li_half] конфига поверх DEFAULTS (cand — поверх DEFAULTS["cand"]). enabled — строго bool."""
    own = dict((cfg or {}).get("li_half") or {})
    out = {**DEFAULTS, **{k: v for k, v in own.items() if k != "cand"}}
    out["cand"] = {**DEFAULTS["cand"], **(own.get("cand") or {})}
    out["enabled"] = _enabled_raw(own)
    return out


def enabled(cfg: dict | None) -> bool:
    return params(cfg)["enabled"] is True


#: bot-8: группы порогов других проходов — li_half их не берёт (своя шкала: другая сетка и детектор).
FORBIDDEN_GROUPS = frozenset({"li", "li_coarse", "li_mosaic", "li_ae", "li_cpu", "li_dev", "li_mask", "li_fort",
                              "li_cnx"})


def threshold_problem(cfg: dict | None, hp: dict | None = None) -> str | None:
    """bot-8: None — у li_half своя группа порогов на все его классы; иначе причина (объектов li_half тогда нет).
    Правило то же, что у cnx (lidar_cnx.threshold_problem)."""
    hp = hp or params(cfg)
    group = hp.get("group")
    if not group or not isinstance(group, str):
        return "li_half.group не задан"
    if group in FORBIDDEN_GROUPS:
        return f"li_half.group={group!r} — пороги другого прохода"
    own = ((cfg or {}).get("thresholds_by_markup") or {}).get(group)
    if not isinstance(own, dict) or not own:
        return f"нет [thresholds_by_markup.{group}]"
    lost = [c for c in hp.get("classes") or [] if c not in own]
    if lost:
        return f"в [thresholds_by_markup.{group}] нет порогов для {lost}"
    return None


def has_thresholds(cfg: dict | None, hp: dict | None = None) -> bool:
    """bot-8: своя группа порогов li_half на все классы прохода — без неё объектов li_half нет."""
    return threshold_problem(cfg, hp) is None


def est_s(t_main_s: float, hp: dict) -> float:
    """bot-8: оценка времени прохода на LAS: max(5, 0,35 × время основного li на этом LAS × tta прохода).
    Проход — сетка 1 м (¼ ячеек 0,5 м) и tta 1, плюс второй детектор; 0,35 — с запасом (бот 2: +2 с к 23 с)."""
    return max(5.0, 0.35 * max(0.0, float(t_main_s)) * max(1, int(hp.get("tta") or 1)))


def dump_dir() -> Path | None:
    v = os.environ.get(DUMP_ENV)
    return Path(v) if v else None


def cand_params(base: candidates.CandidateParams | None, hp: dict) -> candidates.CandidateParams:
    """Параметры второго детектора: [candidates] с заменой полей из hp["cand"]."""
    base = base or candidates.CandidateParams()
    fields = {f.name for f in dataclasses.fields(base)}
    return dataclasses.replace(base, **{k: v for k, v in hp["cand"].items() if k in fields})


def n_px(d, res: float) -> int:
    h, w = d.dem.shape
    return int(np.ceil(h * d.res / res) * np.ceil(w * d.res / res))


def _safe(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s, flags=re.UNICODE)


def half_pass(seg, d, region: str, source: str, base_cand: candidates.CandidateParams | None,
              hp: dict, dump: Path | None = None, extra: dict | None = None) -> list[dict]:
    """Один LAS: каналы на шаге hp["res"] → карта курганов → второй детектор → круги группы li_half.

    Уверенности не режутся (пол выдачи применяет вызывающий). seg.tta на время
    прохода ставится hp["tta"] и возвращается.
    """
    from .lidar import relief_channels

    if seg.meta.get("li_source") != "las" or seg.meta.get("target") == "centers":
        return []
    kur = [c for c in hp["classes"] if c in seg.classes and (getattr(seg, "keep", None) is None or c in seg.keep)]
    if not kur:
        return []
    res = float(hp["res"])
    if n_px(d, res) > int(hp["max_px"]):
        log.warning("%s/%s: li_half пропущен — %d ячеек > max_px %d", region, source, n_px(d, res), hp["max_px"])
        return []
    img, tr = relief_channels(d, res, tuple(seg.meta["li_tags"]))
    old_tta = seg.tta
    try:
        seg.tta = int(hp["tta"])
        probs = seg.predict(img)
    finally:
        seg.tta = old_tta
    del img
    kp = np.ascontiguousarray(probs[[seg.classes.index(k) for k in kur]], np.float32)
    del probs
    cp = cand_params(base_cand, hp)
    c = candidates.detect(d, cp)
    s = candidates.sample_probs(c, kp, tr)
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)
        name = f"{_safe(region)}__{_safe(Path(source).stem)}.npz"
        np.savez_compressed(
            dump / name, probs=kp.astype(np.float16), classes=np.array(kur), transform=np.array(tuple(tr)[:6]),
            crs=np.array(str(d.crs)), x=np.asarray(c["x"], np.float64), y=np.asarray(c["y"], np.float64),
            cand_feats=np.asarray(c["features"], np.float32), scores=s.astype(np.float32),
            region=np.array(region), source=np.array(source), dem_res=np.array(float(d.res)),
            meta=np.array(json.dumps({"res": res, "tta": int(hp["tta"]), "cand": dataclasses.asdict(cp),
                                      "dem_info_factor": (d.info or {}).get("factor"),
                                      "dem_tiled": bool((d.info or {}).get("tiled")), **(extra or {})},
                                     ensure_ascii=False)))
    scores = np.column_stack([np.zeros(len(s), np.float32), s])
    feats = candidates.to_features(c, scores, d.crs, region, cp, classes=tuple(kur), source=source)
    for f in feats:
        f["threshold_group"] = hp["group"]
    return feats


def new_only(half: list[dict], main_li: list[dict], hp: dict) -> list[dict]:
    """Только новые курганы: ближе dedupe_m к оставленному кругу основного li того же класса — вон."""
    from .pipeline import drop_near_kurgans, kept_kurgan_anchors

    return drop_near_kurgans(half, kept_kurgan_anchors(main_li), float(hp["dedupe_m"]))


def floor(feats: list[dict], floors: dict[str, float]) -> list[dict]:
    return [f for f in feats if float(f.get("confidence") or 0) >= floors.get(f["class_name"], 0.0)]
