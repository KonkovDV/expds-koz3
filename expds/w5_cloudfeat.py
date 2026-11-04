"""W4 v4: RR1s-cf (переранжирование li-объектов по признакам облака точек, W1) за флагом ``w5_cloudfeat``.

По умолчанию флаг выключен: ни одна функция модуля не меняет объекты и не читает файлы (выход как v3_on).

Состав (docs/PREREG_w4_v4_cloudfeat.md, гейт docs/PREREG_w4_regate_W1J2.md без M_rp):
для li-объекта класса c из групп LI_GROUPS новая уверенность
    S = α·rt + (1 − α)·RB,
где RB — эмпирическая CDF его conf среди li-строк класса c обучающей таблицы (mid-rank), rt — квантиль RB pool-строк
на уровне CDF score ранжировщика W1 среди pool-строк (outer OOF). Без score (не в pool / модуль не оценил) S = RB.
Пороги li-групп в v4_on.toml заданы в этой шкале. Референс — ``models/w5cf/ref.npz`` (отсортированные массивы, без
меток и координат).

Контракт подключаемого модуля W1 (``[w5cf] module``, по умолчанию ``expds.w5cf_infer``), без сети:
* ``region_features(reg, li_feats, cfg) -> None`` — необязательно; на проходе лидара, пока LAS участка доступен,
  кладёт в ``f["w5cf_feat"]`` признаки облака (cf_*), посчитанные из тестового LAS;
* ``score(feats, cfg) -> list[float | None]`` — в finish, по всем объектам прогона (контекст соседей/модальностей);
  возвращает score ранжировщика для каждого объекта (None — не в pool). Длина = len(feats).
Нет модуля при включённом флаге → ``W5CloudFeatMissing`` в ``preflight`` (в начале pipeline.run), с понятным текстом.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MODULE = "expds.w5cf_infer"
DEFAULT_REF = "models/w5cf/ref.npz"
LI_GROUPS = ("li", "li_dev", "li_mosaic", "li_coarse", "li_fort", "li_mask", "gor_phys")
_ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict = {}


class W5CloudFeatMissing(RuntimeError):
    """Флаг w5_cloudfeat включён, а модуля инференса признаков облака (W1) или референса нет в архиве."""


def enabled(cfg: dict) -> bool:
    return bool(cfg.get("w5_cloudfeat", False))


def spec(cfg: dict) -> dict:
    s = cfg.get("w5cf") or {}
    return {"module": s.get("module", DEFAULT_MODULE), "ref": s.get("ref", DEFAULT_REF),
            "alpha": float(s.get("alpha", 0.6)), "groups": tuple(s.get("groups", LI_GROUPS))}


def _plugin(cfg: dict):
    name = spec(cfg)["module"]
    if ("mod", name) not in _CACHE:
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            raise W5CloudFeatMissing(
                f"w5_cloudfeat=true, но модуль инференса признаков облака '{name}' не найден ({exc}). "
                "Его даёт W1 (bot 2): признаки из тестового LAS, без сети, с parity-проверкой. "
                "Соберите архив с модулем или выключите флаг (w5_cloudfeat=false).") from exc
        if not callable(getattr(mod, "score", None)):
            raise W5CloudFeatMissing(f"модуль '{name}' не реализует score(feats, cfg) -> list[float | None]")
        _CACHE[("mod", name)] = mod
    return _CACHE[("mod", name)]


def _ref(cfg: dict) -> dict:
    p = Path(spec(cfg)["ref"])
    p = p if p.is_absolute() else _ROOT / p
    if ("ref", str(p)) not in _CACHE:
        import numpy as np
        if not p.is_file():
            raise W5CloudFeatMissing(f"w5_cloudfeat=true, но нет референса шкалы {p}")
        z = np.load(p, allow_pickle=False)
        ref = {}   # только числовые 1-D массивы (условие бота 1): классы — из имён ключей "<поле>__<класс>"
        for c in sorted({k.split("__", 1)[1] for k in z.files if "__" in k}):
            ref[c] = {k: z[f"{k}__{c}"] for k in ("conf_li", "rb_pool", "r_pool")}
        _CACHE[("ref", str(p))] = ref
    return _CACHE[("ref", str(p))]


def _verify_manifest() -> None:
    """Архив: WEIGHTS_MANIFEST.json (пишет сборщик) — sha256 ref.npz и моделей ранжировщика; расхождение — ошибка."""
    import hashlib
    import json
    p = _ROOT / "WEIGHTS_MANIFEST.json"
    if not p.is_file():
        return
    for rel, v in json.loads(p.read_text(encoding="utf-8")).get("w5cf", {}).items():
        f = _ROOT / rel
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != v["sha256"]:
            raise W5CloudFeatMissing(f"w5cf: {rel} не совпадает с WEIGHTS_MANIFEST.json")


def preflight(cfg: dict) -> None:
    """В начале прогона: при включённом флаге модуль и референс обязаны быть, иначе понятная ошибка."""
    if not enabled(cfg):
        return
    _verify_manifest()
    _plugin(cfg)
    _ref(cfg)
    log.info("w5_cloudfeat: модуль %s, α=%.2f, группы %s", spec(cfg)["module"], spec(cfg)["alpha"],
             ",".join(spec(cfg)["groups"]))


def _is_target(f: dict, groups) -> bool:
    return f.get("markup_type") == "li" and (f.get("threshold_group") or "li") in groups


def region_features(reg, feats: list[dict], cfg: dict) -> list[dict]:
    """Проход лидара: признаки облака для li-объектов участка. Ошибка модуля — лог, объекты без признаков."""
    if not enabled(cfg):
        return feats
    fn = getattr(_plugin(cfg), "region_features", None)
    if fn is None:
        return feats
    groups = spec(cfg)["groups"]
    try:
        fn(reg, [f for f in feats if _is_target(f, groups)], cfg)
    except Exception:
        log.exception("w5_cloudfeat: признаки облака участка %s не посчитаны", getattr(reg, "name", "?"))
    return feats


def _cdf(sorted_vals, x):
    """mid-rank эмпирическая CDF, как (rankdata(average) − 0.5) / n на обучающей таблице."""
    import numpy as np
    n = len(sorted_vals)
    lo = np.searchsorted(sorted_vals, x, side="left"); hi = np.searchsorted(sorted_vals, x, side="right")
    return (lo + hi) / 2.0 / n


def blend_scores(cls: str, conf, r, cfg: dict):
    """S для li-объектов одного класса: conf и r (nan — нет score) — массивы. Без референса класса — conf как есть."""
    import numpy as np
    conf = np.asarray(conf, float); r = np.asarray(r, float)
    ref = _ref(cfg).get(cls)
    if ref is None or len(ref["conf_li"]) == 0:
        return conf
    a = spec(cfg)["alpha"]
    rb = _cdf(ref["conf_li"], conf)
    s = rb.copy()
    q, rp = ref["rb_pool"], ref["r_pool"]
    m = np.isfinite(r)
    if m.any() and len(rp) >= 2:
        # ordinal-ранг score среди pool → тот же индекс в отсортированных RB pool
        idx = np.clip(np.searchsorted(rp, r[m], side="right") - 1, 0, len(q) - 1)
        s[m] = a * q[idx] + (1 - a) * rb[m]
    return s


def apply(feats: list[dict], cfg: dict) -> list[dict]:
    """finish(), до порогов: новая уверенность li-объектов. Флаг выкл. — тот же список без изменений."""
    if not enabled(cfg):
        return feats
    import numpy as np
    groups = spec(cfg)["groups"]
    scores = list(_plugin(cfg).score(feats, cfg))
    if len(scores) != len(feats):
        raise ValueError(f"w5_cloudfeat: score вернул {len(scores)} значений на {len(feats)} объектов")
    by_cls: dict[str, list[int]] = {}
    for i, f in enumerate(feats):
        if _is_target(f, groups):
            by_cls.setdefault(f["class_name"], []).append(i)
    out = [dict(f) for f in feats]
    for f in out:
        f.pop("w5cf_feat", None)
    for c, ii in by_cls.items():
        conf = [float(feats[i].get("confidence", 1.0)) for i in ii]
        r = [np.nan if scores[i] is None else float(scores[i]) for i in ii]
        for i, s in zip(ii, blend_scores(c, conf, r, cfg)):
            out[i]["confidence"] = float(s)
    log.info("w5_cloudfeat: переоценено li-объектов %d (со score ранжировщика %d)",
             sum(len(v) for v in by_cls.values()),
             sum(1 for v in by_cls.values() for i in v if scores[i] is not None))
    return out


def li_floor(cfg: dict, markup: str, cls: str, thr: float) -> float:
    """Ранний пол li-группы в шкале conf при включённом флаге: порог thr задан в шкале S, а S ≥ thr достижимо
    при RB ≥ max(0, (thr − α)/(1 − α)) → пол = квантиль conf_li класса на этом уровне. Флаг выкл. — thr как есть."""
    if not enabled(cfg) or markup not in spec(cfg)["groups"]:
        return thr
    import numpy as np
    ref = _ref(cfg).get(cls)
    if ref is None or len(ref["conf_li"]) == 0:
        return thr
    a = spec(cfg)["alpha"]
    need = max(0.0, (thr - a) / (1 - a)) if a < 1 else 0.0
    if need <= 0:
        return 0.0
    v = ref["conf_li"]; k = int(np.floor(need * len(v)))
    return float(v[min(max(k - 1, 0), len(v) - 1)]) if k > 0 else 0.0
