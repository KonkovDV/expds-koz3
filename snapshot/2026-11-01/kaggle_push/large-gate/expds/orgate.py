"""bot-6 ORGATE G: гейт FP архитектуры (or / sp_arch) — ЗАМЕНА rerank_arch (R), не добавка. По умолчанию выключен.

Замер: bot-6/ORGATE.md, PREREG_ORGATE.md (вариант G — предрегистрированный, с правилом LAS); приёмка
bot-10/REPORT12_orgate.md (PACKAGE-ONLY как замена R; k10 dR +0,001933, P_R 0,993; семена 5/5).

    [orgate]
    enabled = false
    model = "models/orgate_g.txt"    # LightGBM, финальная подгонка G на всех семьях (bot-6/work/orgate_final/fit.py)
    meta = "models/orgate_g.json"    # признаки (35 X2 + 22 гейта), словари, пороги групп, правило LAS

Место в цепочке — то же, что у R (watchdog._finish_v3 / _safe_v3): после only_known_classes, ДО site_calibration,
kurgan_neighbour_prior, frame_prior (FR1h) и порогов; при enabled R не применяется (гейт его заменяет).
Объекты dfine_ae/dfine_or сюда не попадают (отдельная цепочка _aeor_split), группу sp_dfine и любые другие
гейт не оценивает: цель — только arkhitektury групп из meta.thresholds (or, sp_arch; sp_or → sp_arch).
Обучение — на стеке rerank/work/table.pkl общей папки (стек A, без sp_dfine/dfine_aeor); при смене стека
кандидатов arch гейт нужно переобучить и перемерить (REPORT12 §3, оговорка 3).

Признаки 1:1 с bot-6/work/orgate/feats_gate.py (трейн) и rerank_arch.features (X2). Правило LAS и r_las —
по наличию LAS во входном каталоге участка (pipeline.run кладёт cfg["_orgate_catalog"] = {участок: {las, utm_json}};
на трейне это ровно notes/las_boxes_59.json — проверка bot-6/work/orgate_final/equiv.py). Без каталога (например,
вызов finish вне pipeline.run) — запасное правило rerank_arch: «у участка есть кандидаты li» (на трейне 209/209 то же).
Широта/долгота — как в трейне: CRS из UTM.json участка (если в нём не 4326), иначе NaN.

Нет файлов / meta не той версии / ошибка → OrGateUnavailable; вызывающий пишет маркер ORGATE_UNAVAILABLE в лог
и откатывается на R (если [rerank_arch] enabled) или базу. check_submission на маркер даёт FAIL.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from . import rerank_arch as R

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL, META = "models/orgate_g.txt", "models/orgate_g.json"
MARKER = "ORGATE_UNAVAILABLE"
CATALOG_KEY = "_orgate_catalog"
GATE_NAMES = ["sh_pp", "sh_rect", "ch_oth5_max", "ch_oth5_n", "ch_oth15_max", "ch_oth15_n",
              "kur15_n", "kur30_n", "kur15_max", "gor30_max", "fort30_max", "li15_n",
              "r_las", "r_fli", "r_fae", "r_fgeo", "r_ncls", "r_km2", "r_arch_dens", "r_arch_hi", "lat", "lon"]


class OrGateUnavailable(RuntimeError):
    pass


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("orgate") or {}).get("enabled"))


def catalog_info(regions) -> dict:
    """Region из catalog.scan_root → {имя: {"las": есть ли LAS, "utm_json": путь|None}}."""
    return {r.name: {"las": bool(r.las), "utm_json": Path(r.utm_json).as_posix() if getattr(r, "utm_json", None) else None}
            for r in regions}


def _lat_lon(utm_json: str | None, x: float, y: float) -> tuple[float, float]:
    """Как feats_gate.py: CRS из UTM.json (если файл есть и в нём не 4326) → градусы; иначе NaN."""
    if not utm_json or not Path(utm_json).is_file():
        return np.nan, np.nan
    text = Path(utm_json).read_text(encoding="utf-8", errors="replace")
    if "4326" in text:
        return np.nan, np.nan
    import pyproj

    crs = json.loads(text)["crs"].split("::")[-1]
    lon, lat = pyproj.Transformer.from_crs(f"EPSG:{crs}", "EPSG:4326", always_xy=True).transform(x, y)
    return float(lat), float(lon)


def gate_features(T: dict, target: np.ndarray, las_of: dict, utm_of: dict) -> np.ndarray:
    """22 признака гейта для строк target (порядок GATE_NAMES), повтор feats_gate.py.
    T — rerank_arch.table() (с алиасом групп arch), las_of/utm_of — по имени участка."""
    from scipy.spatial import cKDTree

    cid = {c: i for i, c in enumerate(R.METRIC_CLASSES)}
    cls = T["cls"].astype(int); conf = T["conf"].astype(np.float32); reg = T["reg"].astype(str)
    grp = T["grp"].astype(str); mk = T["mk"].astype(str); XY = np.c_[T["cx"], T["cy"]]
    tgt = np.flatnonzero(target)
    pos = {int(p): k for k, p in enumerate(tgt)}
    F = np.zeros((tgt.size, len(GATE_NAMES)), np.float32); col = {n: i for i, n in enumerate(GATE_NAMES)}
    ar = np.maximum(T["area"][tgt], 1e-6)
    F[:, 0] = T["perim"][tgt] / np.sqrt(ar); F[:, 1] = T["bw"][tgt] * T["bh"][tgt] / ar
    KUR = [cid["kurgany_povrezhdennye"], cid["kurgany_tselye"]]; G = cid["gorodishcha"]; FO = cid["fortifikatsii"]

    def qmax(idx, pts, r):
        if idx.size == 0:
            return np.zeros(len(pts)), np.zeros(len(pts))
        L = cKDTree(XY[idx]).query_ball_point(pts, r)
        return (np.array([conf[idx[l]].max() if l else 0 for l in L], np.float32),
                np.array([len(l) for l in L], np.float32))

    for r in np.unique(reg[tgt]):
        ir = np.flatnonzero(reg == r); ia = ir[target[ir]]
        rows = np.array([pos[int(p)] for p in ia]); P = XY[ia]
        for g in np.unique(grp[ia]):
            sel = grp[ia] == g; oth = ia[grp[ia] != g]
            for rad, a, b in ((5, "ch_oth5_max", "ch_oth5_n"), (15, "ch_oth15_max", "ch_oth15_n")):
                m, n = qmax(oth, P[sel], rad); F[rows[sel], col[a]] = m; F[rows[sel], col[b]] = n
        ik = ir[np.isin(cls[ir], KUR)]
        F[rows, col["kur15_max"]], F[rows, col["kur15_n"]] = qmax(ik, P, 15); F[rows, col["kur30_n"]] = qmax(ik, P, 30)[1]
        F[rows, col["gor30_max"]] = qmax(ir[cls[ir] == G], P, 30)[0]; F[rows, col["fort30_max"]] = qmax(ir[cls[ir] == FO], P, 30)[0]
        F[rows, col["li15_n"]] = qmax(ir[mk[ir] == "li"], P, 15)[1]
        F[rows, col["r_las"]] = float(las_of[r])
        F[rows, col["r_fli"]] = np.mean(mk[ir] == "li"); F[rows, col["r_fae"]] = np.mean(mk[ir] == "ae")
        F[rows, col["r_fgeo"]] = np.mean(np.isin(mk[ir], ["maggr", "etgr", "terfl"]))
        F[rows, col["r_ncls"]] = len(np.unique(cls[ir][cls[ir] >= 0]))
        xy = XY[ir]; km2 = max((np.ptp(xy[:, 0]) * np.ptp(xy[:, 1])) / 1e6, 1e-3)
        F[rows, col["r_km2"]] = km2; F[rows, col["r_arch_dens"]] = ia.size / km2
        F[rows, col["r_arch_hi"]] = np.mean(conf[ia] >= 0.2)
        mx, my = XY[ia].mean(0)
        F[rows, col["lat"]], F[rows, col["lon"]] = _lat_lon(utm_of.get(r), mx, my)
    return F


def load(opts: dict):
    """(Booster, meta). Любая нестыковка → OrGateUnavailable."""
    mpath, jpath = R._rooted(str(opts.get("model", MODEL))), R._rooted(str(opts.get("meta", META)))
    if not mpath.is_file() or not jpath.is_file():
        raise OrGateUnavailable(f"нет файлов гейта: {mpath} / {jpath}")
    meta = json.loads(jpath.read_text(encoding="utf-8"))
    if meta.get("names") != R.NAMES + GATE_NAMES or tuple(meta.get("classes", ())) != R.METRIC_CLASSES:
        raise OrGateUnavailable("meta гейта не от этой версии признаков")
    try:
        import lightgbm as lgb

        bst = lgb.Booster(model_file=str(mpath))
    except Exception as e:
        raise OrGateUnavailable(f"LightGBM не загрузил {mpath}: {e}") from e
    if bst.num_feature() != len(meta["names"]):
        raise OrGateUnavailable(f"в модели {bst.num_feature()} признаков, в meta {len(meta['names'])}")
    return bst, meta


def decide_table(T: dict, opts: dict, catalog: dict | None, model=None) -> dict:
    """Ядро гейта на массивах стека T (rerank_arch.table или table.pkl трейна). Возвращает
    target (индексы строк), kept_idx, removed_idx, score (по target; −1 = правило LAS), las_regions, las_source.
    gq и пул (rerank_arch.group_quantile / pool_mask) считаются по всему входу — как у R."""
    bst, meta = model if model is not None else load(opts)
    taus = {str(g): float(t) for g, t in meta["thresholds"].items()}
    info = dict(target=[], kept_idx=[], removed_idx=[], score=[], las_regions=[], las_source=None)
    arch = T["cls"] == R.METRIC_CLASSES.index(R.CLASS)
    alias = {**meta.get("arch_group_alias", {}), **(opts.get("arch_group_alias") or {})}
    if alias:
        T = dict(T, grp=np.array([alias.get(g, g) if a else g for g, a in zip(T["grp"], arch)]))
    target = arch & np.isin(T["grp"], list(taus))
    if not target.any():
        return info
    regs = sorted(set(T["reg"][target].tolist()))
    if catalog:
        las_of = {r: bool((catalog.get(r) or {}).get("las")) for r in regs}
        utm_of = {r: (catalog.get(r) or {}).get("utm_json") for r in regs}
        info["las_source"] = "catalog"
    else:   # запасное правило rerank_arch (на трейне совпадает с las_boxes_59: 209/209)
        las_of = {r: bool(T["has_las"][T["reg"] == r].any()) for r in regs}
        utm_of = {}
        info["las_source"] = "li_candidates"
    las_set = {r for r, v in las_of.items() if v}
    T = dict(T, has_las=np.isin(T["reg"], list(las_set)) if catalog else np.asarray(T["has_las"], bool))
    pool = R.pool_mask(T["grp"], T["cls"], T["conf"])
    X = np.c_[R.features(T, target, meta["grp_vocab"], meta["mk_vocab"]), gate_features(T, target, las_of, utm_of)]
    score = bst.predict(X)
    idx = np.flatnonzero(target)
    las_row = np.isin(T["reg"][idx], list(las_set))
    score = np.where(las_row, -1.0, score)                       # правило LAS: arch на участке с LAS → −1
    keep = pool[idx] & (score >= np.array([taus[g] for g in T["grp"][idx]]))
    info.update(target=idx.tolist(), kept_idx=idx[keep].tolist(), removed_idx=idx[~keep].tolist(),
                score=score.tolist(), las_regions=sorted(las_set))
    return info


_CACHE: dict = {}


def _cached_load(opts: dict):
    """load() один раз на процесс для пары путей (finish зовётся на каждый участок)."""
    key = (str(R._rooted(str(opts.get("model", MODEL)))), str(R._rooted(str(opts.get("meta", META)))))
    if key not in _CACHE:
        _CACHE[key] = load(opts)
    return _CACHE[key]


def decide(feats: list[dict], opts: dict, catalog: dict | None, model=None):
    """Решения гейта на объектах пайплайна: (rest, kept, info). kept — arch, прошедшие гейт (confidence = скор гейта,
    conf_stack — исходный), базовые пороги к ним не применяются (как к выходу R)."""
    if not any(f.get("class_name") == R.CLASS for f in feats):
        return feats, [], dict(target=[], kept_idx=[], removed_idx=[], score=[], las_regions=[], las_source=None)
    info = decide_table(R.table(feats), opts, catalog, model)
    if not info["target"]:
        return feats, [], info
    sc = dict(zip(info["target"], info["score"]))
    kept = [dict(feats[i], confidence=float(sc[i]), conf_stack=float(feats[i].get("confidence", 1.0)))
            for i in info["kept_idx"]]
    drop = set(info["target"])
    rest = [f for k, f in enumerate(feats) if k not in drop]
    log.info("ORGATE G: кандидатов arch %d, LAS-участков %d (%s), оставлено %d, удалено %d", len(drop),
             len(info["las_regions"]), info["las_source"], len(kept), len(info["removed_idx"]))
    return rest, kept, info


def apply_or_fallback(feats: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Шаг цепочки finish: гейт вместо R. Недоступен → маркер в лог и R (если включён) / база."""
    opts = cfg.get("orgate") or {}
    try:
        # модель грузим всегда (и на участке без arch): недоступный гейт должен давать маркер в логе в любом прогоне
        model = _cached_load(opts)
        rest, kept, _ = decide(feats, opts, cfg.get(CATALOG_KEY), model)
        return rest, kept
    except Exception as e:   # noqa: BLE001 — любой сбой гейта не должен ронять сдачу
        log.error("%s: %s — откат на rerank_arch/базу", MARKER, e)
    if (cfg.get("rerank_arch") or {}).get("enabled"):
        return R.apply(feats, cfg["rerank_arch"])
    return feats, []
