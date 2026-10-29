"""Реранкер кандидатов arkhitektury (LightGBM поверх стека). По умолчанию выключен.

Источник: исследование реранка (feats.py, feats2.py, nested2.py/nested3.py, REPORT_rerank.md).
Признаки X2 (35) повторены здесь один в один, но считаются только для строк архитектуры:
у неё нет «близнецов» (twin только у курганов) и d_maxq_othersub (только курганы).
Соседи — все кандидаты участка. Метры — UTM по центроиду, как в официальной метрике.

    [rerank_arch]
    enabled = false                         # включать только после подтверждения bot-10
    model = "models/rerank_arch_lgbm.txt"   # от корня архива
    meta = "models/rerank_arch_lgbm.json"   # признаки, словари grp/mk, пороги по группам
    # arch_group_alias = { sp_or = "sp_arch" }  # по умолчанию из meta: группа сдачи → группа стека

Новая уверенность архитектуры = вероятность модели; вне «пула» (топ-15 % группы) — объект
снимается, как в nested3 (там -1). Пороги по группам — из meta; базовые пороги к этим
объектам не применяются. Любая ошибка → объекты как есть (база).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
#: Порядок классов официальной метрики (F1Calculator.target_classes) — так кодировался признак cls.
METRIC_CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury",
                  "object_poly", "fortifikatsii", "finds_points")
MODEL, META = "models/rerank_arch_lgbm.txt", "models/rerank_arch_lgbm.json"   # пути от корня архива
CLASS = "arkhitektury"
NAMES = ["conf", "logit", "cls", "grp", "mk", "area", "compact", "bw", "aspect", "nv", "has_twin", "twin_conf",
         "twin_diff", "has_las", "rg_n", "rg_rank", "rg_pct", "rg_max", "conf_rel", "rg_mean", "gq", "nb_same10",
         "nb_same30", "nb_same100", "nb_strong30", "nb_strong100", "og_max5", "og_n5", "nn_dist", "nb_otherfam50",
         "d_nhigher_g", "d_maxq_other_g", "d_localmax", "d_maxq_othersub", "d_n8"]


def _rooted(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def group_quantile(grp: np.ndarray, cls: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """gq: ранг уверенности внутри (группа, класс) в [0, 1] (feats.py)."""
    gk = np.char.add(grp.astype(str), cls.astype(str))
    _, gi = np.unique(gk, return_inverse=True)
    q = np.zeros(len(conf))
    for g in range(gi.max() + 1 if len(gi) else 0):
        ii = np.flatnonzero(gi == g)
        o = np.argsort(conf[ii], kind="stable")
        qq = np.empty(len(ii))
        qq[o] = np.arange(len(ii)) / max(len(ii) - 1, 1)
        q[ii] = qq
    return q


def pool_mask(grp: np.ndarray, cls: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Пул: всё в мелких (группа, класс) ≤ 5000, иначе топ max(1000, 15 %) по уверенности (feats2.py)."""
    pool = np.zeros(len(conf), bool)
    key = np.char.add(grp.astype(str), cls.astype(str))
    for k in np.unique(key):
        ii = np.flatnonzero(key == k)
        if len(ii) <= 5000:
            pool[ii] = True
            continue
        m = max(1000, int(0.15 * len(ii)))
        pool[ii[np.argsort(-conf[ii], kind="stable")[:m]]] = True
    return pool & (cls >= 0)


def features(T: dict, target: np.ndarray, grp_vocab: list[str], mk_vocab: list[str]) -> np.ndarray:
    """X2 для строк target. T — массивы стека: cls (индекс METRIC_CLASSES), conf, grp, mk, reg,
    area, perim, cx, cy, bw, bh, nv (метры UTM), has_las (bool на строку)."""
    from scipy.spatial import cKDTree

    cls = T["cls"].astype(int); conf = T["conf"].astype(float); grp = T["grp"].astype(str); reg = T["reg"].astype(str)
    N = len(cls); tgt = np.flatnonzero(target)
    F = {n: np.zeros(N) for n in NAMES}
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    F["conf"] = conf; F["logit"] = lg(conf); F["cls"] = cls.astype(float)
    gmap = {g: i for i, g in enumerate(grp_vocab)}; mmap = {m: i for i, m in enumerate(mk_vocab)}
    F["grp"] = np.array([gmap.get(g, np.nan) for g in grp]); F["mk"] = np.array([mmap.get(m, np.nan) for m in T["mk"].astype(str)])
    A, Pm = T["area"], T["perim"]
    F["area"] = np.log1p(A); F["compact"] = 4 * np.pi * A / np.maximum(Pm, 1e-6) ** 2
    F["bw"] = np.log1p(np.maximum(T["bw"], T["bh"]))
    F["aspect"] = np.minimum(T["bw"], T["bh"]) / np.maximum(np.maximum(T["bw"], T["bh"]), 1e-6)
    F["nv"] = np.log1p(T["nv"])
    twin = T.get("twin")
    if twin is None:                     # близнецы — только курганы; у архитектуры их нет
        twin = np.full(N, -1)
    ht = twin >= 0
    F["has_twin"] = ht.astype(float); F["twin_conf"] = np.where(ht, conf[np.maximum(twin, 0)], -1)
    F["twin_diff"] = np.where(ht, conf - conf[np.maximum(twin, 0)], 0)
    F["has_las"] = np.asarray(T["has_las"], float)
    key = np.char.add(np.char.add(np.char.add(reg, "|"), np.char.add(grp, "|")), cls.astype(str))
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv); F["rg_n"] = np.log1p(cnt[inv])
    order = np.lexsort((-conf, inv)); starts = np.r_[0, np.cumsum(cnt)[:-1]]
    rank = np.empty(N); rank[order] = np.arange(N) - starts[inv[order]]
    F["rg_rank"] = np.log1p(rank); F["rg_pct"] = rank / np.maximum(cnt[inv] - 1, 1)
    mx = np.full(len(uk), -1.0); np.maximum.at(mx, inv, conf); F["rg_max"] = mx[inv]
    F["conf_rel"] = conf / np.maximum(mx[inv], 1e-6)
    F["rg_mean"] = np.bincount(inv, conf)[inv] / cnt[inv]
    q = group_quantile(grp, cls, conf); F["gq"] = q
    fam = np.where(cls <= 1, 0, cls); strong = q >= 0.98
    XY = np.c_[T["cx"], T["cy"]]
    for r in np.unique(reg[tgt]):
        ir = np.flatnonzero(reg == r)
        for f_ in np.unique(fam[target & (reg == r)]):
            ii = ir[fam[ir] == f_]; jt = ii[target[ii]]
            tr = cKDTree(XY[ii])
            for rad, nm in ((10, "nb_same10"), (30, "nb_same30"), (100, "nb_same100")):
                F[nm][jt] = np.log1p(tr.query_ball_point(XY[jt], rad, return_length=True))
            d, _ = tr.query(XY[jt], k=min(3, len(ii)))
            d = np.atleast_2d(d)
            if len(ii) == 1:
                d = d.reshape(len(jt), 1)
            dd = np.where(d > 0.5, d, np.inf).min(1) if d.shape[1] > 1 else np.full(len(jt), np.inf)
            F["nn_dist"][jt] = np.log1p(np.minimum(dd, 1e4))
            js = ii[strong[ii]]
            if len(js):
                ts = cKDTree(XY[js])
                for rad, nm in ((30, "nb_strong30"), (100, "nb_strong100")):
                    F[nm][jt] = np.log1p(ts.query_ball_point(XY[jt], rad, return_length=True))
            for a, nb in zip(jt, tr.query_ball_point(XY[jt], 5.0)):
                if len(nb) <= 1:
                    continue
                nb = ii[np.asarray(nb)]; m = grp[nb] != grp[a]
                if m.any():
                    F["og_max5"][a] = q[nb[m]].max(); F["og_n5"][a] = len(set(grp[nb[m]]))
            js = ir[(fam[ir] != f_) & strong[ir]]
            if len(js):
                F["nb_otherfam50"][jt] = np.log1p(cKDTree(XY[js]).query_ball_point(XY[jt], 50, return_length=True))
        for c in np.unique(cls[target & (reg == r)]):
            ii = ir[cls[ir] == c]; jt = ii[target[ii]]
            tr = cKDTree(XY[ii])
            for a, nb in zip(jt, tr.query_ball_point(XY[jt], 8.0)):
                nb = np.asarray(nb, dtype=int); nb = ii[nb]; nb = nb[nb != a]
                F["d_n8"][a] = len(nb)
                if not len(nb):
                    F["d_localmax"][a] = 1
                    continue
                same = grp[nb] == grp[a]
                F["d_nhigher_g"][a] = (q[nb[same]] > q[a]).sum()
                F["d_maxq_other_g"][a] = q[nb[~same]].max() if (~same).any() else 0
                F["d_localmax"][a] = float(q[a] >= q[nb].max())
            # d_maxq_othersub — только для курганов (c in 0, 1); архитектура остаётся 0
    return np.c_[tuple(F[n][tgt] for n in NAMES)].astype(np.float32)


def table(feats: list[dict]) -> dict:
    """Объекты пайплайна (EPSG:3857) → массивы стека в метрах UTM (зона по центроиду, как в метрике)."""
    import shapely
    from pyproj import Transformer

    from .metric import threshold_group

    cid = {c: i for i, c in enumerate(METRIC_CLASSES)}
    geoms = np.array([f["geom"] for f in feats], dtype=object)
    cen = shapely.centroid(geoms)
    lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(shapely.get_x(cen), shapely.get_y(cen))
    zone = (np.floor((np.asarray(lon) + 180) / 6).astype(int) % 60) + 1
    epsg = np.where(np.asarray(lat) >= 0, 32600, 32700) + zone
    utm = np.empty(len(geoms), dtype=object)
    for e in np.unique(epsg):
        m = epsg == e
        tf = Transformer.from_crs(3857, int(e), always_xy=True)
        utm[m] = shapely.transform(geoms[m], lambda xy: np.c_[tf.transform(xy[:, 0], xy[:, 1])])
    c2 = shapely.centroid(utm); b = shapely.bounds(utm)
    reg = np.array([str(f.get("region_name", "default_source")) for f in feats])
    mk = np.array([str(f.get("markup_type") or "") for f in feats])
    li_regions = set(reg[mk == "li"])
    return dict(cls=np.array([cid.get(f["class_name"], -1) for f in feats]),
                conf=np.array([float(f.get("confidence", 1.0)) for f in feats]),
                grp=np.array([threshold_group(f) for f in feats]), mk=mk, reg=reg,
                area=shapely.area(utm), perim=shapely.length(utm), cx=shapely.get_x(c2), cy=shapely.get_y(c2),
                bw=b[:, 2] - b[:, 0], bh=b[:, 3] - b[:, 1], nv=shapely.get_num_coordinates(utm),
                # на трейне has_las = участок из notes/las_boxes_59.json; там это ровно «есть кандидаты li» (209/209)
                has_las=np.isin(reg, list(li_regions)))


def apply(feats: list[dict], opts: dict) -> tuple[list[dict], list[dict]]:
    """(остальные объекты, архитектура после реранка и своих порогов). Ошибка/нет файлов → (feats, [])."""
    try:
        import lightgbm as lgb

        mpath, jpath = _rooted(str(opts.get("model", MODEL))), _rooted(str(opts.get("meta", META)))
        if not mpath.is_file() or not jpath.is_file():
            log.warning("реранк архитектуры включён, нет файлов (%s, %s) — без реранка", mpath, jpath)
            return feats, []
        meta = json.loads(jpath.read_text(encoding="utf-8"))
        assert meta["names"] == NAMES and tuple(meta["classes"]) == METRIC_CLASSES, "meta не от этой версии признаков"
        taus = {str(g): float(t) for g, t in meta["thresholds"].items()}
        if not any(f.get("class_name") == CLASS for f in feats):
            return feats, []
        T = table(feats)
        arch = T["cls"] == METRIC_CLASSES.index(CLASS)
        alias = {**meta.get("arch_group_alias", {}), **(opts.get("arch_group_alias") or {})}
        T["grp"] = np.array([alias.get(g, g) if a else g for g, a in zip(T["grp"], arch)])
        target = (T["cls"] == METRIC_CLASSES.index(CLASS)) & np.isin(T["grp"], list(taus))
        if not target.any():
            return feats, []
        pool = pool_mask(T["grp"], T["cls"], T["conf"])
        X = features(T, target, meta["grp_vocab"], meta["mk_vocab"])
        score = lgb.Booster(model_file=str(mpath)).predict(X)
        idx = np.flatnonzero(target)
        kept = []
        for i, s in zip(idx, score):
            if pool[i] and s >= taus[T["grp"][i]]:
                kept.append(dict(feats[i], confidence=float(s), conf_stack=float(feats[i].get("confidence", 1.0))))
        drop = set(idx.tolist())
        rest = [f for k, f in enumerate(feats) if k not in drop]
        log.info("реранк архитектуры: %d кандидатов, в пуле %d, оставлено %d", len(idx), int(pool[idx].sum()), len(kept))
        return rest, kept
    except Exception:
        log.exception("реранк архитектуры упал — без реранка")
        return feats, []
