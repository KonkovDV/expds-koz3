"""Городища ↔ фортификации: арбитр «фортификация → городище» и грубый проход городищ по стеку кандидатов.

Зачем (bot-4/REPORT.md). Метрика сопоставляет только внутри класса, а маленький
полигон, целиком лежащий в эталонном городище, засчитывается городищу (≥50 % меньшей
фигуры + внутренняя точка в эталоне). Генераторы фортификаций (li_fort, li_coarse,
ae_fort) стреляют по валам городищ, а 44 из 696 эталонных фортификаций сами лежат
внутри эталонных городищ — валы и рвы размечены отдельно. Поэтому фортификацию не
перекрашиваем и не удаляем (это теряет её TP), а добавляем копию-городище:

* ``fort_to_gor`` — для каждой фортификации-кандидата круг 3 м класса gorodishcha в
  её внутренней точке, группа порога ``fort2gor``; уверенность выставляет классификатор
  по признакам формы и окружения (``features``).
* ``coarse_gor`` — грубый проход: все кандидаты городищ и фортификаций участка
  сливаются в скопления (связность в радиусе ``link_m``); на скопление — один круг
  3 м в центре, взвешенном уверенностью, группа ``gor_coarse``. Валы окружают площадку,
  поэтому центр скопления обычно внутри городища, даже если ни одна часть не
  «городище».

Обе функции не знают эталона; классификатор обучается вне этого модуля (скрипт
bot-4/work/gorfort_eval.py) с фолдами по семьям участков. Без обученной модели
уверенность — правило ``rule_score`` (не подобрано по эталону, только по смыслу).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GOR, FORT = "gorodishcha", "fortifikatsii"
GOR_GROUPS = ("gor_phys", "li_mask", "li_coarse", "li", "sp_or", "ae", "or")
FORT_GROUPS = ("ae_fort", "li_fort", "li_coarse", "li", "ae", "sp_or", "or")
_R_EARTH = 6378137.0


@dataclass
class GorFortParams:
    radius_m: float = 3.0          # выходной круг, как у gor_phys
    ctx_m: tuple = (40.0, 120.0)   # окна окружения для признаков
    link_m: float = 40.0           # связность скопления в грубом проходе
    min_members: int = 2           # скопление из одного кандидата — не грубый кандидат
    fort_groups: tuple = ("li_fort", "li_coarse", "li", "ae_fort")
    ae_floor: float = 0.9          # ae_fort очень много (61 тыс.), ниже — шум


def _group(props: dict) -> str:
    from .metric import threshold_group
    return threshold_group(props)


def _lat(y: float) -> float:
    return math.degrees(2 * math.atan(math.exp(y / _R_EARTH)) - math.pi / 2)


def table(fc: dict, p: GorFortParams | None = None) -> dict:
    """Кандидаты gor/fort → столбцы: класс, группа, участок, уверенность, внутренняя точка (3857),
    площадь м², компактность 4πS/P², вытянутость (стороны min-rect), масштаб 3857→м."""
    from shapely.geometry import shape

    p = p or GorFortParams()
    rows = []
    for i, f in enumerate(fc["features"]):
        pr = f.get("properties") or {}
        cls = pr.get("class_name")
        if cls not in (GOR, FORT):
            continue
        g = _group(pr)
        conf = float(pr.get("confidence", 0.0))
        if cls == FORT and g == "ae_fort" and conf < p.ae_floor:
            continue
        geom = shape(f["geometry"])
        if geom.is_empty:
            continue
        pt = geom.representative_point()
        k = math.cos(math.radians(_lat(pt.y)))          # метры = единицы 3857 × k
        area = geom.area * k * k
        per = max(geom.length * k, 1e-6)
        try:
            mr = geom.minimum_rotated_rectangle
            xs, ys = mr.exterior.coords.xy
            s = sorted([math.hypot(xs[1] - xs[0], ys[1] - ys[0]), math.hypot(xs[2] - xs[1], ys[2] - ys[1])])
            elong = s[1] / max(s[0], 1e-6)
        except Exception:
            elong = 1.0
        rows.append((i, cls, g, str(pr.get("region_name", "")), conf, pt.x, pt.y, k, area,
                     4 * math.pi * area / per ** 2, elong, str(pr.get("markup_type", ""))))
    cols = list(zip(*rows)) if rows else [[]] * 12
    names = ("idx", "cls", "group", "region", "conf", "x", "y", "k", "area", "compact", "elong", "markup")
    return {n: np.asarray(c, dtype=object if n in ("cls", "group", "region", "markup") else float) for n, c in zip(names, cols)}


def _ctx(t: dict, sel: np.ndarray, qx, qy, qk, qreg, radius: float) -> dict:
    """Окружение точек (qx, qy) по кандидатам t[sel] того же участка в радиусе radius м:
    число и максимум уверенности по каждой паре (класс, группа)."""
    from scipy.spatial import cKDTree

    out = {}
    keys = sorted({(c, g) for c, g in zip(t["cls"][sel], t["group"][sel])})
    for c, g in keys:
        m = sel & (t["cls"] == c) & (t["group"] == g)
        n = np.zeros(len(qx)); mx = np.zeros(len(qx))
        for reg in np.unique(qreg):
            q = qreg == reg
            mm = m & (t["region"] == reg)
            if not mm.any() or not q.any():
                continue
            tree = cKDTree(np.c_[t["x"][mm], t["y"][mm]])
            confs = t["conf"][mm]
            for j, nb in zip(np.nonzero(q)[0], tree.query_ball_point(np.c_[qx[q], qy[q]], r=radius / np.maximum(qk[q], 1e-6))):
                if nb:
                    n[j] = len(nb); mx[j] = confs[nb].max()
        out[f"n_{c[:4]}_{g}_{int(radius)}"] = n
        out[f"max_{c[:4]}_{g}_{int(radius)}"] = mx
    return out


def fort_features(t: dict, p: GorFortParams | None = None) -> tuple[np.ndarray, dict]:
    """Признаки фортификаций-кандидатов для арбитра. Возвращает (маску строк t, признаки)."""
    p = p or GorFortParams()
    q = (t["cls"] == FORT) & np.isin(t["group"], list(p.fort_groups))
    allm = np.ones(len(t["cls"]), bool)
    feats = {"conf": t["conf"][q], "log_area": np.log1p(t["area"][q]), "compact": t["compact"][q],
             "elong": np.log(t["elong"][q])}
    for g in p.fort_groups:
        feats[f"is_{g}"] = (t["group"][q] == g).astype(float)
    for r in p.ctx_m:
        feats.update(_ctx(t, allm, t["x"][q], t["y"][q], t["k"][q], t["region"][q], r))
    return q, feats


def clusters(t: dict, p: GorFortParams | None = None) -> list[np.ndarray]:
    """Скопления кандидатов участка: связные компоненты графа «ближе link_m»."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    p = p or GorFortParams()
    out = []
    for reg in np.unique(t["region"]):
        idx = np.nonzero(t["region"] == reg)[0]
        if idx.size == 0:
            continue
        k = float(np.median(t["k"][idx]))
        tree = cKDTree(np.c_[t["x"][idx], t["y"][idx]])
        pairs = tree.query_pairs(r=p.link_m / k, output_type="ndarray")
        from scipy.sparse import coo_matrix
        a = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(idx.size, idx.size)) if len(pairs) else coo_matrix((idx.size, idx.size))
        n, lab = connected_components(a, directed=False)
        for c in range(n):
            mem = idx[lab == c]
            if mem.size >= p.min_members:
                out.append(mem)
    return out


def cluster_features(t: dict, cl: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Центр (взвешенный уверенностью) и признаки каждого скопления."""
    cx = np.zeros(len(cl)); cy = np.zeros(len(cl))
    keys = [(GOR, g) for g in GOR_GROUPS] + [(FORT, g) for g in FORT_GROUPS]
    F = {f"n_{c[:4]}_{g}": np.zeros(len(cl)) for c, g in keys}
    F.update({f"max_{c[:4]}_{g}": np.zeros(len(cl)) for c, g in keys})
    F["log_n"] = np.zeros(len(cl)); F["log_extent"] = np.zeros(len(cl)); F["log_area_sum"] = np.zeros(len(cl))
    for j, mem in enumerate(cl):
        w = np.maximum(t["conf"][mem], 1e-3)
        cx[j] = np.average(t["x"][mem], weights=w); cy[j] = np.average(t["y"][mem], weights=w)
        k = float(np.median(t["k"][mem]))
        F["log_n"][j] = math.log(mem.size)
        F["log_extent"][j] = math.log1p(k * max(np.ptp(t["x"][mem]), np.ptp(t["y"][mem])))
        F["log_area_sum"][j] = math.log1p(t["area"][mem].sum())
        for c, g in keys:
            m = mem[(t["cls"][mem] == c) & (t["group"][mem] == g)]
            if m.size:
                F[f"n_{c[:4]}_{g}"][j] = m.size
                F[f"max_{c[:4]}_{g}"][j] = t["conf"][m].max()
    return cx, cy, F


def rule_score_fort(F: dict) -> np.ndarray:
    """Правило без обучения: фортификация в окружении других фортификаций (вал по кругу)
    и/или кандидатов городищ похожа на часть городища."""
    n_fort = sum(v for k, v in F.items() if k.startswith("n_fort") and k.endswith("_120"))
    gor = sum(v for k, v in F.items() if k.startswith("max_goro") and k.endswith("_120"))
    return np.clip(F["conf"] * (1 - np.exp(-n_fort / 4.0)) * 0.5 + 0.5 * np.minimum(gor, 1.0), 0, 1)


def circles(x, y, k, conf, region, group: str, radius_m: float = 3.0, markup: str = "li") -> list[dict]:
    """Круги радиуса radius_m (м) в EPSG:3857 → признаки GeoJSON класса gorodishcha."""
    from shapely.geometry import Point, mapping

    out = []
    for xi, yi, ki, ci, ri in zip(x, y, k, conf, region):
        g = Point(float(xi), float(yi)).buffer(radius_m / float(ki), 16)
        out.append({"type": "Feature", "geometry": mapping(g),
                    "properties": {"class_name": GOR, "confidence": round(float(ci), 6), "region_name": str(ri),
                                   "crs": "EPSG:3857", "markup_type": markup, "threshold_group": group}})
    return out


def nms(x, y, k, score, region, dist_m: float = 60.0) -> np.ndarray:
    """Жадное подавление: в радиусе dist_m (м) на участке остаётся кандидат с наибольшей оценкой.
    Много кругов на одном городище — один TP и остальные FP, поэтому дубли снимаем до порога."""
    from scipy.spatial import cKDTree

    keep = np.zeros(len(score), bool)
    for reg in np.unique(region):
        idx = np.nonzero(region == reg)[0]
        idx = idx[np.argsort(-score[idx], kind="stable")]
        tree = cKDTree(np.c_[x[idx], y[idx]])
        dead = np.zeros(idx.size, bool)
        for a in range(idx.size):
            if dead[a]:
                continue
            keep[idx[a]] = True
            for b in tree.query_ball_point([x[idx[a]], y[idx[a]]], r=dist_m / k[idx[a]]):
                if b != a:
                    dead[b] = True
    return keep
