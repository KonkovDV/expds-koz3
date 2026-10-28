"""bot-8: быстрые версии метрики против прежних (_reference) — совпадение побайтно.

Проверяется: пакетное построение геометрий (Rust flatten_polygons + from_ragged_array)
против официальной _extract_geometry (WKB байт в байт), _collect против
_collect_reference, build_curves, sweep_groups, honest_counts, bootstrap,
paired_bootstrap, paired_bootstrap_counts (точное равенство всех чисел и порядка
ключей). Данные: синтетика с краевыми случаями и реальные выборки бота 8
(если есть). Нужен официальный compute_metrics.py: EXPDS_METRIC или materials/.
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import shapely

from expds import fastgeom, metric

ROOT = Path(__file__).resolve().parents[1]
METRIC = os.environ.get("EXPDS_METRIC") or str(ROOT / "materials" / "compute_metrics.py")
B8 = Path("/workspace/koz3_shared/bot-8")
REAL_PRED = B8 / "data" / "stack_sub.geojson"
REAL_GT = B8 / "data" / "gt_sub.geojson"
REAL_PRED20 = B8 / "data" / "stack_every20.geojson"
REAL_GT_FULL = Path("/workspace/rerank/gt_train_v2.geojson")

pytestmark = pytest.mark.skipif(not Path(METRIC).exists(), reason="нет compute_metrics.py")


@pytest.fixture(autouse=True)
def _no_shm(tmp_path, monkeypatch):
    monkeypatch.setattr(metric, "_SHM_CACHE_DIR", tmp_path / "shm")
    metric._UTM_GEOM_CACHE.clear()
    yield
    metric._UTM_GEOM_CACHE.clear()


def mod():
    return metric.load_official(METRIC)


def same_geom(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.geom_type == b.geom_type and shapely.to_wkb(a) == shapely.to_wkb(b)


def canon(x):
    """JSON с порядком ключей — для сравнения «и значения, и порядок»."""
    return json.dumps(x, sort_keys=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# ───────────────────────────── геометрии ─────────────────────────────

X0, Y0 = 4_200_000.0, 7_400_000.0   # EPSG:3857, где-то в Центральной России


def ring(cx, cy, r, n=12, close=True, rot=0.0):
    pts = [[cx + r * math.cos(rot + 2 * math.pi * i / n), cy + r * math.sin(rot + 2 * math.pi * i / n)]
           for i in range(n)]
    return pts + [pts[0]] if close else pts


def edge_geoms():
    nan = float("nan")
    sq = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    sqx = [[X0 + x, Y0 + y] for x, y in sq]
    return [
        {"type": "Polygon", "coordinates": [ring(X0, Y0, 5)]},
        {"type": "Polygon", "coordinates": [ring(X0, Y0, 5, close=False)]},          # незамкнутое кольцо
        {"type": "Polygon", "coordinates": [ring(X0, Y0, 10), ring(X0, Y0, 3)[::-1]]},  # с дыркой
        {"type": "Polygon", "coordinates": [ring(X0, Y0, 10), ring(X0, Y0, 3, close=False)]},
        {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + 10, Y0 + 10], [X0 + 10, Y0], [X0, Y0 + 10], [X0, Y0]]]},  # бабочка
        {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + 1, Y0], [X0, Y0]]]},       # 3 точки (замкнуто) — ошибка
        {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + 1, Y0], [X0 + 1, Y0 + 1]]]},  # 3 точки → замыкается до 4
        {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + 1, Y0]]]},
        {"type": "Polygon", "coordinates": [[[X0, Y0]] * 5]},                           # вырожденный (одна точка)
        {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + 5, Y0], [X0 + 10, Y0], [X0, Y0]]]},  # нулевая площадь
        {"type": "Polygon", "coordinates": [[[int(X0), int(Y0)], [int(X0) + 7, int(Y0)], [int(X0) + 7, int(Y0) + 7],
                                             [int(X0), int(Y0)]]]},                     # целые числа
        {"type": "Polygon", "coordinates": [[(X0, Y0), (X0 + 3, Y0), (X0 + 3, Y0 + 3), (X0, Y0)]]},  # кортежи
        {"type": "Polygon", "coordinates": [[[np.float64(X0), np.float64(Y0)], [X0 + 3, Y0], [X0, Y0 + 3]]]},
        {"type": "Polygon", "coordinates": [[[X0, Y0, 1.0], [X0 + 3, Y0, 1.0], [X0, Y0 + 3, 1.0], [X0, Y0, 1.0]]]},  # 3D
        {"type": "Polygon", "coordinates": [[[True, False], [X0, Y0], [X0 + 1, Y0 + 1], [True, False]]]},  # bool
        {"type": "Polygon", "coordinates": [[["a", "b"], [1, 2], [3, 4], ["a", "b"]]]},
        {"type": "Polygon", "coordinates": [[[nan, nan], [X0, Y0], [X0 + 1, Y0], [nan, nan]]]},
        {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + 4, Y0], [X0 + 4, Y0 + 4], [X0, Y0 + 4], [nan, Y0]]]},
        {"type": "Polygon", "coordinates": []},
        {"type": "Polygon", "coordinates": [[]]},
        {"type": "Polygon", "coordinates": [sqx, []]},
        {"type": "Polygon", "coordinates": [sqx, [[X0 + 20, Y0 + 20], [X0 + 21, Y0 + 20], [X0 + 21, Y0 + 21], [X0 + 20, Y0 + 20]]]},  # дыра снаружи — невалидно
        {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + 10, Y0], [X0 + 10, Y0 + 10], [X0 + 5, Y0], [X0, Y0 + 10], [X0, Y0]]]},  # самокасание
        {"type": "Polygon", "coordinates": [sq]},                                       # маленькие координаты (не 3857)
        {"type": "Polygon", "coordinates": [[[1e300, 1e300], [-1e300, 1e300], [0, -1e300], [1e300, 1e300]]]},
        {"type": "Polygon", "coordinates": "abc"},
        {"type": "Polygon"},
        {"type": "polygon", "coordinates": [ring(X0, Y0, 5)]},
        {"type": "MultiPolygon", "coordinates": [[ring(X0, Y0, 5)], [ring(X0 + 50, Y0, 8)]]},
        {"type": "MultiPolygon", "coordinates": [[ring(X0, Y0, 5)], [ring(X0 + 50, Y0, 5, rot=0.0)]]},   # равные площади
        {"type": "Point", "coordinates": [X0, Y0]},
        {"type": "Point", "coordinates": [X0]},
        {"type": "LineString", "coordinates": [[X0, Y0], [X0 + 1, Y0]]},
        None,
        {},
        {"type": "Polygon", "coordinates": [ring(X0, Y0, 5)], "extra": 1},
        {"type": "Polygon", "coordinates": ([[X0, Y0], [X0 + 2, Y0], [X0, Y0 + 2], [X0, Y0]],)},   # кортеж колец
    ]


def test_extract_many_edge_cases_bitwise():
    if not fastgeom.enabled():
        pytest.skip("expds_core без flatten_polygons")
    feats = [{"type": "Feature", "geometry": g, "properties": {}} for g in edge_geoms()]
    m = mod()
    fast = fastgeom.extract_many(m, feats)
    for f, g in zip(feats, fast):
        assert same_geom(g, m._extract_geometry(f)), f["geometry"]


def test_extract_many_random_polygons_bitwise():
    if not fastgeom.enabled():
        pytest.skip("expds_core без flatten_polygons")
    rng = np.random.default_rng(1)
    feats = []
    for i in range(3000):
        n = int(rng.integers(3, 40))
        r = rng.uniform(0.5, 30, n)            # «звёзды» — часть самопересекается
        ang = np.sort(rng.uniform(0, 2 * np.pi, n)) if i % 3 else rng.uniform(0, 2 * np.pi, n)
        pts = np.c_[X0 + r * np.cos(ang), Y0 + r * np.sin(ang)].tolist()
        if i % 2:
            pts.append(pts[0])
        rings = [pts]
        if i % 5 == 0:
            rings.append(ring(X0, Y0, 0.3, close=bool(i % 10)))
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": rings}})
    m = mod()
    fast = fastgeom.extract_many(m, feats)
    n_invalid = sum(1 for f in feats if m._extract_geometry(f) is not None
                    and not shapely.is_valid(shapely.Polygon(f["geometry"]["coordinates"][0])))
    assert n_invalid > 100   # проверка действительно задевает make_valid
    for f, g in zip(feats, fast):
        assert same_geom(g, m._extract_geometry(f))


def test_max_abs_x_matches_official():
    m = mod()
    geoms = [m._extract_geometry({"geometry": g}) for g in edge_geoms()]
    geoms = [g for g in geoms if g is not None]
    vec = fastgeom.max_abs_x(geoms)
    ref = [m._geom_max_abs_x(g) for g in geoms]
    for a, b in zip(vec.tolist(), ref):
        assert (math.isnan(a) and math.isnan(b)) or a == b


def test_parallel_utm_transform_bitwise(monkeypatch):
    m = mod()
    rng = np.random.default_rng(3)
    c = np.c_[rng.uniform(4.0e6, 4.4e6, 300_000), rng.uniform(7.2e6, 7.6e6, 300_000)]
    tr = m._get_transformer_3857_to_utm(32637)
    ref = np.column_stack(tr.transform(c[:, 0], c[:, 1]))
    monkeypatch.setenv("EXPDS_METRIC_THREADS", "4")
    out = fastgeom.utm_coords_fn(tr, 32637, min_chunk=10_000)(c)
    assert out.tobytes() == ref.tobytes()


# ───────────────────────────── _collect / build_curves ─────────────────────────────

def synth_fc(seed: int, n_pred: int = 400, gt: bool = False):
    rng = random.Random(seed)
    classes = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "arkhitektury", "finds_points",
               "fortifikatsii", "object_poly", "Курганы", "unknown_class"]
    geoms = edge_geoms()
    feats = []
    for i in range(n_pred):
        reg = f"region_{rng.randint(1, 6):04d}"
        cx, cy = X0 + rng.uniform(0, 300), Y0 + rng.uniform(0, 300)
        k = rng.random()
        if k < 0.6:
            g = {"type": "Polygon", "coordinates": [ring(cx, cy, rng.uniform(1, 9), n=rng.randint(4, 20),
                                                         close=rng.random() < 0.8)]}
        elif k < 0.75:
            g = {"type": "Point", "coordinates": [cx, cy]}
        elif k < 0.85:
            g = {"type": "MultiPolygon", "coordinates": [[ring(cx, cy, 3)], [ring(cx + 20, cy, rng.uniform(1, 5))]]}
        else:
            g = geoms[rng.randrange(len(geoms))]
        props = {"class_name": rng.choice(classes), "region_name": reg,
                 "markup_type": rng.choice(["li", "ae", "op", None])}
        if rng.random() < 0.3:
            props["threshold_group"] = rng.choice(["li_a", "li_b", ""])
        if not gt:
            props["confidence"] = rng.choice([0.5, 0.25, 0.9, rng.random(), "bad", None, 1])
        feats.append({"type": "Feature", "geometry": g, "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def collect_both(fc, is_gt):
    m = mod()
    calc = m.F1Calculator()
    metric._UTM_GEOM_CACHE.clear()
    new = metric._collect(m, calc, fc, is_gt)
    metric._UTM_GEOM_CACHE.clear()
    ref = metric._collect_reference(m, calc, fc, is_gt)
    return new, ref


def assert_collect_equal(new, ref):
    assert list(new) == list(ref)
    for cls in ref:
        assert len(new[cls]) == len(ref[cls])
        for (g1, c1, r1, p1), (g2, c2, r2, p2) in zip(new[cls], ref[cls]):
            assert (c1 == c2 or (c1 != c1 and c2 != c2)) and r1 == r2 and p1 == p2
            assert same_geom(g1, g2)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("is_gt", [False, True])
def test_collect_synthetic(seed, is_gt):
    assert_collect_equal(*collect_both(synth_fc(seed, 500, is_gt), is_gt))


def test_collect_empty():
    assert_collect_equal(*collect_both({"type": "FeatureCollection", "features": []}, False))
    assert_collect_equal(*collect_both({}, True))


def build_both(pred, gt):
    metric._UTM_GEOM_CACHE.clear()
    new = metric.build_curves(pred, gt, METRIC)
    metric._UTM_GEOM_CACHE.clear()
    orig = metric._collect
    try:
        metric._collect = metric._collect_reference
        ref = metric.build_curves(pred, gt, METRIC)
    finally:
        metric._collect = orig
    return new, ref


def assert_curves_equal(a, b):
    assert list(a) == list(b)
    for cls in b:
        x, y = a[cls], b[cls]
        assert x.conf.tobytes() == y.conf.tobytes() and x.hit.tobytes() == y.hit.tobytes()
        assert x.pred_region == y.pred_region and x.pred_group == y.pred_group
        assert x.n_gt == y.n_gt and list(x.gt_regions.items()) == list(y.gt_regions.items())


@pytest.mark.parametrize("seed", [0, 5])
def test_build_curves_synthetic(seed):
    assert_curves_equal(*build_both(synth_fc(seed, 600), synth_fc(seed + 100, 300, gt=True)))


def test_build_curves_empty_sides():
    empty = {"type": "FeatureCollection", "features": []}
    pred, gt = synth_fc(7, 200), synth_fc(8, 100, gt=True)
    for p, g in ((empty, gt), (pred, empty), (empty, empty)):
        assert_curves_equal(*build_both(p, g))


def _load(p):
    return json.loads(Path(p).read_bytes())


@pytest.mark.skipif(not (REAL_PRED.exists() and REAL_GT.exists()), reason="нет реальной выборки бота 8")
def test_build_curves_real_subset_and_official():
    pred, gt = _load(REAL_PRED), _load(REAL_GT)
    new, ref = build_both(pred, gt)
    assert_curves_equal(new, ref)
    assert sum(int(c.hit.sum()) for c in new.values()) > 0


# ───────────────────────────── пороги, честный балл, бутстрепы ─────────────────────────────

def synth_curves(seed: int, n: int = 3000, n_regions: int = 25, ties: bool = True):
    rng = np.random.default_rng(seed)
    regions = [f"r{i:03d}" for i in range(n_regions)]
    groups = ["li", "ae", "op", "li_b"]
    out = {}
    for ci, cls in enumerate(["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "arkhitektury",
                              "finds_points", "fortifikatsii", "object_poly"]):
        m = int(rng.integers(0, n)) if ci != 3 else 0        # пустой класс
        conf = rng.random(m)
        if ties:
            conf = np.round(conf, 1 + ci % 3)                  # много равных уверенностей
        order = np.argsort(-conf, kind="stable")
        conf = conf[order]
        hit = rng.random(m) < conf * 0.6
        g_pool = groups[: 1 + ci % 4]
        pred_region = [regions[int(rng.integers(0, n_regions - 3))] for _ in range(m)]   # 3 участка без предсказаний
        pred_group = [g_pool[int(rng.integers(0, len(g_pool)))] for _ in range(m)]
        if ci == 2:   # группа, встречающаяся только на одном участке → в его фолде «невиданная»
            for i in range(0, m, 7):
                pred_group[i], pred_region[i] = "rare", regions[0]
        gtr = Counter()
        if ci != 5:   # класс без эталона
            for r in regions:
                k = int(rng.integers(0, 6))
                if k:
                    gtr[r] = k
            gtr[regions[-1]] = 0
        out[cls] = metric.ClassCurve(conf.astype(np.float64), hit, pred_region, int(sum(gtr.values())), gtr,
                                     pred_group)
    return out


def family(r: str) -> str:
    return r[:3]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("pick", ["max", "plateau"])
def test_sweep_groups_equal(seed, pick):
    cv = synth_curves(seed)
    assert canon(metric.sweep_groups(cv, pick=pick)) == canon(metric.sweep_groups_reference(cv, pick=pick))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("unseen", ["drop", "keep"])
@pytest.mark.parametrize("k", [3, 10, None])
def test_honest_counts_equal(seed, unseen, k):
    cv = synth_curves(seed, 2000)
    regs = sorted({r for c in cv.values() for r in c.pred_region} | {r for c in cv.values() for r in c.gt_regions})
    folds = metric.family_folds(regs, lambda r: r[:3] if k else r, k)
    a = metric.honest_counts(cv, folds, unseen=unseen)
    b = metric.honest_counts_reference(cv, folds, unseen=unseen)
    assert canon(a) == canon(b)
    assert canon(metric.score_counts(a[0])) == canon(metric.score_counts(b[0]))


def test_honest_counts_no_folds_and_empty():
    cv = synth_curves(4, 300)
    assert canon(metric.honest_counts(cv, [])) == canon(metric.honest_counts_reference(cv, []))
    empty = {"x": metric.ClassCurve(np.zeros(0), np.zeros(0, bool), [], 0, Counter(), [])}
    assert canon(metric.honest_counts(empty, [{"a"}])) == canon(metric.honest_counts_reference(empty, [{"a"}]))


@pytest.mark.parametrize("seed", [0, 1])
def test_paired_bootstrap_equal(seed):
    a, b = synth_curves(seed), synth_curves(seed + 10)
    ta = {c: 0.3 for c in a}
    tb = {"kurgany_tselye": 0.5, "gorodishcha": None}
    for n in (0, 1, 50, 300):
        assert canon(metric.paired_bootstrap(a, b, ta, tb, n)) == canon(metric.paired_bootstrap_reference(a, b, ta, tb, n))
    # классы только в одной из моделей
    b2 = {k: v for k, v in b.items() if k != "kurgany_tselye"}
    assert canon(metric.paired_bootstrap(a, b2, ta, tb, 100)) == canon(metric.paired_bootstrap_reference(a, b2, ta, tb, 100))
    assert metric.paired_bootstrap({}, {}, {}, {}, 10) == metric.paired_bootstrap_reference({}, {}, {}, {}, 10)


@pytest.mark.parametrize("seed", [0, 1])
def test_bootstrap_equal(seed):
    cv = synth_curves(seed, 1500)
    assert canon(metric.bootstrap(cv, 60)) == canon(metric.bootstrap_reference(cv, 60))


@pytest.mark.parametrize("seed", [0, 1])
def test_paired_bootstrap_counts_equal(seed):
    ca, cb = synth_curves(seed, 1500), synth_curves(seed + 3, 1500)
    regs = sorted({r for c in ca.values() for r in c.pred_region} | {r for c in ca.values() for r in c.gt_regions})
    folds = metric.family_folds(regs, family, 4)
    ha = metric.honest_counts_reference(ca, folds)[0]
    hb = metric.honest_counts_reference(cb, folds)[0]
    for n, regions in ((500, None), (200, set(regs[:10])), (0, None), (50, set())):
        assert (canon(metric.paired_bootstrap_counts(ha, hb, family, n, regions=regions))
                == canon(metric.paired_bootstrap_counts_reference(ha, hb, family, n, regions=regions)))
    hb2 = {k: v for k, v in hb.items() if k != "gorodishcha"}
    assert (canon(metric.paired_bootstrap_counts(ha, hb2, lambda r: r, 300))
            == canon(metric.paired_bootstrap_counts_reference(ha, hb2, lambda r: r, 300)))


@pytest.mark.skipif(not (REAL_PRED20.exists() and REAL_GT_FULL.exists()) or os.environ.get("BOT8_REAL") != "1",
                    reason="реальный прогон: BOT8_REAL=1 и выборка бота 8")
def test_real_scoreboard_flow_equal():
    """Весь поток scoreboard --honest на реальной выборке (каждое 20-е предсказание, весь эталон)."""
    pred, gt = _load(REAL_PRED20), _load(REAL_GT_FULL)
    new, ref = build_both(pred, gt)
    assert_curves_equal(new, ref)
    assert canon(metric.sweep_groups(new)) == canon(metric.sweep_groups_reference(ref))
    regs = sorted({r for c in ref.values() for r in c.pred_region} | {r for c in ref.values() for r in c.gt_regions})
    fam = lambda r: r.rsplit("_", 1)[0]  # noqa: E731
    folds = metric.family_folds(regs, fam, 10)
    a, b = metric.honest_counts(new, folds), metric.honest_counts_reference(ref, folds)
    assert canon(a) == canon(b)
    thr = {c: 0.3 for c in ref}
    assert (canon(metric.paired_bootstrap(new, new, thr, {}, 200))
            == canon(metric.paired_bootstrap_reference(ref, ref, thr, {}, 200)))
    assert (canon(metric.paired_bootstrap_counts(a[0], a[0], fam, 200))
            == canon(metric.paired_bootstrap_counts_reference(b[0], b[0], fam, 200)))


def test_par_is_valid_equal(monkeypatch):
    rng = np.random.default_rng(5)
    polys = []
    for i in range(5000):
        n = int(rng.integers(3, 12))
        ang = rng.uniform(0, 2 * np.pi, n)
        pts = np.c_[np.cos(ang), np.sin(ang)]
        polys.append(shapely.Polygon(pts))
    polys = np.asarray(polys, dtype=object)
    monkeypatch.setenv("EXPDS_METRIC_THREADS", "3")
    a = fastgeom.par_is_valid(polys, min_chunk=500)
    assert a.tolist() == shapely.is_valid(polys).tolist() and (~a).sum() > 100
