"""Быстрый to_feature_collection даёт тот же JSON до байта, что и исходный (base/)."""
import os, sys, json, time, importlib.util
import numpy as np, shapely
from shapely.geometry import Polygon, Point
HERE = os.path.dirname(__file__)


def load(root, name):
    sys.path.insert(0, os.path.join(HERE, "..", root))
    try:
        for k in [k for k in sys.modules if k == "expds" or k.startswith("expds.")]:
            del sys.modules[k]
        return importlib.import_module("expds.writer")
    finally:
        sys.path.pop(0)


def feats(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x, y = rng.uniform(3e6, 5e6), rng.uniform(5e6, 8e6)
        g = Point(x, y).buffer(rng.uniform(1, 30), quad_segs=int(rng.integers(2, 8)))
        if i % 7 == 0:
            g = Polygon(g.exterior.coords, [Point(x, y).buffer(0.5).exterior.coords])
        if i % 11 == 0:   # координаты на ровной половине сотой: проверка округления
            g = shapely.transform(g, lambda c: np.round(c, 2) + 0.005)
        out.append({"geom": g, "class_name": "kurgany", "confidence": float(rng.uniform()), "region_name": "r",
                    "markup_type": "Li" if i % 2 else None, "original_crs": "EPSG:32637" if i % 3 else None})
    return out


def test_bytes_equal():
    fs = feats()
    old, new = load("base", "base"), load("repo", "repo")
    t0 = time.perf_counter(); a = json.dumps(old.to_feature_collection(fs), ensure_ascii=False); ta = time.perf_counter() - t0
    t0 = time.perf_counter(); b = json.dumps(new.to_feature_collection(fs), ensure_ascii=False); tb = time.perf_counter() - t0
    assert a == b
    print(f"to_feature_collection+dumps {len(fs)} объектов: было {ta:.2f}s, стало {tb:.2f}s, x{ta / tb:.1f}")
    # не-Polygon — старый путь
    fs2 = fs[:5] + [{"geom": shapely.MultiPolygon([fs[0]["geom"]]), "class_name": "kurgany"}]
    try:
        old.to_feature_collection(fs2); ok_old = True
    except Exception:
        ok_old = False
    if ok_old:
        assert json.dumps(old.to_feature_collection(fs2)) == json.dumps(new.to_feature_collection(fs2))
