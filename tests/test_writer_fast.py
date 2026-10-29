"""Флаг fast_writer ([speed]): to_feature_collection даёт тот же JSON до байта, что обычный путь.

Харвест speedup-2026-10-28 (29.10.2026). Эталон — тот же модуль с выключенным флагом (без git-истории,
работает и на shallow-checkout в CI).
"""
import json

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon

from expds import writer


def feats(n=3000, seed=0):
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


def _dump(fs, monkeypatch, fast: bool) -> str:
    monkeypatch.setenv("EXPDS_SPEED_FAST_WRITER", "1" if fast else "0")
    return json.dumps(writer.to_feature_collection(fs), ensure_ascii=False)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("EXPDS_SPEED_FAST_WRITER", raising=False)
    monkeypatch.delenv("EXPDS_SPEEDUP", raising=False)
    from expds import speed

    speed.configure(None)
    assert not speed.on("fast_writer")


def test_bytes_equal(monkeypatch):
    fs = feats()
    assert _dump(fs, monkeypatch, False) == _dump(fs, monkeypatch, True)


def test_nan_skipped_and_fid_dense(monkeypatch):
    fs = feats(50)
    fs[3] = dict(fs[3], geom=Polygon([(0, 0), (1, 0), (float("nan"), 1), (0, 0)]))
    a, b = _dump(fs, monkeypatch, False), _dump(fs, monkeypatch, True)
    assert a == b
    fids = [f["properties"]["fid"] for f in json.loads(b)["features"]]
    assert fids == list(range(49))


@pytest.mark.parametrize("extra", ["multi", "empty"])
def test_fallback_same(monkeypatch, extra):
    fs = feats(20)
    g = shapely.MultiPolygon([fs[0]["geom"]]) if extra == "multi" else Polygon()
    fs2 = fs + [{"geom": g, "class_name": "kurgany"}]
    try:
        ref = _dump(fs2, monkeypatch, False)
    except Exception:
        with pytest.raises(Exception):
            _dump(fs2, monkeypatch, True)
        return
    assert ref == _dump(fs2, monkeypatch, True)
