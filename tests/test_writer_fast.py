"""Быстрый to_feature_collection даёт тот же JSON до байта, что версия до ускорения (git 14d419a)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
REF = "14d419a"


def _old_writer(tmp_path):
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{REF}:expds/writer.py"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("нет git-истории с эталоном")
    p = tmp_path / "expds" / "writer_ref.py"
    p.parent.mkdir()
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("expds.writer_ref", p, submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "expds"
    sys.modules["expds.writer_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


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


def test_bytes_equal(tmp_path):
    from expds import writer as new

    old = _old_writer(tmp_path)
    fs = feats()
    assert json.dumps(old.to_feature_collection(fs), ensure_ascii=False) == \
        json.dumps(new.to_feature_collection(fs), ensure_ascii=False)
    fs2 = fs[:5] + [{"geom": shapely.MultiPolygon([fs[0]["geom"]]), "class_name": "kurgany"}]
    try:
        ref = json.dumps(old.to_feature_collection(fs2))
    except Exception:
        return
    assert ref == json.dumps(new.to_feature_collection(fs2))
