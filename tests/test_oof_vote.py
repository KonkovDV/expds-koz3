"""Голосование моделей по OOF-файлам (scripts/oof_vote.py)."""
import importlib.util
from pathlib import Path

from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("oof_vote", ROOT / "scripts" / "oof_vote.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def obj(x, y, conf, cls="kurgany_povrezhdennye", region="001_A"):
    # UTM-подобные малые координаты: ground_point берёт внутреннюю точку как есть
    return {"props": {"class_name": cls, "region_name": region, "confidence": conf},
            "geom": Point(x, y).buffer(4), "conf": conf}


def test_agreement_raises_a_weak_object_above_a_lonely_strong_one():
    m = _mod()
    base = [obj(0, 0, 0.40), obj(100, 0, 0.90)]
    other1 = [obj(2, 1, 0.70), obj(500, 500, 0.95)]
    other2 = [obj(-1, 2, 0.80)]
    conf = m.fuse(base, [other1, other2], radius=6.0, mode="rank-mean")
    assert conf[0] > conf[1]                       # слабый, но подтверждённый двумя — выше
    votes = m.fuse(base, [other1, other2], radius=6.0, mode="votes", vote_rank=0.5)
    assert votes[0] > votes[1]


def test_neighbours_must_share_region_and_class():
    m = _mod()
    base = [obj(0, 0, 0.5)]
    other = [obj(0, 0, 0.9, cls="kurgany_tselye"), obj(0, 0, 0.9, region="002_B")]
    conf = m.fuse(base, [other], radius=6.0, mode="rank-mean")
    assert conf[0] == (1.0 + 0.0) / 2              # ранг базовой 1,0; соседа нет


def test_model_without_data_abstains_instead_of_voting_against():
    """Лидар молчит там, где нет LAS: спутниковый курган не теряет уверенность."""
    import numpy as np

    m = _mod()
    base = [obj(0, 0, 0.40), obj(100, 0, 0.90)]
    lidar = [obj(1, 1, 0.80)]
    cover = np.array([[True, False]])
    plain = m.fuse(base, [lidar], radius=6.0, mode="rank-mean")
    aware = m.fuse(base, [lidar], radius=6.0, mode="rank-mean", cover=cover)
    assert plain[1] == 0.5
    assert aware[1] == 1.0
    assert aware[0] == plain[0] == (0.5 + 1.0) / 2
    votes = m.fuse(base, [lidar], radius=6.0, mode="votes", cover=cover)
    assert votes[1] == 1.0 and votes[0] == (1 + 0.5) / 2


def test_coverage_reads_las_box_and_raster_pixels(tmp_path):
    """Покрытие: рамка LAS для li, непустые пиксели снимка для остальных."""
    import json as _json

    import numpy as np
    from conftest import UTM, X0, Y0, write_las, write_mag_tif
    from shapely.geometry import Point as P

    from expds.crs import WEB_MERCATOR, to_crs

    m = _mod()
    reg = tmp_path / "root" / "001_A"
    write_las(reg / "01_001_A_La_точки" / "01_001_A_La.las", n=5000)
    write_mag_tif(reg / "07_001_A_MagGr_карты" / "01_001_A_MagGr.tif")
    (reg / "UTM.json").write_text(_json.dumps({"utm": "36N"}), encoding="utf-8")
    inside = to_crs(P(X0 + 20, Y0 + 20), UTM, WEB_MERCATOR)
    outside = to_crs(P(X0 + 5000, Y0 + 5000), UTM, WEB_MERCATOR)
    base = [{"props": {"region_name": "001_A", "class_name": "c"}, "geom": g.buffer(1), "conf": 0.5}
            for g in (inside, outside)]
    li = [{"props": {"markup_type": "li"}}]
    mag = [{"props": {"markup_type": "maggr"}}]
    cov = m.coverage(base, [li, mag], tmp_path / "root")
    assert cov.tolist() == [[True, False], [True, False]]
    assert isinstance(cov, np.ndarray)
