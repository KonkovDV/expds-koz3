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
