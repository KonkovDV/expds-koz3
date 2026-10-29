#!/usr/bin/env python3
"""Задача 11, поправка 29.10: отбор li_half перекрёстный по классу, якоря li / li_mosaic / li_dev.
Запуск: python test_li_half_eval.py (или pytest); li_half_run.sh запускает его перед оценкой."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import li_half_eval as E  # noqa: E402
from shapely.geometry import Point  # noqa: E402

R = "016_Хога"
P, T = "kurgany_povrezhdennye", "kurgany_tselye"


def _c(x, y, cls):
    return {"geom": Point(x, y).buffer(2.0), "class_name": cls, "region_name": R}


def test_cross_class_drop():
    anchors = [Point(500_000.0, 6_000_000.0)]                       # принятый ЦЕЛЫЙ курган основного li (метры)
    cands = [_c(500_001.0, 6_000_000.0, P),                          # повреждённый в 1 м — тот же холм → вон
             _c(500_000.0, 6_000_011.0, T),                          # целый в 11 м → вон
             _c(500_013.0, 6_000_000.0, P),                          # 13 м → остаётся
             _c(500_000.0, 6_000_013.5, T)]                          # 13,5 м → остаётся
    kept = E.drop_near_any(cands, anchors, 12.0)
    assert [(round(f["geom"].centroid.x, 1), round(f["geom"].centroid.y, 1)) for f in kept] == [(500013.0, 6000000.0), (500000.0, 6000013.5)]
    assert E.drop_near_any(cands, [], 12.0) == cands


def test_anchor_groups():
    regs = {R}
    base = {"markup_type": "li", "region_name": R}
    assert E.is_anchor({**base, "class_name": T, "confidence": 0.2}, regs)                              # li без группы
    assert not E.is_anchor({**base, "class_name": T, "confidence": 0.19}, regs)
    assert E.is_anchor({**base, "class_name": P, "confidence": 0.96, "threshold_group": "li_mosaic"}, regs)
    assert E.is_anchor({**base, "class_name": P, "confidence": 0.9914, "threshold_group": "li_dev"}, regs)  # σ
    assert E.is_anchor({**base, "class_name": T, "confidence": 0.9996, "threshold_group": "li_dev"}, regs)
    assert not E.is_anchor({**base, "class_name": T, "confidence": 0.999, "threshold_group": "li_dev"}, regs)
    assert not E.is_anchor({**base, "class_name": T, "confidence": 0.99, "threshold_group": "li_coarse"}, regs)
    assert not E.is_anchor({**base, "class_name": "gorodishcha", "confidence": 0.99}, regs)
    assert not E.is_anchor({**base, "class_name": T, "confidence": 0.9, "region_name": "018_Курган_Хога"}, regs)
    assert not E.is_anchor({"markup_type": "ae", "region_name": R, "class_name": T, "confidence": 0.99}, regs)


def test_cross_class_drop_web_mercator():
    # якорь из стека (3857) и кандидат li_half — оба через ground_point в UTM; ~1 м по долготе на 57° с.ш.
    from shapely.geometry import box
    from expds.pipeline import ground_point
    x0, y0 = 4_000_000.0, 7_800_000.0
    a = ground_point(box(x0 - 3, y0 - 3, x0 + 3, y0 + 3))
    near = {"geom": Point(x0 + 1.8, y0).buffer(4.0), "class_name": P, "region_name": R}
    far = {"geom": Point(x0 + 30.0, y0).buffer(4.0), "class_name": P, "region_name": R}
    assert E.drop_near_any([near, far], [a], 12.0) == [far]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
