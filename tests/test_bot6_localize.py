""" LOCALIZE: микрокруги фортификаций через [post.micro_circles] (словарь радиусов)."""
import math

from shapely.geometry import Point, box

from expds import post


def test_micro_circles_fort_dict_radius():
    y = 7_500_000.0                                   # ~55° с. ш. в EPSG:3857
    s = post._ground_scale(y)
    wall = box(0, y, 40 / s, y + 1 / s)                # вал 40 × 1 м на земле
    fort = {"class_name": "fortifikatsii", "confidence": 0.7, "geom": wall}
    gor = {"class_name": "gorodishcha", "confidence": 0.6, "geom": Point(0, y).buffer(30 / s)}
    kur = {"class_name": "kurgany_tselye", "confidence": 0.5, "geom": Point(100, y).buffer(4 / s)}
    spec = {"radius_m": {"kurgany_povrezhdennye": 0.5, "kurgany_tselye": 0.5, "fortifikatsii": 0.5}}
    out = post.micro_circles([fort, gor, kur], spec)
    f, g, k = out
    assert g is gor                                    # городища не трогаем
    assert wall.contains(f["geom"].centroid)           # центр внутри вала
    r_m = math.sqrt(f["geom"].area / math.pi) * s
    assert abs(r_m - 0.5) < 0.01
    assert f["geom"].intersection(wall).area / f["geom"].area >= 0.5   # cov ≥ 0,5 к валу
    assert abs(math.sqrt(k["geom"].area / math.pi) * s - 0.5) < 0.01
    assert [x["confidence"] for x in out] == [0.7, 0.6, 0.5]
