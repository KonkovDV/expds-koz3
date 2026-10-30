from shapely.geometry import Point, box
from w3_post import gor_from_fort_arch, kurgan_subclass_twins, POVR, TSEL, GOR

Y = 7_400_000.0  # ~55 N, scale ~0.57


def F(cls, geom, mk="sp_or", grp=None, conf=0.9, **kw):
    d = dict(class_name=cls, geom=geom, markup_type=mk, region_name="r", confidence=conf)
    if grp: d["threshold_group"] = grp
    d.update(kw); return d


def test_flags_off_identity():
    fs = [F("fortifikatsii", box(0, Y, 300, Y + 300))]
    assert gor_from_fort_arch(fs) is fs and kurgan_subclass_twins(fs) is fs


def test_gor_twin_area_and_dup():
    big = F("fortifikatsii", box(0, Y, 200, Y + 200))       # 40000*0.33 ~ 13000 m2
    small = F("arkhitektury", box(0, Y, 50, Y + 50))        # ~800 m2
    covered = F("arkhitektury", box(1000, Y, 1200, Y + 200))
    gor = F(GOR, box(1000, Y, 1200, Y + 200))
    out = gor_from_fort_arch([big, small, covered, gor], {"enabled": True})
    tw = [f for f in out if f.get("threshold_group", "").startswith("gor_fa_")]
    assert len(tw) == 1 and tw[0]["class_name"] == GOR and tw[0]["gor_fa_src"] == "fortifikatsii"
    assert len(out) == 5


def test_kurgan_twins():
    a = F(TSEL, Point(0, Y).buffer(3), mk="li", grp="li", conf=0.5, subclass_p_povr=0.8)
    b = F(POVR, Point(100, Y).buffer(3), mk="li", grp="li", conf=0.5, subclass_p_povr=0.1)
    c = F(TSEL, Point(200, Y).buffer(3), mk="li", grp="li", subclass_p_povr=0.9)
    c_pair = F(POVR, Point(200.5, Y).buffer(3), mk="ae", grp="ae")   # opposite already there -> no twin
    d = F(TSEL, Point(300, Y).buffer(3), mk="li", grp="li", subclass_p_povr=0.5)  # undecided
    out = kurgan_subclass_twins([a, b, c, c_pair, d], {"enabled": True})
    tw = out[5:]
    assert [(t["class_name"], t["threshold_group"]) for t in tw] == [(POVR, "li_sc_povr"), (TSEL, "li_sc_tsel")]
    assert abs(tw[0]["confidence"] - 0.4) < 1e-9 and abs(tw[1]["confidence"] - 0.45) < 1e-9
