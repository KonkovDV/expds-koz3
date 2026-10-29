from shapely.geometry import box

from expds import post


def f(x, g, c, cls="kurgany_tselye"):
    return {"geom": box(x, 0, x + 1, 1), "region_name": "r", "markup_type": "li",
            "threshold_group": g, "class_name": cls, "confidence": c}


def test_off_by_default():
    feats = [f(0, "a", 0.9), f(0.5, "b", 0.3)]
    assert post.fuse_cross_group(feats) == feats
    assert post.calibrate_group_conf(feats, {("kurgany_tselye", "a"): ("platt", 1.0, 1.0)}) == feats


def test_fuse_only_different_groups():
    feats = [f(0, "a", 0.9), f(0.5, "b", 0.3), f(0.2, "a", 0.8), f(5, "a", 0.5)]
    out = post.fuse_cross_group(feats, enabled=True)
    # кластер {0,1,2} (разные группы через 1) -> один объект; 3 не тронут
    assert len(out) == 2
    fused = [o for o in out if o.get("fused")][0]
    assert abs(fused["confidence"] - (0.9 + 0.3) / 2) < 1e-9   # N=T=2
    same = [f(0, "a", 0.9), f(0.5, "a", 0.3)]
    assert post.fuse_cross_group(same, enabled=True) == same


def test_calib_iso_platt():
    feats = [f(0, "a", 0.5), f(3, "b", 0.5)]
    cal = {("kurgany_tselye", "a"): ("iso", [0.0, 1.0], [0.0, 0.2])}
    out = post.calibrate_group_conf(feats, cal, enabled=True)
    assert abs(out[0]["confidence"] - 0.1) < 1e-9 and out[1]["confidence"] == 0.5
    out = post.calibrate_group_conf(feats, {("kurgany_tselye", "b"): ("platt", 1.0, 0.0)}, enabled=True)
    assert abs(out[1]["confidence"] - 0.5) < 1e-9
