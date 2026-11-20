"""Канал σ: одно битое облако не выбрасывает круги с остальных LAS участка."""
from types import SimpleNamespace

from shapely.geometry import Point

from expds.catalog import Region
from expds.pipeline import li_dev_feats


def test_sigma_keeps_circles_from_a_good_las(tmp_path, monkeypatch):
    a = tmp_path / "a.las"
    b = tmp_path / "b.las"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    reg = Region("r", tmp_path, las=[a, b])
    seen = {"n": 0}

    def dem(las, **_k):
        seen["n"] += 1
        if las.name == "b.las":
            raise RuntimeError("облако не читается")
        return SimpleNamespace(crs="EPSG:32636")

    monkeypatch.setattr("expds.lidar.build_dem", dem)
    feat = {"class_name": "kurgany_tselye", "confidence": 0.4,
            "geom": Point(0, 0).buffer(1), "markup_type": "li", "region_name": "r", "source": "a.las"}
    monkeypatch.setattr("expds.pipeline._li_with_model", lambda *args, **kwargs: [dict(feat)])
    monkeypatch.setattr("expds.pipeline._cached_segmenters",
                        lambda *args, **kwargs: [SimpleNamespace(meta={}, classes=["kurgany_tselye"])])
    guard = SimpleNamespace(ok=lambda *_a, **_k: True)
    out = li_dev_feats(reg, {"models": {"li_dev": "m.pt"}}, guard, {}, tmp_path)
    assert seen["n"] == 2
    assert len(out) == 1
    assert out[0]["threshold_group"] == "li_dev"
    assert out[0]["source"] == "a.las"
