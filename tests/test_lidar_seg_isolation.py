"""Основной лидар: сбой одной сети не стирает курганы другой и не зовёт запасной детектор."""
from types import SimpleNamespace

from shapely.geometry import Point

from expds.catalog import Region
from expds.pipeline import lidar_feats


def test_one_lidar_network_failure_keeps_the_other_and_skips_cpu(tmp_path, monkeypatch):
    las = tmp_path / "a.las"
    las.write_bytes(b"x")
    reg = Region("r", tmp_path, las=[las])
    cpu = {"n": 0}

    def dem(*_a, **_k):
        return SimpleNamespace(crs="EPSG:32636", info={})

    def model(seg, *_a, **_k):
        if getattr(seg, "bad", False):
            raise RuntimeError("сеть")
        return [{"class_name": "kurgany_tselye", "confidence": 0.5, "geom": Point(0, 0).buffer(1),
                 "markup_type": "li", "region_name": "r", "source": "a.las", "threshold_group": "li"}]

    monkeypatch.setattr("expds.lidar.build_dem", dem)
    monkeypatch.setattr("expds.pipeline._li_with_model", model)
    monkeypatch.setattr("expds.pipeline._cached_segmenters",
                        lambda *_a, **_k: [SimpleNamespace(meta={}, bad=False), SimpleNamespace(meta={}, bad=True)])
    monkeypatch.setattr("expds.pipeline.cpu_kurgans", lambda *_a, **_k: cpu.__setitem__("n", cpu["n"] + 1) or [])
    guard = SimpleNamespace(ok=lambda *_a, **_k: True, soft_ceiling=None)
    cfg = {"models": {"li": "m.pt"}, "hillfort": {"enabled": False}, "li": {"mode": "rescore"},
           "lidar": {"mosaic": False}}
    out = lidar_feats(reg, cfg, guard, {}, tmp_path)
    assert len(out) == 1
    assert out[0]["source"] == "a.las"
    assert cpu["n"] == 0
