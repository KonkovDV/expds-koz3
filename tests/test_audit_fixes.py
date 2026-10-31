"""Исправления аудита 27.10: флаги обучения, которые молча не работали или портили цель."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # W4: окружение без torch — skip, а не ошибка сбора
from shapely.geometry import LineString, Point

from expds.centers import focal_center_loss, gaussians_from_binary, small_component_weights
from expds.open_ideas import medial_fort_circles, mercator_scale
from expds.train import CHANNEL_DROP_FILL, drop_channel, elastic_deform


def two_mounds():
    y = torch.zeros(1, 1, 64, 64)
    y[0, 0, 10:13, 10:13] = 1          # 9 пикселей: мельче 0,2 % кадра 64×64 (8,2) не считается
    y[0, 0, 30:50, 30:50] = 1          # 400 пикселей
    y[0, 0, 5:7, 50:52] = 1            # 4 пикселя — мелкая
    return y


def test_small_weight_reaches_the_loss_not_the_target():
    y = two_mounds()
    assert small_component_weights(y, 1.0) is None
    w = small_component_weights(y, 3.0, max_frac=0.002, grow_px=0)
    assert w[0, 0, 5, 50] == 3.0 and w[0, 0, 40, 40] == 1.0 and w[0, 0, 11, 11] == 1.0
    t = gaussians_from_binary(y, 1.5)
    logits = torch.zeros_like(t)
    assert focal_center_loss(logits, t, weight=w) > focal_center_loss(logits, t)


def test_small_weight_rejected_for_mask_target(tmp_path):
    from expds.train import train

    (tmp_path / "meta.json").write_text(json.dumps({"classes": ["kurgany_tselye"], "channels": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="target=centers"):
        train(str(tmp_path), str(tmp_path / "m.pt"), target="mask", small_object_weight=2.0)


def test_within_site_rank_rejected_outside_focal_centers(tmp_path):
    from expds.train import train

    (tmp_path / "meta.json").write_text(json.dumps({"classes": ["kurgany_tselye"], "channels": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="фокальной"):
        train(str(tmp_path), str(tmp_path / "m.pt"), target="mask", within_site_rank=1.0)


def test_elastic_keeps_a_mound_in_one_piece():
    torch.manual_seed(0)
    y = torch.zeros(4, 1, 128, 128)
    yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
    y[:, 0] = (((yy - 64) ** 2 + (xx - 64) ** 2) < 15 ** 2).float()
    x = y.repeat(1, 3, 1, 1)
    _, yd = elastic_deform(x, y, 0.03)
    from scipy import ndimage

    for i in range(4):
        _, n = ndimage.label(yd[i, 0].numpy() > 0.5)
        assert n == 1
        assert abs(float(yd[i, 0].sum()) - float(y[i, 0].sum())) < 0.25 * float(y[i, 0].sum())


def test_cpu_dataset_applies_elastic(tmp_path):
    """Без видеокарты --elastic не должен молча пропадать: маска сдвигается до нормировки."""
    from expds.train import TileDataset

    img = np.zeros((1, 3, 32, 32), np.uint8)
    msk = np.zeros((1, 32, 32), np.uint16)
    msk[:, 8:24, 8:24] = 1
    np.save(tmp_path / "images_0000.npy", img)
    np.save(tmp_path / "masks_0000.npy", msk)
    rows = [{"shard": 0, "idx": 0}]
    plain = TileDataset(tmp_path, rows, 1, flips=False)
    torch.manual_seed(0)
    moved = TileDataset(tmp_path, rows, 1, flips=False, elastic=0.25)
    _, y0 = plain[0]
    _, y1 = moved[0]
    assert y0.sum() == 16 * 16 and not torch.equal(y0, y1)


def test_dropped_channel_is_mid_scale():
    torch.manual_seed(1)
    x = torch.rand(8, 3, 4, 4)
    out = drop_channel(x, 1.0)
    dropped = [(out[i] == CHANNEL_DROP_FILL).all(dim=(1, 2)) for i in range(8)]
    assert all(int(d.sum()) == 1 for d in dropped)
    assert (out.sub(0.5).div(0.25)[torch.stack(dropped)] == 0).all()


def test_fort_axis_circles_are_spaced_and_in_ground_meters():
    y0 = 7_500_000.0                       # около 55° с. ш. в EPSG:3857
    k = mercator_scale(y0)
    assert 1.6 < k < 1.9
    wall = LineString([(0, y0), (200 * k, y0)]).buffer(3 * k, cap_style=2)
    feats = [{"class_name": "fortifikatsii", "geom": wall, "confidence": 0.5}]
    out = medial_fort_circles(feats, radius_m=2.0, spacing_m=25.0)
    xs = sorted(f["geom"].centroid.x for f in out)
    assert 5 <= len(out) <= 10
    assert min(np.diff(xs)) >= 25.0 * k * 0.95
    assert abs(out[0]["geom"].area - np.pi * (2.0 * k) ** 2) < 0.1 * np.pi * (2.0 * k) ** 2
    other = [{"class_name": "gorodishcha", "geom": Point(0, y0).buffer(10), "confidence": 0.5}]
    assert medial_fort_circles(other, 2.0) == other


def test_clip_ignore_records_the_bit(val_dir, tmp_path):
    from conftest import UTM, X0, Y0
    from shapely.geometry import box

    from expds.crs import WEB_MERCATOR, to_crs
    from expds.tiles import IGNORE_BIT, build_cache

    poly = to_crs(box(X0 + 5, Y0 + 5, X0 + 35, Y0 + 35), UTM, WEB_MERCATOR)
    gt = tmp_path / "gt.geojson"
    gt.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": poly.__geo_interface__,
         "properties": {"class_name": "object_poly", "markup_type": "maggr", "region_name": "region_0002"}}]}),
        encoding="utf-8")
    meta = build_cache(str(val_dir), str(gt), "maggr", str(tmp_path / "c"), res=0.25, tile=64,
                       min_clip_frac=0.5, seed=0)
    assert meta["ignore_bit"] == IGNORE_BIT and meta["min_clip_frac"] == 0.5
    assert meta["crs_by_source"] and all(v for v in meta["crs_by_source"].values())


def test_grid_centers_end_flush_with_the_source():
    """Шаг, который не доходит до конца, дописывает окно встык с краем, а не за ним."""
    from expds.tiles import grid_centers

    xs = sorted({round(x, 6) for x, _ in grid_centers(0, 0, 50, 50, 16, 8)})
    assert xs == [8.0, 16.0, 24.0, 32.0, 40.0, 42.0]


def test_grid_cache_covers_the_source(val_dir, tmp_path):
    from expds.tiles import build_cache

    gt = tmp_path / "gt.geojson"
    gt.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    meta = build_cache(str(val_dir), str(gt), "maggr", str(tmp_path / "g"), res=0.25, tile=64,
                       grid_overlap=0.5, seed=0, only_classes=None)
    import csv

    rows = list(csv.DictReader((tmp_path / "g" / "index.csv").open(encoding="utf-8")))
    # Карта 40 м при окне 16 м и шаге 8 м: инференс ставит 4 окна встык, не 5 с выносом за край.
    assert meta["grid_overlap"] == 0.5 and len(rows) == 16
    xs = sorted({float(r["cx"]) for r in rows})
    assert np.allclose(np.diff(xs), 8.0)
