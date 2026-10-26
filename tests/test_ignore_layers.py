"""Объект того же класса из разметки другого слоя — «не учить», а не фон (tiles --ignore-other-layers)."""
import numpy as np
import pytest
from test_gt_tiles import _gt_file, make_train

from expds import tiles


def test_other_layer_objects_get_the_ignore_bit_only_where_own_labels_are_empty(tmp_path):
    root = make_train(tmp_path / "train")
    gt = str(_gt_file(root, tmp_path))
    plain = tiles.build_cache(str(root), gt, "li", str(tmp_path / "plain"), res=0.5, tile=64)
    meta = tiles.build_cache(str(root), gt, "li", str(tmp_path / "ign"), res=0.5, tile=64,
                             ignore_other_layers=True)
    assert meta["ignore_bit"] == tiles.IGNORE_BIT and "ignore_bit" not in plain
    assert meta["positive"] == plain["positive"]                        # бит «не учить» не делает тайл позитивным
    m = np.load(tmp_path / "ign" / "masks_0000.npy")
    ign = (m >> tiles.IGNORE_BIT) & 1
    own = m & tiles.CLASS_BITS_MASK
    assert ign.sum() > 0                                                 # курган аэрофото попал в тайлы лидара
    assert not (ign.astype(bool) & (own > 0)).any()                     # и никогда не поверх своей метки
    assert np.array_equal(own, np.load(tmp_path / "plain" / "masks_0000.npy") & tiles.CLASS_BITS_MASK)


def test_training_reads_the_ignore_bit_as_minus_one(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    from expds.train import train

    root = make_train(tmp_path / "train")
    gt = str(_gt_file(root, tmp_path))
    tiles.build_cache(str(root), gt, "li", str(tmp_path / "ign"), res=0.5, tile=64, ignore_other_layers=True)
    res = train(str(tmp_path / "ign"), str(tmp_path / "m.pt"), epochs=1, batch=2, workers=0, pretrained=False,
                amp=False, gpu_data=False, target="centers", center_sigma_m=1.0, center_loss_name="focal",
                use_ignore=True)
    assert (tmp_path / "m.pt").exists() and res["history"]
    with pytest.raises(ValueError):
        train(str(tmp_path / "ign"), str(tmp_path / "m2.pt"), epochs=1, batch=2, workers=0, pretrained=False,
              amp=False, gpu_data=False, target="mask", use_ignore=True)
