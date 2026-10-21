"""GpuTiles: тайлы в памяти устройства, распаковка масок и аугментации на устройстве."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from expds.train import GpuTiles, TileDataset, paste_objects


def test_dice_does_not_dilute_a_missed_class_with_an_empty_one():
    from expds.train import dice_loss

    logits = torch.zeros(2, 2, 4, 4)
    target = torch.zeros_like(logits)
    target[:, 0, 1:3, 1:3] = 1
    missed = dice_loss(logits[:, :1], target[:, :1])
    both = dice_loss(logits, target)
    assert torch.allclose(both, missed, atol=1e-5)
    assert float(missed) > 0.5


def make_cache(tmp_path, n=10, c=3, t=16, k=5):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, (n, c, t, t), dtype=np.uint8)
    masks = rng.integers(0, 1 << k, (n, t, t), dtype=np.uint8)
    np.save(tmp_path / "images_0000.npy", imgs[:6])
    np.save(tmp_path / "masks_0000.npy", masks[:6])
    np.save(tmp_path / "images_0001.npy", imgs[6:])
    np.save(tmp_path / "masks_0001.npy", masks[6:])
    rows = [{"shard": str(i // 6), "idx": str(i % 6)} for i in range(n)]
    return rows, imgs, masks


def test_unpacked_like_tile_dataset(tmp_path):
    rows, imgs, masks = make_cache(tmp_path)
    g = GpuTiles(tmp_path, rows, 5, flips=False, device=torch.device("cpu"))
    ds = TileDataset(tmp_path, rows, 5, flips=False)
    x, y = g.get(torch.arange(len(rows)), augment=False)
    for i in range(len(rows)):
        xr, yr = ds[i]
        assert torch.allclose(x[i], xr) and torch.equal(y[i], yr)
    assert g.nbytes(tmp_path, rows) == imgs.nbytes + masks.nbytes


def test_same_transform_for_image_and_mask(tmp_path):
    rows, _, _ = make_cache(tmp_path)
    g = GpuTiles(tmp_path, rows, 5, flips=True, device=torch.device("cpu"))
    g.x[:] = torch.from_numpy(np.broadcast_to(np.arange(256, dtype=np.uint8).reshape(16, 16), g.x.shape).copy())
    g.m[:] = g.x[:, 0] % 32
    torch.manual_seed(0)
    seen = set()
    for _ in range(20):
        x, y = g.get(torch.arange(len(rows)), augment=True)
        raw = torch.round((x * 0.25 + 0.5) * 255).to(torch.uint8)
        bits = sum((y[:, k].to(torch.uint8) << k) for k in range(5))
        assert torch.equal(raw[:, 0] % 32, bits)                 # маска преобразована так же, как тайл
        seen |= {tuple(raw[i, 0, 0, :2].tolist()) for i in range(len(rows))}
    assert len(seen) == 8                                         # все 8 поворотов и отражений


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нет CUDA")
def test_predict_on_device_matches_cpu(tmp_path):
    smp = pytest.importorskip("segmentation_models_pytorch")
    from expds.infer_nn import Segmenter

    net = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
    path = tmp_path / "m.pt"
    torch.save({"state_dict": net.state_dict(), "encoder": "resnet18", "in_channels": 3,
                "classes": ["kurgany_povrezhdennye", "kurgany_tselye"], "res": 0.5, "tile": 64}, path)
    img = np.random.default_rng(1).integers(0, 255, (3, 150, 210), dtype=np.uint8)
    cpu = Segmenter(path, device="cpu").predict(img)
    gpu_seg = Segmenter(path, device="cuda")
    gpu_seg.amp = False
    gpu = gpu_seg.predict(img)
    assert gpu.shape == cpu.shape == (2, 150, 210)
    assert np.abs(gpu - cpu).max() < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нет CUDA")
def test_larger_batch_matches_small_batch(tmp_path):
    """Больший батч в режиме eval не смешивает окна: вероятности те же, что при батче 4."""
    pytest.importorskip("segmentation_models_pytorch")
    from expds.infer_nn import Segmenter

    net = __import__("segmentation_models_pytorch").Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
    path = tmp_path / "m.pt"
    torch.save({"state_dict": net.state_dict(), "encoder": "resnet18", "in_channels": 3,
                "classes": ["a", "b"], "res": 0.5, "tile": 64}, path)
    img = np.random.default_rng(2).integers(0, 255, (3, 180, 200), dtype=np.uint8)
    seg = Segmenter(path, device="cuda")
    seg.amp = False
    small = seg.predict(img, batch=4)
    large = seg.predict(img, batch=32)
    # Сложение перекрытий идёт в другом порядке, последняя цифра float32 гуляет примерно на 1e-5.
    assert np.abs(small - large).max() < 1e-4


def test_paste_copies_object_pixels_only():
    x = torch.zeros(2, 1, 8, 8)
    y = torch.zeros(2, 2, 8, 8)
    x[0, 0, 2:4, 3:5] = 0.8
    y[0, 0, 2:4, 3:5] = 1
    x[1] = 0.2
    same_x, same_y = paste_objects(x.clone(), y.clone(), (0,), p=0.0)
    assert torch.equal(same_x, x) and torch.equal(same_y, y)
    found = False
    for seed in range(40):
        torch.manual_seed(seed)
        xp, yp = paste_objects(x.clone(), y.clone(), (0,), p=1.0)
        obj = yp[1, 0] > 0.5
        if not bool(obj.any()):
            continue
        assert torch.allclose(xp[1, 0][obj], torch.full((int(obj.sum()),), 0.8))
        assert torch.allclose(xp[1, 0][~obj], torch.full((int((~obj).sum()),), 0.2))
        assert float(yp[1, 1].sum()) == 0
        assert torch.equal(xp[0], x[0]) and torch.equal(yp[0], y[0])
        found = True
        break
    assert found


def test_copy_paste_on_gpu_tiles(tmp_path):
    rows, _, _ = make_cache(tmp_path, n=4)
    g = GpuTiles(tmp_path, rows, 5, flips=False, device=torch.device("cpu"),
                 copy_paste=1.0, paste_channels=(0,))
    g.x.zero_()
    g.m.zero_()
    g.x[0, 0, 4:8, 4:8] = 200
    g.m[0, 4:8, 4:8] = 1
    painted = (200 / 255.0 - 0.5) / 0.25
    found = False
    for seed in range(40):
        torch.manual_seed(seed)
        x, y = g.get(torch.arange(4), augment=True)
        obj = y[1, 0] > 0.5
        if not bool(obj.any()):
            continue
        assert torch.allclose(x[1, 0][obj], torch.full((int(obj.sum()),), painted), atol=1e-5)
        assert float(y[1, 1:].sum()) == 0
        found = True
        break
    assert found


def test_epoch_covers_each_tile_once(tmp_path):
    rows, _, _ = make_cache(tmp_path)
    g = GpuTiles(tmp_path, rows, 5, flips=False, device=torch.device("cpu"))
    sizes = [len(x) for x, _ in g.batches(3, shuffle=False, drop_last=False)]
    assert sizes == [3, 3, 3, 1]
    assert [len(x) for x, _ in g.batches(3, shuffle=True, drop_last=True)] == [3, 3, 3]
