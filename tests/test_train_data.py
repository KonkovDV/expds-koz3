"""GpuTiles: тайлы в памяти устройства, распаковка масок и аугментации на устройстве."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from expds.train import GpuTiles, TileDataset


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


def test_epoch_covers_each_tile_once(tmp_path):
    rows, _, _ = make_cache(tmp_path)
    g = GpuTiles(tmp_path, rows, 5, flips=False, device=torch.device("cpu"))
    sizes = [len(x) for x, _ in g.batches(3, shuffle=False, drop_last=False)]
    assert sizes == [3, 3, 3, 1]
    assert [len(x) for x, _ in g.batches(3, shuffle=True, drop_last=True)] == [3, 3, 3]
