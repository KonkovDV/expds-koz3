"""Насыпь, обрезанная краем тайла, не учится (gaussians_from_binary edge_ignore)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from expds.centers import center_loss, focal_center_loss, gaussians_from_binary, heatmap_counts


def tile_with_clipped_mound():
    m = np.zeros((1, 1, 64, 64), np.float32)
    yy, xx = np.mgrid[0:64, 0:64]
    m[0, 0][(yy - 32) ** 2 + (xx - 32) ** 2 <= 16] = 1          # целая насыпь в середине
    m[0, 0][(yy - 30) ** 2 + (xx - 1) ** 2 <= 25] = 1           # центр почти на краю: видна половина
    return torch.from_numpy(m)


def test_default_target_is_unchanged_and_peaks_the_fragment():
    t = gaussians_from_binary(tile_with_clipped_mound(), 2.0)
    assert float(t.min()) >= 0
    assert float(t[0, 0, 32, 32]) == 1.0
    assert float(t[0, 0, 30, 0:6].max()) > 0.9                   # пик в центре обрезка, а не кургана


def test_edge_ignore_drops_the_fragment_peak_and_marks_it():
    t = gaussians_from_binary(tile_with_clipped_mound(), 2.0, edge_ignore=True)
    assert float(t[0, 0, 32, 32]) == 1.0
    assert float(t[0, 0, 30, 0:6].max()) < 0                     # область обрезка — «не учить»
    assert float(t[0, 0, 30, 0:6].min()) == -1.0


def test_losses_and_counts_skip_ignored_pixels():
    t = gaussians_from_binary(tile_with_clipped_mound(), 2.0, edge_ignore=True)
    base = torch.full_like(t, -6.0)
    base[0, 0, 32, 32] = 6.0
    hot = base.clone()
    hot[0, 0, 30, 2] = 6.0                                       # уверенный пик в зоне «не учить»
    assert torch.isclose(focal_center_loss(hot, t), focal_center_loss(base, t))
    assert torch.isclose(center_loss(hot, t), center_loss(base, t))
    tp, fp, fn = heatmap_counts(hot.sigmoid(), t, 2)
    assert int(fp.sum()) == 0 and int(tp.sum()) == 1 and int(fn.sum()) == 0
