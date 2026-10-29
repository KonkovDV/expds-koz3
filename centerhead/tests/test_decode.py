import numpy as np

from centerhead import cfg_with
from centerhead.decode import decode, find_peaks, nms_distance, stitch, tile_origins
from centerhead.targets import render_targets_strided
from centerhead.targets import render_targets

LEFT, TOP, STEP = 500_000.0, 6_200_000.0, 0.5


def test_decode_recovers_centres_with_offsets():
    rng = np.random.default_rng(0)
    n = 25
    g = np.stack(np.meshgrid(np.arange(5), np.arange(5)), -1).reshape(-1, 2)
    xy = 20 + g * 40 + rng.uniform(-3, 3, (n, 2))
    objs = np.column_stack([xy, rng.uniform(6, 14, n), rng.integers(2, size=n)])
    t = render_targets(objs, 220, 220)
    c = decode(t["hm"], LEFT, TOP, STEP, wh=t["wh"], off=t["off"])
    assert len(c["X"]) == n
    X = LEFT + objs[:, 0] * STEP
    Y = TOP - objs[:, 1] * STEP
    for i in range(n):
        j = np.argmin(np.hypot(c["X"] - X[i], c["Y"] - Y[i]))
        assert np.hypot(c["X"][j] - X[i], c["Y"][j] - Y[i]) < 1e-4
        assert c["cls"][j] == objs[i, 3]
        assert abs(c["radius_m"][j] - objs[i, 2] / 2 * STEP) < 1e-4
        assert c["score"][j] == 1.0


def test_decode_without_regression_uses_pixel_centre():
    hm = np.zeros((2, 32, 32), np.float32)
    hm[1, 7, 9] = 0.8
    c = decode(hm, LEFT, TOP, STEP)
    assert list(c["cls"]) == [1]
    assert c["X"][0] == LEFT + 9.5 * STEP and c["Y"][0] == TOP - 7.5 * STEP
    assert c["radius_m"][0] == 4.0                                     # default_diam_m/2


def test_find_peaks_threshold_topk():
    p = np.zeros((20, 20), np.float32)
    p[2, 2], p[10, 10], p[15, 3] = 0.9, 0.04, 0.5
    r, c, s = find_peaks(p, 3, 0.05, 1)
    assert list(zip(r, c)) == [(2, 2)] and s[0] == np.float32(0.9)
    r, c, s = find_peaks(p, 3, 0.05, 10)
    assert len(r) == 2


def test_nms():
    X = np.array([0.0, 0.5, 3.0, 0.2, 10.0])
    Y = np.zeros(5)
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cls = np.array([0, 0, 0, 1, 0])
    keep = nms_distance(X, Y, s, 2.0, cls, min_m=1.0, frac=0.8)      # порог max(1, 1.6) = 1.6 м
    assert sorted(keep) == [0, 2, 3, 4]                              # 1 подавлен; 3 — другой класс, выжил
    keep = nms_distance(X, Y, s, 2.0, cls, 1.0, 0.8, cross_class=True)
    assert sorted(keep) == [0, 2, 4]
    keep = nms_distance(X, Y, s, np.array([0.1, 0.1, 5, 0.1, 5]), cls, 1.0, 0.8)
    assert sorted(keep) == [0, 2, 3, 4]                              # 0,5 м < 1 м → подавлен
    assert len(nms_distance([], [], [], 1.0, [])) == 0


def test_decode_plateau_collapses():
    hm = np.zeros((2, 16, 16), np.float32)
    hm[0, 5:7, 5:7] = 0.7
    c = decode(hm, LEFT, TOP, STEP, cfg=cfg_with(nms_min_m=1.0))
    assert len(c["X"]) == 1


def test_strided_decode_and_stitch():
    """Цели на stride 4 как «предсказание» → декод с шагом 2 м восстанавливает центры (через offset) точно."""
    rng = np.random.default_rng(3)
    H, W, S = 900, 1100, 512
    xy = np.column_stack([rng.uniform(5, W - 5, 30), rng.uniform(5, H - 5, 30)])
    objs = np.column_stack([xy, rng.uniform(6, 14, 30), rng.integers(2, size=30)])
    origins = tile_origins(H, W, S, 384, 4)
    assert all(y % 4 == 0 and x % 4 == 0 for y, x in origins)
    outs = []
    for y0, x0 in origins:
        o = objs.copy()
        o[:, 0] -= x0
        o[:, 1] -= y0
        t = render_targets_strided(o, S, 4)
        outs.append(np.concatenate([t["hm"], t["wh"], t["off"]]))
    m = stitch(outs, origins, H, W, S, 4)
    assert m.shape == (5, 225, 275)
    c = decode(m[:2], LEFT, TOP, STEP * 4, wh=m[2:3], off=m[3:5], cfg=cfg_with(min_score=0.5))
    X, Y = LEFT + objs[:, 0] * STEP, TOP - objs[:, 1] * STEP
    d = [np.min(np.hypot(c["X"] - X[i], c["Y"] - Y[i])) for i in range(30)]
    # объекты, разделившие одну клетку 2×2 м с другим объектом того же класса, теряются — пропускаем их
    ok = np.array(d) < 1e-3
    assert ok.mean() >= 0.9, np.sort(d)
