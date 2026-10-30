"""Задача 14: тесты оконных признаков кольца morph_ringw_* на маленьких синтетических ЦМР (без данных)."""
import numpy as np
import morph_fast as MF

RES = MF.RES


def _dem(n=200, mound=True, crater_at=None, crater_r=1.2, crater_d=0.5, tilt=0.0):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    c = n / 2
    x = (xx + 0.5 - c) * RES; y = (c - yy - 0.5) * RES          # м, центр ЦМР = (0, 0)
    z = 100.0 + tilt * x
    if mound:
        r = np.hypot(x, y)
        z += np.where(r < 5, 1.0 * (1 - (r / 5) ** 2), 0)        # насыпь r=5 м, h=1 м
    if crater_at is not None:
        rc = np.hypot(x - crater_at[0], y - crater_at[1])
        z -= np.where(rc < crater_r, crater_d * (1 - (rc / crater_r) ** 2), 0)
    return z.astype(np.float32), -c * RES, c * RES                # dem, left, top


def _run(dem, left, top, cx, cy):
    cx = np.asarray(cx, float); cy = np.asarray(cy, float)
    rw = np.empty((cx.size, 4), np.float32)
    F = MF.features(*MF.windows(dem, left, top, cx, cy), rw=rw)
    return F, rw


def test_shifts():
    assert MF.RW_SY.size == 29 and MF.RW_SD.max() == 3.0 and MF.RW_SD[0] == 0
    assert set(np.unique(np.abs(MF.RW_SY))) <= {0, 2, 4, 6}        # шаг 1 м = 2 пикс.
    assert MF.RW_IDX.shape == (29, MF.nRW)
    print("ok test_shifts")


def test_offcentre_crater_found_by_shift():
    # кратер в центре насыпи; кандидат смещён на 2 м — неоконное кольцо его недооценивает, оконное — нет
    dem, l, t = _dem(crater_at=(0.0, 0.0))
    F, rw = _run(dem, l, t, [0.25 + 2.0, 0.25], [0.25, 0.25])
    ring_w, depth_w, rimbg, off = rw[0]
    assert ring_w > 0.2 and depth_w > 0.1 and rimbg > 0.3, rw
    assert abs(rw[0, 0] - rw[1, 0]) < 1e-5                          # сдвиг 2 м восстанавливает кольцо центра
    assert F[0, MF.J["morph_ring"]] < ring_w - 0.1, F[0, MF.J["morph_ring"]]
    assert abs(off - 2.0) <= 1.0 + 1e-6, off
    assert ring_w >= F[0, MF.J["morph_ring"]] - 1e-6
    _, rw0 = _run(dem, l, t, [0.25], [0.25])                         # кандидат точно в центре
    assert rw0[0, 3] <= 1.0 and rw0[0, 0] > 0.2
    print("ok test_offcentre_crater_found_by_shift", rw[0].round(3), rw0[0].round(3))


def test_intact_mound_and_flat():
    dem, l, t = _dem()                                              # целая насыпь без ямы
    _, rw = _run(dem, l, t, [0.25], [0.25])
    assert rw[0, 1] < 0, rw                                         # кольцо 1,5–5 м ниже центра → глубина < 0
    assert rw[0, 0] < 0.2, rw                                       # лишь склон насыпи при сдвиге 3 м
    demf, l, t = _dem(mound=False, tilt=0.05)                       # ровный склон 5 %: плоскость вычитается
    _, rwf = _run(demf, l, t, [0.25], [0.25])
    assert np.all(np.abs(rwf[0, :3]) < 0.02), rwf
    print("ok test_intact_mound_and_flat", rw[0].round(3), rwf[0].round(3))


def test_nan_and_no_change_to_raw():
    dem, l, t = _dem(crater_at=(1.0, 0.0))
    F1 = MF.features(*MF.windows(dem, l, t, np.array([0.25]), np.array([0.25])))
    F2, _ = _run(dem, l, t, [0.25], [0.25])
    assert np.array_equal(F1, F2, equal_nan=True)                   # флаг не меняет 30 сырых признаков
    nan = np.full_like(dem, np.nan)
    _, rwn = _run(nan, l, t, [0.25], [0.25])
    assert np.isnan(rwn).all()
    print("ok test_nan_and_no_change_to_raw")


if __name__ == "__main__":
    test_shifts(); test_offcentre_crater_found_by_shift(); test_intact_mound_and_flat(); test_nan_and_no_change_to_raw()
