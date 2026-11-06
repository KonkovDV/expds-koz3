"""C2 (ensemble_dirs) и SATf (sat_finds) — opt-in; выключены → те же бустеры и карты, что без них."""
import numpy as np
import pytest

G = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (module is not in this tree)")
if not hasattr(G, "ensemble_use"):
    pytest.skip("geo_pixgbm without the C2/SATf patch from bots/bot3-opt", allow_module_level=True)


def test_off_is_identity():
    use = {"poly": object(), "finds": object()}
    assert G.ensemble_use(use, []) is use
    P = {"poly": np.ones((2, 2), np.float32)}
    assert G.ensemble_mean(P, use, []) is P
    assert G.ensemble_members([], "lightgbm") == []
    assert G.sat_finds_setup(None) is None and G.sat_finds_setup({}) is None


def test_mean_of_members():
    a, b, c = object(), object(), object()
    ens = [{"poly": a}, {"poly": b}, {"poly": c}]
    u = G.ensemble_use({"poly": a}, ens)
    assert list(u) == ["poly#0", "poly#1", "poly#2"] and u["poly#1"] is b
    P = {"poly#0": np.full((2, 2), 0.3, np.float32), "poly#1": np.full((2, 2), 0.6, np.float32),
         "poly#2": np.full((2, 2), 0.9, np.float32)}
    out = G.ensemble_mean(P, {"poly": a}, ens)
    assert set(out) == {"poly"} and np.allclose(out["poly"], 0.6) and out["poly"].dtype == np.float32


def test_single_member_exact():
    """Среднее одной модели — те же байты (деление на 1 в float32 точно)."""
    x = np.random.default_rng(0).random((5, 7)).astype(np.float32)
    out = G.ensemble_mean({"poly#0": x.copy()}, {"poly": 1}, [{"poly": 1}])
    assert out["poly"].tobytes() == x.tobytes()


def test_missing_member_disables(tmp_path):
    assert G.ensemble_members([str(tmp_path)], "lightgbm", ("poly",)) == []
