"""[dfine_ae_tile] default OFF; train-matched windows + fuse patch."""
from __future__ import annotations

import tomllib
from pathlib import Path

from expds import dfine_ae_tile as M
from expds import dfine_aeor as A
from expds import sp_dfine as S


def test_default_off_in_toml():
    raw = Path("configs/default.toml").read_bytes()
    cfg = tomllib.loads(raw.decode("utf-8"))
    assert (cfg.get("dfine_ae_tile") or {}).get("enabled") is False
    # dfine_aeor remains aeor windows when tile flag OFF
    st = A.settings(cfg)
    assert st.get("windows_mode", "aeor") == "aeor"
    assert st.get("fuse") == "wbf"


def test_apply_noop_when_off():
    cfg = {"dfine_aeor": {"enabled": False}, "dfine_ae_tile": {"enabled": False}}
    out = M.apply(cfg)
    assert out.get("dfine_aeor") == cfg["dfine_aeor"]


def test_apply_patches_windows_and_fuse():
    cfg = {
        "dfine_aeor": {"enabled": True, "fuse": "wbf", "windows_mode": "aeor"},
        "dfine_ae_tile": {"enabled": True},
    }
    out = M.apply(cfg)
    aeor = out["dfine_aeor"]
    assert aeor["windows_mode"] == "train"
    assert aeor["fuse"] == "nms"
    assert aeor["nms_iou"] == 0.5
    st = A.settings(out)
    assert st["windows_mode"] == "train"
    assert st["fuse"] == "nms"


def test_windows_train_matches_sp_dfine_and_oof():
    cases = [(640, 640), (641, 641), (1000, 800), (1471, 1442), (16656, 16569)]
    for W, H in cases:
        train = A.windows_train(W, H, 640, 160)
        aeor = A.windows_aeor(W, H, 640, 160)
        sp = S.windows(W, H, 640, 160)
        via = A.windows(W, H, 640, 160, mode="train")
        assert train == sp == via, (W, H)
        # same count usually, but last starts differ when W not on grid
        if W not in (640,):
            assert train != aeor or W <= 640


def test_windows_aeor_default_unchanged():
    # legacy pin-edge: W=1000 → last x=360 (not 480)
    wins = A.windows(1000, 800, 640, 160)  # default mode aeor
    xs = sorted({x for x, _ in wins})
    assert xs[-1] == 360
    wins_t = A.windows(1000, 800, 640, 160, mode="train")
    xs_t = sorted({x for x, _ in wins_t})
    assert xs_t[-1] == 480


def test_optional_scales_override():
    cfg = {
        "dfine_aeor": {},
        "dfine_ae_tile": {"enabled": True, "scales": [1.0, 1.5]},
    }
    out = M.apply(cfg)
    assert out["dfine_aeor"]["scales"] == [1.0, 1.5]
