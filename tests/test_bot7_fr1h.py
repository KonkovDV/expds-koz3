"""bot-7 v3: приор кадра FR1h (bot-10 REPORT11_audit.md C3h) включён в конфиг сдачи ровно в принятом виде,
стоит до порогов и не трогает городища."""
import tomllib
from pathlib import Path

from shapely.geometry import Point, box

from expds import post

ROOT = Path(__file__).resolve().parents[1]
# W4: FR1h за флагом — в default.toml выключен, включённый вид лежит в configs/v3_on.toml.
CFG = tomllib.loads((ROOT / "configs" / "v3_on.toml").read_text(encoding="utf-8"))
DEFAULT = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def test_fr1h_off_by_default():
    assert not (DEFAULT.get("post") or {}).get("frame_prior")
    assert not DEFAULT["sp_dfine"].get("enabled")


def test_fr1h_config_exact():
    fp = CFG["post"]["frame_prior"]
    assert fp["groups"] == ["sp_or", "sp_focal", "sp_focal2"]
    assert set(fp["classes"]) == {"kurgany_povrezhdennye", "kurgany_tselye", "arkhitektury", "object_poly",
                                  "fortifikatsii", "finds_points"}
    assert "gorodishcha" not in fp["classes"]
    assert fp["d0"] == 0.5 and fp["edge"] == 0.5
    post_cfg = CFG["post"]
    for off in ("site_calibration", "kurgan_neighbour_prior", "kurgans_outside_hillforts"):
        assert not post_cfg.get(off), off


def test_fr1h_skips_gorodishcha_and_scales_edge():
    fp = CFG["post"]["frame_prior"]
    B = {"sp_or": [box(0, 0, 100, 100)]}
    def f(cls, x):
        return {"markup_type": "sp_or", "class_name": cls, "confidence": 0.8, "geom": Point(x, 50)}
    feats = post.annotate_scene_drel([f("gorodishcha", 99.9), f("kurgany_tselye", 99.9), f("kurgany_tselye", 50),
                                      {**f("kurgany_tselye", 99.9), "markup_type": "li"}], B)
    out = post.frame_prior(feats, dict(fp))
    assert out[0]["confidence"] == 0.8                      # городища не трогаются
    assert abs(out[1]["confidence"] - 0.8 * 0.5) < 5e-3     # у края сцены (d 0,998) ≈ × edge
    assert out[2]["confidence"] == 0.8                      # центр
    assert out[3]["confidence"] == 0.8                      # не sp-группа


def test_fr1h_before_thresholds_in_finish():
    src = (ROOT / "expds" / "watchdog.py").read_text(encoding="utf-8")
    for name in ("def finish", "def safe_finish"):
        body = src[src.index(name):]
        a, b, c = body.index("frame_prior("), body.index("apply_thresholds("), body.index("kurgan_cross_modal_twins")
        assert a < b < c, name
