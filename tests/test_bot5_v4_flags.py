"""bot-5 v4: D-FINE SpOr ([sp_dfine]) отделён от Ae/Or ([dfine_aeor]); оба OFF по умолчанию.

Контракт SpOr-off (bot1_v3_spor_diag): enabled=false — фаза не входит → drop_groups не снимает
U-Net sp_or/sp_focal/sp_arch/sp_focal2. drop_groups=[] нужен только при enabled=true без варианта R.
"""
import inspect
import tomllib
from pathlib import Path
from types import SimpleNamespace

from expds import sp_dfine

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def test_sp_dfine_and_dfine_aeor_default_off_independent():
    cfg = _cfg()
    assert cfg["sp_dfine"] == {"enabled": False}
    assert cfg["dfine_aeor"] == {"enabled": False}
    # Включение одного в оверлее не трогает другой ключ конфига.
    only_sp = {**cfg, "sp_dfine": {"enabled": True, "weights": ["models/dfine_f0"]}}
    assert only_sp["dfine_aeor"]["enabled"] is False
    only_ae = {**cfg, "dfine_aeor": {"enabled": True}}
    assert only_ae["sp_dfine"]["enabled"] is False


def test_v4_optional_blocks_default_off():
    cfg = _cfg()
    assert cfg["gor_dense"]["enabled"] is False
    assert cfg["gor_gate"]["enabled"] is False
    assert cfg["orgate"]["enabled"] is False


def test_v4_spor_off_is_enabled_false():
    """Выбранный безопасный выключатель v4: [sp_dfine] enabled=false (не drop_groups=[] при enabled).

    pipeline.run гейтит всю фазу по enabled → drop_groups внутри _sp_dfine_phase не вызывается.
    DEFAULTS.drop_groups уже [] — на случай будущего enabled=true без R.
    """
    cfg = _cfg()
    assert cfg["sp_dfine"]["enabled"] is False
    assert not sp_dfine.enabled(cfg)
    st = sp_dfine.settings(cfg)
    assert st["drop_groups"] == []
    assert sp_dfine.DEFAULTS["drop_groups"] == []


def test_v4_enabled_false_skips_phase_even_if_drop_groups_r(monkeypatch):
    """Даже при drop_groups=R в оверлее enabled=false → фаза не вызывается → U-Net слой не дропается."""
    from expds import pipeline

    src = inspect.getsource(pipeline.run)
    assert 'if (cfg.get("sp_dfine") or {}).get("enabled"):' in src
    # Источник: drop_groups только внутри фазы (до monkeypatch).
    phase_src = inspect.getsource(pipeline._sp_dfine_phase)
    assert 'st["drop_groups"]' in phase_src and "drop_groups(" in phase_src

    called = []

    def boom(*_a, **_k):
        called.append(1)
        raise AssertionError("_sp_dfine_phase must not run when [sp_dfine] enabled=false")

    monkeypatch.setattr(pipeline, "_sp_dfine_phase", boom)

    cfg_r = {
        "sp_dfine": {
            "enabled": False,
            "drop_groups": ["sp_or", "sp_focal", "sp_arch", "sp_focal2"],
        }
    }
    assert not sp_dfine.enabled(cfg_r)
    # Прямой drop при R снимает U-Net — это то, чего фаза НЕ должна делать при enabled=false.
    feats = [
        {"markup_type": "sp_or", "class_name": "kurgany_tselye", "region_name": "a"},
        {"markup_type": "sp_focal", "class_name": "kurgany_tselye", "region_name": "a"},
        {"markup_type": "li", "class_name": "gorodishcha", "region_name": "a"},
    ]
    dropped = sp_dfine.drop_groups(feats, cfg_r["sp_dfine"]["drop_groups"], ["a"])
    assert [f["markup_type"] for f in dropped] == ["li"]
    assert sp_dfine.drop_groups(feats, [], ["a"]) == feats
    assert called == []  # boom не вызывали — фаза не входила


def test_v4_phase_drop_only_when_enabled_and_drop_groups(monkeypatch):
    """Контроль: при enabled=true + drop_groups=R фаза снимает U-Net на fully-complete участках."""
    from expds import pipeline

    class St:
        def __init__(self, feats):
            self.feats = list(feats)

        def add(self, f):
            self.feats += f

        def snapshot(self):
            return list(self.feats)

        def replace(self, f):
            self.feats = list(f)

    class G:
        soft_ceiling = None

        def ok(self, s=0):
            return True

        def elapsed(self):
            return 0.0

    def fake(reg, cfg, guard, cache, root, crs_for=None, soft_ok=None):
        cache["models"] = ["m"]
        return (
            [{
                "markup_type": "sp_or",
                "class_name": "kurgany_tselye",
                "region_name": reg.name,
                "threshold_group": "sp_dfine",
                "confidence": 0.9,
            }],
            True,
        )

    monkeypatch.setattr(sp_dfine, "region_feats", fake)
    old = [{"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": "a"}]
    st = St(old)
    regs = [SimpleNamespace(name="a", rasters={"sp_or": ["x"]})]
    cfg = {"sp_dfine": {"enabled": True, "drop_groups": ["sp_or"]}}
    assert pipeline._sp_dfine_phase(regs, cfg, G(), st, ROOT) == 1
    assert all(f.get("threshold_group") == "sp_dfine" for f in st.feats)
    assert not any(f.get("threshold_group") is None and f["markup_type"] == "sp_or" for f in st.feats)
