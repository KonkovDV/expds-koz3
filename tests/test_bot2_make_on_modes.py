"""harvest make_on — дефолт C (без R); R явно включает drop_groups."""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKE = ROOT / "archive_review/analysis/2026-10-29/bots/bot-7/work6/on/make_on.py"
if not MAKE.is_file():
    pytestmark = pytest.mark.skip(reason="скрипт на ветке archive/worked-20261031")


def test_make_on_default_mode_is_c():
    src = MAKE.read_text(encoding="utf-8")
    assert 'else "C"' in src or "else 'C'" in src
    assert "по умолчанию C" in src or 'дефолт C' in src


def test_make_on_c_vs_r_drop_groups_injection():
    """Реплика логики подстановки конфига из make_on (без zip/весов)."""
    THR = {"kurgany_povrezhdennye": 0.55, "kurgany_tselye": 0.67, "gorodishcha": 0.68,
           "arkhitektury": 0.37, "fortifikatsii": 0.86}
    MNAME = "dfine_f0"
    cfg0 = "[sp_dfine]\nenabled = false\n\n[thresholds_by_markup.or]\narkhitektury = 0.065565\n"
    for mode, want_drop in (("C", False), ("R", True)):
        cfg = cfg0
        if mode == "R":
            drop = '\ndrop_groups = ["sp_or", "sp_focal", "sp_arch", "sp_focal2"]\n'
        else:
            drop = "\n# вариант C — без drop_groups (v2 SpOr + D-FINE рядом)\n"
        cfg, n = re.subn(
            r"(?m)^\[sp_dfine\]\nenabled = false\n",
            (
                "[sp_dfine]\nenabled = true\nweights = [\"models/MNAME\"]\n"
                "flip_tta = true\nfp16 = true\nconvert = \"circle\"\n"
                "DROPPLACEHOLDER\n[thresholds_by_markup.sp_dfine]\nTHRLINES\n"
            )
            .replace("DROPPLACEHOLDER", drop)
            .replace("MNAME", MNAME)
            .replace("THRLINES", "\n".join(f"{k} = {v}" for k, v in THR.items())),
            cfg,
        )
        assert n == 1
        assert ("drop_groups = [" in cfg) is want_drop, (mode, cfg)


def test_make_on_overlays_pipeline():
    src = MAKE.read_text(encoding="utf-8")
    assert 'pipeline.py' in src
    tree = ast.parse(src)
    assert tree is not None
