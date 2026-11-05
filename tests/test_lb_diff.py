"""Сравнение логов площадки (scripts/lb_diff.py) на трёх настоящих логах публичного теста."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BG = ROOT / "docs" / "background"
V2 = BG / "LB_SCORING_v2_eb268d6_0.41238.log"
E1 = BG / "LB_SCORING_e1g1_ed51cae4_0.38988.log"
V4 = BG / "LB_SCORING_v4c1f_f0def982_0.34418.log"


def _mod():
    spec = importlib.util.spec_from_file_location("lb_diff", ROOT / "scripts" / "lb_diff.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _mod()


@pytest.fixture(scope="module")
def res(m):
    texts = [p.read_text(encoding="utf-8") for p in (V2, E1, V4)]
    return m.compare(texts, ["v2", "e1g1", "v4"])


def test_geophysics_gap_of_e1g1_is_dropped_output_that_was_mostly_hits(res):
    geo = res["types"]["геофизика"]
    assert geo["gt"] == 1441
    v2, e1, v4 = geo["runs"]
    assert (v2["tp"], v2["fp"]) == (946, 567)
    assert (e1["tp"], e1["fp"]) == (568, 167)
    v = e1["vs_base"]
    assert (v["dtp"], v["dfp"]) == (-378, -400)
    assert v["kind"] == "drops" and v["pays"] is True
    assert v["hit_share"] == pytest.approx(378 / 778)
    assert v["break_even"] == pytest.approx(e1["f1"] / (4 - e1["f1"]))
    assert (v4["tp"], v4["fp"]) == (448, 355)


def test_v4_resolves_with_unknown_ortho_fp_and_satellite_noise(res):
    # лог v4 не сходится по FP ratio без ложных ортофото: их 2
    assert res["fp_unknown"] == [0, 0, 2]
    sp = res["types"]["спутник"]["runs"]
    assert (sp[2]["tp"], sp[2]["fp"]) == (112, 915)
    v = sp[2]["vs_base"]
    assert v["kind"] == "adds" and v["pays"] is False
    assert v["hit_share"] == pytest.approx(16 / 702)


def test_lidar_and_aerial_identical_across_v2_and_e1g1(res):
    for g in ("лидар", "аэрофото"):
        runs = res["types"][g]["runs"]
        assert runs[1]["vs_base"]["kind"] == "same"


def test_size_buckets_hold_polygons_only(res):
    b = res["sizes"]
    assert b["≤500"]["gt_min"] == 248 and b["≤500"]["tp"] == [80, 69, 70]
    assert b["≤100"]["gt_min"] == 614 and b["≤100"]["tp"] == [265, 268, 274]
    # нулевой recall v2/E1_G1 в корзине 5–10 тыс. м² совместим с эталоном v4 (1 из 5)
    assert b["≤10000"]["gt_min"] == 5 and b["≤10000"]["tp"] == [0, 0, 1]
    # изменение TP корзин между v2 и E1_G1 мало против −378 TP геофизики: потеря — точки (находки)
    d = sum(x["tp"][1] - x["tp"][0] for x in b.values())
    assert abs(d) <= 15


def test_contour_shows_object_poly_pairs_changed(res):
    c = res["contour"]["object_poly"]
    assert c[0][1] == pytest.approx(6.882874) and c[1][1] == pytest.approx(2.229542)
    assert res["contour"]["arkhitektury"][2] is None


def test_joint_bucket_counts_zero_is_compatible_with_any_n(m):
    assert m.joint_bucket_counts([0.0, 0.2]) == (5, [0, 1])
    assert m.joint_bucket_counts([0.0, 0.0]) == (None, [0, 0])


def test_block_verdict_matches_exact_f1_change(m):
    """h > F/(4−F) того прогона, к которому добавляют блок, ⇔ F1 растёт (FP весит 0,5)."""
    def f1(tp, fp, n):
        return 2 * tp / (n + tp + 0.5 * fp)

    n = 1000
    for tp0, fp0, a, b in itertools.product((100, 400, 700), (50, 400, 900), (1, 30, 200), (1, 60, 500)):
        if tp0 + a > n:
            continue
        base_f = f1(tp0, fp0, n)
        more_f = f1(tp0 + a, fp0 + b, n)
        v = m.block_verdict(a, b, base_f, more_f)          # вариант добавляет блок
        assert v["kind"] == "adds"
        assert v["pays"] == (more_f > base_f)
        w = m.block_verdict(-a, -b, more_f, base_f)        # вариант снимает тот же блок
        assert w["kind"] == "drops"
        assert w["pays"] == (more_f > base_f)


def test_render_and_cli(m, tmp_path, capsys):
    out = tmp_path / "d.json"
    assert m.main([str(V2), str(E1), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "ΔTP -378 ΔFP -400" in text and "слишком осторожен" in text
    assert out.exists() and '"геофизика"' in out.read_text(encoding="utf-8")
