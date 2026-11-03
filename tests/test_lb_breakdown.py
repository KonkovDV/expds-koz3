"""Офлайн-разбивка как scoring.log (scripts/lb_breakdown.py)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    path = ROOT / "scripts" / "lb_breakdown.py"
    spec = importlib.util.spec_from_file_location("lb_breakdown", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _square(x0, y0, s, cls, markup, conf=0.9):
    return {
        "type": "Feature",
        "properties": {"class_name": cls, "markup_type": markup, "confidence": conf},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s], [x0, y0],
            ]],
        },
    }


def _fc(feats):
    return {"type": "FeatureCollection", "features": feats}


@pytest.fixture(scope="module")
def m():
    return _mod()


def test_size_bucket_labels_match_lb_log(m):
    bins = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
    assert m.size_bucket_label(100.0, bins) == "≤ 100 м²"
    assert m.size_bucket_label(500.0, bins) == "≤ 500 м²"
    assert m.size_bucket_label(1000.0, bins) == "≤ 1000 м²"
    assert m.size_bucket_label(5000.0, bins) == "≤ 5000 м²"
    assert m.size_bucket_label(10000.0, bins) == "≤ 10000 м²"
    assert m.size_bucket_label(None, bins) == "> 10000 м²"


def test_size_bucket_edges_via_calculator(m):
    """Границы корзин: area==100 → ≤100; 100+ε → ≤500; >10000 → последняя."""
    from expds.metric import score

    # UTM-подобные координаты в метрах (CRS auto часто трактует большие числа как метры)
    # Площадь квадрата со стороной s = s².
    gt_feats = [
        _square(0, 0, 10, "kurgany_tselye", "li"),       # 100 м² → bin0
        _square(100, 0, 10.0001, "kurgany_tselye", "li"),  # ~100.002 → bin1 (если >100)
        _square(200, 0, 22.4, "kurgany_tselye", "li"),    # ~501.76 → bin2 (>500)
        _square(300, 0, 110, "kurgany_tselye", "ae"),     # 12100 → >10000
    ]
    # Полное перекрытие pred=gt → все TP
    pred = _fc([dict(f) for f in gt_feats])
    gt = _fc(gt_feats)
    res = score(pred, gt, str(ROOT / "materials" / "compute_metrics.py"))
    bd = m.extract_breakdown(res)
    cats = bd["size_categories"]
    assert len(cats) == 6
    # Ровно на границе 100: ≤100
    assert cats[0]["gt"] >= 1
    # Квадрат 110×110 = 12100 → последняя корзина
    assert cats[-1]["gt"] >= 1
    # recall при полном матче = 1.0 там где gt>0
    for c in cats:
        if c["gt"] > 0:
            assert c["recall"] == pytest.approx(1.0)


def test_soft_recall_hit_and_miss(m):
    from expds.metric import score

    gt = _fc([_square(0, 0, 20, "kurgany_tselye", "li")])
    # hit: полное перекрытие
    pred_hit = _fc([_square(0, 0, 20, "kurgany_tselye", "li")])
    r_hit = score(pred_hit, gt, str(ROOT / "materials" / "compute_metrics.py"))
    assert r_hit["auxiliary"]["soft_recall"] == pytest.approx(1.0, abs=1e-5)

    # miss: далеко
    pred_miss = _fc([_square(1000, 1000, 20, "kurgany_tselye", "li")])
    r_miss = score(pred_miss, gt, str(ROOT / "materials" / "compute_metrics.py"))
    assert r_miss["auxiliary"]["soft_recall"] == pytest.approx(0.0, abs=1e-5)


def test_fp_ratio_and_center_deviation(m):
    from expds.metric import score

    gt = _fc([_square(0, 0, 20, "kurgany_tselye", "li")])
    # 1 TP + 1 FP
    pred = _fc([
        _square(0, 0, 20, "kurgany_tselye", "li"),
        _square(500, 500, 20, "kurgany_tselye", "li"),
    ])
    r = score(pred, gt, str(ROOT / "materials" / "compute_metrics.py"))
    aux = r["auxiliary"]
    assert aux["fp_ratio"] == pytest.approx(1.0, abs=1e-4)
    assert aux["center_deviation"] >= 0.0
    # TP совпадает по центру → deviation ≈ 0
    assert aux["center_deviation"] == pytest.approx(0.0, abs=1e-6)


def test_data_type_mapping_li_ae(m):
    from expds.metric import score

    gt = _fc([
        _square(0, 0, 20, "kurgany_tselye", "li"),
        _square(100, 0, 20, "kurgany_tselye", "ae"),
    ])
    pred = _fc([
        _square(0, 0, 20, "kurgany_tselye", "li"),
        _square(100, 0, 20, "kurgany_tselye", "ae"),
    ])
    r = score(pred, gt, str(ROOT / "materials" / "compute_metrics.py"))
    bd = m.extract_breakdown(r)
    assert "лидар" in bd["by_data_type"]
    assert "аэрофото" in bd["by_data_type"]
    assert bd["by_data_type"]["лидар"]["recall"] == pytest.approx(1.0, abs=1e-4)
    assert bd["by_data_type"]["аэрофото"]["recall"] == pytest.approx(1.0, abs=1e-4)


def test_format_scoring_log_diffable_headers(m):
    """Каркас строк совпадает с LB-логом (заголовки + 5 типов + 6 корзин)."""
    fake = {
        "final_score": 0.4123779189,
        "per_class": {},
        "parameters": {"target_classes": ["kurgany_tselye", "finds_points"], "size_bins": m.DEFAULT_SIZE_BINS},
        "auxiliary": {
            "soft_recall": 0.622291,
            "center_deviation": 1.133383,
            "fp_ratio": 1.139323,
            "contour": {"boundary_iou": 0.001875, "hausdorff95_m": 7.947175, "polis_m": 4.735235, "pairs": 1},
            "contour_by_class": {
                "kurgany_tselye": {
                    "boundary_iou": 0.000499, "hausdorff95_m": 5.821141,
                    "polis_m": 3.944770, "pairs": 1,
                },
            },
            "by_data_type": {
                "лидар": {"tp": 1, "fp": 0, "fn": 0, "precision": 0.516472, "recall": 0.504149},
                "аэрофото": {"tp": 0, "fp": 0, "fn": 1, "precision": 0.090909, "recall": 0.013333},
                "спутник": {"tp": 0, "fp": 0, "fn": 1, "precision": 0.456057, "recall": 0.254642},
                "ортофото": {"tp": 0, "fp": 0, "fn": 1, "precision": 0.368421, "recall": 0.140000},
                "геофизика": {"tp": 0, "fp": 0, "fn": 1, "precision": 0.769418, "recall": 0.656489},
            },
            "size_categories": [
                {"max_area": 100.0, "gt": 1, "tp": 0, "recall": 0.431596},
                {"max_area": 500.0, "gt": 1, "tp": 0, "recall": 0.322581},
                {"max_area": 1000.0, "gt": 1, "tp": 0, "recall": 0.326087},
                {"max_area": 5000.0, "gt": 1, "tp": 0, "recall": 0.404762},
                {"max_area": 10000.0, "gt": 1, "tp": 0, "recall": 0.0},
                {"max_area": None, "gt": 1, "tp": 0, "recall": 0.294118},
            ],
        },
    }
    text = m.format_scoring_log(fake, archive="submission_bot7cand_v2_eb268d6.zip",
                                inference_time="3 мин 48 с / 30 мин")
    assert text.startswith("===== ИТОГИ ОЦЕНКИ · КОЗ-3 =====")
    assert "Soft recall:        0.622291" in text
    assert "Center deviation:   1.133383 м" in text
    assert "FP ratio:           1.139323" in text
    assert "лидар      precision 0.516472   recall 0.504149" in text
    assert "геофизика  precision 0.769418   recall 0.656489" in text
    assert "(сумма всех классов: maggr, etgr, terfl)" in text
    assert "≤ 100 м²     recall 0.431596" in text
    assert "≤ 10000 м²   recall 0.000000" in text
    assert "> 10000 м²   recall 0.294118" in text
    assert "finds_points           — (контур не применим)" in text
    assert "kurgany_tselye         BIoU 0.000499   HD95 5.821141 м   PoLiS 3.944770 м" in text
    # parseable by lb_counts
    import importlib.util as iu
    spec = iu.spec_from_file_location("lb_counts", ROOT / "scripts" / "lb_counts.py")
    lc = iu.module_from_spec(spec)
    spec.loader.exec_module(lc)
    parsed = lc.parse(text)
    assert parsed["final_score"] == pytest.approx(0.4123779189)
    assert parsed["soft_recall"] == pytest.approx(0.622291)
    assert parsed["groups"]["лидар"] == (pytest.approx(0.516472), pytest.approx(0.504149))


def test_write_outputs_and_side_by_side(m, tmp_path):
    fake = {
        "final_score": 0.5,
        "per_class": {},
        "parameters": {"size_bins": m.DEFAULT_SIZE_BINS, "target_classes": ["kurgany_tselye"]},
        "auxiliary": {
            "soft_recall": 0.5,
            "center_deviation": 1.0,
            "fp_ratio": 1.0,
            "contour": {"boundary_iou": None, "hausdorff95_m": None, "polis_m": None, "pairs": 0},
            "contour_by_class": {},
            "by_data_type": {
                "лидар": {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5},
            },
            "size_categories": [
                {"max_area": b, "gt": 0, "tp": 0, "recall": None} for b in m.DEFAULT_SIZE_BINS
            ] + [{"max_area": None, "gt": 0, "tp": 0, "recall": None}],
        },
    }
    paths = m.write_outputs(fake, tmp_path, stem="scoring_mock")
    assert paths["log"].exists() and paths["json"].exists() and paths["csv"].exists()
    bd = json.loads(paths["json"].read_text(encoding="utf-8"))
    table = m.side_by_side({"LB": bd, "mock": bd})
    assert "лидар.recall" in table
    assert "size ≤ 100 м²" in table


def test_roundtrip_real_lb_log_headers(m):
    """Заголовки секций реального LB-лога на месте (для diff-контракта)."""
    lb = Path("/home/box/shared/koz3/reports/lb/scoring_v2_eb268d6_0.41238.log")
    if not lb.exists():
        pytest.skip("shared LB log missing")
    text = lb.read_text(encoding="utf-8")
    for needle in (
        "===== ИТОГИ ОЦЕНКИ · КОЗ-3 =====",
        "Сводные показатели:",
        "Soft recall:",
        "По типам данных:",
        "По размерам GT-объектов:",
        "≤ 100 м²",
        "> 10000 м²",
        "лидар",
        "геофизика",
    ):
        assert needle in text


def test_final_score_from_per_class_weights(m):
    """final_score = Σ w·F1 / Σ w; ∂/∂F1_c = w_c/W."""
    per = {
        "kurgany_povrezhdennye": {"f1": 0.5, "gt_count": 10},
        "kurgany_tselye": {"f1": 0.4, "gt_count": 10},
        "gorodishcha": {"f1": 0.3, "gt_count": 10},
        "arkhitektury": {"f1": 0.2, "gt_count": 10},
        "object_poly": {"f1": 0.1, "gt_count": 10},
        "fortifikatsii": {"f1": 0.0, "gt_count": 10},
        "finds_points": {"f1": 1.0, "gt_count": 10},
    }
    # 30*0.5 + 20*0.4 + 15*0.3 + 10*0.2 + 10*0.1 + 5*0.0 + 10*1.0 = 15+8+4.5+2+1+0+10 = 40.5
    assert m.final_score_from_per_class(per) == pytest.approx(0.405)
    partials = m.class_partial_derivatives()
    assert partials["kurgany_povrezhdennye"] == pytest.approx(0.30)
    assert partials["fortifikatsii"] == pytest.approx(0.05)
    assert sum(partials.values()) == pytest.approx(1.0)


def test_reconcile_lb_log_cannot_reproduce_from_types(m):
    lb = Path("/home/box/shared/koz3/reports/lb/scoring_v2_eb268d6_0.41238.log")
    if not lb.exists():
        pytest.skip("shared LB log missing")
    v = m.reconcile_final_score_with_lb_log(lb.read_text(encoding="utf-8"))
    assert v["log_matches_expected"] is True
    assert v["reproduces_from_type_size"] is False
    assert v["type_share_weighted_f1"] == pytest.approx(0.566868, abs=1e-3)
    assert abs(v["type_weighted_gap"]) > 0.1  # 0.567 vs 0.412
    # shares sum ~1
    assert abs(sum(v["lb_type_shares"].values()) - 1.0) < 1e-6
    assert v["lb_type_shares"]["геофизика"] == pytest.approx(1441 / 2907, abs=1e-4)


def test_expected_delta_lb_proxy_and_assumptions(m):
    """ΔLB_hat(X) = s_X^LB · ΔF1_X; labelled assumptions present."""
    base = {
        "by_data_type": {
            "лидар": {"precision": 0.5, "recall": 0.5},
            "аэрофото": {"precision": 0.1, "recall": 0.1},
            "спутник": {"precision": 0.4, "recall": 0.4},
            "ортофото": {"precision": 0.2, "recall": 0.2},
            "геофизика": {"precision": 0.7, "recall": 0.7},
        },
        "per_class": {
            "kurgany_povrezhdennye": {"f1": 0.4, "gt_count": 1},
            "kurgany_tselye": {"f1": 0.4, "gt_count": 1},
            "gorodishcha": {"f1": 0.4, "gt_count": 1},
            "arkhitektury": {"f1": 0.4, "gt_count": 1},
            "object_poly": {"f1": 0.4, "gt_count": 1},
            "fortifikatsii": {"f1": 0.4, "gt_count": 1},
            "finds_points": {"f1": 0.4, "gt_count": 1},
        },
    }
    exp = {
        "by_data_type": {
            "лидар": {"precision": 0.6, "recall": 0.6},  # F1 0.5 → 0.6, Δ=+0.1
            "аэрофото": {"precision": 0.1, "recall": 0.1},
            "спутник": {"precision": 0.4, "recall": 0.4},
            "ортофото": {"precision": 0.2, "recall": 0.2},
            "геофизика": {"precision": 0.7, "recall": 0.7},
        },
        "per_class": {
            "kurgany_povrezhdennye": {"f1": 0.41, "gt_count": 1},
            "kurgany_tselye": {"f1": 0.4, "gt_count": 1},
            "gorodishcha": {"f1": 0.4, "gt_count": 1},
            "arkhitektury": {"f1": 0.4, "gt_count": 1},
            "object_poly": {"f1": 0.4, "gt_count": 1},
            "fortifikatsii": {"f1": 0.4, "gt_count": 1},
            "finds_points": {"f1": 0.4, "gt_count": 1},
        },
    }
    d = m.delta_f1_by_type(base, exp)
    assert d["лидар"] == pytest.approx(0.1, abs=1e-9)
    table = m.expected_delta_lb_from_type_deltas(d)
    li = next(r for r in table["rows"] if r["type"] == "лидар")
    assert li["expected_delta_lb"] == pytest.approx((964 / 2907) * 0.1, abs=1e-9)
    assert "type metrics are auxiliary" in table["assumptions"][0]
    text = m.format_expected_delta_lb(table)
    assert "expected_delta_lb" in text
    assert "∂S/∂F1" in text
    # exact class delta: only kurgany_povrezhdennye +0.01 → Δ = 0.30*0.01 = 0.003
    assert m.final_score_from_per_class(exp["per_class"]) - m.final_score_from_per_class(base["per_class"]) == pytest.approx(0.003)


def test_type_f1_harmonic(m):
    assert m.type_f1(0.5, 0.5) == pytest.approx(0.5)
    assert m.type_f1(1.0, 0.0) == pytest.approx(0.0)
    assert m.type_f1(None, 0.5) is None
