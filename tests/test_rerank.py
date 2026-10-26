"""Второй этап по контексту стека (scripts/rerank_oof.py)."""
import importlib.util
import json

import numpy as np
from conftest import ROOT, UTM, X0, Y0
from shapely.geometry import Point

from expds.crs import WEB_MERCATOR, to_crs

METRIC = str(ROOT / "materials" / "compute_metrics.py")
FAMILIES = ["Альфа", "Бета", "Гамма", "Дельта", "Эпсилон", "Дзета", "Эта", "Тета"]


def load_script():
    spec = importlib.util.spec_from_file_location("rerank_oof", ROOT / "scripts" / "rerank_oof.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def feat(x, y, r, cls, region, conf=None, group=None, markup="li"):
    g = to_crs(Point(X0 + x, Y0 + y).buffer(r), UTM, WEB_MERCATOR)
    props = {"class_name": cls, "region_name": region, "markup_type": markup}
    if conf is not None:
        props["confidence"] = conf
    if group:
        props["threshold_group"] = group
    return {"type": "Feature", "geometry": g.__geo_interface__, "properties": props}


def scene(seed: int = 0):
    """Могильники по 5 курганов; ложные кандидаты одиночны. Уверенность сети случайна у тех и других."""
    rng = np.random.default_rng(seed)
    p, t = "kurgany_povrezhdennye", "kurgany_tselye"
    gt, stack = [], []
    for k, fam in enumerate(FAMILIES):
        region = f"{101 + k:03d}_{fam}_1"
        ox = k * 5000.0
        for c in range(5):
            cx, cy = ox + c * 300.0, 0.0
            for j in range(5):
                x, y = cx + j * 20.0, cy + (j % 2) * 15.0
                gt.append(feat(x, y, 5.0, p, region, markup="li"))
                conf = float(rng.uniform(0.2, 0.8))
                stack.append(feat(x + rng.normal(0, 0.5), y + rng.normal(0, 0.5), 1.5, p, region, conf, "li"))
                stack.append(feat(x, y, 1.5, t, region, conf * 0.5, "li"))
                if rng.random() < 0.6:
                    stack.append(feat(x, y, 4.0, p, region, float(rng.uniform(0.3, 0.9)), "sp_or", "sp_or"))
        for j in range(25):
            x, y = ox + 150.0 + (j % 5) * 300.0, 200.0 + (j // 5) * 120.0
            stack.append(feat(x, y, 1.5, p, region, float(rng.uniform(0.2, 0.8)), "li"))
        for j in range(8):
            stack.append(feat(ox + 60.0 + j * 170.0, -200.0, 4.0, p, region, float(rng.uniform(0.3, 0.9)), "sp_or", "sp_or"))
    return {"type": "FeatureCollection", "features": stack}, {"type": "FeatureCollection", "features": gt}


def best_score(fc: dict, gt: dict) -> float:
    from expds.metric import build_curves, score_group_thresholds, sweep_groups

    curves = build_curves(fc, gt, METRIC)
    res = sweep_groups(curves)
    taus: dict = {}
    for cls, v in res.items():
        for g, gv in v["groups"].items():
            taus.setdefault(g, {})[cls] = gv["threshold"]
    return score_group_thresholds(curves, taus)[0]


def test_context_rerank_separates_cemeteries_from_lone_peaks():
    rr = load_script()
    stack, gt = scene()
    out, report = rr.rerank(stack, gt, METRIC, folds=4, min_pos=30)
    pv = report["kurgany_povrezhdennye"]
    assert pv["reranked"] and pv["auc_base"] < 0.65 and pv["auc_rerank"] > pv["auc_base"] + 0.2
    assert report["kurgany_tselye"]["reranked"] is False                      # попаданий нет — класс не трогается
    props = [f["properties"] for f in out["features"]]
    assert all("confidence_base" in q for q in props)
    assert all(q["confidence"] == q["confidence_base"] for q in props if q["class_name"] == "kurgany_tselye")
    assert best_score(out, gt) > best_score(stack, gt) + 0.02


def test_rerank_cli_writes_the_stack_with_base_confidence(tmp_path):
    import subprocess
    import sys

    stack, gt = scene(seed=1)
    (tmp_path / "s.geojson").write_text(json.dumps(stack), encoding="utf-8")
    (tmp_path / "g.geojson").write_text(json.dumps(gt), encoding="utf-8")
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "rerank_oof.py"), "--stack", str(tmp_path / "s.geojson"),
                          "--gt", str(tmp_path / "g.geojson"), "--out", str(tmp_path / "o.geojson"), "--folds", "4"],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    out = json.loads((tmp_path / "o.geojson").read_text(encoding="utf-8"))
    assert len(out["features"]) == len(stack["features"])
    assert json.loads(run.stdout)["kurgany_povrezhdennye"]["reranked"] is True


def test_related_class_nearby_is_a_feature():
    """Вал у городища видит городище в 200 м; одиночный вал в поле — нет."""
    rr = load_script()

    def row(cls, x, r, conf, group):
        return {"cls": cls, "geom": Point(x, 0.0).buffer(r), "conf": conf, "region": "101_Альфа_1",
                "group": group, "markup": "li"}

    rows = [row("fortifikatsii", 0.0, 5.0, 0.3, "li_fort"), row("gorodishcha", 150.0, 20.0, 0.8, "li_mask"),
            row("fortifikatsii", 5000.0, 5.0, 0.3, "li_fort")]
    X = rr.features(rows)
    rel = X[:, rr.FEATURES.index("rel_max200")]
    assert list(rel) == [0.8, 0.3, 0.0]
    assert list(X[:, rr.FEATURES.index("n200_all")]) == [1.0, 1.0, 0.0]
