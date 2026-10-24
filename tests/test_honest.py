"""Честный балл: порог группы подбирается без семьи, на которой его проверяют."""
from collections import Counter

import numpy as np

from expds.metric import (
    ClassCurve,
    _f1,
    family_folds,
    honest_counts,
    paired_bootstrap_counts,
    score_counts,
    sweep_groups,
)

KUR = "kurgany_povrezhdennye"


def fam(region: str) -> str:
    return region.split("_")[0]


def curve(rows):
    """rows: (уверенность, попадание, участок, группа) — сортируются по убыванию уверенности."""
    rows = sorted(rows, key=lambda r: -r[0])
    return ClassCurve(conf=np.array([r[0] for r in rows]), hit=np.array([r[1] for r in rows]),
                      pred_region=[r[2] for r in rows], n_gt=0, gt_regions=Counter(),
                      pred_group=[r[3] for r in rows])


def with_gt(c: ClassCurve, gt: dict[str, int]) -> ClassCurve:
    c.gt_regions = Counter(gt)
    c.n_gt = sum(gt.values())
    return c


def test_family_folds_never_split_a_family():
    regions = ["A_1", "A_2", "B_1", "C_1", "C_2", "C_3", "D_1"]
    lofo = family_folds(regions, fam)
    assert sorted(map(sorted, lofo)) == [["A_1", "A_2"], ["B_1"], ["C_1", "C_2", "C_3"], ["D_1"]]
    two = family_folds(regions, fam, 2)
    assert len(two) == 2 and set().union(*two) == set(regions)
    for f in two:
        fams_in = {fam(r) for r in f}
        for other in two:
            if other is not f:
                assert not fams_in & {fam(r) for r in other}
    assert family_folds(regions, fam, 2) == two


def test_same_distribution_gives_the_in_sample_score():
    """Если семьи одинаковы, порог с других семей совпадает со своим, и балл тот же."""
    rows, gt = [], {}
    for f in ("A", "B", "C", "D"):
        r = f"{f}_1"
        rows += [(0.9, True, r, "li"), (0.8, True, r, "li"), (0.6, False, r, "li"), (0.3, False, r, "li")]
        gt[r] = 3
    c = with_gt(curve(rows), gt)
    curves = {KUR: c}
    counts, info = honest_counts(curves, family_folds(gt, fam))
    honest, _ = score_counts(counts)
    res = sweep_groups(curves)
    assert info["unseen_dropped"] == 0
    assert abs(honest - res[KUR]["f1"]) < 1e-9


def test_threshold_fit_on_one_family_does_not_see_it():
    """Семья B — сплошные ложные между 0,66 и 0,7. Порог на всех участках встаёт выше них;
    порог с A и C ставится посередине между 0,9 и 0,4 и ложные B оставляет."""
    rows, gt = [], {}
    for f in ("A", "C"):
        r = f"{f}_1"
        rows += [(0.95, True, r, "sp"), (0.9, True, r, "sp"), (0.4, False, r, "sp")]
        gt[r] = 2
    rows += [(0.7, False, "B_1", "sp"), (0.68, False, "B_1", "sp"), (0.66, False, "B_1", "sp")]
    gt["B_1"] = 1
    curves = {KUR: with_gt(curve(rows), gt)}
    in_sample = sweep_groups(curves)[KUR]["groups"]["sp"]["threshold"]
    assert in_sample > 0.7
    counts, _ = honest_counts(curves, family_folds(gt, fam))
    assert counts[KUR]["B_1"] == [0, 3, 1]
    assert counts[KUR]["A_1"][0] == 2


def test_group_unseen_elsewhere_is_dropped_or_kept():
    rows = [(0.9, True, "A_1", "li"), (0.9, True, "B_1", "li"), (0.8, True, "B_1", "only_b")]
    gt = {"A_1": 1, "B_1": 2}
    curves = {KUR: with_gt(curve(rows), gt)}
    dropped, info = honest_counts(curves, family_folds(gt, fam), unseen="drop")
    assert info["unseen_dropped"] == 1 and dropped[KUR]["B_1"] == [1, 0, 2]
    kept, info = honest_counts(curves, family_folds(gt, fam), unseen="keep")
    assert info["unseen_kept"] == 1 and kept[KUR]["B_1"] == [2, 0, 2]


def test_score_counts_matches_the_metric_formula():
    counts = {KUR: {"A_1": [3, 2, 5], "B_1": [1, 4, 2]}, "gorodishcha": {"A_1": [0, 1, 0]}}
    s, per = score_counts(counts)
    f1 = float(_f1(4, 6, 3))
    assert abs(per[KUR]["f1"] - f1) < 1e-12
    assert abs(s - f1) < 1e-12          # городищ в эталоне нет: класс выпадает из веса


def test_family_bootstrap_is_certain_when_a_wins_everywhere():
    a = {KUR: {f"{f}_1": [3, 0, 4] for f in "ABCDE"}}
    b = {KUR: {f"{f}_1": [2, 1, 4] for f in "ABCDE"}}
    res = paired_bootstrap_counts(a, b, fam, n=200)
    assert res["p_gt0"] == 1.0 and res["families"] == 5 and res["mean"] > 0


def test_scoreboard_honest_uses_the_stored_curves(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("scoreboard", Path(__file__).resolve().parents[1] / "scripts" / "scoreboard.py")
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    rows, gt = [], {}
    for f in ("001_A", "002_B", "003_C"):
        rows += [(0.9, True, f, "li"), (0.5, False, f, "li")]
        gt[f] = 1
    sb.LAST_CURVES["база"] = {KUR: with_gt(curve(rows), gt)}
    scores, counts = sb.honest("база", {"001_A"}, {"003_C"}, -1)
    assert abs(scores[0] - 1.0) < 1e-5 and counts[KUR]["002_B"] == [1, 0, 1]
