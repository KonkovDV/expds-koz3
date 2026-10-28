"""Повтор тайлов выбранных участков в индексе фолда. Модель не запускается."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nn_oof import repeat_rows


def test_repeat_copies_only_named_regions_and_skips_the_val_family():
    rows = [
        {"region": "hr_a", "family": "А", "idx": "0"},
        {"region": "hr_b", "family": "Б", "idx": "1"},
        {"region": "low", "family": "А", "idx": "2"},
    ]
    out = repeat_rows(rows, {"hr_a", "hr_b"}, 3, skip_families={"Б"})
    assert [r["idx"] for r in out] == ["0", "0", "0", "1", "2"]
    assert out[0] is not out[1]
    assert repeat_rows(rows, {"hr_a"}, 1) == rows


def test_cloud_gpu_farm_build_fold_plan():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import cloud_gpu_farm

    plan = cloud_gpu_farm.build_fold_plan("work/cache/li", "li", "work/oof", "work/oof.geojson",
                                          epochs=10, batch=32, n_nodes=4)
    assert len(plan) == 4
    assert plan[0]["folds"] == [0, 1] and "--fold-indices 0 1 --train-only" in plan[0]["command"]
    assert plan[1]["folds"] == [2, 3] and "--fold-indices 2 3 --train-only" in plan[1]["command"]
    assert plan[2]["folds"] == [4] and "--fold-indices 4 --train-only" in plan[2]["command"]
    assert plan[3]["all_data"] is True and "--all-data" in plan[3]["command"]
