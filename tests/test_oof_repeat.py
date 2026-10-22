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
