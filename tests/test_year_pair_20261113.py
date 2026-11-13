"""Пара 2024/2025: общий сдвиг яркости — не насыпь, локальный круг — пик."""
import numpy as np

from expds.year_pair import disk_contrast, local_change, pick_year_pair, strongest_mark, year_from_tags


def test_numbered_files_are_not_a_year_pair():
    names = [
        "04_Шумгора_SpOr_спутник/01_Шумгора_SpOr_спутник.tif",
        "04_Шумгора_SpOr_спутник/02_Шумгора_SpOr_спутник.tif",
    ]
    assert pick_year_pair(names) is None


def test_years_in_the_name_or_the_tiff_date_select_the_pair():
    names = ["site_2024.tif", "site_2025.tif", "site_2023.tif"]
    assert pick_year_pair(names) == ("site_2024.tif", "site_2025.tif")
    assert year_from_tags({"DateTime": "2024:06:11 08:30:00"}) == 2024
    assert year_from_tags({}) is None


def test_a_global_brightness_shift_is_not_a_mound_and_a_new_disk_is():
    rng = np.random.default_rng(0)
    early = rng.normal(80, 3, (80, 80)).astype(np.float32)
    late = early + 25
    flat = local_change(early, late)
    assert float(flat.max()) < 1e-3

    marked = early.copy()
    rr, cc = np.ogrid[:80, :80]
    disk = (rr - 40) ** 2 + (cc - 50) ** 2 <= 6 ** 2
    marked[disk] += 40
    change = local_change(early, marked)
    row, col, contrast = strongest_mark(change, radius_px=6)
    assert abs(row - 40) <= 2 and abs(col - 50) <= 2
    assert contrast > 15
    edge = np.zeros((80, 80), np.float32)
    edge[:, 40:] = 40
    assert float(disk_contrast(edge, 6).max()) < contrast / 2
