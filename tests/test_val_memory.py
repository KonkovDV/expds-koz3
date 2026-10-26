"""Эталон трейна внутри рамок выборки (scripts/val_memory.py): память, дыры, вырезанные данные."""
import importlib.util

import numpy as np
from conftest import ROOT


def load_script():
    spec = importlib.util.spec_from_file_location("val_memory", ROOT / "scripts" / "val_memory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grid(x0, x1, y0, y1, step, skip=None):
    pts = [(x, y) for x in np.arange(x0, x1, step) for y in np.arange(y0, y1, step)]
    if skip:
        pts = [(x, y) for x, y in pts if not (skip[0] <= x <= skip[2] and skip[1] <= y <= skip[3])]
    lon, lat = (np.array(v) for v in zip(*pts))
    return lon, lat, np.array(["kurgany_tselye"] * len(lon)), np.array(["li"] * len(lon))


def test_regimes_memory_hole_and_cut_data():
    vm = load_script()
    train = [("li", (37.0, 55.0, 37.02, 55.02))]
    vbox = (37.008, 55.008, 37.012, 55.012)
    lon, lat, cls, mt = grid(37.0, 37.02, 55.0, 55.02, 0.0005)
    a = vm.assess(vbox, "li", train, lon, lat, cls, mt)
    assert a["regime"] == "A" and a["train_cover"] == 1.0 and 0.7 < a["ratio"] < 1.4 and a["gt_same_type"] > 0
    lon, lat, cls, mt = grid(37.0, 37.02, 55.0, 55.02, 0.0005, skip=vbox)
    b = vm.assess(vbox, "li", train, lon, lat, cls, mt)
    assert b["regime"] == "B" and b["gt_same_type"] == 0 and b["ratio"] == 0.0
    c = vm.assess(vbox, "sp_or", train, lon, lat, cls, mt)
    assert c["regime"] == "C" and c["train_cover"] == 0.0


def test_geophysics_types_pool_and_summary_compares_platform():
    vm = load_script()
    assert vm.platform_type("MagGr") == "geophysics" and vm.platform_type("etgr") == "geophysics"
    rows = [{"type": "li", "gt_same_type": 482, "regime": "A"}, {"type": "maggr", "gt_same_type": 10, "regime": "B"}]
    s = vm.summarize(rows)
    assert s["li"]["share_of_platform_gt"] == 0.5 and s["geophysics"]["regimes"] == {"B": 1}


def test_cli_on_same_folder_sees_full_cover_and_counts_gt(val_dir, tmp_path):
    import json
    import subprocess
    import sys

    from conftest import MOUNDS, UTM, X0, Y0
    from shapely.geometry import Point

    from expds.crs import WEB_MERCATOR, to_crs

    feats = [{"type": "Feature", "geometry": to_crs(Point(X0 + x, Y0 + y), UTM, WEB_MERCATOR).__geo_interface__,
              "properties": {"class_name": "kurgany_tselye", "markup_type": "li", "region_name": "region_0001"}}
             for x, y, _, _ in MOUNDS]
    gt = tmp_path / "gt.geojson"
    gt.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    out = tmp_path / "vm.json"
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "val_memory.py"), "--train", str(val_dir),
                          "--gt", str(gt), "--val", str(val_dir), "--sample", "50", "--out", str(out)],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    res = json.loads(out.read_text(encoding="utf-8"))
    by_type = {r["type"]: r for r in res["files"]}
    assert by_type["li"]["train_cover"] == 1.0 and by_type["li"]["gt_same_type"] == len(MOUNDS)
    assert by_type["maggr"]["train_valid"] == 1.0 and by_type["maggr"]["gt_same_type"] == 0
    assert all(r["region"].startswith("val_") for r in res["files"])
    assert res["summary"]["li"]["platform_gt"] == 964
