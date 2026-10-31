"""ground.rasterise via Rust expds_fast (las_grid_fast) vs numpy (EXPDS_FAST=0 path = ground.rasterise_numpy) vs the
reference oracle DTM (tests/lidar_a_oracle.py, read-only competitor copy, loaded as a test oracle only).
Bit-equality on real sites + × (median of 3, wall and process CPU, 2 threads). RSS guard 1950 MiB.

usage: python bench_las_grid_fast.py LAS...   → bench/las_grid_fast/site_<id>.json
"""
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("LIDAR_A_REPO", ROOT / "wt-cpua-clean"))
sys.path.insert(0, str(REPO / "inference" / "lib")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(ROOT / "bench"))
from bench_cpu_a_clean import PEAK, _guard, bits, meas  # noqa: E402

import numpy as np  # noqa: E402

OUT = ROOT / "bench" / "las_grid_fast"


def with_env(val, fn):
    old = os.environ.get("EXPDS_FAST")
    os.environ["EXPDS_FAST"] = val
    try:
        return fn()
    finally:
        os.environ.pop("EXPDS_FAST") if old is None else os.environ.__setitem__("EXPDS_FAST", old)


def site(path, extra_modes):
    import torch
    from expds import las_grid_fast
    from expds.lidar_a import ground, runner
    from lidar_a_oracle import oracle
    assert las_grid_fast._fast is not None, "expds_fast not importable"
    r = {"site": path, "backend": las_grid_fast.backend(), "rayon": os.environ.get("RAYON_NUM_THREADS"),
         "torch_threads": torch.get_num_threads(), "expds_cpu_threads": os.environ.get("EXPDS_CPU_THREADS")}
    mins, maxs, npts = ground.header_box(path); grid = ground.grid_of(mins, maxs); mode = ground.ground_mode(path)
    r.update(points=npts, cells=grid.rows * grid.cols, mode=mode)
    # rasterise: numpy vs rust, median of 3
    m_np, a = meas(lambda: ground.rasterise_numpy(path, grid, mode)); gc.collect()
    m_rs, b = meas(lambda: ground.rasterise(path, grid, mode))
    c = with_env("0", lambda: ground.rasterise(path, grid, mode))
    r["rasterise_numpy"], r["rasterise_rust"] = m_np, m_rs
    r["x_rasterise"] = [m_np["wall_med"] / m_rs["wall_med"], m_np["cpu_med"] / m_rs["cpu_med"]]
    r["rasterise_rust_eq_numpy"] = bits(a, b); r["rasterise_fast0_eq_numpy"] = bits(a, c)
    del a, b, c; gc.collect()
    for m in extra_modes:   # other ground modes on the same real cloud (equality only)
        a = ground.rasterise_numpy(path, grid, m); b = ground.rasterise(path, grid, m)
        r[f"rasterise_{m}_rust_eq_numpy"] = bits(a, b); del a, b; gc.collect()
    # whole DTM: numpy path vs rust path, median of 3; both vs the reference oracle
    m_dn, dn = with_env("0", lambda: meas(lambda: ground.make_dtm(path))); gc.collect()
    m_dr, dr = meas(lambda: ground.make_dtm(path))
    r["dtm_numpy"], r["dtm_rust"] = m_dn, m_dr
    r["x_dtm"] = [m_dn["wall_med"] / m_dr["wall_med"], m_dn["cpu_med"] / m_dr["cpu_med"]]
    r["dtm_rust_eq_numpy"] = bits(dn[0], dr[0]) and bits(dn[1], dr[1]) and dn[2] == dr[2]
    del dn; gc.collect()
    rd, rv, ro = oracle("dtm").build(path)
    r["dtm_rust_eq_oracle"] = bits(dr[0], rd) and bits(dr[1], rv) and tuple(dr[2]) == tuple(ro)
    del rd, rv; gc.collect()
    if os.environ.get("BENCH_RELIEF") == "1":   # CPU relief through torch_all_cores: 1 vs EXPDS_CPU_THREADS threads
        d, v, cell = dr[0], dr[1], dr[2][2]
        with torch.no_grad():
            os.environ["EXPDS_CPU_THREADS"] = "1"; m1, s1 = meas(lambda: runner.stacks_for(d, v, cell, [""], "cpu"))
            os.environ["EXPDS_CPU_THREADS"] = "2"; m2, s2 = meas(lambda: runner.stacks_for(d, v, cell, [""], "cpu"))
        r["relief_1t"], r["relief_2t"] = m1, m2
        r["relief_2t_eq_1t"] = bits(s1[0][""], s2[0][""]) and bits(s1[1], s2[1])
        r["x_relief_2t"] = [m1["wall_med"] / m2["wall_med"], m1["cpu_med"] / m2["cpu_med"]]
    r["peak_rss_mb"] = PEAK[0]
    return r


if __name__ == "__main__":
    threading.Thread(target=_guard, daemon=True).start()
    OUT.mkdir(parents=True, exist_ok=True)
    extra = [m for m in os.environ.get("EXTRA_MODES", "").split(",") if m]
    for s in sys.argv[1:]:
        PEAK[0] = 0
        r = site(s, extra)
        name = f"site_{Path(s).parent.parent.name.split('_')[0]}"
        (OUT / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(name, {k: v for k, v in r.items() if not isinstance(v, dict)}, flush=True)
