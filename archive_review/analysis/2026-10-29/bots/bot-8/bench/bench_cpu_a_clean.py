"""Clean-room branch bench (bot8-cpu-a-clean, expds.lidar_a): bit-equality vs the reference on real sites and × of
items (2) (3) (6), median of 3 (wall + process CPU), 2 threads. The reference is loaded only as a test oracle from the
read-only competitor copy (tests/lidar_a_oracle.py); nothing of it is copied.

usage: python bench_cpu_a_clean.py site LAS...          → cpu_a_clean/site_<stem>.json
       python bench_cpu_a_clean.py pool TAG LAS...      → cpu_a_clean/pool_<TAG>.json
"""
import hashlib
import json
import os
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("LIDAR_A_REPO", ROOT / "wt-cpua-clean"))
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests"))
OUT = ROOT / "bench" / "cpu_a_clean"
RSS_LIMIT_MB = 1950


def _rss_tree_mb():
    me, tot = os.getpid(), 0
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                st = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
                if int(d) == me or int(st[1]) == me:
                    tot += int(st[21]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, IndexError, ValueError):
                pass
    return tot / 2 ** 20


PEAK = [0.0]


def _guard():
    while True:
        r = _rss_tree_mb(); PEAK[0] = max(PEAK[0], r)
        if r > RSS_LIMIT_MB:
            sys.stderr.write(f"RSS guard: {r:.0f} MiB > {RSS_LIMIT_MB}, abort\n"); os._exit(3)
        time.sleep(0.05)


def cpu_all():
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def meas(fn, n=3):
    walls, cpus, load, res = [], [], [], None
    for _ in range(n):
        res = None
        load.append(os.getloadavg()[0]); c0 = cpu_all(); w0 = time.perf_counter()
        res = fn()
        walls.append(time.perf_counter() - w0); cpus.append(cpu_all() - c0)
    return {"wall": walls, "cpu": cpus, "wall_med": statistics.median(walls), "cpu_med": statistics.median(cpus), "load1": load}, res


def bits(a, b):
    return bool(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())


def site(path):
    from lidar_a_oracle import oracle, oracle_relief
    from expds.lidar_a import ground, runner, terrain
    ref_dtm, ref_rel = oracle("dtm"), oracle("relief")
    r = {"site": path}
    # DTM: reference build vs clean (exact and exact_fast fill); one run each (whole build), fill-only median of 3
    t0 = time.perf_counter(); rd, rv, ro = ref_dtm.build(path); r["ref_build_wall"] = time.perf_counter() - t0
    mins, maxs, _ = ground.header_box(path); grid = ground.grid_of(mins, maxs)
    mode = ground.ground_mode(path)
    t0 = time.perf_counter(); cellz = ground.rasterise(path, grid, mode)
    if mode == "lowest":
        cellz = ground.morph_ground(cellz, grid.cell)
    r["clean_ground_wall"] = time.perf_counter() - t0
    r.update(mode=mode, cells=int(cellz.size), nonground_share=float(np.isnan(cellz).mean()))
    m_ex, (d_ex, v_ex) = meas(lambda: ground.fill_holes(cellz, grid.cell, "exact"))
    m_fa, (d_fa, v_fa) = meas(lambda: ground.fill_holes(cellz, grid.cell, "exact_fast"))
    r["fill_exact"], r["fill_exact_fast"] = m_ex, m_fa
    r["x_fill_fast"] = [m_ex["wall_med"] / m_fa["wall_med"], m_ex["cpu_med"] / m_fa["cpu_med"]]
    d_ex = d_ex.astype(np.float32)
    r["dtm_bitwise_exact_vs_ref"] = bits(d_ex, rd) and bits(v_ex, rv) and grid.origin == ro
    r["dtm_bitwise_fast_vs_ref"] = bits(d_fa.astype(np.float32), rd) and bits(v_fa, rv)
    del d_fa, v_fa, rd, rv
    m_bd, (d_bd, v_bd) = meas(lambda: ground.fill_holes(cellz, grid.cell, "band"))
    r["fill_band"] = m_bd
    r["x_fill_band"] = [m_ex["wall_med"] / m_bd["wall_med"], m_ex["cpu_med"] / m_bd["cpu_med"]]
    del cellz
    d, v, cell = d_ex, v_ex, grid.cell
    # relief: reference stacks (one run each) vs clean; item (2): 9 + 12 separately vs 12 sliced
    with __import__("torch").no_grad():
        t0 = time.perf_counter(); rx9, rvv = oracle_relief(ref_rel, False).channels_tiled(d, v, cell, dev="cpu")
        rx12 = oracle_relief(ref_rel, True).channels_tiled(d, v, cell, dev="cpu")[0]; oracle_relief(ref_rel, False)
        r["ref_relief_9_12_wall"] = time.perf_counter() - t0
    m_sep, (x9, x12, vv) = meas(lambda: (lambda a, b: (a[0], b[0], a[1]))(
        terrain.relief_stack_windowed(d, v, cell, dev="cpu"), terrain.relief_stack_windowed(d, v, cell, extra=True, dev="cpu")))
    r["relief_bitwise_vs_ref"] = {"x9": bits(x9, rx9), "x12": bits(x12, rx12), "valid": bits(vv, rvv)}
    del rx9, rx12, x9
    m_sh, (st, vv2) = meas(lambda: runner.stacks_for(d, v, cell, ["", "x"], "cpu"))
    r["relief_separate"], r["relief_shared"] = m_sep, m_sh
    r["x_relief_shared"] = [m_sep["wall_med"] / m_sh["wall_med"], m_sep["cpu_med"] / m_sh["cpu_med"]]
    r["shared_bitwise"] = bits(st[""], terrain.relief_stack_windowed(d, v, cell, dev="cpu")[0]) and bits(st["x"], x12)
    del st
    xb = terrain.relief_stack_windowed(d_bd, v_bd, cell, extra=True, dev="cpu")[0]
    diff = xb.view(np.uint32) != x12.view(np.uint32)
    r["band_x12_equal_value"] = bool(np.array_equal(xb, x12)); r["band_x12_bitdiff_all_zero"] = bool((x12[diff] == 0).all())
    r["band_x12_bitdiff_count"] = int(diff.sum())
    r["peak_rss_mb"] = PEAK[0]
    return r


def pool(tag, sites, workers=2, budget=int(float(os.environ.get("POOL_BUDGET_GB", "1.4")) * 2 ** 30)):
    from expds.lidar_a import ground
    from expds.lidar_a.dtm_workers import DtmPool, est_bytes, plan

    def h(x):
        return hashlib.sha256(x[0].tobytes() + x[1].tobytes() + repr(x[2]).encode()).hexdigest()
    m_seq, r_seq = meas(lambda: [h(ground.make_dtm(p)) for p in sites])
    p_seq = PEAK[0]; PEAK[0] = 0

    def par():
        out = [None] * len(sites)
        for i, _, x, _ in DtmPool(sites, workers=workers, ram_budget=budget, ordered=False):
            out[i] = h(x); del x
        return out
    m_par, r_par = meas(par)
    sz = [plan(p) for p in sites]
    return {"sites": [Path(p).parent.parent.name for p in sites], "points": [a for a, _ in sz], "cells": [b for _, b in sz],
            "est_mb": [est_bytes(*x) / 2 ** 20 for x in sz], "workers": workers, "budget_mb": budget / 2 ** 20,
            "old": m_seq, "new": m_par, "bitwise": r_seq == r_par, "peak_tree_rss_mb_seq": p_seq, "peak_tree_rss_mb_pool": PEAK[0],
            "x_wall": m_seq["wall_med"] / m_par["wall_med"], "x_cpu": m_seq["cpu_med"] / m_par["cpu_med"]}


if __name__ == "__main__":
    threading.Thread(target=_guard, daemon=True).start()
    OUT.mkdir(parents=True, exist_ok=True)
    item = sys.argv[1]
    if item == "pool":
        r = pool(sys.argv[2], sys.argv[3:]); name = f"pool_{sys.argv[2]}"
        print(name, r["bitwise"], round(r["x_wall"], 3), round(r["x_cpu"], 3), r["old"]["wall_med"], r["new"]["wall_med"], r["peak_tree_rss_mb_pool"])
        r["threads"] = os.environ.get("OMP_NUM_THREADS")
        (OUT / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        for s in sys.argv[2:]:
            PEAK[0] = 0
            r = site(s); r["threads"] = os.environ.get("OMP_NUM_THREADS")
            import torch; r["torch_threads"] = torch.get_num_threads()
            name = f"site_{Path(s).parent.parent.name.split('_')[0]}"
            (OUT / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
            print(name, {k: v for k, v in r.items() if not isinstance(v, dict) or k.startswith("relief_bitwise")}, flush=True)
