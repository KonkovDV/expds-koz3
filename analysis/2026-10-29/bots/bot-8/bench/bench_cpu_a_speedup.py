"""§7.12 bench: scenario (a) CPU speedups, old vs new path, median of 3 (wall + process CPU), 2 threads.

usage: python bench_cpu_a_speedup.py ITEM SITE...   ITEM ∈ {x9, edt, pool, gpu}; SITE = LAS/LAZ path or a cached
       .npz (d, v) DTM. Writes bench/cpu_a_speedup/<item>_<site>.json. Code: wt-cpua (branch bot8-cpu-a-speedup).
"""
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wt-cpua"))
OUT = ROOT / "bench" / "cpu_a_speedup"
from expds.li_a import dtm  # noqa: E402  (relief/expert import torch: only where needed)


def meas(fn, n=3):
    walls, cpus, res = [], [], None
    for _ in range(n):
        l0 = os.getloadavg()[0]; c0 = time.process_time(); w0 = time.perf_counter()
        res = fn()
        walls.append(time.perf_counter() - w0); cpus.append(time.process_time() - c0)
    return {"wall": walls, "cpu": cpus, "wall_med": statistics.median(walls), "cpu_med": statistics.median(cpus),
            "load1_start": l0, "load1_end": os.getloadavg()[0]}, res


def load_dtm(p):
    if p.endswith(".npz"):
        z = np.load(p); return z["d"], z["v"], 0.25
    d, v, (_, _, res) = dtm.build(p); return d, v, res


def item_x9(site):
    from expds.li_a import expert, relief
    d, v, res = load_dtm(site)

    def old():
        x9, vv = relief.channels_tiled(d, v, res, dev="cpu")
        x12 = expert._channels_extra(d, v, res, dev="cpu")
        return x9, vv, x12

    def new():
        x12, vv = expert._channels_extra(d, v, res, dev="cpu", with_valid=True)
        return x12[expert.X9_OF_X12], vv, x12
    mo, ro = meas(old); mn, rn = meas(new)
    eq = all(np.array_equal(a, b) for a, b in zip(ro, rn))
    return {"cells": int(d.size), "px_net": int(ro[1].size), "old": mo, "new": mn, "byte_equal": eq,
            "x_wall": mo["wall_med"] / mn["wall_med"], "x_cpu": mo["cpu_med"] / mn["cpu_med"]}


def _tree_rss_mb():
    """RSS of this process + its direct children (pool workers), MiB, from /proc."""
    me, tot = os.getpid(), 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
            if int(d) == me or int(st[1]) == me:
                tot += int(st[21]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            pass
    return tot / 2 ** 20


def item_pool(sites, workers=2, budget=int(float(os.environ.get("POOL_BUDGET_GB", "1.4")) * 2 ** 30)):
    """DTM stage of a run: sequential dtm.build (upstream) vs DtmPool(2 workers, completion order), CPU = self+children."""
    import threading
    from expds.li_a.dtm_pool import DtmPool, est_bytes, plan
    peak = [0.0]; stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], _tree_rss_mb()); time.sleep(0.05)
    threading.Thread(target=sampler, daemon=True).start()

    def cpu():
        a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime

    def run(fn):
        walls, cpus, load = [], [], []
        for _ in range(3):
            load.append(os.getloadavg()[0]); c0 = cpu(); w0 = time.perf_counter(); r = fn()
            walls.append(time.perf_counter() - w0); cpus.append(cpu() - c0)
        return {"wall": walls, "cpu": cpus, "wall_med": statistics.median(walls), "cpu_med": statistics.median(cpus), "load1": load}, r

    import hashlib

    def h(r):
        return hashlib.sha256(r[0].tobytes() + r[1].tobytes() + repr(r[2]).encode()).hexdigest()

    def seq():
        return [h(dtm.build(p)) for p in sites]       # hashes only: the results must not hold RAM (box ≤ 2 GB)
    per = []

    def pool():
        out = [None] * len(sites); per.clear()
        for i, _, r, st in DtmPool(sites, workers=workers, ram_budget=budget, ordered=False):
            out[i] = h(r); del r; per.append((Path(sites[i]).parent.parent.name, st["wall"], st["cpu"]))
        return out
    peak[0] = 0; mo, ro = run(seq); peak_seq = peak[0]
    peak[0] = 0; mn, rn = run(pool); peak_pool = peak[0]; stop.set()
    eq = ro == rn
    sz = [plan(p) for p in sites]
    return {"sites": [Path(p).parent.parent.name for p in sites], "points": [a for a, _ in sz], "cells": [b for _, b in sz],
            "est_mb": [est_bytes(*x) / 2 ** 20 for x in sz], "largest_point_share": max(a for a, _ in sz) / sum(a for a, _ in sz),
            "workers": workers, "budget_mb": budget / 2 ** 20, "old": mo, "new": mn, "byte_equal": eq,
            "peak_tree_rss_mb_seq": peak_seq, "peak_tree_rss_mb_pool": peak_pool, "per_job_last_run": per,
            "x_wall": mo["wall_med"] / mn["wall_med"], "x_cpu": mo["cpu_med"] / mn["cpu_med"]}


if __name__ == "__main__":
    item, sites = sys.argv[1], sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    if item == "pool":
        tag, sites = sites[0], sites[1:]
        r = item_pool(sites)
        (OUT / f"pool_{tag}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print("pool", tag, {k: r[k] for k in ("largest_point_share", "byte_equal", "x_wall", "x_cpu", "peak_tree_rss_mb_seq", "peak_tree_rss_mb_pool")},
              "old", round(r["old"]["wall_med"], 2), round(r["old"]["cpu_med"], 2), "new", round(r["new"]["wall_med"], 2), round(r["new"]["cpu_med"], 2))
        sys.exit()
    for s in sites:
        r = globals()["item_" + item](s)
        r.update(site=s, threads=os.environ.get("OMP_NUM_THREADS"), torch_threads=sys.modules["torch"].get_num_threads() if "torch" in sys.modules else None,
                 maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        name = Path(s).stem
        (OUT / f"{item}_{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(item, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("old", "new")},
              "old", round(r["old"]["wall_med"], 2), round(r["old"]["cpu_med"], 2), "new", round(r["new"]["wall_med"], 2), round(r["new"]["cpu_med"], 2), flush=True)
