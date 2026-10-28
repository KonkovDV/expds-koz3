"""Site procedure for the LiDAR expert (spec §4): DTM → relief stacks → tiled ensemble → probability maps.

Decoding of the maps into objects is platform glue and is passed in as `decode(prob, valid, meta) → features`.
models = [(net, kind, weight)], kind '' = 9-channel stack, 'x' = 12-channel, 'u' = 12-channel with asinh LRMs.
"""
from __future__ import annotations

import time

import numpy as np

from . import ground, terrain, tiling

KIND_ARGS = {"": dict(extra=False, unclip=False), "x": dict(extra=True, unclip=False), "u": dict(extra=True, unclip=True)}
ONE_MODEL_LEFT_S = 120
FLIPS_ONLY_LEFT_S = 300


def guard_models(models, deadline):
    return models[:1] if time.time() > deadline - ONE_MODEL_LEFT_S else models


def guard_tta(tta, deadline):
    return True if tta == "d8" and time.time() > deadline - FLIPS_ONLY_LEFT_S else tta


def stacks_for(dtm, valid, cell, kinds, dev, share=True):
    """{kind: stack} for the kinds in use and the valid mask at NET_CELL.

    share (item (2)): the 9-channel stack is planes terrain.BASE_OF_EXTRA of the 12-channel one (same operations in
    the same order), so with both '' and 'x' in use the 12-channel stack is computed once and sliced."""
    got, vmask = {}, None
    todo = list(kinds)
    if share and "" in todo and "x" in todo:
        got["x"], vmask = terrain.relief_stack_windowed(dtm, valid, cell, dev=dev, **KIND_ARGS["x"])
        got[""] = got["x"][terrain.BASE_OF_EXTRA]
        todo = [k for k in todo if k not in ("", "x")]
    for kind in todo:
        got[kind], vk = terrain.relief_stack_windowed(dtm, valid, cell, dev=dev, **KIND_ARGS[kind])
        vmask = vk if vmask is None else vmask
    return got, vmask


def ensemble(maps, models):
    wts = np.array([w for _, _, w in models], np.float32)
    return np.tensordot(wts / wts.sum(), np.stack(maps), axes=1)


def site_maps(dtm, valid, cell, models, n_out, tta=True, deadline=float("inf"), dev="cuda", log=print):
    """→ (blended maps (n_out, h, w) float32, valid (h, w) bool) at terrain.NET_CELL."""
    models = guard_models(models, deadline)
    kinds = list(dict.fromkeys(k for _, k, _ in models))
    stacks, vmask = stacks_for(dtm, valid, cell, kinds, dev)
    tta = guard_tta(tta, deadline)
    maps = [tiling.with_oom_ladder(lambda t, m=m, s=stacks[k]: tiling.infer_map(
        m, s, n_out, tile=t, overlap=tiling.overlap_for(t), tta=tta, dev=dev), log=log) for m, k, _ in models]
    return ensemble(maps, models), vmask


def run_site(path, dtm, valid, origin, models, n_out, decode, meta, tta, deadline, dev, log):
    t0 = time.time()
    left, top, cell = origin
    prob, vmask = site_maps(dtm, valid, cell, models, n_out, tta, deadline, dev, log)
    feats = decode(prob, vmask, dict(meta, x0=left, ytop=top, res=terrain.NET_CELL))
    log(f"lidar_a {getattr(path, 'name', path)}: dtm {dtm.shape} cell {cell} → {len(feats)} feats {time.time() - t0:.0f}s")
    return feats


def dtms_in_order(paths, pool=None):
    """(path, (dtm, valid, origin)) in input order: sequential, or built by worker processes (item (6))."""
    if not pool:
        for p in paths:
            yield p, ground.make_dtm(p)
        return
    from .dtm_workers import DtmPool
    for i, _, res, _ in DtmPool(paths, **pool):
        yield paths[i], res


def predict(paths, models, n_out, decode, meta=None, tta=True, deadline=float("inf"), dev="cuda", log=print, pool=None):
    """Features of the LAS files of one region, in input order. pool: None (sequential) or dict(workers=, ram_budget=)."""
    feats = []
    paths = list(paths)
    for p, (d, v, origin) in dtms_in_order(paths, pool):
        feats += run_site(p, d, v, origin, models, n_out, decode, meta or {}, tta, deadline, dev, log)
    return feats


def predict_all(jobs, models, n_out, decode, tta=True, deadline=float("inf"), dev="cuda", log=print, workers=2, ram_budget=1 << 30):
    """Item (6), run level: one DtmPool over the files of ALL regions, jobs = [(meta, paths)] with meta['region'].

    Sites are processed in DTM completion order (largest file first in the pool, the GPU works on small ones
    meanwhile); features are regrouped in input order → {region: features}, equal to per-region predict() unless a
    deadline guard fires at a different moment."""
    from .dtm_workers import DtmPool
    flat = [(meta, p) for meta, paths in jobs for p in paths]
    done = {}
    for i, _, (d, v, origin), _ in DtmPool([p for _, p in flat], workers=workers, ram_budget=ram_budget, ordered=False):
        meta, p = flat[i]
        done[i] = run_site(p, d, v, origin, models, n_out, decode, meta, tta, deadline, dev, log)
        del d, v
    out = {meta["region"]: [] for meta, _ in jobs}
    for i, (meta, _) in enumerate(flat):
        out[meta["region"]] += done[i]
    return out
