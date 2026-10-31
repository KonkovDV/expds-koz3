"""Кэш тайлов li с другими каналами: те же центры и маски, изображения пересчитаны.

    python recut_channels.py SRC DST lrm_s lrm_l slope opns svf

Каналы, общие со старым кэшем, сверяются побайтно — проверка, что окна те же.
"""
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
import numpy as np

from expds.catalog import read_utm_json, scan_root
from expds.lidar import DemPrefetch, relief_channels
from expds.tiles import cut_window

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
tags = sys.argv[3:]
meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
old_tags = meta["li_tags"]
tile, res = int(meta["tile"]), float(meta["res"])
dem = {"res": float(meta["dem_res"]), "fill": meta["dem_fill"]}
with (src / "index.csv").open(encoding="utf-8") as fh:
    rd = csv.DictReader(fh)
    fields, rows = list(rd.fieldnames or []), list(rd)

dst.mkdir(parents=True, exist_ok=True)
shards = sorted({int(r["shard"]) for r in rows})
old = {s: np.load(src / f"images_{s:04d}.npy", mmap_mode="r") for s in shards}
new = {s: np.zeros((old[s].shape[0], len(tags)) + old[s].shape[2:], np.uint8) for s in shards}
for s in shards:
    m = src / f"masks_{s:04d}.npy"
    t = dst / m.name
    if not t.exists():
        try:
            os.link(m, t)
        except OSError:
            shutil.copy2(m, t)

regions = {r.name: r for r in scan_root(r"F:\expds3\train-expds-3")}
groups = defaultdict(list)
for r in rows:
    groups[(r["region"], r["file"])].append(r)
order = []
for (reg_name, file) in groups:
    reg = regions[reg_name]
    las = next(p for p in reg.las if p.name == file)
    order.append((las, read_utm_json(reg.utm_json), reg_name))
pf = DemPrefetch([(p, c) for p, c, _ in order], depth=2)
shared = [(i, old_tags.index(t)) for i, t in enumerate(tags) if t in old_tags]
mismatch, total = 0, 0
t0 = time.perf_counter()
try:
    for k, (las, crs, reg_name) in enumerate(order):
        d = pf.get(las, dem["res"], dem["fill"], crs)
        arr, tr = relief_channels(d, res, tuple(tags))
        for r in groups[(reg_name, las.name)]:
            img, _ = cut_window(arr, tr, float(r["cx"]), float(r["cy"]), tile)
            s, i = int(r["shard"]), int(r["idx"])
            new[s][i] = img
            for ni, oi in shared:
                total += 1
                mismatch += int(not np.array_equal(img[ni], old[s][i][oi]))
        print(f"{k + 1}/{len(order)} {reg_name} тайлов {len(groups[(reg_name, las.name)])} "
              f"расхождений {mismatch}/{total} {time.perf_counter() - t0:.0f} с", flush=True)
finally:
    pf.close()
for s in shards:
    np.save(dst / f"images_{s:04d}.npy", new[s])
shutil.copy2(src / "index.csv", dst / "index.csv")
meta.update(li_tags=list(tags), channels=len(tags))
(dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"rows": len(rows), "shards": shards, "mismatch_planes": mismatch, "planes": total}))
