"""Насколько текущие каналы li отличаются от старого кэша: доля пикселей и величина, по каналам и бэкендам."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
import numpy as np

from expds import accel
from expds.catalog import read_utm_json, scan_root
from expds.lidar import build_dem, relief_channels
from expds.tiles import cut_window

src = Path(r"F:\expds3\work\cache\li_tin_hard2_t128")
region = sys.argv[1] if len(sys.argv) > 1 else "036_Тимерево"
rows = [r for r in csv.DictReader(open(src / "index.csv", encoding="utf-8")) if r["region"] == region]
old = {s: np.load(src / f"images_{s:04d}.npy", mmap_mode="r") for s in {int(r["shard"]) for r in rows}}
reg = next(r for r in scan_root(r"F:\expds3\train-expds-3") if r.name == region)
crs = read_utm_json(reg.utm_json)
d = build_dem(reg.las[0], res=0.1, fill="delaunay", crs=crs)
for label in ("rust", "numpy"):
    saved = accel._rs
    if label == "numpy":
        accel._rs = None
    try:
        arr, tr = relief_channels(d, 0.5, ("lrm_s", "lrm_l", "slope"))
    finally:
        accel._rs = saved
    diff_px = np.zeros(3)
    maxd = np.zeros(3)
    n = 0
    for r in rows:
        img, _ = cut_window(arr, tr, float(r["cx"]), float(r["cy"]), 128)
        o = np.asarray(old[int(r["shard"])][int(r["idx"])]).astype(int)
        dd = np.abs(img.astype(int) - o)
        diff_px += (dd > 0).reshape(3, -1).mean(1)
        maxd = np.maximum(maxd, dd.reshape(3, -1).max(1))
        n += 1
    print(f"{region} {label}: тайлов {n}, доля отличных пикселей {np.round(diff_px / n * 100, 4)} %, макс {maxd}")
    if label == "rust":
        zero_flip = small = big = 0
        worst = (0, None)
        for r in rows:
            img, _ = cut_window(arr, tr, float(r["cx"]), float(r["cy"]), 128)
            o = np.asarray(old[int(r["shard"])][int(r["idx"])]).astype(int)
            dd = np.abs(img.astype(int) - o)
            m = dd > 0
            zero_flip += int((m & ((img == 0) | (o == 0))).sum())
            small += int((m & (dd <= 2) & (img > 0) & (o > 0)).sum())
            big += int((m & (dd > 2) & (img > 0) & (o > 0)).sum())
            if m.mean() > worst[0]:
                worst = (m.mean(), r)
        print(f"  пиксели: граница съёмки {zero_flip}, |Δ|≤2 {small}, |Δ|>2 {big}")
        r = worst[1]
        img, _ = cut_window(arr, tr, float(r["cx"]), float(r["cy"]), 128)
        o = np.asarray(old[int(r["shard"])][int(r["idx"])]).astype(int)
        m = (np.abs(img.astype(int) - o) > 0).any(0)
        ys, xs = np.nonzero(m)
        print(f"  худший тайл {r['shard']}/{r['idx']} доля {worst[0]:.3f} строки {ys.min()}–{ys.max()} столбцы {xs.min()}–{xs.max()}"
              f" нулей новых {(img == 0).mean():.3f} старых {(o == 0).mean():.3f}")
        exact = shifted = other = 0
        step = tr.a
        for r in rows:
            o = np.asarray(old[int(r["shard"])][int(r["idx"])])
            hit = None
            for dy in (0, -1, 1):
                for dx in (0, -1, 1):
                    img, _ = cut_window(arr, tr, float(r["cx"]) + dx * step, float(r["cy"]) - dy * step, 128)
                    if np.array_equal(img, o):
                        hit = (dy, dx)
                        break
                if hit:
                    break
            if hit == (0, 0):
                exact += 1
            elif hit:
                shifted += 1
            else:
                other += 1
        print(f"  тайлов совпало как есть {exact}, со сдвигом на 1 пиксель {shifted}, иначе {other}")
