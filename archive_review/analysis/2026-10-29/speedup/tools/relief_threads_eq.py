"""Каналы 12 (relief.channels_tiled, dev=cpu) при разном числе потоков torch: побитно ли равны и сколько времени.
Код соперника только импортируется (сверка), не копируется."""
import os, sys, time, hashlib
os.environ["RELIEF_EXTRA"] = "1"
import numpy as np, torch
sys.path.insert(0, os.environ.get("COMP", "/workspace/koz3_shared/competitors/EhimenNathan-EXpedition"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from koz3 import dtm, relief
from test_las_grid_fast import laz
site = sys.argv[1]; threads = [int(t) for t in sys.argv[2].split(",")]
d, v, (x0, y1, res) = dtm.build(laz(site))
print(site, d.shape, "Mpx@0.5", d.size / 4e6)
for t in threads:
    torch.set_num_threads(t)
    best = 1e9
    for _ in range(2):
        a = time.perf_counter(); c0 = time.process_time()
        x, vv = relief.channels_tiled(d, v, res, dev="cpu")
        best = min(best, time.perf_counter() - a); cpu = time.process_time() - c0
    print(t, "threads", round(best, 2), "s wall", round(cpu, 2), "cpu", hashlib.sha1(x.tobytes()).hexdigest()[:16], x.shape)
