#!/usr/bin/env python3
"""Добавить/изменить элемент очереди watcher-а (QUEUE.json["dispatch"]) под flock.
  orch/enqueue.py add --id w2-snap --owner bot-W2-lidar --accel cpu --account any-cpu --kernel-dir w2/kernels/koz3-w2-center-snap --ttl 6 [--after id1,id2] [--pull-to w2/out/snap] [--hold]
  orch/enqueue.py add --id w0-prof --owner coord --accel box --cmd "python3 ..." --ttl 3
  orch/enqueue.py set <id> status=ready|hold|done   |   orch/enqueue.py list
Правила: kernel-metadata.json должен совпадать по ускорителю; slug с префиксом koz3-w<N>-; GPU/TPU только elenakonkova|dmitrijkv."""
import argparse, json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from watcher import Queue
ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
a = sp.add_parser("add")
for k in ("id", "owner", "accel"): a.add_argument("--" + k, required=True)
for k in ("account", "kernel-dir", "cmd", "after", "pull-to", "note"): a.add_argument("--" + k)
a.add_argument("--ttl", type=float, default=10); a.add_argument("--hold", action="store_true")
s = sp.add_parser("set"); s.add_argument("id"); s.add_argument("kv", nargs="+")
sp.add_parser("list")
g = ap.parse_args()
with Queue() as d:
    D = d["dispatch"]
    if g.c == "add":
        if any(x["id"] == g.id for x in D): sys.exit(f"id {g.id} уже есть")
        if g.accel in ("gpu", "tpu") and g.account not in ("elenakonkova", "dmitrijkv"): sys.exit("GPU/TPU только elenakonkova|dmitrijkv")
        if g.accel != "box" and not g.kernel_dir: sys.exit("нужен --kernel-dir")
        it = {"id": g.id, "owner": g.owner, "accel": g.accel, "account": g.account or ("box" if g.accel == "box" else "any-cpu"),
              "ttl_h": g.ttl, "status": "hold" if g.hold else "ready", "after": g.after.split(",") if g.after else []}
        for k in ("kernel_dir", "cmd", "pull_to", "note"):
            if getattr(g, k): it[k] = getattr(g, k)
        D.append(it); print("OK", it)
    elif g.c == "set":
        it = next(x for x in D if x["id"] == g.id)
        for kv in g.kv: k, v = kv.split("=", 1); it[k] = v
        print("OK", it)
    else:
        for x in D: print(f"{x['id']:<28} {x['status']:<8} {x['accel']:<4} {x.get('account',''):<14} {x.get('kernel') or x.get('kernel_dir') or x.get('cmd','')[:60]}")
