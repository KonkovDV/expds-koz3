#!/usr/bin/env python3
"""Семьи, чьи участки пересекаются в пространстве (рамки EPSG:3857 + допуск).

Официальное сопоставление глобально: объект участка одной семьи может закрыть
эталон соседней (часто — сцены спутника). Такие семьи надо сопоставлять вместе.
Пишет JSON {кластер: [семьи]} для family_curves.py --clusters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expds.naming import site_family  # noqa: E402


def coords(g):
    t = g.get("type")
    c = g.get("coordinates")
    if t == "Point":
        return [c]
    if t in ("LineString", "MultiPoint"):
        return c
    if t in ("Polygon", "MultiLineString"):
        return [p for r in c for p in r]
    if t == "MultiPolygon":
        return [p for poly in c for r in poly for p in r]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=float, default=100.0, help="допуск рамки, м (EPSG:3857)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    box: dict[str, list[float]] = {}

    def upd(reg, g):
        if not g:
            return
        xy = np.asarray(coords(g), dtype=float)
        if xy.size == 0:
            return
        b = box.setdefault(reg, [np.inf, np.inf, -np.inf, -np.inf])
        b[0] = min(b[0], xy[:, 0].min()); b[1] = min(b[1], xy[:, 1].min())
        b[2] = max(b[2], xy[:, 0].max()); b[3] = max(b[3], xy[:, 1].max())

    for f in json.loads(Path(a.gt).read_text(encoding="utf-8"))["features"]:
        upd(str(f["properties"].get("region_name")), f.get("geometry"))
    for p in Path(a.fam).glob("*.jsonl"):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                f = json.loads(line)
                upd(str(f["properties"].get("region_name")), f.get("geometry"))
    regs = sorted(box)
    parent = {site_family(r): site_family(r) for r in regs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i, r in enumerate(regs):
        bi = box[r]
        for s in regs[i + 1:]:
            bj = box[s]
            fi, fj = site_family(r), site_family(s)
            if fi == fj:
                continue
            if bi[0] - a.tol <= bj[2] and bj[0] - a.tol <= bi[2] and bi[1] - a.tol <= bj[3] and bj[1] - a.tol <= bi[3]:
                pairs.append([r, s])
                parent[find(fi)] = find(fj)
    cl: dict[str, list[str]] = {}
    for f in parent:
        cl.setdefault(find(f), []).append(f)
    multi = {k: sorted(v) for k, v in cl.items() if len(v) > 1}
    Path(a.out).write_text(json.dumps({"clusters": multi, "overlapping_region_pairs": pairs,
                                       "boxes": box}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("пар участков разных семей с пересечением рамок:", len(pairs))
    for k, v in multi.items():
        print(" ", v)


if __name__ == "__main__":
    main()
