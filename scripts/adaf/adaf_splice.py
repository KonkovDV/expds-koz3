#!/usr/bin/env python3
"""Add ADAF zero-shot detections to a copy of the bot-10 family stack (work/fam, read-only) as NEW
threshold groups; nothing is removed from the stack.

  barrow    -> kurgany_povrezhdennye AND kurgany_tselye, group adaf_b
  enclosure -> gorodishcha AND fortifikatsii,            group adaf_e
Geometry: circle r (default 1 m) at the box centre, EPSG:3857 (stack convention), confidence = score.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from pyproj import Transformer

MAP = {"OD_barrow_sd": ("adaf_b", ["kurgany_povrezhdennye", "kurgany_tselye"]),
       "OD_enclosure_sd": ("adaf_e", ["gorodishcha", "fortifikatsii"])}


def circle(x, y, r, n=16):
    import math
    pts = [[round(x + r * math.cos(2 * math.pi * i / n), 2), round(y + r * math.sin(2 * math.pi * i / n), 2)]
           for i in range(n)]
    return {"type": "Polygon", "coordinates": [pts + [pts[0]]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dets", required=True, help="dir with <model>/<region>.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--min-score", type=float, default=0.05)
    a = ap.parse_args()
    lines = defaultdict(list); st = Counter(); fid = 20_000_000
    for mdl in a.models:
        grp, classes = MAP[mdl]
        for fp in sorted((Path(a.dets) / mdl).glob("*.json")):
            d = json.load(open(fp))
            crs = d["crs"]; crs = crs if isinstance(crs, str) else crs.get("crs")
            epsg = "EPSG:" + crs.split(":")[-1]
            tr = Transformer.from_crs(epsg, "EPSG:3857", always_xy=True)
            for det in d["dets"]:
                if det["score"] < a.min_score:
                    continue
                X, Y = tr.transform(det["x"], det["y"])
                # r metres on the ground -> 3857 units (scale 1/cos(lat))
                import math

                k = 1.0 / math.cos(math.atan(math.sinh(Y / 6378137.0)))
                for c in classes:
                    f = {"type": "Feature", "geometry": circle(X, Y, a.r * k),
                         "properties": {"fid": fid, "class_name": c, "confidence": det["score"],
                                        "region_name": d["region"], "crs": "EPSG:3857", "markup_type": "adaf",
                                        "threshold_group": grp, "original_crs": epsg}}
                    lines[d["region"]].append(json.dumps(f, ensure_ascii=False)); fid += 1
                    st[f"{grp}|{c}"] += 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    left = set(lines)
    for fp in sorted(Path(a.src).glob("*.jsonl")):
        regs = set()
        with fp.open() as fh:  # families without ADAF regions: symlink instead of a copy (disk)
            for line in fh:
                if '"region_name"' in line:
                    i = line.find('"region_name": "') + 16
                    regs.add(line[i:line.find('"', i)])
        if not regs & left:
            (out / fp.name).symlink_to(fp.resolve()); st["families_symlinked"] += 1
            continue
        regs = set()
        with fp.open() as fh, (out / fp.name).open("w") as fo:
            for line in fh:
                if '"region_name"' in line:
                    i = line.find('"region_name": "') + 16
                    regs.add(line[i:line.find('"', i)])
                fo.write(line if line.endswith("\n") else line + "\n")
            for r in sorted(regs & left):
                fo.write("\n".join(lines[r]) + "\n"); left.discard(r); st["regions"] += 1
    cp = Path(a.src) / "_counts.json"
    if cp.exists():
        (out / "_counts.json").write_text(cp.read_text())
    st["regions_without_family_file"] = len(left)
    (out / "_adaf_stats.json").write_text(json.dumps({"written": st, "missing": sorted(left)}, ensure_ascii=False, indent=1))
    print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    main()
