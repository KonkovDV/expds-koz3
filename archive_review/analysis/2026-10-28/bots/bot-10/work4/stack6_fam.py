"""sota_stack6 на стеке по семьям: post.multipole_object_poly (+ filter_finds_points_regimes), порядок как в watchdog.finish.

    python work4/stack6_fam.py --src work/fam --dst work4/fam_sep --mode separate|unified [--fams A,B,C]
Пишет только семьи с object_poly/finds_points; счётчики до/после — в <dst>/_counts.json.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))
from shapely.geometry import shape, mapping
from expds import post

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--mode", choices=["separate", "unified"], required=True)
ap.add_argument("--only", choices=["both", "multipole", "finds"], default="both")
ap.add_argument("--fams", default="Гнёздово,Семибратнее,Судьбищи")
a = ap.parse_args()
spec = {"unified_group": "maggr_poly"} if a.mode == "unified" else None
dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
stats = {}
for fam in a.fams.split(","):
    feats = []
    with open(Path(a.src) / f"{fam}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            f = json.loads(line)
            d = dict(f.get("properties") or {}); d["geom"] = shape(f["geometry"]); feats.append(d)
    key = lambda d: f'{d.get("class_name")}|{d.get("threshold_group") or d.get("markup_type")}'
    before = Counter(key(d) for d in feats if d.get("class_name") in ("object_poly", "finds_points"))
    if a.only in ("both", "multipole"):
        feats = post.multipole_object_poly(feats, spec)
    if a.only in ("both", "finds"):
        feats = post.filter_finds_points_regimes(feats, None)
    after = Counter(key(d) for d in feats if d.get("class_name") in ("object_poly", "finds_points"))
    stats[fam] = {"before": dict(before), "after": dict(after)}
    with open(dst / f"{fam}.jsonl", "w", encoding="utf-8") as out:
        for d in feats:
            g = d.pop("geom")
            out.write(json.dumps({"type": "Feature", "geometry": mapping(g), "properties": d}, ensure_ascii=False) + "\n")
    print(fam, json.dumps(stats[fam], ensure_ascii=False), flush=True)
(dst / "_counts.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
