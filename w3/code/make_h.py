"""Build H1-H3 extras (gor_from_fort_arch) from split stack_arch. Output: variants/<name>.jsonl, line = {"fam","feat"}.
Uses code/w3_post.gor_from_fort_arch on post.py-style dicts (geom shapely EPSG:3857)."""
import json, sys, glob, os, collections
from pathlib import Path
import shapely.geometry as sg
sys.path.insert(0, os.path.dirname(__file__))
from w3_post import gor_from_fort_arch
SPLIT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
SPECS_ALL = {"H4": {"enabled": True, "min_area_m2": 0, "markups": ["sp_or", "or"]}}
SPECS_REST = {"H1": {"enabled": True, "min_area_m2": 5000},
         "H2": {"enabled": True, "min_area_m2": 5000, "markups": ["sp_or", "li"]},
         "H3": {"enabled": True, "min_area_m2": 2000}}
SPECS_ALL.update(SPECS_REST); SPECS = {k: SPECS_ALL[k] for k in (sys.argv[3].split(",") if len(sys.argv) > 3 else SPECS_ALL)}
outs = {n: open(OUT / f"{n}.jsonl", "w", encoding="utf-8") for n in SPECS}
cnt = {n: collections.Counter() for n in SPECS}
for p in sorted(SPLIT.glob("*.tsv")):
    fam = p.stem
    feats = []
    for line in open(p, encoding="utf-8"):
        if '"gorodishcha"' not in line and '"fortifikatsii"' not in line and '"arkhitektury"' not in line:
            continue
        f = json.loads(line.split("\t", 1)[1]); pr = f["properties"]
        d = dict(pr); d["geom"] = sg.shape(f["geometry"]); d["_f"] = f
        feats.append(d)
    if not feats:
        continue
    for n, spec in SPECS.items():
        res = gor_from_fort_arch(feats, spec)
        for d in res[len(feats):]:
            pr = {k: v for k, v in d.items() if k not in ("geom", "_f")}
            outs[n].write(json.dumps({"fam": fam, "feat": {"type": "Feature", "geometry": d["_f"]["geometry"], "properties": pr}},
                                     ensure_ascii=False) + "\n")
            cnt[n][pr["threshold_group"]] += 1
for n in SPECS:
    outs[n].close(); print(n, dict(cnt[n]), sum(cnt[n].values()))
