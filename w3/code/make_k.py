"""Build K1..K5 variants (PREREG + A1) from split_arch + kurgan OOF csv.
usage: make_k.py SPLIT OOF.csv OUTDIR [pcol=p_povr] [prefix=K]
out: <prefix>N.jsonl extras {"fam","feat"} and <prefix>N.drop (stack idx) for replace variants (K1, K5)."""
import json, sys, os, csv, math, collections
from pathlib import Path
import shapely.geometry as sg
sys.path.insert(0, os.path.dirname(__file__))
from w3_post import kurgan_subclass_twins, POVR, TSEL, _grp, _ground_scale
SPLIT, OOF, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
PCOL = sys.argv[4] if len(sys.argv) > 4 else "p_povr"; PFX = sys.argv[5] if len(sys.argv) > 5 else "K"
P = {}
for r in csv.DictReader(open(OOF, encoding="utf-8")):
    if r["kind"] == "cand" and r[PCOL] not in ("", "nan"): P[int(r["stack_idx"])] = float(r[PCOL])
TW = {"2": dict(enabled=True, p_hi=.6, p_lo=.2, dedup_m=2.0, groups=["li", "li_mosaic", "li_coarse"]),
      "3": dict(enabled=True, p_hi=.6, p_lo=.2, dedup_m=2.0, groups=["li", "li_mosaic", "li_coarse"], scale_conf=False),
      "4": dict(enabled=True, p_hi=.6, p_lo=.2, dedup_m=0.0, groups=["li", "li_mosaic"])}
ex = {n: open(OUT / f"{PFX}{n}.jsonl", "w", encoding="utf-8") for n in "12345"}
drop = {n: [] for n in "15"}; cnt = collections.Counter()
def feat(d, **upd):
    pr = {k: v for k, v in d.items() if k not in ("geom", "_f", "_i")}; pr.update(upd)
    return {"type": "Feature", "geometry": d["_f"]["geometry"], "properties": pr}
for p in sorted(SPLIT.glob("*.tsv")):
    fam = p.stem; feats = []
    for line in open(p, encoding="utf-8"):
        if '"kurgany_' not in line: continue
        i, _, rest = line.partition("\t"); f = json.loads(rest); d = dict(f["properties"])
        if d.get("class_name") not in (POVR, TSEL): continue
        d["geom"] = sg.shape(f["geometry"]); d["_f"] = f; d["_i"] = int(i); feats.append(d)
    cov = [d for d in feats if d["_i"] in P]
    if not cov: continue
    for d in feats:
        if d["_i"] in P: d["subclass_p_povr"] = P[d["_i"]]
    for n, spec in TW.items():
        for t in kurgan_subclass_twins(feats, spec)[len(feats):]:
            ex[n].write(json.dumps({"fam": fam, "feat": feat(t)}, ensure_ascii=False) + "\n"); cnt[f"{PFX}{n}"] += 1
    # pair copies: same region+group, opposite class, centroid within 0.5 m (ground)
    key = collections.defaultdict(list)
    for d in feats:
        c = d["geom"].centroid; s = _ground_scale(c.y)
        key[(d.get("region_name"), _grp(d), round(c.x * s / 0.5), round(c.y * s / 0.5))].append(d)
    seen = set()
    for d in cov:
        c = d["geom"].centroid; s = _ground_scale(c.y)
        k = (d.get("region_name"), _grp(d), round(c.x * s / 0.5), round(c.y * s / 0.5))
        grp = [e for e in key[k] if e["_i"] not in seen]
        if not grp: continue
        for e in grp: seen.add(e["_i"])
        pp = P[d["_i"]]; g = _grp(d)
        # K5: all copies → <g>_sc, povr conf×p, tsel conf×(1−p)
        for e in grp:
            w = pp if e["class_name"] == POVR else 1 - pp
            drop["5"].append(e["_i"]); cnt[f"{PFX}5_moved"] += 1
            ex["5"].write(json.dumps({"fam": fam, "feat": feat(e, threshold_group=f"{g}_sc", confidence=round(float(e["confidence"]) * w, 6))}, ensure_ascii=False) + "\n")
        # K1: relabel at 0.4 (bot-9 apply) + dedup same class within 2 m (pair copies) keeping max conf
        new = POVR if pp >= 0.4 else TSEL
        best = max(grp, key=lambda e: float(e["confidence"]))
        for e in grp: drop["1"].append(e["_i"])
        ex["1"].write(json.dumps({"fam": fam, "feat": feat(best, class_name=new)}, ensure_ascii=False) + "\n"); cnt[f"{PFX}1_relabeled"] += 1
for n in ex: ex[n].close()
for n, l in drop.items(): open(OUT / f"{PFX}{n}.drop", "w").write("\n".join(map(str, l)) + "\n")
print(PCOL, "covered", len(P), dict(cnt))
