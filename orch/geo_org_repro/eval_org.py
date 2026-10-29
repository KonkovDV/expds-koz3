"""1:1 comparison: KOZ3 classic geo A vs A with parts replaced by organizer-baseline OOF.
Uses w4clean/scripts/pix_clean_eval.py logic (honest_counts thresholds per markup_type group on held-out folds).
usage: eval_org.py A.geojson OUT.json name=path_to_C ..."""
import json, sys, copy
from pathlib import Path
sys.path.insert(0, "/workspace/koz3/w4clean/scripts"); sys.path.insert(0, "/workspace/koz3/w4clean")
import pix_clean_eval as P
from expds import metric
from expds.constants import CLASS_WEIGHTS
from expds.naming import site_family
GT = "/workspace/koz3/org_repro/out/gt_geo.geojson"
A, OUT = sys.argv[1], sys.argv[2]
specs = dict(a.split("=", 1) for a in sys.argv[3:])
gt = P._load(GT); regions = {f["properties"]["region_name"] for f in gt["features"]}
scale = sum(CLASS_WEIGHTS.get(c, 0) for c in P.CLASSES) / sum(CLASS_WEIGHTS.values())
FOLDSETS = {"geo5": P.folds_of(regions),
            "fam": {f: {r for r in regions if site_family(r) == f} for f in sorted({site_family(r) for r in regions})}}
res = {"A": A, "scale": scale, "runs": {}}
for fsn, folds in FOLDSETS.items():
    ac, ai = P.honest(P._load(A, regions), gt, folds); sa = P.summary(ac, folds)
    res["runs"][fsn] = {"A": sa, "C": {}}
    print(fsn, "A", json.dumps(sa, ensure_ascii=False), flush=True)
    for name, path in specs.items():
        cc, ci = P.honest(P._load(path, regions), gt, folds); sc = P.summary(cc, folds)
        d = {"summary": sc, "d_f1": {c: sc["f1"].get(c, 0) - sa["f1"].get(c, 0) for c in P.CLASSES},
             "d_global": (sc["score2"] - sa["score2"]) * scale,
             "d_folds_global": {k: (sc["folds"][k] - sa["folds"][k]) * scale for k in sc["folds"]},
             "boot_family": metric.paired_bootstrap_counts(cc, ac, site_family, 2000, 0),
             "boot_region": metric.paired_bootstrap_counts(cc, ac, lambda r: r, 2000, 0),
             "counts": {c: list(map(float, v)) if hasattr(v, '__iter__') else v for c, v in {}.items()}}
        for b in ("boot_family", "boot_region"):
            for k in ("mean", "p5", "p50", "p95"):
                if k in d[b]: d[b][k + "_global"] = d[b][k] * scale
        res["runs"][fsn]["C"][name] = d
        print(fsn, name, json.dumps({k: d[k] for k in ("d_f1", "d_global", "d_folds_global")}, ensure_ascii=False),
              "p_family", d["boot_family"].get("p_gt0"), "p_region", d["boot_region"].get("p_gt0"), flush=True)
Path(OUT).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
