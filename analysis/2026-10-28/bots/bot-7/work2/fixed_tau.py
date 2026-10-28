"""Вариант «только пост, пороги базы»: пороги фолда подобраны на кривых БАЗЫ (без своего фолда),
применены к кривым кандидата. Так поведёт себя пакет, если включить [post.micro_circles], не перекалибровав пороги."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from expds import metric as M
from expds.naming import site_family
from honest_bootstrap import load_curves, regions_of, weighted_class_delta
def counts_with(base, cand, folds):
    every = set().union(*folds); counts = {c: {} for c in cand}
    for held in folds:
        res = M.sweep_groups({c: M.curve_where(v, every - held) for c, v in base.items()})
        taus = {}
        for cls, v in res.items():
            for g, gv in v["groups"].items(): taus.setdefault(g, {})[cls] = gv["threshold"]
        for cls, c in cand.items():
            hc = M.curve_where(c, held); per = counts[cls]
            for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
            if hc.conf.size == 0: continue
            tau = np.array([taus.get(g, {}).get(cls, np.inf) for g in hc.pred_group], float)
            kept = hc.conf >= tau
            for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
                per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return counts
base = load_curves(Path("work2/curves_base")); regions = regions_of(base)
out = {}
for layout, folds in [("h10", M.family_folds(regions, site_family, 10)), ("lofo", M.family_folds(regions, site_family, None))]:
    b, _ = M.honest_counts(base, folds)
    for v in ["kur", "arch", "bundle"]:
        c = counts_with(base, load_curves(Path(f"work2/curves_{v}")), folds)
        r = {"d": M.score_counts(c)[0] - M.score_counts(b)[0]}
        if layout == "h10":
            r["boot"] = M.paired_bootstrap_counts(c, b, site_family, 2000)
            r["weighted"] = {k: x["weighted"] for k, x in weighted_class_delta(c, b).items() if x["weighted"]}
        out[f"{v}_{layout}"] = r; print(v, layout, r, flush=True)
Path("work2/fixed_tau.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
