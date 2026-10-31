"""Обёртка над scripts/scoreboard.py: тот же расчёт, плюс честные F1 по классам и по семьям."""
import json, sys
from pathlib import Path

REPO = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import scoreboard as sb
from expds.metric import score_counts
from expds.naming import site_family

EXTRA = Path(sys.argv.pop(1))
CAP = {"eval": {}, "honest": {}}
_ev, _ho = sb.evaluate, sb.honest


def evaluate(name, feats, gt, new, optical, metric):
    r = _ev(name, feats, gt, new, optical, metric)
    CAP["eval"][name] = {"scores": r[0], "per_class": r[3]}
    CAP["sets"] = {"new": sorted(new), "optical": sorted(optical)}
    return r


def honest(name, new, optical, folds_k, by="family"):
    r = _ho(name, new, optical, folds_k, by)
    CAP["honest"][name] = {"scores": r[0], "counts": r[1]}
    return r


sb.evaluate, sb.honest = evaluate, honest
sb.main()
out = {"eval": CAP["eval"], "honest": {}}
for name, v in CAP["honest"].items():
    counts = v["counts"]
    regs = {r for by in counts.values() for r in by}
    fams = {}
    for r in regs:
        fams.setdefault(site_family(r), set()).add(r)
    out["honest"][name] = {
        "scores": v["scores"],
        "per_class": score_counts(counts)[1],
        "per_class_new": score_counts(counts, set(CAP["sets"]["new"]))[1],
        "per_family": {f: score_counts(counts, rs)[0] for f, rs in sorted(fams.items())},
        "per_family_class": {f: {c: d for c, d in score_counts(counts, rs)[1].items() if d["n_gt"] or d["tp"] or d["fp"]}
                             for f, rs in sorted(fams.items())},
    }
EXTRA.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("extra ->", EXTRA, flush=True)
import os
sys.stdout.flush(); os._exit(0)
