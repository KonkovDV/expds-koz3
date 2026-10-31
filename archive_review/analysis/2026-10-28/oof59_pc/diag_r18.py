"""Сравнение распределений: сданный li_hard2_fused против нового fused (курганы, по группам и семьям)."""
import json, sys, gc
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(REPO))
from expds.naming import site_family

KUR = {"kurgany_povrezhdennye", "kurgany_tselye"}


def summarize(path: Path) -> dict:
    gc.disable()
    fc = json.loads(path.read_bytes())
    feats = fc["features"]
    del fc
    keys = Counter()
    conf = defaultdict(list)
    fam = defaultdict(Counter)
    regions = set()
    for f in feats:
        p = f.get("properties") or {}
        cls = p.get("class_name")
        mk = p.get("markup_type")
        keys[f"{mk}|{p.get('threshold_group')}|{cls}"] += 1
        if mk == "li":
            regions.add(p.get("region_name"))
        if cls in KUR and mk == "li":
            conf[cls].append(float(p.get("confidence") or 0))
            fam[cls][site_family(str(p.get("region_name")))] += 1
    del feats
    gc.enable(); gc.collect()
    q = [0.5, 0.9, 0.95, 0.99, 0.999]
    out = {"keys": dict(keys), "li_regions": len(regions), "regions": sorted(r for r in regions if r)}
    for cls, v in conf.items():
        a = np.asarray(v)
        out[cls] = {"n": int(a.size), "quantiles": dict(zip(map(str, q), np.quantile(a, q).round(5).tolist())),
                    "n_ge": {str(t): int((a >= t).sum()) for t in (0.05, 0.0641, 0.1, 0.1942, 0.3, 0.5, 0.7, 0.9)},
                    "by_family": dict(fam[cls])}
    return out


if __name__ == "__main__":
    res = {}
    for name, p in (("old", sys.argv[1]), ("new", sys.argv[2])):
        print("читаю", p, flush=True)
        res[name] = summarize(Path(p))
    Path(sys.argv[3]).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok", flush=True)
