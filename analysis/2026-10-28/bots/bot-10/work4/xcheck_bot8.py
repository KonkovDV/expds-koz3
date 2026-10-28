"""Сверка временного EXPDS_MATCH_SCOPE=site (bot-10) с рабочей копией bot-8 (незакоммичено, только чтение) на семьях стека."""
import sys, json, pickle, os
from pathlib import Path
sys.path.insert(0, "/workspace/koz3_shared/bot-10/work4/repo_bot8")
from expds import metric as M8
assert "repo_bot8" in M8.__file__, M8.__file__
if hasattr(M8, "_SHM_CACHE_DIR"): M8._SHM_CACHE_DIR = Path("/proc/expds_no_cache")
from expds.naming import site_family
gt_all = json.loads(Path("/workspace/rerank/gt_train_v2.geojson").read_text(encoding="utf-8"))["features"]
res = {}
for fam in sys.argv[1:]:
    mem = fam.split("+")
    pred = {"type": "FeatureCollection", "features": [json.loads(l) for m in mem for l in open(f"work/fam/{m}.jsonl", encoding="utf-8")]}
    gt = {"type": "FeatureCollection", "features": [f for f in gt_all if site_family(str(f["properties"].get("region_name"))) in mem]}
    cv = M8.build_curves(pred, gt, None, scope="site")
    mine = pickle.loads(Path(f"work4/curves_site/{fam}.pkl").read_bytes())
    r = {}
    for c in mine:
        a, b = mine[c], cv.get(c)
        r[c] = {"bot10_tp": int(a.hit.sum()), "bot8_tp": int(b.hit.sum()) if b is not None else None,
                "n_pred": [int(a.conf.size), int(b.conf.size) if b is not None else None],
                "hit_equal": bool(b is not None and a.conf.size == b.conf.size and (a.hit == b.hit).all())}
    res[fam] = r; print(fam, json.dumps(r, ensure_ascii=False), flush=True)
Path(os.environ.get("XOUT", "work4/xcheck_bot8.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
