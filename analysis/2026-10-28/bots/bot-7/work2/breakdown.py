"""Разбивки пакета bot-7: honest-10 (те же фолды), all/new34/optical, по классам, парный бутстреп по семьям.
python work2/breakdown.py base=DIR kur=DIR arch=DIR bundle=DIR  (первый — база)"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from expds import metric as M
from expds.constants import CLASS_WEIGHTS
from expds.naming import site_family
from honest_bootstrap import load_curves, regions_of, weighted_class_delta
M._SHM_CACHE_DIR = Path("/proc/expds_no_cache")
gt = json.loads(Path("/workspace/rerank/gt_train_v2.geojson").read_text(encoding="utf-8"))
boxes = json.loads((ROOT / "notes" / "las_boxes_59.json").read_text(encoding="utf-8"))
optical = {(f.get("properties") or {}).get("region_name") for f in gt["features"]} - set(boxes) - {None}
new = {l.strip() for l in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
vs = [a.split("=", 1) for a in sys.argv[1:]]
cur = {n: load_curves(Path(p)) for n, p in vs}
regions = set().union(*(regions_of(c) for c in cur.values()))
optical &= regions
folds = M.family_folds(regions, site_family, 10)
cnt, rep = {}, {"n_regions": len(regions), "n_optical": len(optical), "n_new34": len(new & regions), "variants": {}, "paired": {}}
for n, c in cur.items():
    cs, info = M.honest_counts(c, folds); cnt[n] = cs
    s, per = M.score_counts(cs)
    pc = {}
    for cls, by in cs.items():
        tp = sum(v[0] for v in by.values()); fp = sum(v[1] for v in by.values()); ng = sum(v[2] for v in by.values())
        if ng: pc[cls] = {"f1": per[cls]["f1"], "tp": tp, "fp": fp, "n_gt": ng}
    rep["variants"][n] = {"all": s, "new34": M.score_counts(cs, new)[0], "optical": M.score_counts(cs, optical)[0],
                          "unseen_dropped": info["unseen_dropped"], "per_class": pc}
    print(n, {k: v for k, v in rep["variants"][n].items() if k != "per_class"}, flush=True)
b = vs[0][0]
for n, _ in vs[1:]:
    va, vb = rep["variants"][n], rep["variants"][b]
    rep["paired"][f"{n}-{b}"] = {
        "d_all": va["all"] - vb["all"], "d_new34": va["new34"] - vb["new34"], "d_optical": va["optical"] - vb["optical"],
        "boot_all": M.paired_bootstrap_counts(cnt[n], cnt[b], site_family, 2000),
        "boot_new34": M.paired_bootstrap_counts(cnt[n], cnt[b], site_family, 2000, regions=new & regions),
        "boot_optical": M.paired_bootstrap_counts(cnt[n], cnt[b], site_family, 2000, regions=optical),
        "per_class_weighted": {c: x for c, x in weighted_class_delta(cnt[n], cnt[b]).items()},
        "per_class_dtp_dfp": {c: [va["per_class"][c]["tp"] - vb["per_class"][c]["tp"], va["per_class"][c]["fp"] - vb["per_class"][c]["fp"]] for c in va["per_class"]}}
out = Path(__file__).with_name("breakdown.json")
out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(rep["paired"], ensure_ascii=False, indent=1)[:5000])
