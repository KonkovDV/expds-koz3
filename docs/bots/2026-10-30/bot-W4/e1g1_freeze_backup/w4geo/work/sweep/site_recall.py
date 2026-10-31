import sys, json, collections
sys.path.insert(0, "/workspace/koz3/wt/w4-pix"); sys.path.insert(0, "/workspace/koz3/wt/w4-pix/scripts")
from expds import metric; import pix_clean_eval as E
gt_p = sys.argv[1]; files = sys.argv[2:]
gt = E._load(gt_p, None, "maggr"); regs = {f["properties"]["region_name"] for f in gt["features"]}
gt = E._load(gt_p, regs, "maggr")
rows = {}
for p in files:
    c = metric.build_curves(E._load(p, regs, "maggr"), gt)["finds_points"]
    tp = collections.Counter(r for r, h in zip(c.pred_region, c.hit) if h); n = collections.Counter(c.pred_region)
    rows[p.split("/")[-1][-22:]] = {r: (tp[r] / max(1, c.gt_regions[r]), n[r]) for r in c.gt_regions}
names = list(rows); print(f"{'site':28s}" + "".join(f"{x:>24s}" for x in names))
for r in sorted(c.gt_regions, key=lambda r: rows[names[-1]][r][0] - rows[names[0]][r][0]):
    print(f"{r[:26]:28s}" + "".join(f"{rows[x][r][0]:>16.3f} n={rows[x][r][1]:<5d}" for x in names))
