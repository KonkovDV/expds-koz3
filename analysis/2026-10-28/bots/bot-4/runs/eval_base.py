import json, time
from common import *
t=time.time(); gt = gt_sub(load_gt()); pred = load_stack()
print("gt", len(gt["features"]), "pred", len(pred["features"]), f"{time.time()-t:.0f}s")
out, curves, counts = evaluate(pred, gt, tag="base")
json.dump(out, open("res_base.json","w"), ensure_ascii=False, indent=1, default=float)
print(f"{time.time()-t:.0f}s")
