import sys, json, collections, numpy as np
sys.path.insert(0, "/workspace/koz3/wt/w4-pix"); sys.path.insert(0, "/workspace/koz3/wt/w4-pix/scripts")
from expds import metric; import pix_clean_eval as E
W="/workspace/koz3/w4geo/work/"; GTP="/workspace/koz3/w5data/gt_train_v2.geojson"
mod=metric.load_official(None)
def regions():
    return {f["properties"]["region_name"] for f in json.load(open(GTP))["features"] if f["properties"].get("region_name","").split("_")[0] in {str(i) for i in range(200,218)}}
def feat_hits(pred, gt, cls="finds_points"):
    """per-feature hit aligned with pred['features'] (None for non-cls)"""
    c=metric.build_curves(pred,gt)[cls]
    idx=[i for i,f in enumerate(pred["features"]) if mod.normalize_class_name(f["properties"].get("class_name"))==cls and f.get("geometry")]
    order=sorted(idx, key=lambda i: mod._get_confidence(pred["features"][i]), reverse=True)
    assert len(order)==len(c.conf)
    h=[None]*len(pred["features"])
    for k,i in enumerate(order): h[i]=bool(c.hit[k])
    return h, c
