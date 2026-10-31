"""Stream /workspace/rerank/stack_arch.geojson.gz, keep gor/fort features (all groups) + class/group counts."""
import gzip, json, ijson, collections, sys
src = "/workspace/rerank/stack_arch.geojson.gz"
keep = {"gorodishcha", "fortifikatsii"}
cnt = collections.Counter(); keys = collections.Counter(); out = []
with gzip.open(src, "rb") as f:
    for feat in ijson.items(f, "features.item", use_float=True):
        p = feat["properties"]
        cnt[(p.get("class_name"), p.get("threshold_group"), p.get("markup_type"))] += 1
        if p.get("class_name") in keep:
            keys.update(p.keys()); out.append(feat)
json.dump({"type": "FeatureCollection", "features": out}, open("stack_gor_fort.geojson", "w"))
json.dump({"counts": [[*k, v] for k, v in sorted(cnt.items(), key=str)], "keys": keys}, open("stack_counts.json", "w"), ensure_ascii=False, indent=1)
print(len(out)); print(keys)
