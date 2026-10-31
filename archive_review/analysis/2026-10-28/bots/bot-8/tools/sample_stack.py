"""Потоковая выборка реального стека: все объекты из детерминированного подмножества участков.
python sample_stack.py SRC.geojson.gz GT.geojson OUT_PRED OUT_GT frac"""
import gzip, json, sys, zlib, ijson
src, gt_p, out_p, out_g, frac = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
keep = lambda r: (zlib.crc32(str(r).encode()) % 1000) < frac * 1000
gt = json.load(open(gt_p, encoding="utf-8"))
gtf = [f for f in gt["features"] if keep((f.get("properties") or {}).get("region_name"))]
json.dump({"type": "FeatureCollection", "features": gtf}, open(out_g, "w", encoding="utf-8"), ensure_ascii=False)
del gt
n = k = 0
with gzip.open(src, "rb") as fh, open(out_p, "w", encoding="utf-8") as out:
    out.write('{"type": "FeatureCollection", "features": [')
    for f in ijson.items(fh, "features.item", use_float=True):
        n += 1
        if keep((f.get("properties") or {}).get("region_name")):
            out.write(("," if k else "") + json.dumps(f, ensure_ascii=False)); k += 1
    out.write("]}")
print("pred", n, "->", k, "gt", len(gtf))
