"""Все участки, полный эталон, каждое N-е предсказание (детерминированно по номеру).
python sample_stack2.py SRC.geojson.gz OUT_PRED every"""
import gzip, json, sys, ijson
src, out_p, every = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = k = 0
with gzip.open(src, "rb") as fh, open(out_p, "w", encoding="utf-8") as out:
    out.write('{"type": "FeatureCollection", "features": [')
    for f in ijson.items(fh, "features.item", use_float=True):
        if n % every == 0:
            out.write(("," if k else "") + json.dumps(f, ensure_ascii=False)); k += 1
        n += 1
    out.write("]}")
print("pred", n, "->", k)
