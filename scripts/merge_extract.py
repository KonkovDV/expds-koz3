"""bot-2, merge-check: потоково (только чтение) вынимает из стека объекты участков, где у markup
несколько входных файлов (источников) по листингу S3 — только там post.merge_across_sources
может склеить объекты (source = имя файла, expds/pipeline.py:59,64 окна растра; li — один LAS на участок).
python merge_extract.py <stack.geojson> <s3_sources.json> <out.pkl>"""
import json, pickle, sys
src, srcs, out = sys.argv[1:4]
multi = json.load(open(srcs))["multi"]
want = {m: set(v) for m, v in multi.items() if m not in ("etgr",) and v}   # etgr не склеивается (post.py:111)
dec = json.JSONDecoder(); buf = ""; started = False; n = 0; keep = []
with open(src, "r") as f:
    while True:
        chunk = f.read(8 << 20); buf += chunk
        if not started:
            i = buf.find('"features": ['); buf = buf[i + 13:]; started = True
        pos = 0
        while True:
            while pos < len(buf) and buf[pos] in " ,\n\r\t": pos += 1
            if pos >= len(buf) or buf[pos] == "]": break
            try: obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError: break
            p = obj.get("properties") or {}
            if p.get("region_name") in want.get(str(p.get("markup_type")), ()):
                p["_ord"] = n; keep.append(obj)
            n += 1; pos = end
        buf = buf[pos:]
        if not chunk: break
pickle.dump({"n_stack": n, "features": keep, "want": {k: sorted(v) for k, v in want.items()}}, open(out, "wb"))
print(n, len(keep))
