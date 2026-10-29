"""Потоковая разбивка большого стека (FeatureCollection) по участкам в JSONL без загрузки целиком.
python stack_split_regions.py <stack.geojson> <outdir>   — открывает стек только на чтение."""
import json, sys, os, collections
src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
dec = json.JSONDecoder(); buf = ""; started = False; files = {}
stat = collections.Counter(); keys = collections.Counter(); n = 0
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
            r = str(p.get("region_name", "default_source"))
            if p.get("sub_region_name"): stat["has_sub_region"] += 1
            stat["|".join(map(str, (p.get("markup_type"), p.get("threshold_group") or "", p.get("class_name"))))] += 1
            for k in p: keys[k] += 1
            if r not in files: files[r] = open(os.path.join(outdir, r + ".jsonl"), "w")
            p["_ord"] = n                                  # порядок в исходном файле (для тай-брейка как в build_curves)
            files[r].write(json.dumps(obj, ensure_ascii=False) + "\n"); n += 1
            pos = end
        buf = buf[pos:]
        if not chunk: break
for v in files.values(): v.close()
json.dump({"n": n, "regions": len(files), "counts": dict(stat), "prop_keys": dict(keys)},
          open(os.path.join(outdir, "_summary.json"), "w"), ensure_ascii=False, indent=1)
print(n, len(files))
