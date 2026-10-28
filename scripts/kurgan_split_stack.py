# Поток: стек курганов -> файлы по участкам (только conf >= FLOOR) + сводка по всем.
import json, sys, collections, os
src, outdir, floor = sys.argv[1], sys.argv[2], float(sys.argv[3])
os.makedirs(outdir, exist_ok=True)
dec = json.JSONDecoder(); buf = ""; started = False
files = {}; stat = collections.Counter(); kept = collections.Counter(); maxlow = collections.defaultdict(float)
def fh(r):
    if r not in files:
        files[r] = open(os.path.join(outdir, r + ".jsonl"), "w")
    return files[r]
with open(src) as f:
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
            p = obj["properties"]; g = str(p.get("threshold_group") or p.get("markup_type"))
            key = (g, p["class_name"]); stat[key] += 1
            if p["confidence"] >= floor:
                kept[key] += 1; fh(p["region_name"]).write(json.dumps(obj) + "\n")
            pos = end
        buf = buf[pos:]
        if not chunk: break
for v in files.values(): v.close()
json.dump({"floor": floor, "all": {"|".join(k): v for k, v in stat.items()}, "kept": {"|".join(k): v for k, v in kept.items()}},
          open(os.path.join(outdir, "_summary.json"), "w"), ensure_ascii=False, indent=1)
print(sorted(stat.items())); print(sorted(kept.items()))
