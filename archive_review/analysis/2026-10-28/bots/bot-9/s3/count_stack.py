"""Потоковый подсчёт кандидатов стека по участку × классу × markup (и conf ≥ 0.05). Пишет stack_counts.json."""
import ijson, json, sys, collections, time
t = time.time(); src = sys.argv[1]
c_all = collections.Counter(); c05 = collections.Counter(); fids = collections.Counter(); n = 0; keys = collections.Counter()
with open(src, "rb") as f:
    for p in ijson.items(f, "features.item.properties"):
        n += 1; keys.update(p.keys())
        k = (p.get("region_name"), p.get("class_name"), p.get("markup_type"), p.get("threshold_group"))
        c_all[k] += 1
        if float(p.get("confidence", 0)) >= 0.05: c05[k] += 1
        fids[(p.get("fid"), p.get("markup_type"), p.get("region_name"))] += 1
dup = sum(1 for v in fids.values() if v > 1)
out = {"src": src, "n": n, "sec": round(time.time() - t), "prop_keys": dict(keys), "fid_region_markup_dups": dup,
       "rows": [[*k, v, c05[k]] for k, v in sorted(c_all.items(), key=lambda kv: str(kv[0]))]}
json.dump(out, open("stack_counts.json", "w", encoding="utf-8"), ensure_ascii=False)
print(n, out["sec"], dup, dict(keys))
