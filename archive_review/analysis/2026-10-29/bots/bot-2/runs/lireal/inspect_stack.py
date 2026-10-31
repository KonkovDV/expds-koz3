"""Stream bot-10/work/fam/*.jsonl: per region x (markup_type, threshold_group, class) counts, for regions with li features."""
import json, sys, glob, os
from collections import Counter, defaultdict
src = sys.argv[1]
cnt = defaultdict(Counter); props_keys = Counter()
for p in sorted(glob.glob(os.path.join(src, "*.jsonl"))):
    for line in open(p, encoding="utf-8"):
        pr = json.loads(line)["properties"]
        mt = pr.get("markup_type"); tg = pr.get("threshold_group")
        if mt in ("li", "ae") or (tg or "").startswith("li"):
            cnt[pr["region_name"]][(mt, tg, pr["class_name"])] += 1
            props_keys.update(pr.keys())
out = {r: {"|".join(map(str, k)): v for k, v in c.items()} for r, c in cnt.items()}
json.dump({"keys": props_keys, "regions": out}, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
tot = Counter()
for c in cnt.values():
    for k, v in c.items(): tot[k[:2]] += v
print(len(cnt), dict(props_keys)); print({"|".join(map(str,k)): v for k, v in tot.items()})
