# Обзор групп порога стека бота 10 (bot-10/work/fam/*.jsonl): (markup, threshold_group, класс) → число объектов (курганы)
import json, glob, collections, sys
c = collections.Counter(); fams = collections.defaultdict(set)
for p in sorted(glob.glob("/workspace/koz3_shared/bot-10/work/fam/*.jsonl")):
    fam = p.rsplit("/", 1)[1][:-6]
    for l in open(p, encoding="utf-8"):
        pr = json.loads(l)["properties"]
        if pr["class_name"].startswith("kurgany"):
            k = (pr.get("markup_type"), pr.get("threshold_group"), pr["class_name"]); c[k] += 1; fams[k].add(fam)
out = [{"markup": k[0], "group": k[1], "class": k[2], "n": v, "n_fams": len(fams[k])} for k, v in sorted(c.items(), key=lambda x: -x[1])]
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
for r in out: print(r)
