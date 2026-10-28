exec(open("arch_micro_honest.py").read().split("res, cnt = {}, {}")[0])
import collections
_, b = run(st, "b"); _, m = run(micro(st, 0.25), "m")
d = collections.defaultdict(lambda: [0, 0])
for r in set(b[A]) | set(m[A]):
    x = b[A].get(r, [0, 0, 0]); y = m[A].get(r, [0, 0, 0]); f = site_family(r)
    d[f][0] += y[0] - x[0]; d[f][1] += y[1] - x[1]
out = {f: {"dTP": v[0], "dFP": v[1]} for f, v in sorted(d.items()) if v != [0, 0]}
json.dump(out, open("../analysis/arch_micro_honest_families.json", "w"), ensure_ascii=False, indent=1); print(out)
