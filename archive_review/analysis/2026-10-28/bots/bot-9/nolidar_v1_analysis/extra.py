import csv, collections, numpy as np
R = list(csv.DictReader(open("/workspace/kgl_out/nolidar_v/oof_verifier_nolidar.csv", encoding="utf-8")))
for h in ("kurgan", "gor"):
    for k in "01234":
        v = np.array([float(r["p_verifier"]) for r in R if r["group"] == h and r["fold"] == k])
        print(h, k, "n", len(v), "std", round(float(v.std()), 5), "p1/p50/p99", np.round(np.percentile(v, [1, 50, 99]), 5).tolist(), "uniq", len(np.unique(v)))
# дубли: одинаковые (region,cx,cy) у разных классов
g = collections.defaultdict(list)
for r in R:
    if r["group"] == "kurgan": g[(r["region"], r["cx"], r["cy"])].append(r)
d = [x for x in g.values() if len(x) > 1]
print("kurgan точек с >1 кандидатом в одной позиции:", len(d), "строк:", sum(len(x) for x in d),
      "классы:", collections.Counter(tuple(sorted(r["class_name"] for r in x)) for x in d).most_common(3),
      "метки расходятся:", sum(len({r["label"] for r in x}) > 1 for x in d))
print("Курганы регионы:", collections.Counter(r["region"] for r in R if r["family"] == "Курганы" and r["group"] == "kurgan").most_common(12))
