import csv, sys
from collections import Counter, defaultdict
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
B = ["<=100", "100-500", "500-1k", "1k-5k", "5k-10k", ">10k"]
def table(key, title):
    print(f"\n## {title}")
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["bin"] == "point": continue
        for k in (key(r), "ВСЕ"):
            c[(k, r["bin"])][0] += 1; c[(k, r["bin"])][1] += int(r["kept"])
    ks = sorted({k for k, _ in c}, key=lambda k: (k == "ВСЕ", k))
    print("| " + " | ".join(["срез"] + B) + " |"); print("|" + "---|" * (len(B) + 1))
    for k in ks:
        print("| " + " | ".join([k] + [f"{c[(k,b)][1]}/{c[(k,b)][0]} ({c[(k,b)][1]/c[(k,b)][0]:.3f})" if c[(k,b)][0] else "—" for b in B]) + " |")
table(lambda r: "все", "все")
table(lambda r: r["dtype"], "по типу данных (GT markup_type → группа LB)")
table(lambda r: r["cls"], "по классу")
big = [r for r in rows if r["bin"] in ("5k-10k", ">10k")]
print("\n## >5k: класс × тип")
c = Counter((r["cls"], r["dtype"], r["bin"]) for r in big); k = Counter((r["cls"], r["dtype"], r["bin"]) for r in big if r["kept"] == "1")
for key in sorted(c): print(key, f"{k[key]}/{c[key]}")
print("\n## причины пропуска >5k (не найден при честных порогах)")
miss = [r for r in big if r["kept"] != "1"]
print(Counter(r["reason"] for r in miss))
print(Counter((r["cls"], r["reason"]) for r in miss))
print("matched_any (взят в жадном сопоставлении, но ниже порога):", sum(r["matched_any"] == "1" for r in miss))
print("групп below_tau:", Counter((r["cls"], r["ok_grp"], r["ok_tau"]) for r in miss if r["reason"] == "below_tau"))
print("other-class hit:", Counter((r["cls"], r["other_ok_cls"]) for r in miss if r["other_ok_cls"]))
print("\n## найденные >5k: группа")
print(Counter((r["cls"], r["m_grp"]) for r in big if r["kept"] == "1"))
