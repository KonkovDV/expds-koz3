import csv, math, sys
fs = sys.argv[2:]; R = [list(csv.DictReader(open(f))) for f in fs]
lg = lambda p: math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6)))
w = csv.DictWriter(open(sys.argv[1], "w", newline=""), fieldnames=list(R[0][0])[:8]); w.writeheader()
for rows in zip(*R):
    assert len({r["stack_idx"] for r in rows}) == 1
    o = {k: rows[0][k] for k in list(R[0][0])[:7]}; o["p_povr"] = round(1 / (1 + math.exp(-sum(lg(float(r["p_povr"])) for r in rows) / len(rows))), 6); w.writerow(o)
