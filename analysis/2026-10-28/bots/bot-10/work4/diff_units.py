import sys; sys.path.insert(0, "repo")
import pickle, json
from pathlib import Path
out = {}
for p in sorted(Path("work/curves").glob("*.pkl")):
    a = pickle.loads(p.read_bytes()); b = pickle.loads((Path("work4/curves_site") / p.name).read_bytes())
    d = {c: int(a[c].hit.sum()) - int(b[c].hit.sum()) for c in a if int(a[c].hit.sum()) != int(b[c].hit.sum())}
    if d: out[p.stem] = d
tot = {}
for d in out.values():
    for c, v in d.items(): tot[c] = tot.get(c, 0) + v
print(json.dumps({"units": out, "total_tp_global_minus_site": tot, "sum": sum(tot.values())}, ensure_ascii=False))
Path("work4/diff_units.json").write_text(json.dumps({"units": out, "total_tp_global_minus_site": tot, "sum": sum(tot.values())}, ensure_ascii=False, indent=1), encoding="utf-8")
