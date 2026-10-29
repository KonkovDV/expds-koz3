# Каталоги c_<V> по 100 юнитов (симлинки): затронутые юниты — work/curves_<V>, остальные — bot-10/work/curves (= база A).
# Сверка: curves_base == bot-10/work/curves байт в байт на затронутых юнитах.
import sys, os, json
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); A = B10 / "work/curves"
OUT = Path(__file__).resolve().parent; WD = Path(os.environ.get("RE_WORK", OUT / "work")); os.chdir(WD)
FAMS = set(json.load(open("gt.log"))["fams"]); VS = sys.argv[1:]
U = sorted(p.stem for p in (WD / "curves_base").glob("*.pkl") if set(p.stem.split("+")) & FAMS)  # family_curves пишет и пустой юнит кластера — исключить
allu = sorted(p.stem for p in A.glob("*.pkl"))
rep = {"units": U, "skipped": sorted(p.stem for p in (WD / "curves_base").glob("*.pkl") if p.stem not in U), "A_units": len(allu),
       "base_check": {u: (A / f"{u}.pkl").read_bytes() == (WD / "curves_base" / f"{u}.pkl").read_bytes() for u in U}}
assert all(rep["base_check"].values()), rep["base_check"]
for v in ["A"] + VS:
    d = WD / f"c_{v}"; d.mkdir(exist_ok=True)
    for p in d.glob("*.pkl"): p.unlink()
    for u in allu:
        t = WD / f"curves_{v}" / f"{u}.pkl" if v != "A" and u in U else A / f"{u}.pkl"
        assert t.exists(), t; os.symlink(t, d / f"{u}.pkl")
json.dump(rep, open("build.json", "w"), ensure_ascii=False, indent=1); print(json.dumps({"n_units": len(U), "skipped": rep["skipped"], "base_ok": all(rep["base_check"].values())}, ensure_ascii=False))
