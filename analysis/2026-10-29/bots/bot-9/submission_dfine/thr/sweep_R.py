# Фиксированные пороги для пакета: sweep_groups (pick=max) по всем 100 юнитам c5_R (вариант R) и c5_A (база, для сверки).
import sys, json, os
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); os.chdir(B10)
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves
from expds import metric as M
out = {}
for name, d in [("R", sys.argv[1]), ("A", "work/curves")]:
    cv = load_curves(Path(d))
    for pick in ("max", "plateau"):
        sw = M.sweep_groups(cv, pick=pick)
        out[f"{name}_{pick}"] = sw
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1, default=float)
for k, sw in out.items():
    print("==", k)
    for cls, r in sw.items():
        print(f"  {cls}: f1={r.get('f1')} n_gt={r.get('n_gt')} groups={json.dumps(r.get('groups'), default=float)[:600]}")
