import sys
SCRIPT = "/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full/koz3-dfine-spor-full.py"
src = open(SCRIPT, encoding="utf-8").read(); sys.argv = [SCRIPT, "--allow-cpu"]; g = {"__name__": "__main__"}
exec(compile(src[:src.index("class Scene:")], SCRIPT, "exec"), g)
import collections; a = collections.Counter()
for i in g["INV"]: a[g["FOLD"][i["region"]]] += sum(s for _, s in i["tifs"])
print("GB per fold", {k: round(v / 1e9, 2) for k, v in sorted(a.items())}, "total GB", round(sum(a.values()) / 1e9, 2))
