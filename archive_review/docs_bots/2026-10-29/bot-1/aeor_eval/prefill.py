import json, os, sys
V = sys.argv[1]
CL = json.load(open("/workspace/koz3_shared/bot-10/work/family_clusters.json"))["clusters"]
UNIT = {m: "+".join(sorted(ms)) for ms in CL.values() for m in ms}
changed = {UNIT.get(f[:-6], f[:-6]) for f in os.listdir(f"fam_{V}") if not os.path.islink(f"fam_{V}/{f}")}
os.makedirs(f"curves_{V}", exist_ok=True)
for p in os.listdir("curves_base"):
    if p[:-4] not in changed and not os.path.lexists(f"curves_{V}/{p}"):
        os.symlink(os.path.abspath(f"curves_base/{p}"), f"curves_{V}/{p}")
# собрать полный каталог для paired: work/curves + затронутые юниты
T = set(json.load(open("touched.json"))["units"]); A = "/workspace/koz3_shared/bot-10/work/curves"
os.makedirs(f"asm_{V}", exist_ok=True)
for p in os.listdir(A):
    d = f"asm_{V}/{p}"
    if os.path.lexists(d): os.unlink(d)
    os.symlink(os.path.abspath(f"curves_{V}/{p}") if p[:-4] in T else f"{A}/{p}", d)
print(V, "changed", sorted(changed))
