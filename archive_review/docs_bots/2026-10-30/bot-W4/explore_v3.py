import pickle, json, sys
from collections import Counter
sys.path.insert(0,"/workspace/koz3/wt/w1/scripts")
from split_stack_by_family import iter_features
T=pickle.load(open("/workspace/koz3/w5data/work/table.pkl","rb"))
print({k:(type(v).__name__) for k,v in T.items()})
grp=T["grp"].astype(str); reg=T["reg"].astype(str)
sp={"sp_or","sp_focal","sp_arch","sp_focal2"}
spreg=Counter(r for g,r in zip(grp,reg) if g in sp)
print("sp rows",sum(spreg.values()),"regions",len(spreg))
oofreg=set()
for d in ("full","rest"):
    for f,_ in iter_features(__import__("pathlib").Path(f"/workspace/koz3/w4geo/spor_oof/{d}/oof_dfine_spor.geojson")):
        oofreg.add(f["properties"]["region_name"])
print("oof regions",len(oofreg)); unc=set(spreg)-oofreg
print("sp regions not in OOF",len(unc),sorted(unc)[:20],sum(spreg[r] for r in unc))
print("grp x mk etgr", Counter((g,m) for g,m in zip(grp,T["mk"].astype(str)) if g=="etgr" or m=="etgr").most_common(5))
for p in ("A18","C19etonly_orget_s0"):
    c=Counter()
    for f,_ in iter_features(__import__("pathlib").Path(f"/workspace/koz3/w4geo/eval/{p}.geojson")):
        pr=f["properties"]; c[(pr.get("markup_type"),pr.get("threshold_group"))]+=1
    print(p,c.most_common(8))
