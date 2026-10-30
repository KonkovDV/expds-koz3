from lib import *
regs=regions(); gt=E._load(GTP,regs,"maggr")
P=E._load(W+"cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson", regs, "maggr")
h,c=feat_hits(P,gt)
H=E._load(W+"cstar/C_CetO3_Sr8G_noTerFl_L_E1_ht.geojson", regs, "maggr")
keys=set((f["properties"]["source"],f["properties"]["region_name"],json.dumps(f["geometry"]["coordinates"])) for f in H["features"])
rows=collections.defaultdict(lambda:[0,0,0,0,None,None,[],[]])
for f,hh in zip(P["features"],h):
    if hh is None: continue
    p=f["properties"]; s=(p["region_name"][:3],p.get("source")); x=rows[s]; k=(p["source"],p["region_name"],json.dumps(f["geometry"]["coordinates"])) in keys
    x[0]+=1; x[1]+=hh; x[2]+=k; x[3]+=hh and k; x[4]=metric.threshold_group(p); x[6].append(p["confidence"]); x[7].append(hh)
for s,x in sorted(rows.items()):
    v=np.array(x[6]); hv=np.array(x[7]); print(f"{s[0]} {x[4]:15s} n={x[0]:5d} hit={x[1]:4d} | kept={x[2]:4d} hitk={x[3]:4d} | q50={np.median(v):.3f} q90={np.quantile(v,.9):.3f} max={v.max():.3f} hitq50={np.median(v[hv]) if hv.any() else 0:.3f} | {s[1][:55]}")
print("gt per region", dict(c.gt_regions))
