from rx import *
conf=df.conf.astype(float); eps=1e-4
rr=np.clip(np.nan_to_num(mf["ring"]/(mf["s1"]+eps),nan=0.5),0,1.5)
pr=np.clip(np.nan_to_num(mf["prom"],nan=0),0,1)
for b in [0.25,0.5,1.0]:
    run(conf*np.clip(1-rr,0.02,1)**b, f"ringr_b{b}")
for b in [0.25,0.5]:
    run(conf**(1-b)*np.maximum(pr,1e-3)**b, f"prom_mix{b}")
s4=np.nan_to_num(mf["s4"],nan=0)
for b in [0.25,0.5]:
    run(conf**(1-b)*np.maximum(s4,1e-3)**b, f"s4_mix{b}")
