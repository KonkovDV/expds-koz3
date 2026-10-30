import sys, json, numpy as np
sys.path.insert(0,'/workspace/koz3/wt/w4-pix'); sys.path.insert(0,'/workspace/koz3/wt/w4-pix/scripts')
from expds import metric
from pix_clean_eval import _load, folds_of, honest
gt=json.loads(open('/workspace/koz3/w5data/gt_train_v2.geojson').read())
for fn in sys.argv[1:]:
    pr=_load(fn, mt='maggr'); regs={f['properties']['region_name'] for f in pr['features']}
    g={**gt,'features':[f for f in gt['features'] if f['properties'].get('region_name') in regs and f['properties'].get('class_name')=='finds_points' and f['properties'].get('markup_type','maggr')=='maggr']}
    pr={**pr,'features':[f for f in pr['features'] if f['properties']['class_name']=='finds_points']}
    c=metric.build_curves(pr,g)['finds_points']
    tp=np.cumsum(c.hit); fp=np.cumsum(~c.hit); n=c.n_gt
    f1=2*tp/(tp+fp+n); i=int(f1.argmax())
    cnt,_=honest(pr,g,folds_of(regs)); s=np.array(list(cnt['finds_points'].values())).sum(0)
    hf=2*s[0]/(s[0]+s[1]+s[2])
    # per-region oracle
    reg=np.array(c.pred_region); po=[]
    for r in sorted(regs):
        m=reg==r; h=c.hit[m]; nn=c.gt_regions.get(r,0)
        if nn==0: continue
        t=np.cumsum(h); f=np.cumsum(~h); po.append((r[:3],round(float((2*t/(t+f+nn)).max()) if len(t) else 0,3), round(t[-1]/nn if len(t) else 0,2)))
    print(f"{fn.split('/')[-1]:34s} n_gt={n} n_pred={len(c.hit)} Rmax={tp[-1]/n:.3f} oracleF1={f1[i]:.3f}@R{tp[i]/n:.3f}P{tp[i]/(tp[i]+fp[i]):.3f} honestF1={hf:.3f} P={s[0]/(s[0]+s[1]):.3f} R={s[0]/s[2]:.3f}")
    print('   ', po)
