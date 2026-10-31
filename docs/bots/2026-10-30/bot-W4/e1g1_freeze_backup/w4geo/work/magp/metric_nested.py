from rs import *
def auc(y,s):
    o=np.argsort(s,kind="stable"); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1); n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def nested_eval(F,df,X,y,fo,K,nr,grpmode="orig",verbose=True):
    """returns dict(tp,fp,P,R,F1, kept_idx, conf_held)"""
    regs_all=set(df.reg); tot=dict(tp=0,fp=0); kept=[]; held_conf=np.zeros(len(y)); per={}
    grp=df.grp if grpmode=="orig" else np.array(["maggr_pix_one"]*len(y))
    for k in K:
        s=np.zeros(len(y)); te=fo==k
        s[te],_=fit_pred(X,y,~te,te,nr); held_conf[te]=s[te]
        for j in K:
            if j==k: continue
            tj=fo==j; s[tj],_=fit_pred(X,y,~(te|tj),tj,nr)
        Fk=with_conf(F,s,grp=grp)
        c=metric.build_curves({"type":"FeatureCollection","features":Fk},GTF)["finds_points"]
        held={r for r in regs_all if fold_of(r)==k}
        # also GT regions in held without preds
        heldall={r for r in c.gt_regions if fold_of(r)==k}|held
        train=set(c.gt_regions)|regs_all; train-=heldall
        res=metric.sweep_groups({"finds_points":metric.curve_where(c,train)})
        taus={g:gv["threshold"] for g,gv in res["finds_points"]["groups"].items()}
        hc=metric.curve_where(c,heldall)
        t=np.array([taus.get(g,np.inf) for g in hc.pred_group]); kk=hc.conf>=t
        tp=int(hc.hit[kk].sum()); fp=int(kk.sum()-tp); tot["tp"]+=tp; tot["fp"]+=fp
        for i in np.flatnonzero(te):
            if s[i]>=taus.get(grp[i],np.inf): kept.append(i)
        per[k]=dict(tp=tp,fp=fp,taus={g:round(v,3) for g,v in taus.items()})
    ngt=GTF and sum(1 for f in GTF["features"])
    tp,fp=tot["tp"],tot["fp"]; P=tp/(tp+.5*fp); R=tp/3478; F1=2*P*R/(P+R)
    mtp=tp+MP["tp"]; mfp=fp+MP["n"]-MP["tp"]
    out=dict(tp=tp,fp=fp,P=round(P,4),R=round(R,4),F1=round(F1,5),magP=round(mtp/(mtp+.5*mfp),4),magR=round(mtp/3726,4),per=per)
    if verbose: print("NESTED",json.dumps({k:v for k,v in out.items() if k!="per"}),{k:(v["tp"],v["fp"]) for k,v in per.items()},flush=True)
    out["kept"]=kept; out["held_conf"]=held_conf
    return out
