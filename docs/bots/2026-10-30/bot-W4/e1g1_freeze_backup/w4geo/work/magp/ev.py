"""fast screen: replace MagGr finds in E1 by a variant -> honest finds F1 (pix_clean_eval protocol) + gate-like MagGr P/R"""
from lib import *
import copy
regs=regions(); GT=E._load(GTP,regs); folds=E.folds_of(regs); FL=[s for s in folds.values() if s]
BASE=W+"cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson"
def split(path=BASE):
    d=E._load(path,regs); F=[f for f in d["features"] if f["properties"].get("markup_type")=="maggr" and f["properties"]["class_name"]=="finds_points"]
    O=[f for f in d["features"] if not (f["properties"].get("markup_type")=="maggr" and f["properties"]["class_name"]=="finds_points")]
    return d,F,O
GTF={**GT,"features":[f for f in GT["features"] if f["properties"]["class_name"]=="finds_points"]}
MP=dict(tp=0.2903225794*248, n=146)  # maggr poly from E1_ht gate (R .290 of 248; P .66)
def score(F, tag="", verbose=True):
    pred={"type":"FeatureCollection","features":F}
    c=metric.build_curves(pred,GTF)["finds_points"]
    counts,_=metric.honest_counts({"finds_points":c},FL)
    tp=fp=0
    for r,(a,b,g) in counts["finds_points"].items(): tp+=a; fp+=b
    ngt=c.n_gt; f1=2*tp/(2*tp+fp+(ngt-tp)) if tp else 0
    # metric _f1 uses P=tp/(tp+.5fp)
    P=tp/(tp+.5*fp); R=tp/ngt; F1=2*P*R/(P+R)
    mtp=tp+MP["tp"]; mfp=fp+MP["n"]-MP["tp"]; mP=mtp/(mtp+.5*mfp); mR=mtp/(3726)
    per={r:(a,b,g) for r,(a,b,g) in counts["finds_points"].items()}
    out=dict(tag=tag,n_kept=tp+fp,tp=tp,fp=fp,finds_P=round(P,4),finds_R=round(R,4),finds_F1=round(F1,5),mag_P=round(mP,4),mag_R=round(mR,4))
    if verbose: print(json.dumps(out,ensure_ascii=False),flush=True)
    return out,per
if __name__=="__main__":
    d,F,O=split(); o,per=score(F,"E1")
    for r,v in sorted(per.items()): print(r[:3],v, round(v[0]/max(1,v[2]),3))
