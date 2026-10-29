import json,sys
H=["kurgany_povrezhdennye","kurgany_tselye","gorodishcha"]
for f in sys.argv[1:]:
    d=json.load(open(f)); print("==",f, d["base"].split("/")[-1])
    k10=d["layouts"]["k10"]
    for k,s in d["summary"].items():
        b=s["boot"]; pc=s["per_class_weighted"]
        p4c=all(pc.get(c,0)>=-0.0005 for c in H)
        pol=k.split("_")[-1]; base=d["base"].split("=")[0]
        r=s["rule"]
        print(f"{k:8s} base={k10[base+'_'+pol]['all']:.6f} var={k10[k]['all']:.6f} d={s['d_k10']:+.6f} P={b['p_gt0']:.3f} CI=[{b['p2_5']:+.4f};{b['p97_5']:+.4f}] lofo={s['lofo']:+.6f} seeds+={s['seeds_pos']} heavy={s['heavy_weighted']:+.6f} new34={s['d_new34']:+.6f} P1={r['P1_d>=0.002']} P2={r['P2_P>=0.85']} P3={r['P3_lofo>0_and_seeds>=4/5']} P4sum={r['P4_heavy>=-0.0005_and_new34>=-0.001']} P4cls={p4c} P5={'single' if s['d_k10']>=0.006 else 'package'} acc={s['accepted']}")
        print("   cls:", " ".join(f"{c[:12]}={v:+.5f}" for c,v in pc.items()))
