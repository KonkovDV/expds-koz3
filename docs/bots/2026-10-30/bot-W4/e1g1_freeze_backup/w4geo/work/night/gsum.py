import json,sys
W="/workspace/koz3/w4geo/work/cstar/"
for n in sys.argv[1:]:
  for suf in ["","_ht"]:
    try: d=json.load(open(W+f"gate_CetO3_Sr8G_noTerFl_L_E1_{n}{suf}.json"))
    except Exception as e: print(n,suf,"ERR",e); continue
    c=d["C"]; m=c["methods"]["maggr"]; fp=m["classes"]["finds_points"]; e=c["methods"]["etgr"]
    ks=[k for k in d if "pass" in k.lower() or "ready" in k.lower()]
    print(f"{n+suf:10s} geo {c['geo']['precision']:.3f}/{c['geo']['recall']:.3f} MagGr {m['precision']:.4f}/{m['recall']:.4f} (n {m['n_pred']}) finds {fp['precision']:.3f}/{fp['recall']:.3f} EtGr {e['precision']:.3f}/{e['recall']:.3f} hΔ {d['honest_vs_A']['d_global']:.4f}", {k:d[k] for k in ks})
