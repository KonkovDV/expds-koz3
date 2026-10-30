import json,sys
W="/workspace/koz3/w4geo/work/cstar/"
R=json.load(open(W+f"R_{sys.argv[1]}.json"))
def g(n):
    try: d=json.load(open(W+f"gate_CetO3_Sr8G_noTerFl_L_E1_{n}{sys.argv[2] if len(sys.argv)>2 else '_ht'}.json"))
    except Exception as e: return None
    c=d["C"]; m=c["methods"]["maggr"]; fp=m["classes"]["finds_points"]
    return f"geo {c['geo']['precision']:.4f}/{c['geo']['recall']:.4f} MagGr {m['precision']:.4f}/{m['recall']:.4f} finds {fp['precision']:.4f}/{fp['recall']:.4f} harnessΔ {d.get('honest_vs_A',{}).get('d_global',float('nan')):.4f} pass={d.get('PASS_vs_ref',d.get('pass_vs_ref'))}"
print(json.dumps(R,ensure_ascii=False)[:600]) if isinstance(R,dict) and 'C' not in str(list(R)[:3]) else None
