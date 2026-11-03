import json,glob,sys
for f in sys.argv[1:]:
    try: d=json.load(open(f))
    except Exception as e: print(f,'ERR',e); continue
    g=d['C']['geo'];h=d['honest_vs_A']
    print(f"{f.split('gate_CetO3_Sr8G_noTerFl_L_')[-1]:28s} P {g['precision']:.4f} R {g['recall']:.4f} Δ {h['d_global']:+.4f} finds {h['d_f1']['finds_points']:+.4f} poly {h['d_f1']['object_poly']:+.4f} PASSref {d['gate']['PASS_vs_ref']} maggrR {d['gate']['method_deltas_r']['maggr']:+.3f}")
