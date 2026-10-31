import json,sys
d=json.load(open(sys.argv[1]))
for c in sorted(d['C'],key=lambda c:-c['d_global']):
    s=c['summary'];g=c['gate_proxy'];b=c['boot_family']
    print(f"{c['file'][:34]:34s} d={c['d_global']:+.4f} finds={s['f1']['finds_points']:.4f}({c['d_f1']['finds_points']:+.4f}) poly={s['f1']['object_poly']:.4f}({c['d_f1']['object_poly']:+.4f}) "+" ".join(f"{k}:{v['d_global']:+.4f}" for k,v in c['slices_method'].items())+f" folds={g['folds_pos']} p={b['p_gt0']:.2f}")
