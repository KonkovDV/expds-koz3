import json, sys
for fn in sys.argv[1:]:
    d = json.load(open(fn)); v = d["variants"]
    print("==", fn)
    for n, x in v.items():
        print(f"  {n:5s} h10 {x['honest10_all']:.6f} LOFO {x['honestLOFO_all']:.6f} seeds {[round(s,6) for s in x['honest10_random_seeds']]}")
    for k, p in d["paired"].items():
        b = p["bootstrap_families"]; o = k.split("-",1)[1]
        dl = v["A"]["honest10_all"] - v[o]["honest10_all"]; dL = v["A"]["honestLOFO_all"] - v[o]["honestLOFO_all"]
        print(f"  {k}: Δh10 {dl:+.6f} P {b['p_gt0']:.3f} CI [{b['p2_5']:+.5f}; {b['p97_5']:+.5f}] ΔLOFO {dL:+.6f} seeds {[round(s,5) for s in p['seed_deltas']]} lb_fam P {p['lb_sim_families']['delta']['p_gt0']:.3f}")
        pc = {c: round(v['A']['per_class_honest10'][c]['score'] - v[o]['per_class_honest10'].get(c, {}).get('score', 0), 5) for c in v['A']['per_class_honest10'] if 'score' in v['A']['per_class_honest10'][c]}
        print("   per_class Δ", {c: x for c, x in pc.items() if abs(x) > 1e-6})
