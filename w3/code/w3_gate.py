"""Paired honest-10 gate on a split pre-threshold stack (bot-8 stack_scoreboard machinery).
python w3_gate.py --split-dir D --gt GT --clusters base_arch.json --variant NAME=extras.jsonl[@drop.txt] ... --out OUT.json
extras.jsonl lines {"fam","feat"}; drop.txt = base stack indices to remove (replace variants)."""
import argparse, json, sys, time, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "main"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import stack_scoreboard as S
from expds import metric as M
from expds.naming import site_family

ap = argparse.ArgumentParser()
ap.add_argument("--split-dir", type=Path, required=True); ap.add_argument("--gt", type=Path, required=True)
ap.add_argument("--clusters", type=Path, required=True); ap.add_argument("--variant", action="append", default=[])
ap.add_argument("--honest", type=int, default=10); ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
metric = str(ROOT / "materials" / "compute_metrics.py")
cl = json.loads(a.clusters.read_text(encoding="utf-8"))["clusters"]
gt_all = json.loads(a.gt.read_bytes())["features"]
gt_by = {}
for f in gt_all:
    reg = (f.get("properties") or {}).get("region_name")
    gt_by.setdefault(site_family(str(reg)) if reg else "_none", []).append(f)
V = {}
for spec in a.variant:
    name, rest = spec.split("=", 1)
    path, _, drop = rest.partition("@")
    ex = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line); ex.setdefault(d["fam"], []).append(d["feat"])
    dr = set(int(x) for x in open(drop).read().split()) if drop else set()
    V[name] = (ex, dr)
    print(name, "extras", sum(map(len, ex.values())), "drop", len(dr), flush=True)
parts = {"A0": []} | {n: [] for n in V}
li_regs = set(); t0 = time.time()
for group in cl:
    rows = sorted((r for fam in group for r in S.read_family(a.split_dir, fam)), key=lambda r: r[0])
    pf = [f for _, f in rows]
    for f in pf:
        pr = f.get("properties") or {}
        if pr.get("markup_type") == "li": li_regs.add(pr.get("region_name"))
    gf = [f for fam in group for f in gt_by.get(fam, [])]
    if len(group) > 1:
        pos = {id(f): i for i, f in enumerate(gt_all)}; gf.sort(key=lambda f: pos[id(f)])
    cb = M.build_curves(S.fc(pf), S.fc(gf), metric, scope="global")
    parts["A0"].append(cb)
    for n, (ex, dr) in V.items():
        add = [f for fam in group for f in ex.get(fam, [])]
        rm = dr and any(i in dr for i, _ in rows)
        if not add and not rm:
            parts[n].append(cb); continue
        vf = [f for i, f in rows if i not in dr] + add
        parts[n].append(M.build_curves(S.fc(vf), S.fc(gf), metric, scope="global"))
    M._UTM_GEOM_CACHE.clear(); del rows, pf, gf; gc.collect()
    print(f"{'+'.join(group)} done {time.time()-t0:.0f}s rss {S.rss_mb():.0f}MB", flush=True)
new = {ln.strip() for ln in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
optical = {(f.get("properties") or {}).get("region_name") for f in gt_all} - li_regs - {None}
curves = {n: S.merge(p) for n, p in parts.items()}
regions = set()
for n in curves:
    for c in curves[n].values(): regions |= set(c.pred_region) | set(c.gt_regions)
folds = M.family_folds(regions, site_family, a.honest)
H = {}
for n, c in curves.items():
    counts, info = M.honest_counts(c, folds)
    H[n] = (counts, [M.score_counts(counts)[0], M.score_counts(counts, new)[0], M.score_counts(counts, optical)[0]], info)
out = {"base_honest": H["A0"][1], "folds": len(folds), "variants": {}}
print("A0 honest all/new/optical", [round(x, 6) for x in H["A0"][1]], flush=True)
for n in V:
    b = M.paired_bootstrap_counts(H[n][0], H["A0"][0], site_family, a.boot)
    s = H[n][1]; s0 = H["A0"][1]
    per = {}
    for cls in H[n][0]:
        def tot(cs): 
            t = [0, 0, 0]
            for v in cs.get(cls, {}).values(): t = [t[0]+v[0], t[1]+v[1], t[2]+v[2]]
            return t
        per[cls] = {"A0": tot(H["A0"][0]), "V": tot(H[n][0])}
    acc = s[0] - s0[0] >= 0.003 and b.get("p_gt0", 0) >= 0.9 and s[1] >= s0[1] and s[2] >= s0[2]
    out["variants"][n] = {"honest": s, "delta": [x - y for x, y in zip(s, s0)], "boot": b, "gate_w3": acc, "per_class_tp_fp_ngt": per}
    print(f"{n}: honest {s[0]:.6f} Δ {s[0]-s0[0]:+.6f} | 34new Δ {s[1]-s0[1]:+.6f} | optical Δ {s[2]-s0[2]:+.6f} | "
          f"P(A>B) {b.get('p_gt0'):.3f} 90% [{b.get('p5'):+.4f};{b.get('p95'):+.4f}] gate {'PASS' if acc else 'fail'}", flush=True)
a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
