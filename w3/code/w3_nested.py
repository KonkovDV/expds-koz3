"""NST nested per-class selection gate (prereg/C5B_NST_PREREG.md).
python w3_nested.py --split-dir D --gt GT --clusters C --vdir variants/NST --out OUT.json"""
import argparse, json, sys, time, gc, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "main"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import stack_scoreboard as S
from expds import metric as M
from expds.naming import site_family
ap = argparse.ArgumentParser(); ap.add_argument("--split-dir", type=Path, required=True); ap.add_argument("--gt", type=Path, required=True)
ap.add_argument("--clusters", type=Path, required=True); ap.add_argument("--vdir", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--boot", type=int, default=1000); a = ap.parse_args()
metric = str(ROOT / "materials" / "compute_metrics.py"); cl = json.loads(a.clusters.read_text(encoding="utf-8"))["clusters"]
gt_all = json.loads(a.gt.read_bytes())["features"]; gt_by = {}
for f in gt_all:
    reg = (f.get("properties") or {}).get("region_name"); gt_by.setdefault(site_family(str(reg)) if reg else "_none", []).append(f)
NAMES = ["m0", "m1", "m2", "m0_O", "m1_O", "m2_O"]; V = {}
for n in NAMES:
    ex = {}
    for line in open(a.vdir / f"{n}.jsonl", encoding="utf-8"): d = json.loads(line); ex.setdefault(d["fam"], []).append(d["feat"])
    V[n] = (ex, set(int(x) for x in open(a.vdir / f"{n}.drop").read().split()))
parts = {n: [] for n in NAMES}; li_regs = set(); t0 = time.time()
for group in cl:
    rows = sorted((r for fam in group for r in S.read_family(a.split_dir, fam)), key=lambda r: r[0])
    for _, f in rows:
        if (f.get("properties") or {}).get("markup_type") == "li": li_regs.add(f["properties"].get("region_name"))
    gf = [f for fam in group for f in gt_by.get(fam, [])]
    if len(group) > 1:
        pos = {id(f): i for i, f in enumerate(gt_all)}; gf.sort(key=lambda f: pos[id(f)])
    for n, (ex, dr) in V.items():
        vf = [f for i, f in rows if i not in dr] + [f for fam in group for f in ex.get(fam, [])]
        parts[n].append(M.build_curves(S.fc(vf), S.fc(gf), metric, scope="global"))
    M._UTM_GEOM_CACHE.clear(); del rows, gf; gc.collect()
    print(f"{'+'.join(group)} done {time.time()-t0:.0f}s rss {S.rss_mb():.0f}MB", flush=True)
new = {ln.strip() for ln in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
optical = {(f.get("properties") or {}).get("region_name") for f in gt_all} - li_regs - {None}
curves = {n: S.merge(p) for n, p in parts.items()}; del parts; gc.collect()
regions = set()
for c in curves["m0"].values(): regions |= set(c.pred_region) | set(c.gt_regions)
folds = M.family_folds(regions, site_family, 10); every = set().union(*folds)
def cls_counts(c, train, held, fix):
    """counts {region:[tp,fp,ngt]} for regions in held, thresholds swept on train (one class)."""
    rnames, rcodes, gnames, gcodes = M._curve_codes(c)
    lut = np.fromiter((r in train for r in rnames), dtype=bool, count=len(rnames)); m = lut[rcodes] if rcodes.size else np.zeros(0, bool)
    n_gt = int(sum(k for r, k in c.gt_regions.items() if r in train))
    res = M._sweep_one(c.conf[m], c.hit[m], gcodes[m], gnames, n_gt, 20, "max")
    taus = {g: gv["threshold"] for g, gv in (res or {}).get("groups", {}).items()}
    if fix: taus.update({g: 0.0 for g in gnames if "etgr" in g})
    per = {r: [0, 0, int(k)] for r, k in c.gt_regions.items() if r in held}
    lut = np.fromiter((r in held for r in rnames), dtype=bool, count=len(rnames)); m = lut[rcodes] if rcodes.size else np.zeros(0, bool)
    if m.any():
        has = np.array([g in taus for g in gnames], bool); tv = np.array([taus.get(g, 0.0) for g in gnames])
        gh = gcodes[m]; tau = np.where(has[gh], tv[gh], np.inf); kept = c.conf[m] >= tau
        rk = rcodes[m][kept]; hk = c.hit[m][kept].astype(bool)
        tp = np.bincount(rk[hk], minlength=len(rnames)); fp = np.bincount(rk[~hk], minlength=len(rnames))
        for code in np.unique(rk).tolist():
            l = per.setdefault(rnames[code], [0, 0, 0]); l[0] += int(tp[code]); l[1] += int(fp[code])
    return per
def f1(per):
    tp = sum(v[0] for v in per.values()); fp = sum(v[1] for v in per.values()); g = sum(v[2] for v in per.values())
    return 2 * tp / max(2 * tp + fp + (g - tp), 1e-9) if g else 0.0
BIG = ("gorodishcha", "arkhitektury", "fortifikatsii", "object_poly")
def honest(prefix, fix):
    base = {cls: {} for cls in curves[prefix[0]]}; nst = {cls: {} for cls in base}; choice = {}
    for k, held in enumerate(folds):
        train = every - held
        for cls in base:
            pb = cls_counts(curves[prefix[0]][cls], train, held, fix); base[cls].update(pb)
            if cls not in BIG: nst[cls].update(pb); continue
            sc = []
            for n in prefix:
                tot = {}
                for j, hj in enumerate(folds):
                    if j == k: continue
                    for r, v in cls_counts(curves[n][cls], train - hj, hj, fix).items(): tot[r] = v
                sc.append(f1(tot))
            b = int(np.argmax(sc)) if max(sc) > sc[0] + 1e-12 else 0; choice.setdefault(cls, []).append(prefix[b])
            nst[cls].update(pb if b == 0 else cls_counts(curves[prefix[b]][cls], train, held, fix))
    return base, nst, choice
out = {"folds": len(folds)}
for tag, pre, fix in (("main", ["m0", "m1", "m2"], False), ("org3", ["m0_O", "m1_O", "m2_O"], True)):
    b, n, ch = honest(pre, fix)
    sb = [M.score_counts(b)[0], M.score_counts(b, new)[0], M.score_counts(b, optical)[0]]
    sn = [M.score_counts(n)[0], M.score_counts(n, new)[0], M.score_counts(n, optical)[0]]
    bt = M.paired_bootstrap_counts(n, b, site_family, a.boot); d = [x - y for x, y in zip(sn, sb)]
    acc = d[0] >= 0.003 and bt.get("p_gt0", 0) >= 0.9 and d[1] >= 0 and d[2] >= 0
    pc = {c: [round(M.score_counts(b)[1][c]["f1"], 4), round(M.score_counts(n)[1][c]["f1"], 4)] for c in BIG if c in n}
    out[tag] = {"base": sb, "nst": sn, "delta": d, "boot": bt, "gate_w3": acc, "choice": ch, "f1_base_nst": pc}
    print(f"{tag}: B {sb[0]:.6f} NST {sn[0]:.6f} Δ {d[0]:+.6f} | 34new {d[1]:+.6f} | optical {d[2]:+.6f} | P {bt.get('p_gt0'):.3f} "
          f"[{bt.get('p5'):+.4f};{bt.get('p95'):+.4f}] {'PASS' if acc else 'fail'} | choice {ch} | f1 {pc}", flush=True)
a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
