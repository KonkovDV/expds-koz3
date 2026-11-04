"""OOF gap breakdown (diagnostic): stack B=A+M_rp+ORG3(N0) vs own-candidate oracle; honest-10 thresholds.
python gap.py --split-dir D --gt GT --clusters C --variant variants/NST/m0_O --out runs/gap_breakdown --dump runs/gap_cands.pkl"""
import argparse, json, sys, time, gc, pickle, numpy as np, shapely
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1] / "main"; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import stack_scoreboard as S
from expds import metric as M
from expds.naming import site_family
ap = argparse.ArgumentParser(); ap.add_argument("--split-dir", type=Path, required=True); ap.add_argument("--gt", type=Path, required=True)
ap.add_argument("--clusters", type=Path, required=True); ap.add_argument("--variant", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dump", required=True); a = ap.parse_args()
metric = str(ROOT / "materials" / "compute_metrics.py"); mod = M.load_official(metric); calc = mod.F1Calculator()
cl = json.loads(a.clusters.read_text(encoding="utf-8"))["clusters"]
gt_all = json.loads(a.gt.read_bytes())["features"]; gt_by = {}
for f in gt_all:
    reg = (f.get("properties") or {}).get("region_name"); gt_by.setdefault(site_family(str(reg)) if reg else "_none", []).append(f)
ex = {}
for line in open(a.variant + ".jsonl", encoding="utf-8"): d = json.loads(line); ex.setdefault(d["fam"], []).append(d["feat"])
dr = set(int(x) for x in open(a.variant + ".drop").read().split())
# per class accumulators: candidates and gts with relations
C = {}; li_regs = set(); t0 = time.time(); off = Counter(); goff = Counter()
def acc(cls):
    return C.setdefault(cls, {k: [] for k in ("conf", "reg", "grp", "hit", "poly", "okg", "nearg", "asg", "g_reg", "g_ok", "g_near", "g_by", "g_x")})
for group in cl:
    rows = sorted((r for fam in group for r in S.read_family(a.split_dir, fam)), key=lambda r: r[0])
    for _, f in rows:
        if (f.get("properties") or {}).get("markup_type") == "li": li_regs.add(f["properties"].get("region_name"))
    gf = [f for fam in group for f in gt_by.get(fam, [])]
    if len(group) > 1:
        pos = {id(f): i for i, f in enumerate(gt_all)}; gf.sort(key=lambda f: pos[id(f)])
    vf = [f for i, f in rows if i not in dr] + [f for fam in group for f in ex.get(fam, [])]
    preds, gts = M._collect(mod, calc, S.fc(vf), False), M._collect(mod, calc, S.fc(gf), True)
    ST = {}
    for cls in calc.target_classes:
        p, g = preds.get(cls, []), gts.get(cls, []); A = acc(cls)
        order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True); n = len(p); po = off[cls]; go = goff[cls]
        hits = np.zeros(n, bool); asg = -np.ones(n, np.int64); okg = [[] for _ in range(n)]; nearg = [[] for _ in range(n)]
        g_ok = [[] for _ in g]; g_near = [[] for _ in g]; g_by = -np.ones(len(g), np.int64)
        pg_ord = [p[i][0] for i in order]; arr = np.empty(n, dtype=object); arr[:] = pg_ord
        poly = shapely.get_type_id(arr) == 3 if n else np.zeros(0, bool)
        if p and g:
            gg = [x[0] for x in g]; tree = M._gt_tree(calc, gg); qp, qg = tree.query(pg_ord)
            keep = poly[qp]; qp, qg = qp[keep], qg[keep]; o = np.lexsort((qg, qp)); qp, qg = qp[o].tolist(), qg[o].tolist()
            oks, ss = M._pair_quality(mod, calc, pg_ord, gg, qp, qg); matched = set(); k = 0
            for r_, j_, ok_, s_ in zip(qp, qg, oks, ss):
                if ok_: okg[r_].append(go + j_); g_ok[j_].append(po + r_)
                elif s_ > 0 or shapely.get_type_id(gg[j_]) == 0: nearg[r_].append(go + j_); g_near[j_].append(po + r_)
            while k < len(qp):
                r_int = qp[k]; best, bj = float("-inf"), -1
                while k < len(qp) and qp[k] == r_int:
                    if oks[k] and qg[k] not in matched and ss[k] > best + 1e-9: best, bj = ss[k], qg[k]
                    k += 1
                if bj >= 0: matched.add(bj); hits[r_int] = True; asg[r_int] = go + bj; g_by[bj] = po + r_int
        A["conf"] += [p[i][1] for i in order]; A["reg"] += [p[i][2] for i in order]; A["grp"] += [p[i][3] for i in order]
        A["hit"] += hits.tolist(); A["poly"] += poly.tolist(); A["okg"] += okg; A["nearg"] += nearg; A["asg"] += asg.tolist()
        A["g_reg"] += [x[2] for x in g]; A["g_ok"] += g_ok; A["g_near"] += g_near; A["g_by"] += [int(v) if v < 0 else int(v) for v in g_by]
        ST[cls] = (po, order, len(g)); off[cls] += n; goff[cls] += len(g)
    # wrong class: GT of cls with an ok pair (cov>=0.5 & interior point) to a candidate of another class
    for cls in calc.target_classes:
        g = gts.get(cls, []); gx = [[] for _ in g]
        if g:
            gg = [x[0] for x in g]; tree = M._gt_tree(calc, gg)
            for c2 in calc.target_classes:
                p2 = preds.get(c2, [])
                if c2 == cls or not p2: continue
                po2, ord2, _ = ST[c2]; pg2 = [p2[i][0] for i in ord2]; a2 = np.empty(len(pg2), dtype=object); a2[:] = pg2
                pl2 = shapely.get_type_id(a2) == 3; qp, qg = tree.query(pg2); k2 = pl2[qp]; qp, qg = qp[k2].tolist(), qg[k2].tolist()
                if not qp: continue
                oks, _ = M._pair_quality(mod, calc, pg2, gg, qp, qg)
                for r_, j_, ok_ in zip(qp, qg, oks):
                    if ok_: gx[j_].append((c2, po2 + r_))
        acc(cls)["g_x"] += gx
    M._UTM_GEOM_CACHE.clear(); M._GT_TREES.clear(); del rows, gf, vf, preds, gts; gc.collect()
    print(f"{'+'.join(group)[:40]} {time.time()-t0:.0f}s rss {S.rss_mb():.0f}MB", flush=True)
for cls, A in C.items():
    for k in ("conf", "hit", "poly", "asg", "g_by"): A[k] = np.asarray(A[k])
new = {ln.strip() for ln in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
pickle.dump({"C": C, "li": li_regs, "new": new}, open(a.dump, "wb"))
print("dumped", flush=True)
