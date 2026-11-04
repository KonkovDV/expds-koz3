"""WS-K1 tie-break variants (see prereg/WS-K1_PREREG.md). usage: make_kt.py SPLIT NPZ OOF OUTDIR"""
import sys, json, csv, math, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
SPLIT, NPZ, OOF, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]); OUT.mkdir(parents=True, exist_ok=True)
D = np.load(NPZ); X = D["X"].astype(np.float32); N = len(X)
yy, xx = np.mgrid[:64, :64] - 31.5; r = np.hypot(yy, xx)
disc3, disc5, ring = r <= 6, r <= 10, (r >= 12) & (r <= 16)
def feats(x):
    out = []
    for c in (0, 1):
        a = x[c]; rm = np.median(a[ring]); p10 = np.percentile(a[ring], 10); pk = a[disc3].max()
        out += [pk - rm, a[disc3].mean() - rm, pk - p10]
    a = x[0]; rm = np.median(a[ring]); pk = a[disc3].max(); thr = rm + 0.5 * max(pk - rm, 1e-3)
    lab, _ = ndi.label(a > thr); iy, ix = np.unravel_index(np.argmax(np.where(disc3, a, -1e9)), a.shape); m = lab == lab[iy, ix] if lab[iy, ix] else np.zeros_like(a, bool)
    A = m.sum(); P = (m ^ ndi.binary_erosion(m)).sum()
    out += [4 * math.pi * A / max(P, 1) ** 2 if A else 0.0, (a[disc5] > thr).mean(), x[2][disc5].mean()]
    return out
F = np.array([feats(x) for x in X])
O = list(csv.DictReader(open(OOF)))
assert len(O) == N, (len(O), N)
fold = np.array([int(o["fold"]) for o in O]); y = D["y"].astype(int); pp = np.array([float(o["p_povr"]) for o in O])
for o, s in zip(O, D["stack_idx"]): assert int(o["stack_idx"]) == int(s)
pg = np.zeros(N)
for k in sorted(set(fold)):
    tr = (fold != k) & (y >= 0); te = fold == k
    sc = StandardScaler().fit(F[tr]); m = LogisticRegression(C=1, max_iter=2000).fit(sc.transform(F[tr]), y[tr]); pg[te] = m.predict_proba(sc.transform(F[te]))[:, 1]
cand = (D["kind"] == "cand") & (y >= 0); lab = y >= 0
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
pb = 1 / (1 + np.exp(-(lg(pg) + lg(pp)) / 2))
rep = {"auc_geo_all": roc_auc_score(y[lab], pg[lab]), "auc_geo_cand": roc_auc_score(y[cand], pg[cand]), "auc_blend_all": roc_auc_score(y[lab], pb[lab]), "auc_blend_cand": roc_auc_score(y[cand], pb[cand]), "auc_cnn_cand": roc_auc_score(y[cand], pp[cand])}
print(json.dumps(rep))
Z = {}
VARS = (("KST", pp, 1e-3),) if len(sys.argv) > 5 and sys.argv[5] == "KST" else (("KT1", pg, 1e-3), ("KT2", pb, 1e-3), ("KT3", pb, 1e-2))
for name, p, eps in VARS:
    Z[name] = (eps, {int(s): float(np.clip(lg(q), -3, 3) / 3) for s, q, kd in zip(D["stack_idx"], p, D["kind"]) if kd == "cand" and s >= 0})
cov = set(next(iter(Z.values()))[1])
fh = {n: open(OUT / f"{n}.jsonl", "w", encoding="utf-8") for n in Z}; drop = []
for pth in sorted(SPLIT.glob("*.tsv")):
    for line in open(pth, encoding="utf-8"):
        i, _, rest = line.partition("\t"); i = int(i)
        if i not in cov: continue
        f = json.loads(rest); pr = f["properties"]; sgn = 1 if pr["class_name"] == "kurgany_povrezhdennye" else -1 if pr["class_name"] == "kurgany_tselye" else 0
        if not sgn: continue
        drop.append(i)
        for n, (eps, z) in Z.items():
            g = json.loads(rest); g["properties"]["confidence"] = round(min(1.0, max(0.0, pr["confidence"] + sgn * eps * z[i])), 7)
            fh[n].write(json.dumps({"fam": pth.stem, "feat": g}, ensure_ascii=False) + "\n")
for h in fh.values(): h.close()
for n in Z: open(OUT / f"{n}.drop", "w").write("\n".join(map(str, drop)) + "\n")
rep["replaced"] = len(drop); json.dump(rep, open(OUT / "kt_report.json", "w"), indent=1); print("replaced", len(drop))
