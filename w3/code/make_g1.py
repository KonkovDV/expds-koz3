"""WS-G1 variants (prereg/WS-G1_PREREG.md). usage: make_g1.py SPLIT OUTDIR"""
import sys, json, csv, glob, os, math, collections, numpy as np
from pathlib import Path
import shapely.geometry as sg
from shapely.strtree import STRtree
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/workspace/koz3/orch/bot-W3/main"); from expds.naming import site_family
K = "/workspace/koz3"; SPLIT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
CL = ("gorodishcha", "fortifikatsii")
def load(d):
    return np.load(f"{K}/out/{d}/emb.npy", mmap_mode="r"), np.load(f"{K}/out/{d}/cid.npy"), np.load(f"{K}/out/{d}/ok.npy")
Eg, Cg, Og = load("w5-dino-emb-gpu-1030"); Ec, Cc, Oc = load("w5-dino-emb")
pos_g = {int(c): i for i, c in enumerate(Cg)}; pos_c = {int(c): i for i, c in enumerate(Cc)}
C = {}  # stack idx -> row
for p in sorted(SPLIT.glob("*.tsv")):
    for line in open(p, encoding="utf-8"):
        if '"gorodishcha"' not in line and '"fortifikatsii"' not in line: continue
        i, _, rest = line.partition("\t"); i = int(i)
        if i not in pos_g: continue
        f = json.loads(rest); pr = f["properties"]
        if pr.get("class_name") not in CL: continue
        C[i] = (p.stem, f, rest)
ids = sorted(C); N = len(ids); print("pool gor/fort", N, flush=True)
E = np.zeros((N, 768), np.float32); okv = np.zeros(N, bool)
for k, i in enumerate(ids):
    j = pos_g[i]
    if Og[j]: E[k] = Eg[j]; okv[k] = True
    elif i in pos_c and Oc[pos_c[i]]: E[k] = Ec[pos_c[i]]; okv[k] = True
E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
cls = np.array([C[i][1]["properties"]["class_name"] for i in ids]); reg = np.array([C[i][1]["properties"]["region_name"] for i in ids])
fam = np.array([site_family(r) for r in reg]); conf = np.array([float(C[i][1]["properties"]["confidence"]) for i in ids])
geo = [sg.shape(C[i][1]["geometry"]) for i in ids]
# labels
gt = json.load(open(f"{K}/w5data/gt_train_v2.geojson"))["features"]; G = collections.defaultdict(list)
for f in gt:
    pr = f["properties"]
    if pr.get("class_name") in CL and f.get("geometry"): G[(pr["region_name"], pr["class_name"])].append(sg.shape(f["geometry"]))
T = {k: STRtree(v) for k, v in G.items()}
y = np.zeros(N, int)
for k in range(N):
    t = T.get((reg[k], cls[k]))
    if t is None: continue
    c = geo[k].centroid; y[k] = int(any(G[(reg[k], cls[k])][q].contains(c) for q in t.query(c)))
print("pos", dict(collections.Counter(zip(cls, y))), "emb ok", okv.mean(), flush=True)
# kNN LOFO
sk = np.zeros(N)
for c in CL:
    m = (cls == c) & okv
    for fm in np.unique(fam[m]):
        te = np.flatnonzero(m & (fam == fm)); P = np.flatnonzero(m & (fam != fm) & (y == 1)); Ng = np.flatnonzero(m & (fam != fm) & (y == 0))
        if not len(P) or not len(Ng): continue
        def top5(R):
            out = np.zeros(len(te))
            for s in range(0, len(te), 2048):
                S = E[te[s:s+2048]] @ E[R].T; kk = min(5, S.shape[1]); out[s:s+2048] = np.sort(np.partition(S, -kk, 1)[:, -kk:], 1).mean(1)
            return out
        sk[te] = top5(P) - top5(Ng)
# rampart profile (li)
RF = {os.path.basename(q)[:-4]: q for q in glob.glob(f"{K}/w2/relief/cnx_relief/*.npy")}
ring = np.zeros((N, 4)); RC = {}
for k in range(N):
    pr = C[ids[k]][1]["properties"]
    if pr.get("markup_type") != "li" or reg[k] not in RF: continue
    if reg[k] not in RC:
        j = json.load(open(RF[reg[k]][:-4] + ".json")); RC = {reg[k]: (np.load(RF[reg[k]], mmap_mode="r"), j, Transformer.from_crs("EPSG:3857", j["crs"], always_xy=True))}
    a, j, tr = RC[reg[k]]; c = geo[k].centroid; X, Y = tr.transform(c.x, c.y)
    b = geo[k].bounds; sc = 1 / math.cosh(c.y / 6378137.0); R = max(5.0, max(b[2] - b[0], b[3] - b[1]) * sc / 2)
    st = j["step"]; half = 1.9 * R; n = int(2 * half / st); c0 = int((X - half - j["left"]) / st); r0 = int((j["top"] - (Y + half)) / st)
    H, W = a.shape[1:]
    if r0 < 0 or c0 < 0 or r0 + n > H or c0 + n > W or n < 8: continue
    L = np.asarray(a[j["tags"].index("lrm_s"), r0:r0+n, c0:c0+n], np.float32)
    yy, xx = (np.mgrid[:n, :n] - (n - 1) / 2) * st; rr = np.hypot(yy, xx); az = ((np.arctan2(yy, xx) + math.pi) / (2 * math.pi) * 16).astype(int) % 16
    rg = (rr >= .6*R) & (rr <= 1.1*R); ce = rr < .4*R; bg = (rr >= 1.3*R) & (rr <= 1.8*R)
    if not ce.any() or not bg.any(): continue
    bgm = L[bg].mean(); clo = np.mean([L[rg & (az == s)].mean() > L[bg & (az == s)].mean() for s in range(16) if (rg & (az == s)).any() and (bg & (az == s)).any()])
    ring[k] = [L[rg].mean() - L[ce].mean(), L[rg].mean() - bgm, clo, 1.0]
lg = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
F_A = np.c_[lg(conf), sk]; F_B = np.c_[F_A, ring]
pA = conf.copy(); pB = conf.copy(); pC = conf.copy()
for c in CL:
    m = (cls == c) & okv
    for fm in np.unique(fam[m]):
        te = m & (fam == fm); tr_ = m & (fam != fm)
        if len(set(y[tr_])) < 2: continue
        for F, out in ((F_A, pA), (F_B, pB)):
            s = StandardScaler().fit(F[tr_]); out[te] = LogisticRegression(C=1, max_iter=2000).fit(s.transform(F[tr_]), y[tr_]).predict_proba(s.transform(F[te]))[:, 1]
        mu, sd = sk[tr_].mean(), sk[tr_].std() + 1e-9; pC[te] = np.minimum(1, conf[te] * 2 / (1 + np.exp(-(sk[te] - mu) / sd)))
rep = {"N": N, "pos": int(y.sum()), "ring_cov": float(ring[:, 3].mean())}
for c in CL:
    m = (cls == c) & okv
    if len(set(y[m])) == 2:
        for nm, v in (("conf", conf), ("s_knn", sk), ("G1A", pA), ("G1B", pB), ("G1C", pC), ("ring_contrast", ring[:, 1])): rep[f"auc_{c[:3]}_{nm}"] = round(roc_auc_score(y[m], v[m]), 4)
print(json.dumps(rep), flush=True)
rep_ids = [k for k in range(N) if okv[k]]
for nm, v in (("G1A", pA), ("G1B", pB), ("G1C", pC)):
    with open(OUT / f"{nm}.jsonl", "w", encoding="utf-8") as h:
        for k in rep_ids:
            fm_, f, rest = C[ids[k]]; g = json.loads(rest); g["properties"]["confidence"] = round(float(v[k]), 6)
            h.write(json.dumps({"fam": fm_, "feat": g}, ensure_ascii=False) + "\n")
    open(OUT / f"{nm}.drop", "w").write("\n".join(str(ids[k]) for k in rep_ids) + "\n")
json.dump(rep, open(OUT / "g1_report.json", "w"), indent=1)
