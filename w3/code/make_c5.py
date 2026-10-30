"""CI2-5 CPU variant (prereg/CI2-5_PREREG.md). usage: make_c5.py SPLIT OUTDIR"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
import sys, json, math, collections, numpy as np
from pathlib import Path
import shapely.geometry as sg
from shapely.strtree import STRtree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/workspace/koz3/orch/bot-W3/main"); from expds.naming import site_family
K = "/workspace/koz3"; SPLIT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
CL = ("gorodishcha", "fortifikatsii", "arkhitektury", "object_poly")
def load(d): return np.load(f"{K}/out/{d}/emb.npy", mmap_mode="r"), np.load(f"{K}/out/{d}/cid.npy"), np.load(f"{K}/out/{d}/ok.npy")
Eg, Cg, Og = load("w5-dino-emb-gpu-1030"); Ec, Cc, Oc = load("w5-dino-emb")
pos_g = {int(c): i for i, c in enumerate(Cg)}; pos_c = {int(c): i for i, c in enumerate(Cc)}
C = {}
for p in sorted(SPLIT.glob("*.tsv")):
    for line in open(p, encoding="utf-8"):
        if not any(f'"{c}"' in line for c in CL): continue
        i, _, rest = line.partition("\t"); i = int(i); f = json.loads(rest); pr = f["properties"]
        if pr.get("class_name") in CL and pr.get("markup_type") != "li": C[i] = (p.stem, f, rest)
ids = sorted(C); N = len(ids); E = np.zeros((N, 768), np.float32); okv = np.zeros(N, bool)
for k, i in enumerate(ids):
    if i in pos_g and Og[pos_g[i]]: E[k] = Eg[pos_g[i]]; okv[k] = True
    elif i in pos_c and Oc[pos_c[i]]: E[k] = Ec[pos_c[i]]; okv[k] = True
E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
P_ = lambda i: C[i][1]["properties"]
cls = np.array([P_(i)["class_name"] for i in ids]); reg = np.array([P_(i)["region_name"] for i in ids]); mt = np.array([P_(i)["markup_type"] for i in ids])
fam = np.array([site_family(r) for r in reg]); conf = np.array([float(P_(i)["confidence"]) for i in ids])
geo = [sg.shape(C[i][1]["geometry"]) for i in ids]
area = np.zeros(N); asp = np.zeros(N)
for k, g in enumerate(geo):
    sc = 1 / math.cosh(g.centroid.y / 6378137.0); b = g.bounds; w, h = (b[2]-b[0])*sc, (b[3]-b[1])*sc
    area[k] = max(g.area * sc * sc, 1.0); asp[k] = max(w, h) / max(min(w, h), 0.5)
gt = json.load(open(f"{K}/w5data/gt_train_v2.geojson"))["features"]; G = collections.defaultdict(list)
for f in gt:
    pr = f["properties"]
    if pr.get("class_name") in CL and f.get("geometry"): G[(pr["region_name"], pr["class_name"])].append(sg.shape(f["geometry"]))
T = {k: STRtree(v) for k, v in G.items()}; y = np.zeros(N, int)
for k in range(N):
    t = T.get((reg[k], cls[k]))
    if t is not None: c = geo[k].centroid; y[k] = int(any(G[(reg[k], cls[k])][q].contains(c) for q in t.query(c)))
print("N", N, "emb ok", round(okv.mean(), 4), "pos", dict(collections.Counter(zip(cls, mt, y))), flush=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
B = np.c_[lg(conf), np.log(area), np.log(asp)]; p = conf.copy(); done = np.zeros(N, bool)
for c in CL:
    m = (cls == c) & okv
    for fm in np.unique(fam[m]):
        te = m & (fam == fm); tr = m & (fam != fm)
        if y[tr].sum() < 5 or (1 - y[tr]).sum() < 5: continue
        pca = PCA(32, random_state=0).fit(E[tr]); F = lambda s: np.c_[B[s], pca.transform(E[s])]
        sc_ = StandardScaler().fit(F(tr)); lr = LogisticRegression(C=0.5, max_iter=3000).fit(sc_.transform(F(tr)), y[tr])
        p[te] = lr.predict_proba(sc_.transform(F(te)))[:, 1]; done[te] = True
V = {"C5A": p, "C5B": np.where(area >= 2000, p, conf), "C5C": 0.5 * conf + 0.5 * p}
rep = {"N": N, "pos": int(y.sum()), "done": int(done.sum()), "big2000": int((area >= 2000).sum())}
for c in CL:
    for lay in [None] + sorted(set(mt[cls == c])):
        m = (cls == c) & done & ((mt == lay) if lay else True)
        if len(set(y[m])) == 2:
            for nm, v in (("conf", conf), ("C5A", p)): rep[f"auc_{c[:4]}_{lay or 'all'}_{nm}"] = round(roc_auc_score(y[m], v[m]), 4)
print(json.dumps(rep), flush=True)
sel = [k for k in range(N) if done[k]]
for nm, v in V.items():
    with open(OUT / f"{nm}.jsonl", "w", encoding="utf-8") as h:
        for k in sel:
            fm_, f, rest = C[ids[k]]; g = json.loads(rest); g["properties"]["confidence"] = round(float(v[k]), 6)
            h.write(json.dumps({"fam": fm_, "feat": g}, ensure_ascii=False) + "\n")
    open(OUT / f"{nm}.drop", "w").write("\n".join(str(ids[k]) for k in sel) + "\n")
json.dump(rep, open(OUT / "c5_report.json", "w"), indent=1)
