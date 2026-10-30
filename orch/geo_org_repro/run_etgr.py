"""Organizer EtGr baseline (02_baseline_etgr_anomalies.ipynb), headless.
MODE=hgb (default): map candidates scored by the same HGB on the 16 hand features instead of the ResNet18 CNN
(CPU-only box; documented deviation). Profiles: organizer HGB verbatim.
Runs holdout-216 + LOSO, then OOF on KOZ3 geo5 folds with the organizer's own inner-LOSO threshold."""
import json, os, sys, time, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import runnb
MODE = os.environ.get("MODE", "hgb")
cs = json.load(open("02_baseline_etgr_anomalies.ipynb"))["cells"]
g = {"__name__": "nb"}
src1 = "".join(cs[1]["source"])
if MODE == "hgb":  # no torch needed
    src1 = "display = print\n" + src1
    src1 = "\n".join(l for l in src1.split("\n") if "IPython" not in l and "timm" not in l and "torch" not in l and "CUBLAS" not in l)
    src1 = src1.replace('DEVICE = "cuda" if torch.cuda.is_available() else "cpu"', 'DEVICE = "cpu"')
exec(compile(src1, "cell1", "exec"), g)
runnb.run("02_baseline_etgr_anomalies.ipynb", [2, 6, 14, 17, 20], g)
src22 = "".join(cs[22]["source"])
if MODE == "hgb":
    head, tail = src22.split("def nms(polys, scores, thr):", 1)
    tail = "def nms(polys, scores, thr):" + tail
    tail = tail.replace('d["crops"] = candidate_crops(d)', 'pass')
    exec('''
def fit_scorer(dd, kind):
    y = np.concatenate([d["cand_pos"] for d in dd])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                         random_state=SEED).fit(np.concatenate([d["cand_X"] for d in dd]), y.astype(int))
    return (lambda d: clf.predict_proba(d["cand_X"])[:, 1] if len(d["cand_X"]) else np.zeros(0)), clf
''', g)
    src22 = tail
exec(compile(src22, "cell22", "exec"), g)
print("[cell 22 done]", flush=True)
runnb.run("02_baseline_etgr_anomalies.ipynb", [25, 28], g)
exec("""
import time
from pathlib import Path
from shapely.geometry import mapping
FOLDS = {'F3': ['208','209','210','215'], 'F4': ['211','212','216'], 'F5': ['213','214','217']}
feats = []
for fk, regs in FOLDS.items():
    for kind in ('карта', 'профиль'):
        test = [d for d in DS if d['kind'] == kind and d['region'] in regs]
        train = [d for d in DS if d['kind'] == kind and d['region'] not in regs]
        if not test or not train: continue
        trr = sorted({d['region'] for d in train}); oof = {}
        for r in trr:
            sc, _ = fit_scorer([d for d in train if d['region'] != r], kind)
            for d in train:
                if d['region'] == r: oof[id(d)] = sc(d)
        thr = max(THRS, key=lambda t: evaluate(train, [oof[id(d)] for d in train], t)[1][2])
        sc, _ = fit_scorer(train, kind)
        for d in test:
            P, C = predict_polys(d, sc(d), thr)
            for p, c in zip(P, C):
                feats.append({'type': 'Feature', 'geometry': transform_geom(d['crs'], 'EPSG:3857', mapping(p)),
                              'properties': {'class_name': 'object_poly', 'region_name': f"{d['region']}_{d['site']}",
                                             'markup_type': 'etgr', 'confidence': round(float(c), 5)}})
        print(fk, kind, 'thr', thr, len(test), flush=True)
Path('../out').mkdir(exist_ok=True)
json.dump({'type': 'FeatureCollection', 'features': feats}, open(f'../out/etgr_oof_geo5_hgb.geojson', 'w'), ensure_ascii=False)
print('etgr feats', len(feats))
""", g)
