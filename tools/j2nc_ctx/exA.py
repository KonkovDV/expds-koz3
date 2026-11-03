import importlib.util, pickle
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
M = G.D.M if hasattr(G.D, 'M') else None
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
dfx, dreg = G.dfine_feats()
from expds import metric as MM
ex = []
for f in c19 + dfx:
    p = f["properties"]
    ex.append(dict(geometry=f["geometry"], class_name=p["class_name"], confidence=float(p["confidence"]), region_name=p["region_name"],
                   markup_type=p.get("markup_type"), threshold_group=MM.threshold_group(p)))
pickle.dump(dict(ex=ex, dreg=sorted(dreg), SP=sorted(G.SP)), open("extras.pkl", "wb"))
print(len(c19), len(dfx), len(dreg), sorted(G.SP))
