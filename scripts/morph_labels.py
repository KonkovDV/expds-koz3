"""Задача 10, только диагностика: из table.pkl берём официальные пары (P, J, S) и сохраняем маленький npz с метками.
Признаки рельефа эти метки не используют."""
import pickle, sys, numpy as np
T = pickle.load(open("/workspace/rerank/work/table.pkl", "rb"))
KUR = {"kurgany_povrezhdennye", "kurgany_tselye"}
pairs = T["pairs"]
kur_true, tse_pos = set(), set()
for (pc, gc), v in pairs.items():
    P = np.asarray(v[0]).ravel()
    if pc in KUR and gc in KUR:
        kur_true.update(P.tolist())
    if gc == "kurgany_tselye":
        tse_pos.update(P.tolist())
np.savez(sys.argv[1], kur_true=np.array(sorted(kur_true), np.int64), tse_pos=np.array(sorted(tse_pos), np.int64),
         keys=np.array([f"{a}|{b}" for a, b in pairs.keys()]))
print(len(kur_true), len(tse_pos), list(pairs.keys())[:6])
