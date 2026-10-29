"""Вторичная (перевзвешенная) приёмка bot-10, задача 10.
Вес участка: крупная SpOr-сцена (max SpOr tif >= 20e6 байт по /workspace/data/s3_listing.jsonl,
work10/rw/big_scenes.json) -> w_big = (17/30)/(n_big/N); прочие -> w_small = (13/30)/(n_small/N),
N = участки табло. Доли 17/30 и 20/132 — research/OSINT_2026-10-29.md:54 (DATA_INSIGHTS §Drift EN).
Пороги НЕ перевзвешиваются (sweep_groups как в сабмите, honest_counts); веса — только в подсчёте балла.
Бутстреп по семьям: вес = кратность семьи × вес состава. Лидарная часть сдвига (LAS) не моделируется.
    cd bot-10; python work10/rw/rw_paired.py --base A=work/curves --b X=dir ... --pols keep --out f.json
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves, regions_of, random_family_folds, fam_index, stats
from expds import metric as M
from expds.naming import site_family
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--b", action="append", required=True)
ap.add_argument("--pols", default="drop,keep"); ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
bn, bd = a.base.split("=", 1)
V = {bn: load_curves(Path(bd))}
for s in a.b:
    n, d = s.split("=", 1); V[n] = load_curves(Path(d))
big = set(json.load(open("work10/rw/big_scenes.json"))["big_ge20MB"])
regions = set().union(*(regions_of(c) for c in V.values()))
isbig = {r: (r in big) for r in regions}
nb = sum(isbig.values()); N = len(regions)
wb, ws = (17 / 30) / (nb / N), (13 / 30) / ((N - nb) / N)
W = {r: (wb if isbig[r] else ws) for r in regions}
lay = {"k10": M.family_folds(regions, site_family, 10), "lofo": M.family_folds(regions, site_family, None)}
for s in range(5):
    lay[f"seed{s}"] = random_family_folds(regions, 10, s)
HEAVY = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha")
rep = {"N": N, "n_big": nb, "big_in_board": sorted(r for r in regions if isbig[r]), "w_big": wb, "w_small": ws, "layouts": {}}
C = {}
for ln, folds in lay.items():
    row = {}
    for nm, cv in V.items():
        for pol in a.pols.split(","):
            c, _ = M.honest_counts(cv, folds, unseen=pol)
            row[f"{nm}_{pol}"] = {"unw": M.score_counts(c)[0], "rw": M.score_counts(c, None, W)[0]}
            if ln == "k10": C[f"{nm}_{pol}"] = c
    rep["layouts"][ln] = row
fams = fam_index(list(C.values())); names = sorted(fams)
rng = np.random.default_rng(0); B = {k: [] for k in C}
for _ in range(a.boot):
    pick = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
    w = {r: float(k) * W.get(r, ws) for f, k in zip(names, pick) if k for r in fams[f]}
    for k, c in C.items(): B[k].append(M.score_counts(c, None, w)[0])
rep["summary"] = {}
for nm in V:
    if nm == bn: continue
    for pol in a.pols.split(","):
        k, kb = f"{nm}_{pol}", f"{bn}_{pol}"
        d = {ln: rep["layouts"][ln][k]["rw"] - rep["layouts"][ln][kb]["rw"] for ln in lay}
        _, pa = M.score_counts(C[k], None, W); _, pb = M.score_counts(C[kb], None, W)
        den = sum(M.CLASS_WEIGHTS.get(c, 0) for c in pa if pa[c]["n_gt"] > 0)
        pc = {c: M.CLASS_WEIGHTS.get(c, 0) * (pa[c]["f1"] - pb[c]["f1"]) / den for c in pa if pa[c]["n_gt"] > 0}
        st = stats(np.asarray(B[k]) - np.asarray(B[kb]))
        seeds = [d[f"seed{s}"] for s in range(5)]
        rule = {"P1": d["k10"] >= 0.002, "P2": st["p_gt0"] >= 0.85, "P3": d["lofo"] > 0 and sum(x > 0 for x in seeds) >= 4,
                "P4": all(pc.get(c, 0) >= -0.0005 for c in HEAVY)}
        rep["summary"][k] = {"d_rw_k10": d["k10"], "d_unw_k10": rep["layouts"]["k10"][k]["unw"] - rep["layouts"]["k10"][kb]["unw"],
                             "P": st["p_gt0"], "ci": [st["p2_5"], st["p97_5"]], "lofo": d["lofo"], "seeds": seeds,
                             "per_class_w": pc, "rule": rule, "ok": all(rule.values())}
        print(k, f"rw={d['k10']:+.6f} unw={rep['summary'][k]['d_unw_k10']:+.6f} P={st['p_gt0']:.3f} lofo={d['lofo']:+.6f} seeds+={sum(x>0 for x in seeds)}/5 heavy={[round(pc.get(c,0),6) for c in HEAVY]} ok={all(rule.values())}", flush=True)
print("N", N, "n_big", nb, "w_big", round(wb, 4), "w_small", round(ws, 4))
Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
