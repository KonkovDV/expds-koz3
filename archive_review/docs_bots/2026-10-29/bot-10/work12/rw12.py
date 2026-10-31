"""bot-10 work12: вторичная перевзвешенная проверка ORGATE по счётчикам (как work10/rw/rw_paired.py:
w_big=(17/30)/(n_big/N), w_small=(13/30)/(n_small/N), big = work10/rw/big_scenes.json, OSINT:54).
Пороги не перевзвешиваются — счётчики из gate_eval12.py (honest, unseen=drop). Плюс П4: тяжёлые классы побитно равны базе.
    cd bot-10; python work12/rw12.py --var G_noLAS"""
import sys, json, pickle, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, "repo")
from expds import metric as M
from expds.naming import site_family
ap = argparse.ArgumentParser(); ap.add_argument("--var", default="G_noLAS"); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--gdir", default="work12/res")
a = ap.parse_args()
R12 = Path("work12/res"); R4 = Path("work4/rerank")
big = set(json.load(open("work10/rw/big_scenes.json"))["big_ge20MB"])
HEAVY = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha")
L = lambda p: pickle.load(open(p, "rb"))
out = {}
for SS in ("", "_site"):
    for lay in ["k10", "lofo"] + [f"seed{s}" for s in range(5)]:
        pb, pr, pg = R12 / f"counts_{lay}{SS}_base.pkl", R4 / f"counts_{lay}{SS}_full_orig.pkl", Path(a.gdir) / f"counts_{lay}{SS}_{a.var}.pkl"
        if not (pb.exists() and pg.exists()): continue
        B, R, G = L(pb), L(pr), L(pg)
        regs = sorted({r for c in (B, G) for by in c.values() for r in by})
        N = len(regs); nb = sum(r in big for r in regs)
        W = {r: ((17 / 30) / (nb / N) if r in big else (13 / 30) / ((N - nb) / N)) for r in regs}
        heavy_eq = all(B[c] == G[c] for c in HEAVY) and all(R[c] == G[c] for c in HEAVY)
        other_eq = all(B[c] == G[c] for c in B if c != "arkhitektury")
        sc = lambda c, w=None: M.score_counts(c, None, w)[0]
        row = {"N": N, "n_big": nb, "heavy_eq_A_R_G": heavy_eq, "non_arch_eq_A": other_eq}
        for nm, c in (("A", B), ("R", R), ("G", G)):
            row[nm] = {"unw": sc(c), "rw": sc(c, W)}
        row["dA_unw"] = row["G"]["unw"] - row["A"]["unw"]; row["dR_unw"] = row["G"]["unw"] - row["R"]["unw"]
        row["dA_rw"] = row["G"]["rw"] - row["A"]["rw"]; row["dR_rw"] = row["G"]["rw"] - row["R"]["rw"]
        if lay == "k10":
            fams = {}
            for r in regs: fams.setdefault(site_family(r), []).append(r)
            names = sorted(fams); rng = np.random.default_rng(0); dA = []; dR = []; uA = []; uR = []
            for _ in range(a.boot):
                pick = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
                w = {r: float(k) * W[r] for f, k in zip(names, pick) if k for r in fams[f]}
                g = sc(G, w); dA.append(g - sc(B, w)); dR.append(g - sc(R, w))
                wu = {r: float(k) for f, k in zip(names, pick) if k for r in fams[f]}
                g = sc(G, wu); uA.append(g - sc(B, wu)); uR.append(g - sc(R, wu))
            for k, v in (("rw_A", dA), ("rw_R", dR), ("unw_A", uA), ("unw_R", uR)):
                v = np.asarray(v); row[f"boot_{k}"] = {"P": float((v > 0).mean()), "sd": float(v.std()), "ci": [float(np.quantile(v, .025)), float(np.quantile(v, .975))]}
        out[f"{lay}{SS}"] = row
        print(f"{lay}{SS:5s} N={N} big={nb} heavyEq={heavy_eq} nonArchEqA={other_eq} dA unw={row['dA_unw']:+.6f} rw={row['dA_rw']:+.6f} | dR unw={row['dR_unw']:+.6f} rw={row['dR_rw']:+.6f}"
              + (f" | P_rw(A)={row['boot_rw_A']['P']:.3f} P_rw(R)={row['boot_rw_R']['P']:.3f} ciR_rw={[round(x,5) for x in row['boot_rw_R']['ci']]} | P_unw(R)={row['boot_unw_R']['P']:.3f} sdR_unw={row['boot_unw_R']['sd']:.5f} sdA_unw={row['boot_unw_A']['sd']:.5f}" if lay == "k10" else ""), flush=True)
Path(f"work12/rw12_{a.var}{"_b6" if a.gdir != "work12/res" else ""}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
