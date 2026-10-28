"""Парные Δ (вариант − база) при unseen=drop и keep: honest10, LOFO, 5 случайных раскладок, бутстреп по семьям, Δ по классам, П1–П4.

    python work4/paired.py --base A=work/curves --b sep=work4/curves_g_sep --b uni=work4/curves_g_uni --out work4/x.json
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves, regions_of, random_family_folds, boot, fam_index, stats, weighted_class_delta
from expds import metric as M
from expds.naming import site_family

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--b", action="append", required=True)
ap.add_argument("--pols", default="drop,keep"); ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
bn, bd = a.base.split("=", 1)
V = {bn: load_curves(Path(bd))}
for s in a.b:
    n, d = s.split("=", 1); V[n] = load_curves(Path(d))
pols = a.pols.split(",")
new = {l.strip() for l in Path("repo/notes/regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
regions = set().union(*(regions_of(c) for c in V.values()))
lay = {"k10": M.family_folds(regions, site_family, 10), "lofo": M.family_folds(regions, site_family, None)}
for s in range(5):
    lay[f"seed{s}"] = random_family_folds(regions, 10, s)
HEAVY = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha")
rep = {"base": a.base, "variants": a.b, "layouts": {}}
C = {}
for ln, folds in lay.items():
    row = {}
    for nm, cv in V.items():
        for pol in pols:
            c, info = M.honest_counts(cv, folds, unseen=pol)
            sc, per = M.score_counts(c)
            row[f"{nm}_{pol}"] = {"all": sc, "new34": M.score_counts(c, new)[0],
                                  "f1": {k: per[k]["f1"] for k in ("object_poly", "finds_points") if k in per},
                                  "tp_fp": {k: [per[k]["tp"], per[k]["fp"]] for k in ("object_poly", "finds_points") if k in per},
                                  "info": info}
            if ln == "k10":
                C[f"{nm}_{pol}"] = c
    for nm in V:
        if nm == bn:
            continue
        for pol in pols:
            row[f"d_{nm}_{pol}"] = row[f"{nm}_{pol}"]["all"] - row[f"{bn}_{pol}"]["all"]
    rep["layouts"][ln] = row
    print(ln, {k: round(v, 6) if not isinstance(v, dict) else round(v["all"], 6) for k, v in row.items()}, flush=True)
fams = fam_index(list(C.values()))
keys = sorted(C)
b = dict(zip(keys, boot([C[k] for k in keys], a.boot, 0, fams)))
rep["summary"] = {}
for nm in V:
    if nm == bn:
        continue
    for pol in pols:
        k, kb = f"{nm}_{pol}", f"{bn}_{pol}"
        st = stats(b[k] - b[kb])
        wc = weighted_class_delta(C[k], C[kb])
        d10 = rep["layouts"]["k10"][f"d_{k}"]
        lofo = rep["layouts"]["lofo"][f"d_{k}"]
        seeds = [rep["layouts"][f"seed{s}"][f"d_{k}"] for s in range(5)]
        heavy = sum(wc[c]["weighted"] for c in HEAVY if c in wc)
        dnew = rep["layouts"]["k10"][k]["new34"] - rep["layouts"]["k10"][kb]["new34"]
        rule = {"P1_d>=0.002": d10 >= 0.002, "P2_P>=0.85": st["p_gt0"] >= 0.85,
                "P3_lofo>0_and_seeds>=4/5": lofo > 0 and sum(x > 0 for x in seeds) >= 4,
                "P4_heavy>=-0.0005_and_new34>=-0.001": heavy >= -0.0005 and dnew >= -0.001}
        rep["summary"][k] = {"d_k10": d10, "boot": st, "lofo": lofo, "seeds": seeds, "seeds_pos": sum(x > 0 for x in seeds),
                             "heavy_weighted": heavy, "d_new34": dnew,
                             "per_class_weighted": {c: v["weighted"] for c, v in wc.items()},
                             "unseen_info_variant": rep["layouts"]["k10"][k]["info"], "unseen_info_base": rep["layouts"]["k10"][kb]["info"],
                             "rule": rule, "accepted": all(rule.values())}
        s = rep["summary"][k]
        print(k, f"d10={d10:+.6f} P={st['p_gt0']:.3f} CI=[{st['p2_5']:+.5f};{st['p97_5']:+.5f}] lofo={lofo:+.6f} seeds+={s['seeds_pos']}/5 heavy={heavy:+.6f} new34={dnew:+.6f} acc={s['accepted']}", flush=True)
Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
