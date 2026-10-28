"""Задача 9: копия bot-10/work4/paired.py (запуск из папки bot-10) + срезы no_lidar / lidar59, П4 по каждому
тяжёлому классу отдельно (povr, tselye, gor ≥ −0,0005), Δ без 3 семей с наибольшим вкладом (п. «≤3 семьи»).

    cd bot-10 && python <bot-2>/repo/scripts/li_real_paired.py --base L=<curves> --b P=<curves> --boot 1000 --out x.json
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves, regions_of, random_family_folds, boot, fam_index, stats, weighted_class_delta
from expds import metric as M
from expds.naming import site_family

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--b", action="append", required=True)
ap.add_argument("--pols", default="drop"); ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
bn, bd = a.base.split("=", 1)
V = {bn: load_curves(Path(bd))}
for s in a.b:
    n, d = s.split("=", 1); V[n] = load_curves(Path(d))
pols = a.pols.split(",")
new = {l.strip() for l in Path("repo/notes/regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
lidar59 = set(json.loads(Path("repo/notes/las_boxes_59.json").read_text(encoding="utf-8")))
regions = set().union(*(regions_of(c) for c in V.values()))
nolid = regions - lidar59
SL = {"new34": new, "no_lidar": nolid, "lidar59": regions & lidar59}
lay = {"k10": M.family_folds(regions, site_family, 10), "lofo": M.family_folds(regions, site_family, None)}
for s in range(5):
    lay[f"seed{s}"] = random_family_folds(regions, 10, s)
HEAVY = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha")
rep = {"base": a.base, "variants": a.b, "slices_n": {k: len(v & regions) for k, v in SL.items()}, "layouts": {}}
C = {}
for ln, folds in lay.items():
    row = {}
    for nm, cv in V.items():
        for pol in pols:
            c, info = M.honest_counts(cv, folds, unseen=pol)
            sc, per = M.score_counts(c)
            row[f"{nm}_{pol}"] = {"all": sc, **{k: M.score_counts(c, v)[0] for k, v in SL.items()},
                                  "f1": {k: per[k]["f1"] for k in per}, "info": info}
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
        heavy = {c: wc[c]["weighted"] for c in HEAVY if c in wc}
        dsl = {s: rep["layouts"]["k10"][k][s] - rep["layouts"]["k10"][kb][s] for s in SL}
        # вклад семьи: Δ(всё) − Δ(без семьи); «≤3 семьи» — Δ без трёх с наибольшим вкладом
        contrib = {}
        for f, regs in fams.items():
            rest = regions - set(regs)
            contrib[f] = d10 - (M.score_counts(C[k], rest)[0] - M.score_counts(C[kb], rest)[0])
        top = sorted(contrib, key=lambda f: -abs(contrib[f]))[:3]
        rest = regions - set().union(*(set(fams[f]) for f in top))
        d_wo3 = M.score_counts(C[k], rest)[0] - M.score_counts(C[kb], rest)[0]
        n_changed = sum(1 for f in fams if abs(contrib[f]) > 1e-9)
        rule = {"P1_d>=0.002": d10 >= 0.002, "P2_P>=0.85": st["p_gt0"] >= 0.85,
                "P3_lofo>0_and_seeds>=4/5": lofo > 0 and sum(x > 0 for x in seeds) >= 4,
                "P4_each_heavy>=-0.0005_and_new34>=-0.001": all(v >= -0.0005 for v in heavy.values()) and dsl["new34"] >= -0.001}
        acc = all(rule.values())
        verdict = ("принять" if d10 >= 0.006 else "в пакет") if acc else "отклонить"
        rep["summary"][k] = {"d_k10": d10, "boot": st, "lofo": lofo, "seeds": seeds, "seeds_pos": sum(x > 0 for x in seeds),
                             "heavy_weighted": heavy, "d_slices": dsl,
                             "per_class_weighted": {c: v["weighted"] for c, v in wc.items()},
                             "per_class_dF1": {c: v["dF1"] for c, v in wc.items()},
                             "family_contrib_top": {f: contrib[f] for f in sorted(contrib, key=lambda f: -abs(contrib[f]))[:8]},
                             "d_without_top3_families": d_wo3, "families_changed": n_changed,
                             "unseen_info_variant": rep["layouts"]["k10"][k]["info"], "unseen_info_base": rep["layouts"]["k10"][kb]["info"],
                             "rule": rule, "accepted": acc, "verdict_P5": verdict}
        s = rep["summary"][k]
        print(k, f"d10={d10:+.6f} P={st['p_gt0']:.3f} CI=[{st['p2_5']:+.5f};{st['p97_5']:+.5f}] lofo={lofo:+.6f} "
              f"seeds+={s['seeds_pos']}/5 heavy={ {c: round(v, 5) for c, v in heavy.items()} } "
              f"sl={ {c: round(v, 5) for c, v in dsl.items()} } wo3={d_wo3:+.6f} fams={n_changed} {verdict}", flush=True)
Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
