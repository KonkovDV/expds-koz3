"""bot-10: пакет и абляции «по одному» из посчитанных счётчиков (одни фолды).
Компоненты: kur (bot-7 курганы 0,5 м), archM (bot-7 микрокруги arch, пороги переподобраны внутри фолдов),
R (реранкер arch, вложенный: модель вне фолда, пороги на внутреннем 5-OOF обучения). crop — нет данных на коробке.
Классы аддитивны: kur трогает только kurgany_*, archM/R — только arkhitektury; взаимодействие archM×R посчитано совместно (full_micro).
"""
import sys, json, pickle, argparse
from pathlib import Path
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves, regions_of, random_family_folds, boot, fam_index, stats, weighted_class_delta
from expds import metric as M
from expds.naming import site_family
ap = argparse.ArgumentParser(); ap.add_argument("--rv", default="full"); ap.add_argument("--out", default="work4/package.json"); ap.add_argument("--scope", default="global"); a = ap.parse_args()
RR = Path("work4/rerank"); B7 = Path("/workspace/koz3_shared/bot-7/work2")
new = {l.strip() for l in Path("repo/notes/regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
SS = "_site" if a.scope == "site" else ""
A_cv = load_curves(Path("work4/curves_site" if SS else "work/curves")); K_cv = load_curves(Path("work4/curves_site_b7kur") if SS else B7 / "curves_kur")
regions = regions_of(A_cv) | regions_of(K_cv)
lay = {"k10": M.family_folds(regions, site_family, 10), "lofo": M.family_folds(regions, site_family, None)}
for s in range(5): lay[f"seed{s}"] = random_family_folds(regions, 10, s)
KUR = ("kurgany_povrezhdennye", "kurgany_tselye"); AR = "arkhitektury"
HEAVY = KUR + ("gorodishcha",)
def mix(base, kur=None, arch=None):
    c = dict(base)
    if kur is not None:
        for k in KUR: c[k] = kur[k]
    if arch is not None: c[AR] = arch[AR]
    return c
rep = {"layouts": {}, "rv": a.rv}; CK = {}
for ln, folds in lay.items():
    base, _ = M.honest_counts(A_cv, folds); kur, _ = M.honest_counts(K_cv, folds)
    rb = pickle.load(open(RR / f"counts_{ln}{SS}_base.pkl", "rb"))
    chk = M.score_counts(rb)[0] - M.score_counts(base)[0]
    am = pickle.load(open(RR / f"counts_{ln}{SS}_conf_micro.pkl", "rb"))
    ro = pickle.load(open(RR / f"counts_{ln}{SS}_{a.rv}_orig.pkl", "rb")); rm = pickle.load(open(RR / f"counts_{ln}{SS}_{a.rv}_micro.pkl", "rb"))
    V = {"A": base, "kur": mix(base, kur), "archM": mix(base, arch=am), "bundle(kur+archM)": mix(base, kur, am),
         "R": mix(base, arch=ro), "R+archM": mix(base, arch=rm), "R+kur": mix(base, kur, ro), "PACKAGE(kur+archM+R)": mix(base, kur, rm)}
    sa = M.score_counts(base)[0]
    row = {n: M.score_counts(c)[0] - sa for n, c in V.items()}
    row["_check_base_rerank_minus_mine"] = chk
    rep["layouts"][ln] = row
    if ln == "k10": CK = V
    print(ln, {k: round(v, 6) for k, v in row.items()}, flush=True)
fams = fam_index(list(CK.values())); keys = list(CK)
b = dict(zip(keys, boot([CK[k] for k in keys], 2000, 0, fams)))
rep["summary"] = {}
for k in keys[1:]:
    st = stats(b[k] - b["A"]); wc = weighted_class_delta(CK[k], CK["A"])
    L = rep["layouts"]; seeds = [L[f"seed{s}"][k] for s in range(5)]
    heavy = sum(wc[c]["weighted"] for c in HEAVY if c in wc)
    dnew = M.score_counts(CK[k], new)[0] - M.score_counts(CK["A"], new)[0]
    rule = {"P1": L["k10"][k] >= 0.002, "P2": st["p_gt0"] >= 0.85, "P3": L["lofo"][k] > 0 and sum(x > 0 for x in seeds) >= 4,
            "P4": heavy >= -0.0005 and dnew >= -0.001}
    rep["summary"][k] = dict(d_k10=L["k10"][k], boot=st, lofo=L["lofo"][k], seeds=seeds, heavy=heavy, d_new34=dnew,
                             per_class={c: v["weighted"] for c, v in wc.items() if abs(v["weighted"]) > 0}, rule=rule, accepted=all(rule.values()))
    print(f"{k:22s} d10={L['k10'][k]:+.6f} P={st['p_gt0']:.3f} [{st['p5']:+.5f};{st['p95']:+.5f}] lofo={L['lofo'][k]:+.6f} seeds={[round(x,5) for x in seeds]} heavy={heavy:+.6f} new34={dnew:+.6f} acc={all(rule.values())}", flush=True)
Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
