#!/usr/bin/env python3
"""Гейт c5-in (лидар ConvNeXt-T, ImageNet) на V3P в прод-контексте v3_on — PREREG kaggle/2026-11-05/li_cnx_in/PREREG.md.

Движок и контекст — как у w2_gate/r50c5_v3p/gate_r50c5_ctx.py (W4 v3_gate_lib, d3_gate TableEngine, honest-10 по
семьям, paired bootstrap по site_family, seed 0; полы выдачи архива v3_on). Лидар у v2, v3a_min, v4 и E1_G1 на LB
совпадает до третьего знака, поэтому контекст v3_on для лидарной правки равен контексту v2.

Решающие варианты (m = 2, Бонферрони: P ≥ 0,95):
  K — добавить курганы c5-in (kurgany_povrezhdennye, kurgany_tselye) отдельной группой li_cnxin;
  R — заменить курганы лидарного слоя (группы li, li_coarse, li_mosaic) курганами c5-in.
Городища/форты/архитектура c5-in не берутся: у r50c5 (c5-sc) именно они сломали полы (gor −0,0151, fort −0,0052).

PASS (всё сразу): Δ ≥ +0,003; P(Δ>0) ≥ 0,95; Δnew34 ≥ 0; Δnolidar ≥ −0,001; ΔF1 каждого класса ≥ −0,005 (n_gt ≥ 5).

    nice -n 10 /workspace/koz3/w4venv/bin/python gate_cnxin_ctx.py check     # только проверка OOF, секунды
    nice -n 10 /workspace/koz3/w4venv/bin/python gate_cnxin_ctx.py base      # S0 и V3P_ctx без OOF (санити бокса)
    nice -n 10 /workspace/koz3/w4venv/bin/python gate_cnxin_ctx.py 2000 [OUT.json]

OOF: /workspace/koz3/w2/oof/cnxin/<ядро>/oof_cnx_c5-in_f{0..4}.geojson и region_fold.json (выходы ядер
dmitrijkv/koz3-li-cnx-in-{a,b,c}); путь меняется переменной CNXIN_OOF_DIR.
"""
from __future__ import annotations

import gc
import glob
import importlib.util
import json
import os
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OOF_DIR = Path(os.environ.get("CNXIN_OOF_DIR", "/workspace/koz3/w2/oof/cnxin"))
VARIANT = os.environ.get("CNXIN_VARIANT", "c5-in")   # c12-in — kaggle/2026-11-05/li_c12_in/PREREG.md
# Сравнение c12 − c5 (PREREG c12, «Выбор между c5-in и c12-in»): тот же вариант K/R, парный бутстреп по семьям,
# правило Δ > 0 и P ≥ 0,9. Включается CNXIN_CMP_VARIANT (например c5-in) и CNXIN_CMP_DIR (OOF этого варианта).
CMP_VARIANT = os.environ.get("CNXIN_CMP_VARIANT", "")
CMP_DIR = Path(os.environ.get("CNXIN_CMP_DIR", "/workspace/koz3/w2/oof/cnxin"))
CMP_P_MIN = 0.9
KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
LI_KUR_GROUPS = ("li", "li_coarse", "li_mosaic")
GROUP = "li_cnxin"
V3P_REF = 0.3337867244
# конфиг v3_on (= configs/v3_on.toml; box_restore.sh кладёт копию в w2/r50c5_gate, как у гейта r50c5)
CFG_P = next((p for p in (Path(os.environ.get("CNXIN_V3ON_CFG", "")), HERE / "v3on_src/configs/default.toml",
                          Path("/workspace/koz3/w2/r50c5_gate/v3on_src/configs/default.toml")) if str(p) != "." and p.is_file()),
             HERE / "v3on_src/configs/default.toml")
P_MIN = 0.95                                      # m = 2


def oof_files(root: Path = OOF_DIR, variant: str = VARIANT) -> dict[int, Path]:
    """{фолд: файл}; ровно один файл на фолд 0..4, иначе останов (PREREG: упавший фолд — гейт не считается)."""
    found: dict[int, list[Path]] = {}
    for p in sorted(glob.glob(str(root / "**" / f"oof_cnx_{variant}_f[0-4].geojson"), recursive=True)):
        f = int(Path(p).stem.rsplit("_f", 1)[1])
        found.setdefault(f, []).append(Path(p))
    bad = {f: v for f, v in found.items() if len(v) != 1}
    miss = sorted(set(range(5)) - set(found))
    if bad or miss:
        raise SystemExit(f"OOF {variant}: нет фолдов {miss}, дубли {bad} в {root}")
    return {f: v[0] for f, v in sorted(found.items())}


def check_oof(root: Path = OOF_DIR, variant: str = VARIANT) -> dict:
    """Фолды по region_fold.json ядер: каждый участок OOF — тест своего фолда, семьи не разбиты, классы известны."""
    files = oof_files(root, variant)
    rf: dict[str, int] = {}
    fam: dict[str, str] = {}
    for p in sorted(glob.glob(str(root / "**" / "region_fold.json"), recursive=True)):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for r, f in d["region_fold"].items():
            if rf.setdefault(r, int(f)) != int(f):
                raise SystemExit(f"region_fold.json разных ядер расходятся по {r}")
        fam.update(d.get("region_family", {}))
    if not rf:
        raise SystemExit(f"нет region_fold.json в {root}")
    rep = {"rows": {}, "classes": Counter(), "regions_per_fold": {}}
    fam_fold: dict[str, set] = {}
    for f, p in files.items():
        d = json.loads(p.read_text(encoding="utf-8"))
        regs = Counter()
        for ft in d["features"]:
            q = ft["properties"]
            r = q.get("region_name")
            if rf.get(r) != f:
                raise SystemExit(f"{p.name}: участок {r} — тест фолда {rf.get(r)}, а лежит в OOF фолда {f}")
            regs[r] += 1
            rep["classes"][q.get("class_name")] += 1
            fam_fold.setdefault(fam.get(r, r), set()).add(f)
        rep["rows"][f] = sum(regs.values())
        rep["regions_per_fold"][f] = len(regs)
    split = {k: sorted(v) for k, v in fam_fold.items() if len(v) > 1}
    if split:
        raise SystemExit(f"семьи в нескольких фолдах: {split}")
    rep["classes"] = dict(rep["classes"])
    if not any(c in rep["classes"] for c in KUR):
        raise SystemExit("в OOF нет курганов")
    return rep


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        print(json.dumps(check_oof(), ensure_ascii=False, indent=1))
        return
    import numpy as np
    spec = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
    argv = sys.argv
    sys.argv = sys.argv[:1]
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    sys.argv = argv
    D, M, log, site_family = G.D, G.M, G.log, G.site_family
    base_only = len(sys.argv) > 1 and sys.argv[1] == "base"
    boot = 0 if base_only else (int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / ("gate_cnxin_base.json" if base_only else "gate_cnxin_ctx.json")
    t0 = time.perf_counter()
    res = {"what": f"{VARIANT} на V3P, прод-контекст v3_on (PREREG li_cnx_in; c12 — li_c12_in)", "boot": boot,
           "cfg": str(CFG_P), "oof_check": None if base_only else check_oof(), "variants": {}}
    if CMP_VARIANT and not base_only:
        res["cmp_oof_check"] = check_oof(CMP_DIR, CMP_VARIANT)
    save = lambda: out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))   # noqa: E731
    cfg = tomllib.load(open(CFG_P, "rb"))
    post = cfg.get("post") or {}
    assert not post.get("kurgan_neighbour_prior") and not post.get("site_calibration") and "w5_cloudfeat" not in cfg
    fl_cache: dict = {}

    def floor(g, c):   # == expds/pipeline.output_floors(cfg, g)[c] у v3_on (приор и калибровка выключены)
        if g not in fl_cache:
            thr, own = cfg.get("thresholds", {}), cfg.get("thresholds_by_markup", {}).get(g, {})
            dft = cfg.get("default_threshold", 0.15)
            fl_cache[g] = {cc: float(own.get(cc, thr.get(cc, dft))) for cc in D.CLASSES5 + ("object_poly", "finds_points")}
        return fl_cache[g].get(c, float(cfg.get("default_threshold", 0.15)))

    def filt_engine(tab, keep_row):
        import copy
        t2 = copy.copy(tab)
        t2.base = {}
        n0 = n1 = 0
        for c, b in tab.base.items():
            keep = np.array([keep_row(c, g, cf) for g, cf in zip(b["grp"], b["conf"])], bool)
            n0 += len(keep)
            n1 += int(keep.sum())
            idx = np.flatnonzero(keep)
            new = {int(o): n for n, o in enumerate(idx)}
            t2.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx],
                              pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
        return t2, n0, n1

    def fl_feats(fs):
        return [f for f in fs if float(f["properties"]["confidence"]) >=
                floor(M.threshold_group(f["properties"]), f["properties"]["class_name"])]

    stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
                  Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
    tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
    slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
    cA, _ = tab.curves()
    regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
    folds = M.family_folds(regs, site_family, 10)
    del cA
    c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
    dfx, dreg = G.dfine_feats()
    base3 = G.drop_engine(tab, lambda g, r: g == "etgr" or (g in G.SP and r in dreg))
    del tab
    gc.collect()
    cP, _ = base3.curves(c19 + dfx)
    nP = D.honest(cP, folds)[0]
    del cP
    res["V3P"] = M.score_counts(nP)[0]
    res["S0_ok"] = abs(res["V3P"] - V3P_REF) <= 1e-6 or (c19 == [] and abs(res["V3P"] - 0.334435) <= 2e-6)
    log(f"S0 V3P {res['V3P']:.10f} ok {res['S0_ok']}")
    save()
    assert res["S0_ok"], "V3P не воспроизведён (ни 0,3337867 с C19, ни 0,334435 без C19 — box_restore.sh)"
    def kur_rows(files):
        out = []
        for p in files:
            for f, _ in D.iter_features(p):
                q = f["properties"]
                if q.get("class_name") in KUR:
                    q["threshold_group"] = GROUP
                    q["markup_type"] = "li"
                    out.append(f)
        return out

    rows = kur_rows([] if base_only else oof_files().values())
    res["n_cnxin_kur"] = len(rows)
    bctx, n0, n1 = filt_engine(base3, lambda c, g, cf: cf >= floor(g, c))
    ex_ctx = fl_feats(c19 + dfx)
    res["ctx_rows"] = {"stack_before": n0, "stack_kept": n1, "extras_kept": len(ex_ctx)}
    cA, _ = bctx.curves(ex_ctx)
    nA = D.honest(cA, folds)[0]
    del cA
    res["V3P_ctx"] = M.score_counts(nA)[0]
    log(f"V3P_ctx {res['V3P_ctx']:.10f}")
    save()
    if base_only:
        res["sec"] = round(time.perf_counter() - t0)
        save()
        log("base: готово", res["sec"], "с")
        return

    def score(name, nB, nRef, decisive):
        sc = {}
        for k, s in slices.items():
            sa, sb = M.score_counts(nRef, s)[0], M.score_counts(nB, s)[0]
            bs = M.paired_bootstrap_counts(nB, nRef, site_family, boot, 0, s) if boot else {}
            sc[k] = {"A": sa, "B": sb, "delta": sb - sa, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
        pa, pb = M.score_counts(nRef)[1], M.score_counts(nB)[1]
        fl = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "n_gt": pa[c]["n_gt"], "tp": [pa[c]["tp"], pb[c]["tp"]],
                  "fp": [pa[c]["fp"], pb[c]["fp"]],
                  "ok": (pb[c]["f1"] - pa[c]["f1"] >= -0.005 - 1e-12) or pa[c]["n_gt"] < 5} for c in pa if pa[c]["n_gt"] > 0}
        g = {"delta_ge_0.003": sc["all"]["delta"] >= 0.003, "P_ge_0.95": (sc["all"]["P"] or 0) >= P_MIN,
             "class_floors_ge_-0.005": all(v["ok"] for v in fl.values()),
             "new34_ge_0": sc["new34"]["delta"] >= 0.0, "nolidar_ge_-0.001": sc["nolidar"]["delta"] >= -0.001}
        g["PASS"] = all(g.values())
        res["variants"][name] = {"decisive": decisive, "scores": sc, "floor": fl, "gate": g}
        log(f"{name}: Δ {sc['all']['delta']:+.6f} P {sc['all']['P']} new34 {sc['new34']['delta']:+.6f} "
            f"nolidar {sc['nolidar']['delta']:+.6f} PASS {g['PASS']}")
        save()

    def cmp(name, nMain, nCmp):
        """PREREG c12: тот же вариант, VARIANT против CMP_VARIANT; парный бутстреп по семьям (seed 0)."""
        sc = {}
        for k, s in slices.items():
            sa, sb = M.score_counts(nCmp, s)[0], M.score_counts(nMain, s)[0]
            bs = M.paired_bootstrap_counts(nMain, nCmp, site_family, boot, 0, s) if boot else {}
            sc[k] = {"cmp": sa, "main": sb, "delta": sb - sa, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
        rule = sc["all"]["delta"] > 0 and (sc["all"]["P"] or 0) >= CMP_P_MIN
        res.setdefault("cmp", {"main": VARIANT, "vs": CMP_VARIANT, "rule": f"delta > 0 and P >= {CMP_P_MIN}"})[name] = \
            {"scores": sc, "main_replaces_cmp": rule}
        log(f"{name}: {VARIANT} − {CMP_VARIANT} Δ {sc['all']['delta']:+.6f} P {sc['all']['P']} → замена {rule}")
        save()

    def taus(name, fold_taus):
        """05.11: пороги группы GROUP по 10 честным фолдам и их медиана — для конфига сдачи (на решение не влияют)."""
        per = [ft.get(GROUP, {}) for ft in fold_taus]
        med = {c: float(np.median([p[c] for p in per if c in p and np.isfinite(p[c])]))
               for c in KUR if any(c in p and np.isfinite(p[c]) for p in per)}
        res["variants"][name]["taus_group"] = {"group": GROUP, "folds": per, "median": med}
        save()

    rB = fl_feats(rows)
    res["n_cnxin_kur_ctx"] = len(rB)
    rC = fl_feats(kur_rows(oof_files(CMP_DIR, CMP_VARIANT).values())) if CMP_VARIANT else None
    if rC is not None:
        res["n_cmp_kur_ctx"] = len(rC)
    cK, _ = bctx.curves(ex_ctx + rB)
    hK = D.honest(cK, folds)
    score("K_add_kurgans", hK[0], nA, True)
    taus("K_add_kurgans", hK[2])
    if rC is not None:
        cKc, _ = bctx.curves(ex_ctx + rC)
        cmp("K_add_kurgans", hK[0], D.honest(cKc, folds)[0])
        del cKc
    del cK, hK
    brep, _, n_left = filt_engine(bctx, lambda c, g, cf: not (c in KUR and g in LI_KUR_GROUPS))
    res["R_dropped_rows"] = res["ctx_rows"]["stack_kept"] - n_left
    assert res["R_dropped_rows"] > 0, f"в стеке нет курганов групп {LI_KUR_GROUPS} — имена групп?"
    cR, _ = brep.curves(ex_ctx + rB)
    hR = D.honest(cR, folds)
    score("R_replace_li_kurgans", hR[0], nA, True)
    taus("R_replace_li_kurgans", hR[2])
    if rC is not None:
        cRc, _ = brep.curves(ex_ctx + rC)
        cmp("R_replace_li_kurgans", hR[0], D.honest(cRc, folds)[0])
        del cRc
    del cR, hR
    res["PASS_any"] = any(v["gate"]["PASS"] for v in res["variants"].values() if v["decisive"])
    res["sec"] = round(time.perf_counter() - t0)
    save()
    log("готово", res["sec"], "с; PASS хоть одного:", res["PASS_any"])


if __name__ == "__main__":
    main()
