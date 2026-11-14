#!/usr/bin/env python3
"""P1: диски 0,05 м на уже принятых объектах. Правило — docs/plans/AI_PLAN_20261114.md, шаг P1.

Стек, который здесь считается, — train-стек бокса stack_arch (честная база A), пороги —
eb268d6:configs/default.toml, то есть конфиг внутри архива 0,43274 до секции лидара.
Отдельного файла предсказаний v2 на трейне в боксе нет. Курганы c5-in класс этих трёх
не меняют: балл складывается по классам. Городища не заменяются.

Решение (порог не двигается): объекты трёх классов с уверенностью не ниже пола архива
заменяются диском 0,05 м в representative_point. Счёт — те же семьи и бутстреп, что у
гейта лидара (seed 0). PASS: Δ ≥ +0,003, P(Δ>0) ≥ 0,95, ΔF1 класса с n_gt ≥ 5 не ниже
−0,005, Δnew34 ≥ 0.

Рядом считается прежний протокол (честный перебор порога, четыре класса вместе с
городищами). Это сверка со старым +0,004366, не решение.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import ijson
import numpy as np
from shapely.geometry import mapping, shape

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from expds.metric import threshold_group  # noqa: E402
from expds.post import micro_circles  # noqa: E402

DISK = ("arkhitektury", "object_poly", "fortifikatsii")
SANITY = DISK + ("gorodishcha",)
SPEC = {"radius_m": {c: 0.05 for c in SANITY}, "anchor": "rp", "classes": list(SANITY)}


def load_gate():
    spec = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    return G


def v2_floors():
    raw = subprocess.check_output(["git", "-C", str(REPO), "show", "eb268d6:configs/default.toml"])
    cfg = tomllib.loads(raw.decode())
    thr = cfg.get("thresholds") or {}
    own = cfg.get("thresholds_by_markup") or {}
    dft = float(cfg.get("default_threshold", 0.15))
    classes = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury",
               "fortifikatsii", "object_poly", "finds_points")

    def floor(g, c):
        return float((own.get(g) or {}).get(c, thr.get(c, dft)))

    return floor


def read_classes(path: Path, classes: set[str]) -> list[dict]:
    out = []
    with path.open("rb") as fh:
        for feat in ijson.items(fh, "features.item"):
            q = feat.get("properties") or {}
            if q.get("class_name") not in classes:
                continue
            try:
                geom = shape(feat["geometry"])
            except Exception:
                continue
            if geom.is_empty:
                continue
            out.append({**q, "geom": geom, "class_name": q.get("class_name"),
                        "confidence": float(q.get("confidence") or 0.0)})
    return out


def as_features(feats: list[dict]) -> list[dict]:
    rows = []
    for f in feats:
        q = {k: v for k, v in f.items() if k != "geom"}
        q["class_name"] = f["class_name"]
        q["confidence"] = float(f["confidence"])
        rows.append({"type": "Feature", "geometry": mapping(f["geom"]), "properties": q})
    return rows


def drop_classes(tab, classes: set[str]):
    import copy
    t2 = copy.copy(tab)
    t2.base = {}
    for c, b in tab.base.items():
        if c in classes:
            t2.base[c] = dict(conf=np.zeros(0, np.float64), reg=np.zeros(0, dtype=str),
                              grp=np.zeros(0, dtype=str), pairs={})
        else:
            t2.base[c] = b
    return t2


def score_pair(G, folds, site_family, slices, base_counts, var_counts, boot: int) -> dict:
    M = G.M
    sc = {}
    for name, regions in slices.items():
        sa, sb = M.score_counts(base_counts, regions)[0], M.score_counts(var_counts, regions)[0]
        bs = M.paired_bootstrap_counts(var_counts, base_counts, site_family, boot, 0, regions) if boot else {}
        sc[name] = {"base": sa, "var": sb, "delta": sb - sa, "P": bs.get("p_gt0"),
                    "p5": bs.get("p5"), "p95": bs.get("p95")}
    pa, pb = M.score_counts(base_counts)[1], M.score_counts(var_counts)[1]
    fl = {}
    for c, a in pa.items():
        if a["n_gt"] <= 0:
            continue
        d = pb[c]["f1"] - a["f1"]
        fl[c] = {"d": d, "n_gt": a["n_gt"], "tp": [a["tp"], pb[c]["tp"]], "fp": [a["fp"], pb[c]["fp"]],
                 "ok": d >= -0.005 - 1e-12 or a["n_gt"] < 5}
    return {"scores": sc, "floor": fl}


def decide(block: dict) -> dict:
    sc, fl = block["scores"], block["floor"]
    g = {"delta_ge_0.003": sc["all"]["delta"] >= 0.003,
         "P_ge_0.95": (sc["all"]["P"] or 0) >= 0.95,
         "class_floors_ge_-0.005": all(v["ok"] for v in fl.values()),
         "new34_ge_0": sc["new34"]["delta"] >= 0.0}
    g["PASS"] = all(g.values())
    return g


def main() -> None:
    boot = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("gate_disks_p1.json")
    t0 = time.perf_counter()
    G = load_gate()
    D, M, log, site_family = G.D, G.M, G.log, G.site_family
    floor = v2_floors()
    stack_p = Path("/workspace/koz3/w5data/stack_arch.geojson")
    stk = D.Stack(stack_p, Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
                  Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
    tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
    slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
    src = read_classes(stack_p, set(SANITY))
    log(f"классы диска в стеке: { {c: sum(f['class_name']==c for f in src) for c in SANITY} }")
    kept = [f for f in src if f["class_name"] in DISK and f["confidence"] >= floor(threshold_group(f), f["class_name"])]
    disks = micro_circles(kept, {**SPEC, "classes": list(DISK)})
    log(f"после пола eb268d6: {len(kept)} объектов трёх классов")
    folds_src, _ = tab.curves()
    regs = {r for c in folds_src.values() for r in c.pred_region} | {r for c in folds_src.values() for r in c.gt_regions}
    folds = M.family_folds(regs, site_family, 10)
    del folds_src
    engines = {"decision": drop_classes(tab, set(DISK)), "sanity": drop_classes(tab, set(SANITY))}

    def counts(which: str, feats, fixed: bool):
        curves, _ = engines[which].curves(as_features(feats))
        groups = {g for c in curves.values() for g in c.pred_group}
        if fixed:
            return D.honest(curves, folds, fixed=frozenset(g for g in groups if g))[0]
        return D.honest(curves, folds)[0]

    base_fixed = counts("decision", kept, True)
    var_fixed = counts("decision", disks, True)
    decision = score_pair(G, folds, site_family, slices, base_fixed, var_fixed, boot)
    decision["gate"] = decide(decision)
    decision["n"] = len(kept)
    log(f"P1 frozen Δ {decision['scores']['all']['delta']:+.6f} P {decision['scores']['all']['P']} PASS {decision['gate']['PASS']}")

    four = [f for f in src if f["class_name"] in SANITY]
    four_d = micro_circles(four, SPEC)
    base_h = counts("sanity", four, False)
    var_h = counts("sanity", four_d, False)
    sanity = score_pair(G, folds, site_family, slices, base_h, var_h, boot)
    log(f"sanity honest-10 (4 класса) Δ {sanity['scores']['all']['delta']:+.6f} P {sanity['scores']['all']['P']}")

    res = {"what": "P1 диски 0,05 м, якорь representative_point, стек stack_arch, полы eb268d6",
           "stack": str(stack_p), "boot": boot, "classes": list(DISK), "gorodishcha": "не заменяются",
           "limit": "это train-стек базы A с полами архива 0,43274, не отдельный OOF v2: такого файла в боксе нет",
           "decision": decision, "sanity_honest10_with_gorodishcha": sanity,
           "sec": round(time.perf_counter() - t0)}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log("записано", out, res["sec"], "с")


if __name__ == "__main__":
    main()
