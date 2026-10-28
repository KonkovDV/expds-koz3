#!/usr/bin/env python3
"""bot-8, 28.10.2026: пересчёт замеров T4 (bot-9/bench_t4/*.json) на 4090 и сценарии §7.10 BUDGET.md.
Все допущения (k, число тайлов/чипов, CPU-части) — из BUDGET.md §7.1–7.3, §7.8; помечены [оц] там."""
import json, os
B = "/workspace/koz3_shared/bot-9/bench_t4"
cfgs = {}
for fn in ("bench_t4_cnx.json", "bench_t4_det.json"):
    for m in json.load(open(os.path.join(B, fn)))["models"]:
        for c in m["configs"]:
            if "skipped" in c: continue
            cfgs[(m["model"], c["dtype"], c["batch"])] = c
F_BOOST = 1590.0
K = {"cnx": {"upper": 4.0, "nom": 3.2}, "det": {"upper": 3.5, "nom": 2.8}}
K_SPEC = (2.54, 3.2)   # compute fp16 tensor / memory+SM (§7.8)
def proj(t, f, kind):
    cn = t * f / F_BOOST
    return {"t4": round(t, 2), "f_med": f, "upper": round(t / K[kind]["upper"], 2), "nom": round(t / K[kind]["nom"], 2),
            "clocknorm": [round(cn / K_SPEC[1], 2), round(cn / K_SPEC[0], 2)]}
def g(m, key, bs=4, dt="fp16"):
    c = cfgs[(m, dt, bs)]; return c[key], c["clocks_during_timing"]["sm_clock_mhz"]["median"]
out = {"per_unit_ms": {}}
for m in ("cnx_li9", "cnx_li12", "cnx_sp3"):
    for key in ("ms_per_tile_single_median", "ms_per_tile_d8_median", "ms_per_tile_flip3_median"):
        t, f = g(m, key); out["per_unit_ms"][f"{m}/{key}"] = proj(t, f, "cnx")
for m in ("dfine_l", "rtdetrv2_r50"):
    for key in ("ms_per_chip_single_median", "ms_per_chip_flip2_median"):
        t, f = g(m, key); out["per_unit_ms"][f"{m}/{key}"] = proj(t, f, "det")
P = out["per_unit_ms"]
def comb(parts):  # parts: [(name, mult)]
    r = {}
    for v in ("upper", "nom"): r[v] = sum(P[n][v] * k for n, k in parts)
    r["cn_lo"] = sum(P[n]["clocknorm"][0] * k for n, k in parts); r["cn_hi"] = sum(P[n]["clocknorm"][1] * k for n, k in parts)
    return r
A_TILE = comb([("cnx_li9/ms_per_tile_d8_median", 3), ("cnx_li12/ms_per_tile_d8_median", 1)])   # 3 li + li_x (§7.1)
DET5 = comb([("dfine_l/ms_per_chip_flip2_median", 4), ("rtdetrv2_r50/ms_per_chip_flip2_median", 1)])
DET2 = comb([("dfine_l/ms_per_chip_flip2_median", 1), ("rtdetrv2_r50/ms_per_chip_flip2_median", 1)])
HM = comb([("cnx_sp3/ms_per_tile_flip3_median", 1)])
tiles_a = {"nom": 53, "lo": round(53 * 10 / 30), "hi": round(53 * 75 / 30)}        # §7.1/§7.8 п.3
chips = {"nom": 4340, "lo": 2200, "hi": 6500}                                        # §7.2
hm_tiles = {"nom": round(1e9 / 1024**2), "lo": round(0.5e9 / 1024**2), "hi": round(1.5e9 / 1024**2 * 1.51)}  # без/с перекрытием 192
s = lambda ms, n: round(ms * n / 1e3, 1)
gpu = {
 "a_gpu_s":   {"upper": s(A_TILE["upper"], tiles_a["nom"]), "nom": s(A_TILE["nom"], tiles_a["nom"]),
               "clocknorm": [s(A_TILE["cn_lo"], tiles_a["nom"]), s(A_TILE["cn_hi"], tiles_a["nom"])],
               "range": [s(A_TILE["cn_lo"], tiles_a["lo"]), s(A_TILE["nom"], tiles_a["hi"])]},
 "b_det5_s":  {"upper": s(DET5["upper"], chips["nom"]), "nom": s(DET5["nom"], chips["nom"]),
               "clocknorm": [s(DET5["cn_lo"], chips["nom"]), s(DET5["cn_hi"], chips["nom"])],
               "range": [s(DET5["cn_lo"], chips["lo"]), s(DET5["nom"], chips["hi"])]},
 "b_det2_s":  {"upper": s(DET2["upper"], chips["nom"]), "nom": s(DET2["nom"], chips["nom"]),
               "clocknorm": [s(DET2["cn_lo"], chips["nom"]), s(DET2["cn_hi"], chips["nom"])],
               "range": [s(DET2["cn_lo"], chips["lo"]), s(DET2["nom"], chips["hi"])]},
 "b_hm_s":    {"upper": s(HM["upper"], hm_tiles["nom"]), "nom": s(HM["nom"], hm_tiles["nom"]),
               "range": [s(HM["cn_lo"], hm_tiles["lo"]), s(HM["nom"], hm_tiles["hi"])]},
 "per_dfine_detector_s": {"upper": s(P["dfine_l/ms_per_chip_flip2_median"]["upper"], chips["nom"]),
                          "nom": s(P["dfine_l/ms_per_chip_flip2_median"]["nom"], chips["nom"])},
}
out.update(per_tile_a_ms=A_TILE, per_chip_det5_ms=DET5, per_chip_det2_ms=DET2, hm_tile_ms=HM,
           tiles_a=tiles_a, chips=chips, hm_tiles=hm_tiles, gpu_stage_s=gpu)
# CPU-части [оц] из §7.1–7.3: (номинал, низ, верх)
CPU = {"a_dtm": (55, 30, 80), "a_channels": (10, 2, 46), "b_read": (60, 30, 90), "b_gate": (5, 1, 5), "c": (10, 5, 20),
       "rm_li": (45, 30, 60), "rm_spor": (45, 30, 60)}
BASE = 202
def add(*xs): return round(sum(xs), 0)
a_u = CPU["a_dtm"][0] + CPU["a_channels"][0] + gpu["a_gpu_s"]["upper"]
a_n = CPU["a_dtm"][0] + CPU["a_channels"][0] + gpu["a_gpu_s"]["nom"]
a_lo = CPU["a_dtm"][1] + CPU["a_channels"][1] + gpu["a_gpu_s"]["range"][0]
a_hi = CPU["a_dtm"][2] + CPU["a_channels"][2] + gpu["a_gpu_s"]["range"][1]
def b(det):
    return (gpu[det]["upper"] + gpu["b_hm_s"]["upper"] + CPU["b_gate"][0] + CPU["b_read"][0],
            gpu[det]["nom"] + gpu["b_hm_s"]["nom"] + CPU["b_gate"][0] + CPU["b_read"][0],
            gpu[det]["range"][0] + gpu["b_hm_s"]["range"][0] + CPU["b_gate"][1] + CPU["b_read"][1],
            gpu[det]["range"][1] + gpu["b_hm_s"]["range"][1] + CPU["b_gate"][2] + CPU["b_read"][2])
bf, bc = b("b_det5_s"), b("b_det2_s")
rl, rs, c = CPU["rm_li"], CPU["rm_spor"], CPU["c"]
sc = {
 "a":        (a_u, a_n, a_lo, a_hi),
 "a_net":    (a_u - rl[0], a_n - rl[0], a_lo - rl[2], a_hi - rl[1]),
 "b_full":   bf,
 "b_net":    (bf[0] - rs[0], bf[1] - rs[0], bf[2] - rs[2], bf[3] - rs[1]),
 "bcut_net": (bc[0] - rs[0], bc[1] - rs[0], bc[2] - rs[2], bc[3] - rs[1]),
 "bcut_gross": bc,
 "c":        (c[0], c[0], c[1], c[2]),
 "all_full": (a_u + bf[0] + c[0], a_n + bf[1] + c[0], a_lo + bf[2] + c[1], a_hi + bf[3] + c[2]),
 "all_net_bcut": (a_u - rl[0] + bc[0] - rs[0] + c[0], a_n - rl[0] + bc[1] - rs[0] + c[0],
                  a_lo - rl[2] + bc[2] - rs[2] + c[1], a_hi - rl[1] + bc[3] - rs[1] + c[2]),
}
out["scenarios_add_s"] = {k: [round(v) for v in t] for k, t in sc.items()}
out["scenarios_total_s"] = {k: [round(BASE + v) for v in t] for k, t in sc.items()}
out["columns"] = ["upper_k", "nominal_k", "low", "high"]
json.dump(out, open("/workspace/koz3_shared/bot-8/bench/calc_7_10.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("per_tile_a_ms", "per_chip_det5_ms", "per_chip_det2_ms", "hm_tile_ms", "hm_tiles", "tiles_a", "gpu_stage_s", "scenarios_add_s", "scenarios_total_s")}, ensure_ascii=False))
for k, v in P.items(): print(k, v)
