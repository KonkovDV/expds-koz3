"""BUDGET §7.12: scenario (a) CPU part after items (2), (3), (5), (6). Inputs: §7.11 (calc_7_11.json / BUDGET.md
lines 476–495) and bench/cpu_a_speedup/*.json. Prints and writes calc_7_12.json."""
import json
from pathlib import Path

B = Path(__file__).parent / "cpu_a_speedup"
j = lambda n: json.loads((B / n).read_text())
# §7.11 box CPU-s at 10 / 30 / 75 Mpx (BUDGET.md:482–484) and the fits (:476)
box = {10: dict(dtm=17, tile=7, blend=2), 30: dict(dtm=39, tile=20, blend=5), 75: dict(dtm=89, tile=49, blend=12)}
ch9 = lambda mpx: 0.53 + 0.57 * mpx; ch12 = lambda mpx: 3.68 + 0.91 * mpx
GPU_CH = 2.0                                        # §7.11: channels 9+12 on GPU [оц]
# (3) fill = EDT 43 % + gauss 9 % + valid 4 % of DTM wall (105_Вщиж, BUDGET.md:469); exact_fast speedup on the fill
fill_share = 0.43 + 0.09 + 0.04
f = {s: j(f"edt_fill_{s}_t4096.json") if (B / f"edt_fill_{s}_t4096.json").exists() else None for s in ("001", "022")}
x_fill = [r["exact"]["wall_med"] / r["exact_fast"]["wall_med"] for r in f.values() if r] + \
         [r["exact"]["cpu_med"] / r["exact_fast"]["cpu_med"] for r in f.values() if r]
x_fill_lo, x_fill_hi = min(x_fill), max(x_fill)
dtm_after3 = lambda x: 1 - fill_share * (1 - 1 / x)
# (6) DtmPool: wall = max(largest file, Σ/N) + overhead; largest = 58 % of points (§7.0), workers ≥ 2 on 24 cores
LARGEST, POOL_OVH = 0.58, 1.0                       # overhead: worker start + .npy through tmpfs (pool_big61.json ≈ +1 s)
# (5) host part of tiling left = pad + from_numpy (acc_host_residual_022.json)
resid = j("acc_host_residual_022.json")["residual_share"]
D2H_S = 0.2                                         # 4 maps × 0.84 GB at 30 Mpx over PCIe [оц]
out = {"x_fill": [x_fill_lo, x_fill_hi], "dtm_factor_3": [dtm_after3(x_fill_hi), dtm_after3(x_fill_lo)],
       "tile_residual_share": resid, "cases": {}}
for mpx, k in ((10, 2.0), (30, 2.0), (30, 1.5), (75, 1.5)):
    b = box[mpx]
    old = (b["dtm"] + b["tile"] + b["blend"]) / k + GPU_CH
    x9_share = ch9(mpx) / (ch9(mpx) + ch12(mpx))
    ch_new = GPU_CH * (1 - x9_share)
    dtm_new = [b["dtm"] / k * dtm_after3(x) * LARGEST + POOL_OVH for x in (x_fill_hi, x_fill_lo)]
    tile_new = b["tile"] / k * resid
    blend_exact = b["blend"] / k + D2H_S * mpx / 30
    blend_dev = 0.05 * mpx / 30 + D2H_S * mpx / 30 / 4
    new_exact = [d + tile_new + blend_exact + ch_new for d in dtm_new]
    new_dev = [d + tile_new + blend_dev + ch_new for d in dtm_new]
    out["cases"][f"{mpx}Mpx_k{k}"] = dict(old=old, ch_new=ch_new, dtm_old=b["dtm"] / k, dtm_new=dtm_new, tile_old=b["tile"] / k,
                                          tile_new=tile_new, blend_old=b["blend"] / k, blend_exact=blend_exact, blend_device=blend_dev,
                                          new_device_exact=new_exact, new_device=new_dev,
                                          delta_device_exact=[n - old for n in new_exact], delta_device=[n - old for n in new_dev])
print(json.dumps(out, indent=1, ensure_ascii=False))
(Path(__file__).parent / "calc_7_12.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
