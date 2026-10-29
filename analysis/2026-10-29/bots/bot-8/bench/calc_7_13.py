"""§7.13: full package total = base + EhimenNathan port net (b-cut, clean (a)) + PIXGBM, vs working ceiling 1200 s.

Sources (all seconds, platform estimates):
- base 191–202 [meas, docs/LB_LOG.md via BUDGET.md:16].
- "all net, b-cut" TOTALS from §7.12 (BUDGET.md:531): 402–406 (upper k) / 452–456 (nominal k), wide range 201–750;
  §7.11 (BUDGET.md:495) wide edge 808. These totals INCLUDE base 202 (§7.10 table, BUDGET.md:434: 202 + 249/299 = 451/501),
  so port-only net = total − 202.
- PIXGBM [bot-3/pixgbm/BUDGET_PIXGBM.md §4]: 2 thr 258–431, 8 thr 166–276, findings-only 2 thr 158–263; 1 thr 396–660.
"""
import json
from pathlib import Path

BASE = (191, 202)
BASE_IN_TOTALS = 202
PORT_TOTAL = {"upper_k": (402, 406), "nominal_k": (452, 456)}
PESS_TOTAL = {"7.12": 750, "7.11": 808}
PIX = {"8thr": (166, 276), "2thr": (258, 431), "findings_2thr": (158, 263), "1thr": (396, 660)}
CEIL, WATCHDOG, HARD = 1200, 1620, 1800
OVERSHOOT = 50   # one in-flight raster past the budget check on the platform [est: largest box raster 73 s / f 1.5–2.5]

port = {k: (lo - BASE_IN_TOTALS, hi - BASE_IN_TOTALS) for k, (lo, hi) in PORT_TOTAL.items()}
rows = {}
for kk, (plo, phi) in port.items():
    for pk, (xlo, xhi) in PIX.items():
        lo, hi = BASE[0] + plo + xlo, BASE[1] + phi + xhi
        rows[f"{kk}|{pk}"] = {"total": (lo, hi), "margin_to_1200": (CEIL - hi, CEIL - lo), "fits_1200": hi <= CEIL,
                              "private_x2": (2 * lo, 2 * hi), "private_x2_fits_1200": 2 * hi <= CEIL}
pess = {}
for src, tot in PESS_TOTAL.items():
    for pk, (xlo, xhi) in PIX.items():
        pess[f"{src}|{pk}"] = {"total_hi": tot + xhi, "fits_1200": tot + xhi <= CEIL, "fits_watchdog": tot + xhi <= WATCHDOG}


def guarded(rest, budget):
    """PIXGBM capped by its in-phase ladder: budget + one in-flight raster."""
    return rest + budget + OVERSHOOT


guard = {b: {src: guarded(t, b) for src, t in PESS_TOTAL.items()} for b in (250, 300, 350)}
rec_max = CEIL - max(PESS_TOTAL.values()) - OVERSHOOT
res = {"port_net_excl_base": port, "pessimistic_port_net": {k: v - BASE_IN_TOTALS for k, v in PESS_TOTAL.items()},
       "combos": rows, "pessimistic": pess, "guarded_worst": guard, "pixgbm_budget_max_for_1200": rec_max,
       "ceiling": CEIL, "watchdog": WATCHDOG, "hard": HARD, "overshoot_assumed": OVERSHOOT}
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1))
for k, v in rows.items():
    print(k, v["total"], "margin", v["margin_to_1200"], v["fits_1200"], "x2", v["private_x2"])
for k, v in pess.items():
    print(k, v)
print("guard", guard, "max budget", rec_max)
