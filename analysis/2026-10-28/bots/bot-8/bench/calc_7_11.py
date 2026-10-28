"""§7.11: fit per-unit CPU costs from bench/cpu_a/*.json and extrapolate to the public set (BUDGET §7.0/§7.10)."""
import json, glob, numpy as np, os
H = os.path.dirname(os.path.abspath(__file__))
S = {json.load(open(f))["site"]: json.load(open(f)) for f in sorted(glob.glob(f"{H}/cpu_a/*.json"))}
rows = []
for s, d in S.items():
    i = d["dtm_info"]; c = d["median_cpu_s"]; w = d["median_s"]
    rows.append(dict(site=s, Mpt=i["n_points_M"], Mcell=i["cells_M"], Mpx=d["net_px_M"],
                     las_warm=c["io_las_warm"], las_cold_wall=w["io_las_cold"], laz=c["io_laz_cold"],
                     dtm=c["dtm_total_warm"], dtm_wall=w["dtm_total_warm"], r9=c["relief9_cpu"], r12=c["relief12_cpu"],
                     tile1=c["tiling_host_stub_1view"], blend=c["ensemble_blend4"],
                     pmf_extra_wall=(w.get("dtm_forced_pmf_path", np.nan) - w["dtm_total_warm"]),
                     rss=max(d["peak_rss_mb_max"].values()), rss_dtm=d["peak_rss_mb_max"]["dtm_total_warm"]))
def fit(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    return [float(v) for v in b], float(np.sqrt((r ** 2).mean()))
R = {}
R["las_s_per_Mpt"], R["las_rmse"] = fit([[r["Mpt"]] for r in rows], [r["las_warm"] for r in rows])
R["laz_s_per_Mpt"], _ = fit([[r["Mpt"]] for r in rows], [r["laz"] for r in rows])
R["dtm_coef_Mpt_Mcell"], R["dtm_rmse"] = fit([[r["Mpt"], r["Mcell"]] for r in rows], [r["dtm"] for r in rows])
R["r9_coef_1_Mpx"], R["r9_rmse"] = fit([[1, r["Mpx"]] for r in rows], [r["r9"] for r in rows])
R["r12_coef_1_Mpx"], R["r12_rmse"] = fit([[1, r["Mpx"]] for r in rows], [r["r12"] for r in rows])
R["tile1_s_per_Mpx"], _ = fit([[r["Mpx"]] for r in rows], [r["tile1"] for r in rows])
R["blend_s_per_Mpx"], _ = fit([[r["Mpx"]] for r in rows], [r["blend"] for r in rows])
R["pmf_extra_wall_s_per_Mcell_021"] = S["021_Новенькое"]["median_s"]["dtm_forced_pmf_path"] / S["021_Новенькое"]["dtm_info"]["cells_M"] - S["021_Новенькое"]["median_s"]["dtm_total_warm"] / S["021_Новенькое"]["dtm_info"]["cells_M"]
# public set: 2.46 GB LAS (§7.0) at measured 30–34 B/pt; 13 files; area: BUDGET nominal 30 Mpx (10–75) at 0.5 m → cells ×4 at 0.25 m
N_FILES = 13
pub = {}
for name, mpt, mpx in [("low", 2460 / 34, 10), ("nom", 2460 / 34, 30), ("high", 2460 / 30, 75)]:
    mc = 4 * mpx
    a, b = R["dtm_coef_Mpt_Mcell"]
    t = dict(Mpt=mpt, Mpx=mpx, Mcell=mc,
             las_read=R["las_s_per_Mpt"][0] * mpt,
             dtm=a * mpt + b * mc,
             relief9=R["r9_coef_1_Mpx"][0] * N_FILES + R["r9_coef_1_Mpx"][1] * mpx,
             relief12=R["r12_coef_1_Mpx"][0] * N_FILES + R["r12_coef_1_Mpx"][1] * mpx,
             tiling_host_4models=4 * R["tile1_s_per_Mpx"][0] * mpx,
             blend=R["blend_s_per_Mpx"][0] * mpx)
    t["pmf_worst_wall"] = R["pmf_extra_wall_s_per_Mcell_021"] * mc
    t["cpu_no_relief"] = t["dtm"] + t["tiling_host_4models"] + t["blend"]
    t["cpu_with_relief_cpu"] = t["cpu_no_relief"] + t["relief9"] + t["relief12"]
    for k in (1.5, 2.0):   # per-thread speed i9-14900KS vs box Xeon [оц, as in §7.0]
        t[f"plat_no_relief_k{k}"] = t["cpu_no_relief"] / k
        t[f"plat_with_relief_cpu_k{k}"] = t["cpu_with_relief_cpu"] / k
    pub[name] = {k: round(v, 2) for k, v in t.items()}
R["public"] = pub
R["rows"] = rows
json.dump(R, open(f"{H}/calc_7_11.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "rows"}, ensure_ascii=False, indent=1))
for r in rows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
