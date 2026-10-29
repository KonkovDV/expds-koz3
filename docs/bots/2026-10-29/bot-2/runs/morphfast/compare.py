"""сравнение 45 колонок morph_fast с parquet задачи 10 (e918a3b) на тех же stack_idx."""
import sys, json, numpy as np, pyarrow.parquet as pq, pyarrow.dataset as ds, pyarrow.compute as pc
P = "/workspace/koz3_shared/bot-2/morph/morph_li_kurgan.parquet"
res = {}
for f in sys.argv[1:]:
    d = np.load(f); s = d["stack_idx"]; X = d["feats45"]; names = list(d["names"]); comp = d["computed"]
    t = ds.dataset(P).to_table(columns=["stack_idx"] + names, filter=pc.field("stack_idx").isin(s.tolist()))
    pos = {int(v): i for i, v in enumerate(t["stack_idx"].to_numpy())}
    ii = np.array([pos[int(v)] for v in s])
    r = {}
    for k, n in enumerate(names):
        ref = t[n].to_numpy(zero_copy_only=False)[ii].astype(np.float64); x = X[:, k].astype(np.float64)
        ref, x = ref[comp], x[comp]
        nm = np.isnan(ref) == np.isnan(x); both = np.isfinite(ref) & np.isfinite(x)
        dif = np.abs(ref[both] - x[both]); tol = 1e-3 * (1 + np.abs(ref[both]))
        r[n] = dict(nan_match=float(nm.mean()), max_abs=float(dif.max()) if dif.size else 0.0,
                    frac_tol=float((dif <= tol).mean()) if dif.size else 1.0,
                    p99=float(np.quantile(dif, .99)) if dif.size else 0.0,
                    corr=float(np.corrcoef(ref[both], x[both])[0, 1]) if both.sum() > 2 and ref[both].std() > 0 else None)
    bad = {n: v for n, v in r.items() if v["nan_match"] < 1 or v["frac_tol"] < 0.999}
    res[f] = dict(n=int(comp.sum()), worst_frac_tol=min(v["frac_tol"] for v in r.values()),
                  worst_nan_match=min(v["nan_match"] for v in r.values()), bad=bad, all=r)
    print(f, "n", int(comp.sum()), "worst frac_tol", res[f]["worst_frac_tol"], "worst nan", res[f]["worst_nan_match"], "bad:", {n: (round(v["nan_match"],4), round(v["frac_tol"],4), round(v["max_abs"],4), v["corr"] and round(v["corr"],4)) for n, v in bad.items()})
if len(sys.argv) > 1 and "--json" in sys.argv[-1:]:
    pass
json.dump(res, open(sys.argv[1].rsplit("/", 1)[0] + "/compare.json", "w"), ensure_ascii=False, indent=1)
