"""Сравнение вариантов morph_fast с полным точным прогоном full/<регион>.npz (те же stack_idx и порядок).
conf*/: на вычисленных строках — сырые 30 колонок (должны совпасть) и 15 rz (uniform и mix) — |Δrz|.
demli/: все 45 колонок — median/p99 |Δ|, доля в допуске 1e-3·(1+|x|), корреляция, совпадение NaN."""
import sys, json, numpy as np
res = {}
for f in sys.argv[1:]:
    reg = f.rsplit("/", 1)[1]
    A = np.load("full/" + reg); B = np.load(f)
    assert (A["stack_idx"] == B["stack_idx"]).all()
    names = list(A["names"]); c = B["computed"]
    X, Y = A["feats45"][c].astype(np.float64), B["feats45"][c].astype(np.float64)
    r = {"n_computed": int(c.sum()), "n_region": int(c.size)}
    def stats(x, y):
        both = np.isfinite(x) & np.isfinite(y); d = np.abs(x[both] - y[both])
        return dict(nan_match=float((np.isnan(x) == np.isnan(y)).mean()),
                    med=float(np.median(d)) if d.size else 0., p99=float(np.quantile(d, .99)) if d.size else 0.,
                    max=float(d.max()) if d.size else 0., frac_tol=float((d <= 1e-3 * (1 + np.abs(x[both]))).mean()) if d.size else 1.,
                    corr=float(np.corrcoef(x[both], y[both])[0, 1]) if both.sum() > 2 and x[both].std() > 0 and y[both].std() > 0 else None)
    if "rz_mix" in B.files:
        raw_ok = all(stats(X[:, k], Y[:, k])["frac_tol"] == 1.0 for k in range(30))
        r["raw_identical"] = raw_ok
        for tag, Z in (("uniform", Y[:, 30:]), ("mix", B["rz_mix"][c].astype(np.float64))):
            per = {names[30 + k]: stats(X[:, 30 + k], Z[:, k]) for k in range(15)}
            allp99 = [v["p99"] for v in per.values()]; allmed = [v["med"] for v in per.values()]
            r[f"rz_{tag}"] = dict(worst_p99=max(allp99), median_of_med=float(np.median(allmed)),
                                 worst_max=max(v["max"] for v in per.values()), per=per)
        print(f, r["n_computed"], "/", r["n_region"], "raw_identical", raw_ok,
              {t: (round(r[f"rz_{t}"]["median_of_med"], 4), round(r[f"rz_{t}"]["worst_p99"], 4)) for t in ("uniform", "mix")})
    else:
        per = {n: stats(X[:, k], Y[:, k]) for k, n in enumerate(names)}
        r["per"] = per
        print(f, "worst frac_tol", min(v["frac_tol"] for v in per.values()),
              "worst corr", min(v["corr"] for v in per.values() if v["corr"] is not None),
              "worst nan", min(v["nan_match"] for v in per.values()))
        for n, v in sorted(per.items(), key=lambda kv: kv[1]["frac_tol"])[:12]:
            print(f"  {n:24s} tol={v['frac_tol']:.3f} med={v['med']:.4g} p99={v['p99']:.4g} corr={v['corr'] if v['corr'] is None else round(v['corr'],4)} nan={v['nan_match']:.4f}")
    res[f] = r
json.dump(res, open("compare2.json", "w"), ensure_ascii=False, indent=1)
