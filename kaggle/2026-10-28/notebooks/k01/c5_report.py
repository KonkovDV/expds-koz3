# ── Ячейка 6. AUC: общий, по фолдам, по семьям, бутстреп по семьям; OOF в /kaggle/working ───────
names = [s["name"] for s in CFG["models"]]
P = {n: OOF[n] for n in names}
P["ensemble"] = np.nanmean(np.stack([OOF[n] for n in names]), axis=0)
ok = lab & np.isfinite(P["ensemble"])
cand = kind == "cand"

def fam_table(p, mask):
    rows = []
    for f in sorted(set(family[mask])):
        mm = mask & (family == f)
        rows.append({"family": f, "fold": int(fold[mm][0]), "n": int(mm.sum()), "n_povr": int((y[mm] == 1).sum()),
                     "n_tsel": int((y[mm] == 0).sum()), "auc": auc(y[mm], p[mm])})
    return rows

def boot_auc(p, mask, n=CFG["bootstrap"], seed=0):
    fams = np.array(sorted(set(family[mask]))); rng = np.random.default_rng(seed); vals = []
    by = {f: np.flatnonzero(mask & (family == f)) for f in fams}
    for _ in range(n):
        ii = np.concatenate([by[f] for f in rng.choice(fams, len(fams), replace=True)])
        a = auc(y[ii], p[ii])
        if a is not None: vals.append(a)
    return {"p5": float(np.percentile(vals, 5)), "p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95)),
            "P_auc_ge_0.9": float(np.mean(np.array(vals) >= 0.9)), "n": len(vals)} if vals else None

REP = {"npz": str(NPZ), "cfg": CFG, "local_test": LOCAL_TEST, "folds": [sorted(f) for f in FOLDS],
       "per_fold_log": LOG, "models": {}}
for n, p in P.items():
    ft = fam_table(p, ok)
    fa = [r["auc"] for r in ft if r["auc"] is not None]
    REP["models"][n] = {
        "auc_oof_all": auc(y[ok], p[ok]),
        "auc_oof_cand": auc(y[ok & cand], p[ok & cand]),
        "auc_oof_gt": auc(y[ok & ~cand], p[ok & ~cand]),
        "auc_by_fold": {int(k): auc(y[ok & (fold == k)], p[ok & (fold == k)]) for k in sorted(set(fold[ok]))},
        "auc_family_macro": float(np.mean(fa)) if fa else None,
        "families_with_both_classes": len(fa),
        "by_family": ft,
    }
E = P["ensemble"]
REP["models"]["ensemble"]["bootstrap_by_family"] = boot_auc(E, ok)
thr = CFG["apply_thr"]
pred = (E >= thr).astype(int)
cur = np.array([1 if c == "kurgany_povrezhdennye" else 0 for c in cand_class])
mc = ok & cand & (cand_class != "")
REP["decision_at_thr"] = {
    "thr": thr, "why_thr": CFG["why_thr"],
    "acc_all_labeled": float((pred[ok] == y[ok]).mean()),
    "cand_labeled": int(mc.sum()),
    "cand_wrong_subclass_now": int((cur[mc] != y[mc]).sum()),
    "fixed_by_clf": int(((cur != y) & (pred == y))[mc].sum()),
    "broken_by_clf": int(((cur == y) & (pred != y))[mc].sum()),
}
a_ens = REP["models"]["ensemble"]["auc_oof_all"]
REP["target_auc_0.9_met"] = bool(a_ens is not None and a_ens >= 0.9)
REP["elapsed_sec_oof"] = round(time.time() - T0, 1)

(WORK / "kurgan_subclass_report.json").write_text(json.dumps(REP, ensure_ascii=False, indent=1), encoding="utf-8")
with open(WORK / "kurgan_subclass_auc_by_family.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["model", "family", "fold", "n", "n_povr", "n_tsel", "auc"])
    for n in P:
        for r in REP["models"][n]["by_family"]:
            w.writerow([n, r["family"], r["fold"], r["n"], r["n_povr"], r["n_tsel"], "" if r["auc"] is None else round(r["auc"], 4)])
with open(WORK / "kurgan_subclass_oof.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["stack_idx", "region", "family", "fold", "kind", "cand_class", "y", "p_povr", *[f"p_{n}" for n in names]])
    for i in range(len(y)):
        f6 = lambda v: "" if not np.isfinite(v) else round(float(v), 6)
        w.writerow([int(stack_idx[i]), region[i], family[i], int(fold[i]), kind[i], cand_class[i], int(y[i]), f6(E[i]), *[f6(P[n][i]) for n in names]])
summ = {n: {k: REP["models"][n][k] for k in ("auc_oof_all", "auc_oof_cand", "auc_family_macro", "families_with_both_classes")} for n in P}
print(json.dumps({"summary": summ, "bootstrap_ens": REP["models"]["ensemble"]["bootstrap_by_family"],
                  "decision": REP["decision_at_thr"], "AUC>=0.9": REP["target_auc_0.9_met"]}, ensure_ascii=False, indent=1))
print("→ /kaggle/working/kurgan_subclass_oof.csv, kurgan_subclass_report.json, kurgan_subclass_auc_by_family.csv", flush=True)
