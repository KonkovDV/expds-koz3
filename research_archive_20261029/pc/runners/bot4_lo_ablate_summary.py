"""Summarize bot4 large_objects ablation e2e_off_vs_*.json into CSV + SUMMARY.txt.

Usage: python bot4_lo_ablate_summary.py <JobDir> <comma-separated-variant-names>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load_cmp(job: Path, name: str):
    p = job / f"e2e_off_vs_{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def layer_row(data, markup: str):
    if not data:
        return None
    for r in data.get("rows") or []:
        if r.get("region") == "ALL" and r.get("markup") == markup:
            return r
    return None


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: bot4_lo_ablate_summary.py JobDir name1,name2,...", file=sys.stderr)
        return 2
    job = Path(sys.argv[1])
    names = [n for n in sys.argv[2].split(",") if n]
    layers = ("ALL", "li", "ae", "sp_or")

    first = None
    for n in names:
        if n == "off":
            continue
        d = load_cmp(job, n)
        if d:
            first = d
            break

    off_f1 = {}
    for L in layers:
        r = layer_row(first, L) if first else None
        off_f1[L] = (r or {}).get("f1_off")
    off_counts = (first or {}).get("counts_off") or {}

    rows_out = []
    rows_out.append({
        "variant": "off",
        "n": (layer_row(first, "ALL") or {}).get("n_off"),
        "ALL_F1": off_f1.get("ALL"),
        "li": off_f1.get("li"),
        "ae": off_f1.get("ae"),
        "sp_or": off_f1.get("sp_or"),
        "delta_ALL": 0.0,
        "delta_li": 0.0,
        "delta_ae": 0.0,
        "delta_sp_or": 0.0,
        "verdict": "baseline",
        "fail_reason": "",
        "count_deltas": "",
        "counts": json.dumps(off_counts, ensure_ascii=True),
        "counts_on": "",
    })

    for n in names:
        if n == "off":
            continue
        d = load_cmp(job, n)
        rec = {
            "variant": n,
            "fail_reason": "",
            "count_deltas": "",
            "counts": "",
            "counts_on": "",
        }
        if not d:
            rec["verdict"] = "no_score"
            rows_out.append(rec)
            continue
        for L, key in (("ALL", "ALL_F1"), ("li", "li"), ("ae", "ae"), ("sp_or", "sp_or")):
            r = layer_row(d, L) or {}
            rec[key] = r.get("f1_on")
            dkey = "delta_ALL" if L == "ALL" else ("delta_" + L)
            rec[dkey] = r.get("delta")
        r_all = layer_row(d, "ALL") or {}
        rec["n"] = r_all.get("n_on")
        d_all = float(rec.get("delta_ALL") or 0)
        drops = []
        for L in ("li", "ae", "sp_or"):
            dv = float(rec.get("delta_" + L) or 0)
            if dv < -0.01:
                drops.append("%s:%.5f" % (L, dv))
        if d_all >= 0 and not drops:
            rec["verdict"] = "PASS"
        else:
            rec["verdict"] = "FAIL"
            if d_all < 0:
                drops.insert(0, "ALL:%.5f" % d_all)
            rec["fail_reason"] = ";".join(drops)
        co = d.get("counts_off") or {}
        cn = d.get("counts_on") or {}
        keys = sorted(set(co) | set(cn))
        deltas = {k: cn.get(k, 0) - co.get(k, 0) for k in keys if cn.get(k, 0) != co.get(k, 0)}
        rec["count_deltas"] = json.dumps(deltas, ensure_ascii=True)
        rec["counts_on"] = json.dumps(cn, ensure_ascii=True)
        rows_out.append(rec)

    fields = [
        "variant", "n", "ALL_F1", "li", "ae", "sp_or",
        "delta_ALL", "delta_li", "delta_ae", "delta_sp_or",
        "verdict", "fail_reason", "count_deltas", "counts", "counts_on",
    ]
    with (job / "summary_layers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    with (job / "summary_classes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "markup_class", "count", "delta_vs_off"])
        total = sum(off_counts.values()) if off_counts else ""
        w.writerow(["off", "(all)", total, 0])
        for k, v in sorted(off_counts.items()):
            w.writerow(["off", k, v, 0])
        for n in names:
            if n == "off":
                continue
            d = load_cmp(job, n)
            if not d:
                continue
            cn = d.get("counts_on") or {}
            co = d.get("counts_off") or {}
            for k in sorted(set(co) | set(cn)):
                w.writerow([n, k, cn.get(k, 0), cn.get(k, 0) - co.get(k, 0)])

    lines = [
        "bot4 large_objects ablation",
        "sha_jobdir=%s" % job.name,
        "env=docker:pantum/n7q:20261024",
    ]
    for r in rows_out:
        lines.append(
            "%s: ALL=%s dALL=%s li=%s dli=%s ae=%s sp_or=%s verdict=%s %s"
            % (
                r.get("variant"),
                r.get("ALL_F1"),
                r.get("delta_ALL"),
                r.get("li"),
                r.get("delta_li"),
                r.get("ae"),
                r.get("sp_or"),
                r.get("verdict"),
                r.get("fail_reason") or "",
            )
        )
    (job / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary_ok", len(rows_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
