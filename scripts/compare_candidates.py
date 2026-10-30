#!/usr/bin/env python3
"""Gate: сравнить N кандидатов (папки mock e2e) с базой v2.

На кандидата: F1 ALL/li/sp_or/ae в scope=global и scope=layer (expds.metric),
Δ vs v2 (ALL по умолчанию 0.3125; per-layer из --v2 папки или констант mock),
правило «ни один слой не падает >0.01 vs baseline» (PASS/FAIL),
счётчик cross-layer li↔ae TP на классах kurgany_* (из global pairs).

Примеры:

    python scripts/compare_candidates.py \\
        --gt F:/expds3/mock_val/gt.geojson \\
        --v2 C:/Sloi/pc_jobs/v3e2e/out_v3fix2/res_v2 \\
        --out notes/compare/ \\
        --workers 4 \\
        C:/Sloi/pc_jobs/v3e2e/out_v3fix2/res_v3 \\
        C:/Sloi/pc_jobs/bot2_v3_hotfix_1ec2d7c/out_B \\
        C:/Sloi/pc_jobs/bot5_v4_off_50f6f27

    python scripts/compare_candidates.py ... --experiments-row --experiments-file EXPERIMENTS.csv

    # --out optional: %KOZ3_OUT%/compare или ./compare_out; --env docker,host,... per candidate
    # candidate may be path, label=path, or path:env / label=path:env (env in {docker,host,unknown})
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LAYERS = ("li", "sp_or", "ae")
SCOPES = ("global", "layer")
# документированная база mock v2 (e2e_score) — если --v2 не передан
DEFAULT_BASELINE = {
    "ALL": 0.3125,
    "li": 0.336983,
    "sp_or": 0.242915,
    "ae": 0.095556,
}
LAYER_DROP_MAX = 0.01
MSK = ZoneInfo("Europe/Moscow")

VALID_ENVS = ("docker", "host", "unknown")


def resolve_out(out_arg: str | None) -> Path:
    """--out optional: KOZ3_OUT/compare if set, else ./compare_out. Print resolved path."""
    if out_arg:
        out = Path(out_arg)
    else:
        koz3 = os.environ.get("KOZ3_OUT")
        if koz3:
            out = Path(koz3) / "compare"
        else:
            out = Path("compare_out")
    out = out.resolve()
    print(f"out={out}")
    return out


def parse_candidate_spec(spec: str) -> tuple[str | None, Path, str]:
    """Parse path | label=path | path:env | label=path:env.

    env must be docker|host|unknown when taken from a trailing :suffix
    (avoids eating Windows drive letters like C:\\...).
    Returns (label_or_None, path, env).
    """
    label: str | None = None
    rest = spec
    if "=" in spec:
        label, rest = spec.split("=", 1)
        label = label.strip() or None
        rest = rest.strip()
    env = "unknown"
    # trailing :env only when suffix is a known env token
    if ":" in rest:
        left, right = rest.rsplit(":", 1)
        if right.strip().lower() in VALID_ENVS and left.strip():
            # reject bare drive letter "C" as path
            if not (len(left.strip()) == 1 and left.strip().isalpha()):
                env = right.strip().lower()
                rest = left.strip()
    return label, Path(rest), env


# предпочтение layout e2e: out_B (flags-off) > out_A > out > res_* > корень
_GEOJSON_NAMES = ("result.geojson", "submission.geojson")
_SUBDIR_PREF = ("out_B", "out_b", "out_A", "out_a", "out_off", "out_on", "out", "res_v2", "res_v3", "res")


def find_geojson(folder: Path) -> Path:
    """Найти result/submission.geojson в папке кандидата (типичные e2e layouts)."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"нет папки кандидата: {folder}")
    hits: list[tuple[int, Path]] = []
    for name in _GEOJSON_NAMES:
        direct = folder / name
        if direct.is_file():
            hits.append((0, direct))
        for i, sub in enumerate(_SUBDIR_PREF):
            p = folder / sub / name
            if p.is_file():
                hits.append((10 + i, p))
        # один уровень вглубь: любые out_*/res_*/
        for child in sorted(folder.iterdir()) if folder.is_dir() else []:
            if not child.is_dir():
                continue
            p = child / name
            if p.is_file():
                rank = 10 + (_SUBDIR_PREF.index(child.name) if child.name in _SUBDIR_PREF else 50)
                hits.append((rank, p))
    if not hits:
        # рекурсивно, но ограничим глубину 3
        for name in _GEOJSON_NAMES:
            for p in folder.rglob(name):
                if p.is_file() and len(p.relative_to(folder).parts) <= 3:
                    hits.append((100, p))
                    break
    if not hits:
        raise FileNotFoundError(f"нет result/submission.geojson в {folder}")
    hits.sort(key=lambda t: (t[0], str(t[1])))
    return hits[0][1]


def _load_fc(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _filter_layer(fc: dict, layer: str | None) -> dict:
    """Оставить объекты с layer_key == layer; layer=None → без фильтра (ALL)."""
    from expds.metric import layer_key

    if layer is None:
        return fc
    feats = [f for f in fc["features"]
             if layer_key(f.get("properties") or {}) == layer]
    return {**fc, "features": feats}


def layer_scores(pred: dict, gt: dict, metric_path: str | None) -> dict[str, dict[str, float]]:
    """F1 для ALL/li/sp_or/ae × global/layer. Reuse build_curves + score_at (не F1Calculator вручную)."""
    from expds import metric
    from expds.metric import build_curves, score_at

    out: dict[str, dict[str, float]] = {}
    for scope in SCOPES:
        row: dict[str, float] = {}
        for label, markup in (("ALL", None), ("li", "li"), ("sp_or", "sp_or"), ("ae", "ae")):
            metric._UTM_GEOM_CACHE.clear()
            metric._SCOPE_WARNED.clear()
            p = _filter_layer(pred, markup)
            g = _filter_layer(gt, markup)
            if not g["features"]:
                row[label] = float("nan")
                continue
            curves = build_curves(p, g, metric_path, scope=scope)
            f1, _ = score_at(curves, {})
            row[label] = float(f1)
        out[scope] = row
    return out


def count_liae_kurgany(pred: dict, gt: dict, metric_path: str | None) -> int:
    """Число cross-layer TP li↔ae на классах kurgany_* (global pairs)."""
    from expds import metric
    from expds.metric import build_curves_with_pairs

    metric._UTM_GEOM_CACHE.clear()
    metric._SCOPE_WARNED.clear()
    _, pairs = build_curves_with_pairs(pred, gt, metric_path, scope="global")
    n = 0
    for p in pairs:
        if not p.get("cross_layer"):
            continue
        layers = {p.get("pred_layer"), p.get("gt_layer")}
        if layers != {"li", "ae"}:
            continue
        cls = str(p.get("class") or "")
        if cls.startswith("kurgany_"):
            n += 1
    return n


def evaluate_rule(scores_global: dict[str, float], baseline: dict[str, float],
                  drop_max: float = LAYER_DROP_MAX) -> tuple[str, list[str]]:
    """PASS/FAIL: ни один слой li/sp_or/ae не падает больше drop_max vs baseline."""
    failed: list[str] = []
    for layer in LAYERS:
        b = baseline.get(layer)
        s = scores_global.get(layer)
        if b is None or s is None:
            continue
        if s != s or b != b:  # NaN
            continue
        # «падает больше чем на 0.01»: drop = baseline − score; ровно 0.01 ещё PASS
        drop = round(b - s, 6)
        if drop > drop_max:
            failed.append(layer)
    return ("PASS" if not failed else "FAIL"), failed


def evaluate_candidate(
    folder: Path,
    gt: dict,
    baseline: dict[str, float],
    metric_path: str | None,
    label: str | None = None,
    env: str = "unknown",
) -> dict:
    geo = find_geojson(folder)
    pred = _load_fc(geo)
    scores = layer_scores(pred, gt, metric_path)
    liae = count_liae_kurgany(pred, gt, metric_path)
    g = scores["global"]
    deltas = {k: (g[k] - baseline[k]) if k in baseline and g[k] == g[k] else float("nan")
              for k in ("ALL",) + LAYERS if k in g}
    rule, failed = evaluate_rule(g, baseline)
    name = label or folder.name
    env_s = (env or "unknown").strip().lower()
    if env_s not in VALID_ENVS:
        env_s = "unknown"
    return {
        "name": name,
        "folder": str(folder),
        "geojson": str(geo),
        "env": env_s,
        "scores": scores,
        "delta_vs_v2": deltas,
        "rule": rule,
        "rule_failed_layers": failed,
        "liae_kurgany_cross_tp": liae,
    }


def _fmt(x: float, digits: int = 6) -> str:
    if x != x:
        return "nan"
    return f"{x:.{digits}f}"


def write_md(results: list[dict], baseline: dict[str, float], out: Path) -> None:
    lines = [
        "# compare_candidates",
        "",
        f"baseline ALL={_fmt(baseline.get('ALL', float('nan')))} "
        f"li={_fmt(baseline.get('li', float('nan')))} "
        f"sp_or={_fmt(baseline.get('sp_or', float('nan')))} "
        f"ae={_fmt(baseline.get('ae', float('nan')))}",
        f"rule: no layer drops by more than {LAYER_DROP_MAX} vs baseline",
        "",
        "| candidate | env | scope | ALL | li | sp_or | ae | ΔALL | rule | li↔ae kurgany |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        env_s = r.get("env", "unknown")
        for scope in SCOPES:
            s = r["scores"][scope]
            d = r["delta_vs_v2"] if scope == "global" else {}
            delta_s = _fmt(d["ALL"]) if scope == "global" and "ALL" in d else "—"
            rule_s = r["rule"] if scope == "global" else "—"
            if scope == "global" and r["rule"] == "FAIL":
                rule_s = f"FAIL({','.join(r['rule_failed_layers'])})"
            liae_s = str(r["liae_kurgany_cross_tp"]) if scope == "global" else "—"
            lines.append(
                f"| {r['name']} | {env_s} | {scope} | {_fmt(s['ALL'])} | {_fmt(s['li'])} | "
                f"{_fmt(s['sp_or'])} | {_fmt(s['ae'])} | {delta_s} | {rule_s} | {liae_s} |"
            )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def write_csv(results: list[dict], out: Path) -> None:
    fields = [
        "name", "env", "scope", "ALL", "li", "sp_or", "ae",
        "delta_ALL", "delta_li", "delta_sp_or", "delta_ae",
        "rule", "rule_failed_layers", "liae_kurgany_cross_tp", "geojson",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            for scope in SCOPES:
                s = r["scores"][scope]
                d = r["delta_vs_v2"] if scope == "global" else {}
                w.writerow({
                    "name": r["name"],
                    "env": r.get("env", "unknown"),
                    "scope": scope,
                    "ALL": s["ALL"],
                    "li": s["li"],
                    "sp_or": s["sp_or"],
                    "ae": s["ae"],
                    "delta_ALL": d.get("ALL", ""),
                    "delta_li": d.get("li", ""),
                    "delta_sp_or": d.get("sp_or", ""),
                    "delta_ae": d.get("ae", ""),
                    "rule": r["rule"] if scope == "global" else "",
                    "rule_failed_layers": ",".join(r["rule_failed_layers"]) if scope == "global" else "",
                    "liae_kurgany_cross_tp": r["liae_kurgany_cross_tp"] if scope == "global" else "",
                    "geojson": r["geojson"],
                })


def experiments_line(r: dict, branch: str, sha: str, bot: str = "bot7") -> str:
    """Одна строка EXPERIMENTS.csv (без заголовка)."""
    g = r["scores"]["global"]
    d = r["delta_vs_v2"].get("ALL", float("nan"))
    delta_s = _fmt(d) if d == d else ""
    now = datetime.now(MSK).strftime("%Y-%m-%d %H:%M")
    notes = (
        f"rule={r['rule']}"
        + (f" fail={','.join(r['rule_failed_layers'])}" if r["rule_failed_layers"] else "")
        + f"; liae_kurgany={r['liae_kurgany_cross_tp']}"
        + f"; layer_ALL={_fmt(r['scores']['layer']['ALL'])}"
        + f"; env={r.get('env', 'unknown')}"
    )
    return ",".join([
        now,
        bot,
        branch,
        sha,
        f"compare_candidates:{r['name']}",
        "mock_e2e_gate",
        _fmt(g["ALL"]),
        _fmt(g["li"]),
        _fmt(g["sp_or"]),
        _fmt(g["ae"]),
        delta_s,
        r["rule"],
        notes.replace(",", ";"),
    ])


def compute_baseline(v2_folder: Path | None, gt: dict, metric_path: str | None,
                     baseline_all: float) -> dict[str, float]:
    base = dict(DEFAULT_BASELINE)
    base["ALL"] = baseline_all
    if v2_folder is None:
        return base
    geo = find_geojson(v2_folder)
    pred = _load_fc(geo)
    scores = layer_scores(pred, gt, metric_path)["global"]
    for k in ("ALL",) + LAYERS:
        if scores.get(k) == scores.get(k):  # not NaN
            base[k] = scores[k]
    return base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "candidates", nargs="+",
        help="папки mock e2e (path | label=path | path:env | label=path:env; env=docker|host|unknown)",
    )
    ap.add_argument("--gt", required=True, help="эталон mock (FeatureCollection или папка с gt.geojson)")
    ap.add_argument("--v2", default=None, help="папка v2 baseline для per-layer чисел")
    ap.add_argument("--baseline-all", type=float, default=0.3125, help="база ALL (по умолчанию 0.3125)")
    ap.add_argument("--labels", default=None, help="имена кандидатов через запятую (порядок = candidates)")
    ap.add_argument(
        "--env", default=None,
        help="среды кандидатов через запятую (docker|host|unknown; порядок = candidates; default unknown)",
    )
    ap.add_argument("--metric", default=None, help="materials/compute_metrics.py")
    ap.add_argument(
        "--out", default=None,
        help="каталог для compare.md/csv/summary.json; default %%KOZ3_OUT%%/compare или ./compare_out",
    )
    ap.add_argument("--workers", type=int, default=4, help="потоки CPU (cap 4)")
    ap.add_argument("--experiments-row", action="store_true", help="печатать строки EXPERIMENTS.csv")
    ap.add_argument("--experiments-file", default=None, help="дописать строки в этот CSV")
    ap.add_argument("--branch", default="", help="для EXPERIMENTS строки")
    ap.add_argument("--sha", default="", help="для EXPERIMENTS строки")
    args = ap.parse_args(argv)

    workers = max(1, min(int(args.workers), 4))
    out = resolve_out(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt_path = Path(args.gt)
    if gt_path.is_dir():
        for name in ("gt.geojson", "gt_train_v2.geojson", "result.geojson"):
            cand = gt_path / name
            if cand.is_file():
                gt_path = cand
                break
        else:
            # mock_val layout: иногда разметка в подпапках — ищем любой *gt*.geojson
            found = list(gt_path.rglob("gt*.geojson"))[:1]
            if not found:
                raise SystemExit(f"--gt папка без gt.geojson: {args.gt}")
            gt_path = found[0]
    gt = _load_fc(gt_path)
    metric_path = args.metric or str(ROOT / "materials" / "compute_metrics.py")

    baseline = compute_baseline(
        Path(args.v2) if args.v2 else None, gt, metric_path, args.baseline_all
    )

    labels_cli = [x.strip() for x in args.labels.split(",")] if args.labels else None
    envs_cli = [x.strip().lower() for x in args.env.split(",")] if args.env else None
    if labels_cli is not None and len(labels_cli) != len(args.candidates):
        raise SystemExit("--labels: число имён должно совпадать с числом candidates")
    if envs_cli is not None and len(envs_cli) != len(args.candidates):
        raise SystemExit("--env: число значений должно совпадать с числом candidates")

    jobs = []
    for i, spec in enumerate(args.candidates):
        lab_s, folder, env_s = parse_candidate_spec(spec)
        if labels_cli is not None:
            lab_s = labels_cli[i]
        if envs_cli is not None:
            env_s = envs_cli[i] if envs_cli[i] in VALID_ENVS else "unknown"
        jobs.append((folder, lab_s, env_s))

    results: list[dict] = [None] * len(jobs)  # type: ignore[list-item]

    def _one(idx_folder_label_env):
        idx, folder, lab, env = idx_folder_label_env
        return idx, evaluate_candidate(folder, gt, baseline, metric_path, label=lab, env=env)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, (i, f, lab, env)) for i, (f, lab, env) in enumerate(jobs)]
        for fut in as_completed(futs):
            idx, res = fut.result()
            results[idx] = res

    write_md(results, baseline, out / "compare.md")
    write_csv(results, out / "compare.csv")
    (out / "summary.json").write_text(
        json.dumps({"baseline": baseline, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    for r in results:
        line = experiments_line(r, args.branch or "bots/bot7-compare-candidates", args.sha or "unknown")
        lines.append(line)
        if args.experiments_row:
            print(line)
    if args.experiments_file:
        ep = Path(args.experiments_file)
        with ep.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    print(f"wrote {out / 'compare.md'} {out / 'compare.csv'} ({len(results)} candidates, workers={workers})")
    for r in results:
        g = r["scores"]["global"]
        print(
            f"  {r['name']}: env={r.get('env', 'unknown')} ALL={_fmt(g['ALL'])} "
            f"Δ={_fmt(r['delta_vs_v2'].get('ALL', float('nan')))} "
            f"rule={r['rule']} liae={r['liae_kurgany_cross_tp']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
