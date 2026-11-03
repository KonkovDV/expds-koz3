#!/usr/bin/env python3
"""KOZ3 · bot-W4 / CPU: honest-10 gate for [large_objects] via scripts/large_objects_honest.py.

Slug: elenakonkova/koz3-w4-large-gate
Resource: CPU only (enable_gpu/tpu false). No LB submit.

Tip: bots/bot4-large-objects @ 33791bcc (cloud PR #18 runner).

What xfer-s000..s009 contain (figured out from manifests + rebuild_train_tree):
  s000  — small_files.tar.zst: per-region labels (*.geojson), UTM.json, extras
  s001–s009 — large raster/blob shards (f#### → original relative paths under train tree)
  Together they rebuild train-expds-3 (labels + rasters). They do NOT ship fam0 OOF
  jsonl (data protocol — OOF cache is not in git or xfer).

This kernel produces the runner inputs as follows:
  GT       — koz3-gt-train-v2 (attached) preferred; else consolidate labels from rebuilt tree
  fam0     — attached *fam0* / *fam* dataset of *.jsonl, OR split from attached stack*.geojson,
             OR LARGE_GATE_SMOKE=1 synthetic fixture (schema-only; NOT stack-A numbers)
  clusters — scripts/family_clusters.py over fam0+GT

Then runs:
  python scripts/large_objects_honest.py --fam … --gt … --clusters … --metric … --out …

Success: exit 0 and JSON with off / area_tau / fixc_nested
  (score, precision, recall, delta_*, size_recall, recovered_gt_gt5k).
Expected full-stack A: off.score ≈ 0.314615 ± 0.002; wall 30–90 min / 1 core.
Lease: slot:acc1/cpu → bot-W4-gate (see README).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

T0 = time.time()
OUT = Path(os.environ.get("LARGE_GATE_OUT", "/kaggle/working/large_gate"))
ROOT = Path("/kaggle/working")
# When the script is the only uploaded code_file, sibling dirs may live next to it
# after Kaggle unpacks the kernel directory; also support local dry-run layout.
HERE = Path(__file__).resolve().parent


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def write_status(state: str, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": state,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - T0, 1),
        **extra,
    }
    (OUT / "STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [state, f"ts={payload['ts']}", f"elapsed_s={payload['elapsed_s']}"]
    for k, v in extra.items():
        lines.append(f"{k}={v}")
    (OUT / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_inputs() -> dict[str, Path]:
    root = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
    found: dict[str, Path] = {}
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if p.is_dir():
                found[p.name.lower()] = p
    return found


def find_named(inputs: dict[str, Path], *needles: str) -> Path | None:
    for n in needles:
        nl = n.lower()
        for k, p in inputs.items():
            if nl in k:
                return p
    return None


def ensure_code_layout() -> Path:
    """Return repo-like ROOT that contains scripts/, expds/, materials/, configs/."""
    candidates = [
        HERE,
        ROOT,
        Path("/kaggle/working"),
        Path("."),
    ]
    # If kernel only uploaded the .py, code may be under an attached snap dataset.
    inputs = find_inputs()
    snap = find_named(inputs, "bot8-snap", "bot4-snap", "large-gate-code", "expds-koz3")
    if snap is not None:
        for sub in (snap / "expds-koz3", snap):
            candidates.append(sub)

    for c in candidates:
        if (c / "scripts" / "large_objects_honest.py").is_file() and (c / "expds").is_dir():
            return c
    raise FileNotFoundError(
        "vendored tip not found: need scripts/large_objects_honest.py + expds/ "
        "next to the kernel or in an attached code dataset"
    )


def materialize_gt(inputs: dict[str, Path], dest: Path, train_root: Path | None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    gt_ds = find_named(inputs, "koz3-gt-train-v2", "gt-train-v2", "gt_train_v2")
    if gt_ds is not None:
        for cand in (
            gt_ds / "gt_train_v2.geojson",
            gt_ds / "gt_train.geojson",
            *sorted(gt_ds.glob("*.geojson")),
        ):
            if cand.is_file():
                shutil.copy2(cand, dest)
                log("GT from dataset", cand, "→", dest)
                return dest
    # Fallback: consolidate ObjectPoly/Finds from rebuilt train tree labels
    if train_root is not None and train_root.is_dir():
        feats = []
        for gj in train_root.rglob("*.geojson"):
            rel = str(gj.relative_to(train_root)).replace("\\", "/")
            if "разметка" not in rel and "markup" not in rel.lower():
                # still accept any geojson under region dirs
                pass
            try:
                data = json.loads(gj.read_text(encoding="utf-8"))
            except Exception:
                continue
            for f in data.get("features") or []:
                props = dict(f.get("properties") or {})
                if "region_name" not in props:
                    # infer from top folder
                    top = rel.split("/", 1)[0]
                    props["region_name"] = top
                    f = dict(f)
                    f["properties"] = props
                feats.append(f)
        if feats:
            dest.write_text(
                json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
                encoding="utf-8",
            )
            log("GT consolidated from train tree", len(feats), "features →", dest)
            return dest
    raise FileNotFoundError("GT missing: attach dmitrijkv/koz3-gt-train-v2 or rebuild train labels")


def find_fam_dir(inputs: dict[str, Path]) -> Path | None:
    # Direct fam0 directory dataset
    for k, p in inputs.items():
        if "fam0" in k or k.endswith("-fam") or "fam-oof" in k or "stack-a-fam" in k:
            if any(p.glob("*.jsonl")):
                return p
            for sub in p.rglob("*"):
                if sub.is_dir() and any(sub.glob("*.jsonl")) and (sub / "_counts.json").exists():
                    return sub
                if sub.is_dir() and len(list(sub.glob("*.jsonl"))) >= 3:
                    return sub
    # Generic: any input with many family jsonl
    for p in inputs.values():
        for sub in [p, *p.glob("*"), *p.glob("*/*")]:
            if not sub.is_dir():
                continue
            n = len(list(sub.glob("*.jsonl")))
            if n >= 10:
                return sub
    return None


def find_stack_geojson(inputs: dict[str, Path]) -> Path | None:
    for p in inputs.values():
        for cand in list(p.glob("**/stack*.geojson")) + list(p.glob("**/*dump*stack*.geojson")):
            if cand.is_file() and cand.stat().st_size > 1000:
                return cand
    return None


def build_smoke_fam(dest: Path, gt_path: Path) -> Path:
    """Tiny multi-family fixture so the runner contract can be exercised without OOF cache."""
    dest.mkdir(parents=True, exist_ok=True)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    # pick up to 6 GT polygons of class gorodishcha / fortifikatsii from distinct families
    from collections import defaultdict

    by_fam: dict[str, list] = defaultdict(list)
    for f in gt.get("features") or []:
        props = f.get("properties") or {}
        cls = props.get("class_name")
        if cls not in ("gorodishcha", "fortifikatsii"):
            continue
        geom = f.get("geometry") or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        reg = str(props.get("region_name") or "default_source")
        fam = reg.split("_")[0] if "_" in reg else reg
        # crude family: first token before digit run — naming.site_family later
        by_fam[reg].append(f)
        if sum(1 for v in by_fam.values() if v) >= 6 and len(by_fam) >= 3:
            break
    # If GT too sparse, synthesize absolute squares (EPSG:3857)
    if len(by_fam) < 3:
        by_fam.clear()
        bases = {
            "SmokeA_r1": (4180000.0, 7510000.0),
            "SmokeB_r1": (4190000.0, 7520000.0),
            "SmokeC_r1": (4200000.0, 7530000.0),
        }
        feats = []
        for reg, (x0, y0) in bases.items():
            g = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x0, y0], [x0 + 120, y0], [x0 + 120, y0 + 120], [x0, y0 + 120], [x0, y0]]],
                },
                "properties": {
                    "region_name": reg,
                    "class_name": "gorodishcha",
                    "markup_type": "ae",
                    "crs": "EPSG:3857",
                },
            }
            feats.append(g)
            by_fam[reg] = [g]
        gt_path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False), encoding="utf-8")
        log("SMOKE rewrote GT with 3 synthetic families")

    # site_family from expds
    sys.path.insert(0, str(ensure_code_layout()))
    from expds.naming import site_family  # noqa: WPS433

    handles: dict[str, any] = {}
    counts = {}
    for reg, feats in by_fam.items():
        fam = site_family(reg)
        path = dest / f"{fam}.jsonl"
        fh = handles.get(fam)
        if fh is None:
            fh = handles[fam] = open(path, "w", encoding="utf-8")
            counts[fam] = 0
        for gtf in feats[:2]:
            props = dict(gtf.get("properties") or {})
            geom = gtf["geometry"]
            for conf in (0.25, 0.12, 0.04):
                pred = {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "region_name": reg,
                        "class_name": props.get("class_name", "gorodishcha"),
                        "confidence": conf,
                        "threshold_group": "dfine_ae",
                        "markup_type": props.get("markup_type") or "ae",
                        "crs": props.get("crs") or "EPSG:3857",
                        "box_area_m2": 14000.0,
                    },
                }
                fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
                counts[fam] += 1
    for fh in handles.values():
        fh.close()
    (dest / "_counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=1), encoding="utf-8")
    (dest / "_SMOKE").write_text("synthetic fam0 — NOT stack A; off.score will NOT be ~0.314615\n", encoding="utf-8")
    log("SMOKE fam0 families", len(counts), "objs", sum(counts.values()))
    return dest


def resolve_fam(inputs: dict[str, Path], work: Path, gt_path: Path, code: Path) -> tuple[Path, str]:
    smoke = os.environ.get("LARGE_GATE_SMOKE", "").strip().lower() in ("1", "true", "yes")
    fam = find_fam_dir(inputs)
    if fam is not None and not smoke:
        log("fam0 from input", fam)
        return fam, "attached_fam"
    stack = find_stack_geojson(inputs)
    if stack is not None and not smoke:
        out = work / "fam0"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(code / "scripts" / "split_stack_by_family.py"),
            "--stack",
            str(stack),
            "--out",
            str(out),
        ]
        log("split stack → fam0", stack)
        subprocess.check_call(cmd, cwd=str(code))
        return out, "split_stack"
    # smoke / missing
    out = work / "fam0_smoke"
    build_smoke_fam(out, gt_path)
    return out, "smoke"


def build_clusters(code: Path, fam: Path, gt: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(code / "scripts" / "family_clusters.py"),
        "--fam",
        str(fam),
        "--gt",
        str(gt),
        "--out",
        str(out),
    ]
    log("family_clusters", cmd)
    subprocess.check_call(cmd, cwd=str(code))
    return out


def maybe_rebuild_train(inputs: dict[str, Path], work: Path) -> Path | None:
    shards = [k for k in inputs if "koz3-xfer-s" in k]
    if not shards:
        log("no xfer shards attached — skip rebuild")
        return None
    # Prefer existing rebuild
    dest = Path(os.environ.get("PIXGBM_TREE", "/kaggle/working/train-expds-3"))
    marker = work / "rebuild_done.json"
    if marker.is_file() and dest.is_dir():
        log("rebuild checkpoint hit", marker)
        return dest
    os.environ["PIXGBM_REBUILD_ALL"] = os.environ.get("PIXGBM_REBUILD_ALL", "1")
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    import importlib.util

    mod_path = HERE / "rebuild_train_tree.py"
    if not mod_path.is_file():
        mod_path = ROOT / "rebuild_train_tree.py"
    spec = importlib.util.spec_from_file_location("rebuild_train_tree", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    log("rebuild_train_tree →", dest, "shards", sorted(shards))
    report = mod.rebuild_train_tree(dest=dest)
    marker.write_text(json.dumps(report, ensure_ascii=False, indent=2)[:200000], encoding="utf-8")
    (ROOT / "rebuild_report.json").write_text(marker.read_text(encoding="utf-8"), encoding="utf-8")
    log("rebuild done regions≈", report.get("any_region_dirs_n"), report.get("note"))
    return dest


def validate_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("off", "area_tau", "fixc_nested"):
        if key not in data:
            raise SystemExit(f"result missing key {key}")
        row = data[key]
        for field in ("score", "precision", "recall", "size_recall"):
            if field not in row:
                raise SystemExit(f"{key} missing {field}")
        if key != "off":
            for field in ("delta_score", "delta_precision", "delta_recall", "recovered_gt_gt5k"):
                if field not in row:
                    raise SystemExit(f"{key} missing {field}")
    return data


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "work"
    work.mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", step="start")
    tip = (HERE / "TIP.txt").read_text(encoding="utf-8").strip() if (HERE / "TIP.txt").is_file() else "unknown"
    log("tip", tip, "python", sys.version)

    try:
        code = ensure_code_layout()
        log("code root", code)
        sys.path.insert(0, str(code))
        sys.path.insert(0, str(code / "scripts"))

        inputs = find_inputs()
        log("inputs", sorted(inputs))

        write_status("RUNNING", step="rebuild_train")
        train_root = maybe_rebuild_train(inputs, work)

        write_status("RUNNING", step="gt")
        gt_path = work / "gt_train_v2.geojson"
        smoke = os.environ.get("LARGE_GATE_SMOKE", "").strip().lower() in ("1", "true", "yes")
        try:
            materialize_gt(inputs, gt_path, train_root)
        except FileNotFoundError:
            if not smoke:
                raise
            # placeholder; build_smoke_fam will rewrite with synthetic families
            gt_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False),
                encoding="utf-8",
            )
            log("SMOKE: empty GT placeholder (no gt dataset / train labels)")

        write_status("RUNNING", step="fam0")
        fam_path, fam_src = resolve_fam(inputs, work, gt_path, code)
        log("fam source", fam_src, fam_path)

        write_status("RUNNING", step="clusters")
        clusters_path = work / "family_clusters.json"
        # Prefer attached clusters if present
        for p in inputs.values():
            for cand in p.glob("**/family_clusters.json"):
                shutil.copy2(cand, clusters_path)
                log("clusters from input", cand)
                break
            else:
                continue
            break
        if not clusters_path.is_file():
            build_clusters(code, fam_path, gt_path, clusters_path)

        metric = code / "materials" / "compute_metrics.py"
        if not metric.is_file():
            raise FileNotFoundError(metric)

        out_json = OUT / "large_objects_honest.json"
        write_status("RUNNING", step="honest", fam_src=fam_src)
        cmd = [
            sys.executable,
            str(code / "scripts" / "large_objects_honest.py"),
            "--fam",
            str(fam_path),
            "--gt",
            str(gt_path),
            "--clusters",
            str(clusters_path),
            "--metric",
            str(metric),
            "--out",
            str(out_json),
        ]
        log("run", " ".join(cmd))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(code) + os.pathsep + str(code / "scripts") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, cwd=str(code), env=env, text=True, capture_output=True)
        (OUT / "honest_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (OUT / "honest_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            log("honest failed", proc.returncode)
            print(proc.stdout[-4000:] if proc.stdout else "")
            print(proc.stderr[-4000:] if proc.stderr else "", file=sys.stderr)
            write_status("FAILED", step="honest", returncode=proc.returncode, fam_src=fam_src)
            return proc.returncode

        data = validate_result(out_json)
        summary = {
            "tip": tip,
            "fam_src": fam_src,
            "off_score": data["off"]["score"],
            "area_tau_score": data["area_tau"]["score"],
            "area_tau_delta": data["area_tau"]["delta_score"],
            "fixc_nested_score": data["fixc_nested"]["score"],
            "fixc_nested_delta": data["fixc_nested"]["delta_score"],
            "fixc_recovered_gt_gt5k": data["fixc_nested"]["recovered_gt_gt5k"],
            "smoke": fam_src == "smoke",
        }
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log("SUMMARY", json.dumps(summary, ensure_ascii=False))
        write_status("DONE", step="honest", fam_src=fam_src, off_score=data["off"]["score"])
        # also drop a copy at /kaggle/working root for easy pull (skip if not on Kaggle)
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            if ROOT.resolve() != OUT.resolve():
                shutil.copy2(out_json, ROOT / "large_objects_honest.json")
                shutil.copy2(OUT / "STATUS.txt", ROOT / "STATUS.txt")
                shutil.copy2(OUT / "summary.json", ROOT / "summary.json")
        except OSError as err:
            log("skip root copy", err)
        return 0
    except Exception as e:
        traceback.print_exc()
        write_status("FAILED", error=str(e)[:500])
        return 1


if __name__ == "__main__":
    sys.exit(main())
