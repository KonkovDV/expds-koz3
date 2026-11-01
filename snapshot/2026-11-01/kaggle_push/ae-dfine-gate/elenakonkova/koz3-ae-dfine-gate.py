#!/usr/bin/env python3
"""KOZ3 Ae D-FINE gate: OFF / A+B / A+B+C on v2 base (bot3 steering).

Commit tip: 7b1c272 (bots/bot3-aerial-recall; ancestors 1e7073a A+B, 0bcb039 Fix C).
Weights f4pack sha16 249b86fb. Fix C required: default WBF+flip halves scores vs OOF NMS.

Attaches:
  elenakonkova/koz3-ae-dfine-gate — submission zip + tools + ae_*.toml + regions/015
  dmitrijkv/koz3-gt-train-v2 — full GT
  optional dmitrijkv/koz3-xfer-s00* — extra Ae regions (rebuild if present)

No LB / no main push. Tokens never printed. Don't touch lidar kernels.
"""
from __future__ import annotations
import hashlib, json, os, re, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working/ae_dfine_gate")
PIXGBM_SHA = "35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b"
COMMIT = "7b1c272d147a50f7e2053ab518f6cda6cb774f69"
WEIGHTS_SHA16 = "249b86fb9713cc1c"
VARIANTS = ("OFF", "A+B", "A+B+C")  # mosaic_intact skipped (not cheap)
WALL_INFER_S = 20 * 60
B_PRIME = 0.315800  # documented target; mock/xfer ALL may differ


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def status(state: str, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"status": state, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "commit": COMMIT, "weights_sha16": WEIGHTS_SHA16, **extra}
    (OUT / "STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "STATUS.txt").write_text(state + "\n" + "\n".join(f"{k}={v}" for k, v in payload.items()) + "\n", encoding="utf-8")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_inputs():
    root = Path("/kaggle/input")
    found = {}
    if not root.is_dir():
        return found
    for p in sorted(root.iterdir()):
        name = p.name.lower()
        if "ae-dfine-gate" in name or "ae_dfine_gate" in name:
            found["cand"] = p
        elif "gt-train" in name or "gt_train" in name:
            found["gt"] = p
        elif "e2e-tools" in name:
            found["tools"] = p
        elif re.search(r"xfer-s\d+", name):
            found.setdefault("xfer", []).append(p)
    return found


def pip_transformers():
    import importlib
    need = []
    try:
        import transformers as t
        if getattr(t, "__version__", "") != "5.21.0":
            need.append("transformers==5.21.0")
        from transformers import DFineForObjectDetection  # noqa: F401
    except Exception:
        need.append("transformers==5.21.0")
    if need:
        log("pip", need)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", *need])
        if r.returncode:
            raise SystemExit("pip transformers failed")
        for k in [k for k in list(sys.modules) if k.split(".")[0] in ("transformers", "tokenizers", "huggingface_hub")]:
            del sys.modules[k]
        importlib.invalidate_caches()
    import transformers
    from transformers import DFineForObjectDetection  # noqa: F401
    log("transformers", transformers.__version__, transformers.__file__)


def unpack_cand(cand: Path, dest: Path) -> Path:
    # Dataset may ship as .zip OR already-extracted (kaggle --dir-mode zip).
    # Always materialize under dest (/kaggle/working) — input is read-only.
    dest.mkdir(parents=True, exist_ok=True)
    pre = sorted(cand.glob("submission_ae_abc_*")) + sorted(cand.glob("submission_*"))
    pre = [d for d in pre if d.is_dir() and (d / "inference" / "main.py").is_file()]
    if pre:
        src = pre[0]
        target = dest / src.name
        if target.exists():
            shutil.rmtree(target)
        log("copy pre-extracted submission", src, "->", target)
        shutil.copytree(src, target, symlinks=False)
        return target
    zips = sorted(cand.rglob("submission_ae_abc_*.zip")) + sorted(cand.rglob("submission*.zip"))
    if not zips:
        mains0 = list(cand.rglob("inference/main.py"))
        if mains0:
            src = mains0[0].parents[1]
            target = dest / src.name
            if target.exists():
                shutil.rmtree(target)
            log("copy cand-rooted submission", src, "->", target)
            shutil.copytree(src, target, symlinks=False)
            return target
        raise FileNotFoundError("no submission zip/dir in cand dataset")
    z = zips[0]
    log("unpack", z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    mains = list(dest.rglob("inference/main.py"))
    if not mains:
        raise FileNotFoundError("inference/main.py missing")
    root = mains[0].parents[1]
    return root


def inject_pixgbm(root: Path, tools: Path):
    src = tools / "expds" / "geo_pixgbm.py"
    if not src.is_file():
        hits = list(tools.rglob("geo_pixgbm.py"))
        if not hits:
            raise FileNotFoundError("geo_pixgbm.py missing")
        src = hits[0]
    got = sha256_file(src)
    if got != PIXGBM_SHA:
        raise RuntimeError(f"geo_pixgbm sha mismatch got={got} want={PIXGBM_SHA}")
    dst = root / "expds" / "geo_pixgbm.py"
    shutil.copy2(src, dst)
    log("pixgbm inject ok", got[:16])


def apply_variant(root: Path, cand: Path, variant: str):
    mapping = {"OFF": "ae_off.toml", "A+B": "ae_ab_on.toml", "A+B+C": "ae_abc_on.toml"}
    name = mapping[variant]
    src = cand / name
    if not src.is_file():
        hits = list(cand.rglob(name))
        if not hits:
            raise FileNotFoundError(name)
        src = hits[0]
    dst = root / "configs" / "default.toml"
    shutil.copy2(src, dst)
    # hard-assert key flags via text
    text = dst.read_text(encoding="utf-8")
    if variant == "OFF":
        assert "enabled = true\nmarkups" not in text.split("[dfine_aeor]")[1][:80]
    log("variant", variant, "config", name)


def resolve_regions(cand: Path, xfer: list[Path], work: Path) -> Path:
    bundled = cand / "regions"
    if bundled.is_dir() and any(bundled.iterdir()):
        log("regions from cand bundle", bundled)
        return bundled
    # fallback: empty
    raise FileNotFoundError("no regions/; attach cand dataset with regions/015 or rebuild from xfer")


def find_gt(gt_ds: Path, cand: Path) -> Path:
    for p in list(gt_ds.rglob("*.geojson")) + list(cand.rglob("gt_*.geojson")):
        if p.stat().st_size > 100:
            # prefer full train gt
            if "train" in p.name or "gt_train" in p.name:
                return p
    hits = list(gt_ds.rglob("*.geojson"))
    if hits:
        return max(hits, key=lambda x: x.stat().st_size)
    hits = list(cand.rglob("gt_*.geojson"))
    if hits:
        return hits[0]
    raise FileNotFoundError("gt geojson not found")


def run_infer(root: Path, regions: Path, out_geo: Path, deadline: float):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, str(root / "inference" / "main.py"), str(regions), str(out_geo)]
    log("infer", " ".join(cmd))
    # soft wall
    remain = max(60, int(deadline - time.time()))
    try:
        r = subprocess.run(cmd, cwd=str(root), env=env, timeout=remain)
    except subprocess.TimeoutExpired:
        log("infer TIMEOUT", remain)
        return False
    log("infer exit", r.returncode, "out", out_geo.exists(), out_geo.stat().st_size if out_geo.exists() else 0)
    return r.returncode == 0 and out_geo.is_file()


def score(tools: Path, gt: Path, pred: Path, out_json: Path, name: str):
    score_py = tools / "score" / "score_one.py"
    metric = tools / "score" / "compute_metrics.py"
    if not score_py.is_file():
        score_py = next(tools.rglob("score_one.py"))
        metric = next(tools.rglob("compute_metrics.py"))
    cmd = [sys.executable, str(score_py), "--metric", str(metric), "--gt", str(gt),
           "--pred", str(pred), "--out", str(out_json), "--name", name]
    log("score", name)
    subprocess.run(cmd, check=False)
    if out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))
    return {}


def count_ae(pred: Path) -> dict:
    if not pred.is_file():
        return {"n": 0}
    d = json.loads(pred.read_text(encoding="utf-8"))
    feats = [f for f in d.get("features", []) if (f.get("properties") or {}).get("markup_type") == "ae"]
    from collections import Counter
    return {"n": len(feats), "by_class": dict(Counter((f.get("properties") or {}).get("class_name") for f in feats))}


def main():
    status("RUNNING", phase="start")
    try:
        pip_transformers()
        inp = find_inputs()
        log("inputs", {k: (str(v) if not isinstance(v, list) else [str(x) for x in v]) for k, v in inp.items()})
        if "cand" not in inp:
            raise FileNotFoundError("cand dataset elenakonkova/koz3-ae-dfine-gate missing")
        cand = inp["cand"]
        tools = cand / "tools"
        if "tools" in inp:
            tools = inp["tools"]
        if not (tools / "expds" / "geo_pixgbm.py").is_file() and not list(tools.rglob("geo_pixgbm.py")):
            tools = cand / "tools"
        gt_ds = inp.get("gt", cand)
        root = unpack_cand(cand, OUT / "sub")
        inject_pixgbm(root, tools)
        w = root / "models" / "dfine_ae" / "model.safetensors"
        got = sha256_file(w)[:16]
        log("weights sha16", got, "want", WEIGHTS_SHA16)
        if got != WEIGHTS_SHA16:
            raise RuntimeError(f"weights sha mismatch {got}")
        regions = resolve_regions(cand, inp.get("xfer") or [], OUT)
        gt = find_gt(gt_ds, cand)
        log("gt", gt, "regions", regions)

        # filter GT to regions present
        reg_names = {p.name for p in regions.iterdir() if p.is_dir()}
        gt_all = json.loads(gt.read_text(encoding="utf-8"))
        gt_f = [f for f in gt_all["features"] if (f.get("properties") or {}).get("region_name") in reg_names]
        gt_use = OUT / "gt_regions.geojson"
        gt_use.write_text(json.dumps({"type": "FeatureCollection", "features": gt_f}, ensure_ascii=False), encoding="utf-8")
        log("gt filtered", len(gt_f), "regions", sorted(reg_names))

        deadline = T0 + WALL_INFER_S
        results = {}
        for variant in VARIANTS:
            if time.time() > deadline - 90:
                log("skip variant — wall", variant)
                results[variant] = {"skipped": True}
                continue
            status("RUNNING", phase=f"infer_{variant}")
            apply_variant(root, cand, variant)
            out_geo = OUT / f"result_{variant.replace('+','').replace('/','')}.geojson"
            # normalize filename
            tag = {"OFF": "off", "A+B": "ab", "A+B+C": "abc"}[variant]
            out_geo = OUT / f"result_{tag}.geojson"
            ok = run_infer(root, regions, out_geo, deadline)
            ae = count_ae(out_geo) if ok else {"n": 0}
            sc = score(tools, gt_use, out_geo, OUT / f"scores_{tag}.json", tag) if ok else {}
            results[variant] = {
                "ok": ok,
                "ae": ae,
                "ALL_F1": sc.get(f"ALL_F1_{tag}"),
                "scores_path": str(OUT / f"scores_{tag}.json"),
            }
            log("variant done", variant, results[variant])

        # Gate checks
        off = results.get("OFF", {})
        abc = results.get("A+B+C", {})
        ab = results.get("A+B", {})
        delta_abc = None
        if off.get("ALL_F1") is not None and abc.get("ALL_F1") is not None:
            delta_abc = abc["ALL_F1"] - off["ALL_F1"]
        flag_dead = (abc.get("ae", {}).get("n", 0) == 0) or (
            off.get("ALL_F1") is not None and abc.get("ALL_F1") is not None and abs(delta_abc) < 1e-9
        )
        summary = {
            "commit": COMMIT,
            "weights_sha16": WEIGHTS_SHA16,
            "regions": sorted(reg_names),
            "results": results,
            "delta_abc_vs_off": delta_abc,
            "flag_dead": flag_dead,
            "b_prime_ref": B_PRIME,
            "note": "Fix C (dfine_ae_tile fuse=nms) required; A+B alone WBF-halves scores. mosaic_intact not run.",
            "wall_s": time.time() - T0,
        }
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log("SUMMARY", json.dumps(summary, ensure_ascii=False))
        if flag_dead:
            status("FAILED", phase="flag_dead", **{k: summary[k] for k in ("delta_abc_vs_off", "flag_dead")})
            raise SystemExit("flag dead or zero ae dets on A+B+C")
        # soft assert nonzero
        if abc.get("ae", {}).get("n", 0) < 1:
            status("FAILED", phase="zero_ae")
            raise SystemExit("A+B+C ae dets == 0")
        status("DONE", phase="ok", delta_abc_vs_off=delta_abc, ae_abc=abc.get("ae"))
    except Exception as e:
        traceback.print_exc()
        status("FAILED", phase="exception", error=str(e)[:300])
        raise


if __name__ == "__main__":
    main()
