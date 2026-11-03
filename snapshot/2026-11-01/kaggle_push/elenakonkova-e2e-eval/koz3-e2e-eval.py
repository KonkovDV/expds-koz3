#!/usr/bin/env python3
"""KOZ3 reusable e2e mock-gate eval on Kaggle GPU (account-2 / elenakonkova).

Slug: elenakonkova/koz3-e2e-eval
Resource: NvidiaTeslaT4, internet ON, private.

Inputs (attach):
  elenakonkova/koz3-mock-val-gate   — regions/ + gt.geojson
  elenakonkova/koz3-e2e-tools       — score/, expds/geo_pixgbm.py, requirements.lock
  elenakonkova/koz3-e2e-baseline-v2 — submission_bot7cand_v2_eb268d6.zip (calib default)
  optional: elenakonkova/koz3-e2e-cand-* with candidate.zip + SPEC.json

Candidate resolution order:
  1) /kaggle/input/*/SPEC.json  (fields: zip|commit, env, notes)
  2) env E2E_ZIP / E2E_COMMIT
  3) baseline zip dataset (calibration)

Steps: unpack/checkout → inject geo_pixgbm (sha check) → verify weights non-empty
→ pip-align packages vs requirements.lock (best-effort; record env) → run
inference/main.py on mock regions → score_one.py → scores.json + warn_scan +
wall + exit. STATUS.txt RUNNING|DONE|FAILED.

Calibration gate vs PC docker: |ΔALL| ≤ 0.002 vs 0.312509 (ae/li/sp_or logged).
No LB submissions. Tokens never printed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working/e2e_eval")
PIXGBM_SHA = "35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b"
PC_BASELINE = {
    "ALL": 0.3125090807983978,
    "ae": 0.09555648513894874,
    "li": 0.3369826841738736,
    "sp_or": 0.24291478563833163,
}
CALIB_TOL = 0.002
DOCKER_IMAGE_HINT = "pantum/n7q:20261024 (python3.12, torch==2.11.0+cu128; Kaggle≈closest)"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def write_status(state: str, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"status": state, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    (OUT / "STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [state, f"ts={payload['ts']}"]
    for k, v in extra.items():
        lines.append(f"{k}={v}")
    (OUT / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_inputs() -> dict[str, Path]:
    root = Path("/kaggle/input")
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for p in sorted(root.iterdir()):
        if p.is_dir():
            found[p.name.lower()] = p
    return found


def find_named(inputs: dict[str, Path], *needles: str) -> Path | None:
    for n in needles:
        for k, p in inputs.items():
            if n in k:
                return p
    return None


def load_spec(inputs: dict[str, Path]) -> dict:
    """SPEC.json from any attached dataset; else env; else baseline."""
    for p in inputs.values():
        for cand in (p / "SPEC.json", p / "spec.json"):
            if cand.is_file():
                spec = json.loads(cand.read_text(encoding="utf-8"))
                spec["_spec_path"] = str(cand)
                return spec
    spec: dict = {"env": {}}
    if os.environ.get("E2E_ZIP"):
        spec["zip"] = os.environ["E2E_ZIP"]
    if os.environ.get("E2E_COMMIT"):
        spec["commit"] = os.environ["E2E_COMMIT"]
    # env flags: E2E_ENV_FOO=bar → FOO=bar
    for k, v in os.environ.items():
        if k.startswith("E2E_ENV_"):
            spec.setdefault("env", {})[k[len("E2E_ENV_"):]] = v
    if "zip" not in spec and "commit" not in spec:
        spec["zip"] = "baseline"  # resolve later
        spec["calib"] = True
    return spec


def resolve_mock(inputs: dict[str, Path]) -> tuple[Path, Path]:
    mock = find_named(inputs, "koz3-mock-val-gate", "mock-val-gate", "mock_val")
    if mock is None:
        raise FileNotFoundError("attach elenakonkova/koz3-mock-val-gate")
    regions = None
    for c in (mock / "regions", mock):
        if c.is_dir() and any(x.is_dir() for x in c.iterdir()):
            # prefer explicit regions/
            if (mock / "regions").is_dir():
                regions = mock / "regions"
            else:
                regions = mock
            break
    if regions is None:
        # maybe tar inside
        tars = list(mock.glob("regions_gate.tar")) + list(mock.glob("*.tar"))
        if tars:
            dest = Path("/kaggle/working/regions")
            if not dest.exists():
                import tarfile
                dest.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tars[0]) as tf:
                    tf.extractall(dest.parent)
                if (dest.parent / "regions").is_dir():
                    regions = dest.parent / "regions"
                else:
                    regions = dest
    if regions is None:
        raise FileNotFoundError(f"no regions under {mock}")
    gt = None
    for c in (mock / "gt.geojson", mock / "gt" / "gt.geojson"):
        if c.is_file():
            gt = c
            break
    if gt is None:
        hits = list(mock.rglob("gt.geojson"))
        if hits:
            gt = hits[0]
    if gt is None:
        raise FileNotFoundError(f"gt.geojson missing under {mock}")
    return regions, gt


def resolve_tools(inputs: dict[str, Path]) -> Path:
    tools = find_named(inputs, "koz3-e2e-tools", "e2e-tools")
    if tools is None:
        raise FileNotFoundError("attach elenakonkova/koz3-e2e-tools")
    return tools


def find_zip(inputs: dict[str, Path], spec: dict) -> Path:
    z = spec.get("zip")
    if z and z not in ("baseline", "auto") and Path(z).is_file():
        return Path(z)
    # named in any dataset
    if z and z not in ("baseline", "auto"):
        for p in inputs.values():
            cand = p / z
            if cand.is_file():
                return cand
            hits = list(p.glob(z)) + list(p.rglob(Path(z).name))
            for h in hits:
                if h.is_file():
                    return h
    # baseline dataset
    base = find_named(inputs, "koz3-e2e-baseline-v2", "baseline-v2", "bot7cand_v2")
    if base:
        for pat in ("submission_bot7cand_v2_eb268d6.zip", "*.zip"):
            hits = list(base.glob(pat))
            if hits:
                return hits[0]
    # any submission zip
    for p in inputs.values():
        hits = list(p.glob("submission*.zip")) + list(p.glob("*.zip"))
        for h in hits:
            if h.is_file() and h.stat().st_size > 1000:
                return h
    raise FileNotFoundError("candidate/baseline zip not found")


def unpack_zip(zpath: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    log("unzip", zpath, "->", dest)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest)
    # find inference/main.py
    mains = list(dest.rglob("inference/main.py"))
    if not mains:
        raise FileNotFoundError("inference/main.py missing in zip")
    root = mains[0].parent.parent
    return root


def checkout_commit(commit: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    repo = dest / "expds-koz3"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    # Prefer public clone; if private, token via x-access-token (never log token)
    url = "https://github.com/KonkovDV/expds-koz3.git"
    if token:
        url = f"https://x-access-token:{token}@github.com/KonkovDV/expds-koz3.git"
    if not (repo / ".git").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", url, str(repo)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.check_call(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", commit],
                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.check_call(["git", "-C", str(repo), "checkout", commit],
                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return repo


def inject_pixgbm(root: Path, tools: Path) -> dict:
    src = tools / "expds" / "geo_pixgbm.py"
    if not src.is_file():
        hits = list(tools.rglob("geo_pixgbm.py"))
        if not hits:
            raise FileNotFoundError("geo_pixgbm.py not in tools dataset")
        src = hits[0]
    got = sha256_file(src)
    if got != PIXGBM_SHA:
        raise RuntimeError(f"geo_pixgbm sha mismatch got={got} want={PIXGBM_SHA}")
    dst = root / "expds" / "geo_pixgbm.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    # also models/pixgbm if zip has models but missing poly — leave as-is; warn later
    return {"injected": str(dst), "sha256": got, "ok": True}


def verify_weights(root: Path) -> dict:
    models = root / "models"
    report = {"root": str(models), "files": [], "empty": [], "ok": True}
    if not models.is_dir():
        report["ok"] = False
        report["error"] = "models/ missing"
        return report
    for p in sorted(models.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".pt", ".pth", ".bin", ".safetensors", ".txt", ".json", ".onnx"}:
            continue
        sz = p.stat().st_size
        rel = str(p.relative_to(root))
        report["files"].append({"path": rel, "bytes": sz})
        if sz == 0:
            report["empty"].append(rel)
            report["ok"] = False
    # torch.load spot-check on a few .pt
    try:
        import torch
        checked = 0
        for item in report["files"]:
            if not item["path"].endswith(".pt"):
                continue
            if item["bytes"] == 0:
                continue
            try:
                torch.load(root / item["path"], map_location="cpu", weights_only=True)
            except TypeError:
                torch.load(root / item["path"], map_location="cpu")
            checked += 1
            if checked >= 3:
                break
        report["torch_load_checked"] = checked
    except Exception as e:
        report["torch_load_error"] = f"{type(e).__name__}: {e}"
    return report


def record_env(path: Path) -> dict:
    info = {
        "python": sys.version,
        "platform": sys.platform,
        "docker_image_target": DOCKER_IMAGE_HINT,
        "packages": {},
    }
    try:
        import torch
        info["packages"]["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["packages"]["torch"] = f"ERR {e}"
    for mod in ("numpy", "scipy", "shapely", "pyproj", "rasterio", "laspy",
                "segmentation_models_pytorch", "lightgbm", "timm", "PIL"):
        try:
            m = __import__(mod if mod != "PIL" else "PIL")
            info["packages"][mod] = getattr(m, "__version__", "?")
        except Exception as e:
            info["packages"][mod] = f"MISSING {e}"
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def align_packages(tools: Path) -> list[str]:
    """Best-effort pip install from requirements.lock (skip torch — use Kaggle's CUDA build)."""
    lock = tools / "requirements.lock"
    if not lock.is_file():
        hits = list(tools.rglob("requirements.lock"))
        lock = hits[0] if hits else None
    cmds = []
    if lock and lock.is_file():
        # filter out lines that are comments; skip nothing else — torch not in lock
        cmd = [sys.executable, "-m", "pip", "install", "-q", "-r", str(lock)]
        log("pip", " ".join(cmd[-2:]))
        try:
            subprocess.check_call(cmd)
            cmds.append("pip-install-lock-ok")
        except subprocess.CalledProcessError as e:
            cmds.append(f"pip-install-lock-fail:{e.returncode}")
            log("WARN pip lock install failed; continuing with preinstalled")
    # ensure scoring deps
    for pkg in ("shapely", "pyproj"):
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            cmds.append(f"pip-{pkg}")
    return cmds


def warn_scan(log_path: Path, err_path: Path, out: Path) -> list[str]:
    pats = [
        r"pixgbm включён, но",
        r"ступени \{\}",
        r"SP_DFINE_UNAVAILABLE",
        r"UNAVAILABLE",
        r"WARNING",
        r"ERROR",
        r"ускоритель",
        r"среда:",
        r"platform",
        r"Traceback",
    ]
    rx = re.compile("|".join(pats))
    hits = []
    for p in (log_path, err_path):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            if rx.search(line):
                hits.append(line[:500])
    (out / "warn_scan.txt").write_text("\n".join(hits[:200]) + ("\n" if hits else ""), encoding="utf-8")
    return hits


def run_infer(root: Path, regions: Path, out_geo: Path, env_extra: dict) -> tuple[int, float]:
    out_geo.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPDS_PIXGBM_STRICT"] = env.get("EXPDS_PIXGBM_STRICT", "1")
    for k, v in (env_extra or {}).items():
        env[str(k)] = str(v)
    # Prefer candidate's own packages path
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(root / "inference" / "main.py"), str(regions), str(out_geo)]
    log("infer cmd:", " ".join(cmd))
    log_path = OUT / "main.log"
    err_path = OUT / "main.err"
    t1 = time.time()
    with log_path.open("w", encoding="utf-8") as lo, err_path.open("w", encoding="utf-8") as er:
        p = subprocess.run(cmd, cwd=str(root), env=env, stdout=lo, stderr=er)
    wall = time.time() - t1
    (OUT / "wall_sec.txt").write_text(f"{wall:.3f}\n", encoding="utf-8")
    (OUT / "exit_code.txt").write_text(f"{p.returncode}\n", encoding="utf-8")
    return p.returncode, wall


def score(tools: Path, gt: Path, pred: Path) -> dict:
    metric = tools / "score" / "compute_metrics.py"
    score_py = tools / "score" / "score_one.py"
    if not metric.is_file():
        metric = next(tools.rglob("compute_metrics.py"))
    if not score_py.is_file():
        score_py = next(tools.rglob("score_one.py"))
    out_scores = OUT / "scores.json"
    cmd = [sys.executable, str(score_py),
           "--metric", str(metric), "--gt", str(gt),
           "--pred", str(pred), "--out", str(out_scores), "--name", "cand"]
    log("score", " ".join(cmd))
    subprocess.check_call(cmd)
    return json.loads(out_scores.read_text(encoding="utf-8"))


def calib_compare(scores: dict) -> dict:
    rows = { (r["region"], r["markup"]): r for r in scores.get("rows", []) }
    def f1(markup):
        r = rows.get(("ALL", markup))
        return None if not r else r.get("f1_cand")
    got = {"ALL": f1("ALL"), "ae": f1("ae"), "li": f1("li"), "sp_or": f1("sp_or")}
    deltas = {k: (None if got[k] is None else got[k] - PC_BASELINE[k]) for k in PC_BASELINE}
    ok = got["ALL"] is not None and abs(deltas["ALL"]) <= CALIB_TOL
    return {
        "pc_baseline": PC_BASELINE,
        "kaggle": got,
        "delta": deltas,
        "tol": CALIB_TOL,
        "accept": ok,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", phase="start")
    try:
        inputs = find_inputs()
        log("inputs", sorted(inputs))
        spec = load_spec(inputs)
        (OUT / "spec_resolved.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        regions, gt = resolve_mock(inputs)
        tools = resolve_tools(inputs)
        log("regions", regions, "gt", gt, "tools", tools)

        write_status("RUNNING", phase="packages")
        pip_notes = align_packages(tools)
        env_info = record_env(OUT / "env.json")
        log("env torch", env_info.get("packages", {}).get("torch"), "cuda", env_info.get("cuda_available"))

        write_status("RUNNING", phase="candidate")
        work = Path("/kaggle/working/cand")
        if spec.get("commit") and not spec.get("zip"):
            root = checkout_commit(str(spec["commit"]), work)
            # commit path usually lacks weights — fail loudly unless models attached
        else:
            zpath = find_zip(inputs, spec)
            log("zip", zpath, "bytes", zpath.stat().st_size, "sha", sha256_file(zpath)[:16])
            (OUT / "candidate_zip_sha256.txt").write_text(sha256_file(zpath) + "\n", encoding="utf-8")
            root = unpack_zip(zpath, work / "unzipped")

        write_status("RUNNING", phase="pixgbm_inject")
        inj = inject_pixgbm(root, tools)
        (OUT / "pixgbm_inject.json").write_text(json.dumps(inj, indent=2) + "\n", encoding="utf-8")
        log("pixgbm", inj)

        write_status("RUNNING", phase="weights_check")
        wrep = verify_weights(root)
        (OUT / "weights_check.json").write_text(json.dumps(wrep, indent=2) + "\n", encoding="utf-8")
        if not wrep.get("ok"):
            write_status("FAILED", phase="weights_check", empty=",".join(wrep.get("empty") or []) or wrep.get("error"))
            log("FAILED empty weights", wrep.get("empty") or wrep.get("error"))
            return 5

        write_status("RUNNING", phase="infer")
        out_geo = OUT / "result.geojson"
        ec, wall = run_infer(root, regions, out_geo, spec.get("env") or {})
        hits = warn_scan(OUT / "main.log", OUT / "main.err", OUT)
        log("infer exit", ec, "wall", round(wall, 1), "warn_hits", len(hits))

        # Validity (LESSONS): exit==0, geojson exists, wall not tiny
        if ec != 0:
            write_status("FAILED", phase="infer", exit=ec, wall_s=round(wall, 1))
            return 4
        if not out_geo.is_file() or out_geo.stat().st_size < 10:
            write_status("FAILED", phase="infer", reason="no_geojson")
            return 5
        if wall < 30:
            write_status("FAILED", phase="infer", reason="wall_too_short", wall_s=round(wall, 1))
            log("INVALID wall_s too short for full e2e")
            return 6

        write_status("RUNNING", phase="score")
        scores = score(tools, gt, out_geo)
        calib = calib_compare(scores) if spec.get("calib") or os.environ.get("E2E_CALIB", "") in ("1", "true", "yes") else None
        # Always attach baseline deltas for ALL layers when possible
        if calib is None:
            calib = calib_compare(scores)
            calib["mode"] = "informational"
        else:
            calib["mode"] = "calibration"
        (OUT / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n", encoding="utf-8")
        summary = {
            "exit": ec,
            "wall_s": round(wall, 3),
            "ALL_F1": scores.get("ALL_F1_cand"),
            "calibration": calib,
            "pip_notes": pip_notes,
            "pixgbm": inj,
            "n_warn_lines": len(hits),
            "spec": {k: v for k, v in spec.items() if k != "env"} | {"env_keys": list((spec.get("env") or {}).keys())},
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        log("SUMMARY", json.dumps(summary, ensure_ascii=False)[:800])

        if calib.get("mode") == "calibration" and not calib.get("accept"):
            write_status("FAILED", phase="calib", ALL=scores.get("ALL_F1_cand"),
                         delta_ALL=calib["delta"]["ALL"], accept=False)
            return 7

        write_status("DONE", phase="done", ALL=scores.get("ALL_F1_cand"), wall_s=round(wall, 1),
                     accept=calib.get("accept"), delta_ALL=calib.get("delta", {}).get("ALL"))
        return 0
    except Exception as e:
        traceback.print_exc()
        write_status("FAILED", phase="exception", error=f"{type(e).__name__}: {e}")
        (OUT / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(main())
