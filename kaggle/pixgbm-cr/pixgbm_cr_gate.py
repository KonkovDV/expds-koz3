#!/usr/bin/env python3
"""Kaggle CPU: PIXGBM e2e geo F1 gate on regions 200–217 (bot-6 / pixgbm-cr).

Compares REF (v2 eb268d6 *output* result.geojson) vs CANDIDATE (clean zip) and
applies helpers/bot6_r3_gate.py (|dF1| <= 0.01 on ALL + maggr/etgr/terfl;
INVALID on empty steps / no full / …).

Modes (env GATE_MODE):
  auto  — default. REF=result.geojson + CAND=submission zip → run CAND only, score.
  zip   — unpack two submission zips and run inference twice (legacy).
  tip   — tip packs (NOT used for REF anymore; REF is v2 output only).
  score — skip inference; score pre-dropped out_ref/ + out_cand/ result.geojson.

Geo tree: MagGr/EtGr/TerFl + GT for 200–217 pulled from S3 (train-expds-3 /
selcloud ru-7) inside rebuild_w4_geo_tree.py. Credentials: Kaggle Secrets
SECRET_ID/SECRET_KEY with fallback dataset dmitrijkv/koz3-s3-cred. Never log keys.

enable_gpu=false, enable_internet=TRUE (S3 + optional pip). Prefer offline wheels.
Never submits to LB. «1» pushes this package. Do NOT ship EN geo_pixgbm.py.
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
from datetime import datetime, timezone
from pathlib import Path

T0 = time.time()
OUT = Path(os.environ.get("PIXGBM_CR_OUT", "/kaggle/working/pixgbm_cr"))
TREE = Path(os.environ.get("PIXGBM_TREE", "/kaggle/working/train-expds-3"))
GEO_PIX_SHA = "35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b"
REF_POLY = "d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800"
REF_FINDS = "5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a"
TOL = float(os.environ.get("PIXGBM_DF1_TOL", "0.01"))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(state: str, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"status": state, "ts": utc_now(), **extra}
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
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if p.is_dir():
            out[p.name.lower()] = p
    return out


def find_named(inputs: dict[str, Path], *needles: str) -> Path | None:
    for n in needles:
        for k, p in inputs.items():
            if n in k:
                return p
    return None


def load_rebuild():
    here = Path(__file__).resolve().parent
    mod_path = here / "rebuild_w4_geo_tree.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("rebuild_w4_geo_tree", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def ensure_helpers() -> Path:
    here = Path(__file__).resolve().parent
    helpers = here / "helpers"
    if not (helpers / "bot6_r3_gate.py").is_file():
        raise FileNotFoundError("helpers/bot6_r3_gate.py missing in package")
    return helpers


def find_zip(inputs: dict[str, Path], *needles: str) -> Path | None:
    """Glob /kaggle/input mounts for a submission zip whose mount or name matches needles."""
    for n in needles:
        for k, p in inputs.items():
            if n not in k:
                continue
            hits = sorted(list(p.glob("*.zip")) + [z for z in p.rglob("*.zip") if z not in p.glob("*.zip")])
            if hits:
                return hits[0]
    # Also search any mount for filename needles
    for n in needles:
        for p in inputs.values():
            for z in p.rglob("*.zip"):
                if n.replace("-", "").replace("_", "") in z.name.lower().replace("-", "").replace("_", ""):
                    return z
    return None


def find_file(inputs: dict[str, Path], name: str) -> Path | None:
    for p in inputs.values():
        for hit in p.rglob(name):
            if hit.is_file():
                return hit
    return None


def resolve_ref_cand(inputs: dict[str, Path], mode: str) -> dict:
    """Locate REF and CAND artifacts. Returns dict with keys present/missing + paths."""
    out: dict = {"mode": mode, "ref": None, "cand": None, "ref_kind": None, "cand_kind": None, "errors": []}

    if os.environ.get("REF_ZIP") and Path(os.environ["REF_ZIP"]).is_file():
        out["ref"] = Path(os.environ["REF_ZIP"]); out["ref_kind"] = "zip"
    if os.environ.get("CAND_ZIP") and Path(os.environ["CAND_ZIP"]).is_file():
        out["cand"] = Path(os.environ["CAND_ZIP"]); out["cand_kind"] = "zip"

    if out["ref"] is None:
        # Preferred: v2 ready OUTPUT only (result.geojson) — no EN code/models.
        ref_ds = find_named(inputs, "pixgbm-cr-ref", "pixgbm_cr_ref", "baseline-v2", "pixgbm-ref")
        gj = None
        if ref_ds is not None:
            cands = sorted(ref_ds.rglob("result.geojson"))
            # Prefer geo/result.geojson or out_ref/result.geojson if present
            prefer = [c for c in cands if "out_ref" in c.as_posix() or c.parent.name == "geo"]
            gj = (prefer or cands)[0] if (prefer or cands) else None
        if gj is None:
            gj = find_file(inputs, "result.geojson")
            # Avoid picking CAND outputs by mistake when searching broadly
            if gj is not None and any(x in gj.as_posix().lower() for x in ("cand", "pix_clean", "clean")):
                gj = None
        if gj is not None:
            out["ref"] = gj
            out["ref_kind"] = "output"
        else:
            z = find_zip(
                inputs,
                "pixgbm-cr-ref", "pixgbm_cr_ref", "e2e-baseline-v2", "baseline-v2",
                "ref-v2", "bot7cand_v2", "eb268d6",
            )
            if z is not None:
                out["ref"] = z
                out["ref_kind"] = "zip"

    if out["cand"] is None:
        z = find_zip(
            inputs,
            "pixgbm-cr-cand", "pixgbm_cr_cand", "geo-clean", "pix-clean",
            "pixgbm-cand", "cand-clean", "clean-pack", "pixgbm_clean",
        )
        if z is not None:
            out["cand"] = z; out["cand_kind"] = "zip"
        else:
            cand_ds = find_named(inputs, "pixgbm-cr-cand", "pixgbm_cr_cand", "pix-clean", "pix_clean", "models-pix-clean")
            if cand_ds is not None:
                # need at least one model file
                models = list(cand_ds.rglob("poly.txt")) + list(cand_ds.rglob("finds.txt"))
                if models:
                    out["cand"] = cand_ds; out["cand_kind"] = "tip"

    if out["ref"] is None:
        out["errors"].append(
            "INVALID: REF missing — attach dmitrijkv/koz3-pixgbm-cr-ref with "
            "v2 ready OUTPUT only (result.geojson for regions 200–217). "
            "Do NOT attach EN geo_pixgbm.py / models."
        )
    if out["cand"] is None:
        out["errors"].append(
            "INVALID: CAND missing — attach dmitrijkv/koz3-pixgbm-cr-cand "
            "(submission_pixgbm_clean_*.zip or models_pix_clean/)"
        )
    return out


def ensure_offline_wheels() -> list[str]:
    """Prefer vendored wheels; network pip allowed as fallback (enable_internet=true)."""
    installed: list[str] = []
    root = Path("/kaggle/input")
    wheel_dirs = [d for d in root.rglob("wheels") if d.is_dir()] if root.is_dir() else []

    def _need(mod: str) -> bool:
        try:
            __import__(mod)
            return False
        except ImportError:
            return True

    def _pip(pkgs: list[str], *, offline: bool) -> bool:
        import subprocess as _sp
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if offline:
            if not wheel_dirs:
                return False
            cmd += ["--no-index", "--no-deps"]
            for d in wheel_dirs:
                cmd.extend(["--find-links", str(d)])
        cmd += pkgs
        r = _sp.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log("WARN pip", pkgs, "offline=" + str(offline),
                (r.stderr or r.stdout or "")[:400])
            return False
        return True

    # zstandard (optional; only if a leftover xfer path needs it)
    if _need("zstandard"):
        if _pip(["zstandard"], offline=True):
            installed.append("zstandard")
        elif _pip(["zstandard"], offline=False):
            installed.append("zstandard(net)")

    # boto3 required for S3 geo fetch
    if _need("boto3"):
        if _pip(["boto3"], offline=True):
            installed.append("boto3")
        elif _pip(["boto3"], offline=False):
            installed.append("boto3(net)")
        else:
            log("WARN boto3 not installed — S3 rebuild will fail")
    return installed


def dry_run_syntax() -> int:
    """Local/Kaggle syntax check: compile + import helpers; no /kaggle/input required."""
    import py_compile
    here = Path(__file__).resolve().parent
    files = [
        here / "pixgbm_cr_gate.py",
        here / "rebuild_w4_geo_tree.py",
        here / "scripts" / "rebuild_train_tree.py",
        here / "helpers" / "bot6_r3_gate.py",
        here / "helpers" / "bot6_r3_build_gt.py",
        here / "helpers" / "bot6_r3_select_copy.py",
    ]
    for f in files:
        if not f.is_file():
            print(f"MISSING {f}")
            return 2
        py_compile.compile(str(f), doraise=True)
        print(f"OK compile {f.name}")
    # helpers import
    ensure_helpers()
    load_rebuild()
    print("OK helpers+rebuild load")
    # optional runtime deps (Kaggle CPU image)
    missing = []
    for mod in ("numpy", "scipy", "lightgbm", "rasterio", "shapely"):
        try:
            __import__(mod)
            print(f"OK import {mod}")
        except ImportError:
            missing.append(mod)
            print(f"MISSING import {mod} (needed at inference time; expect on Kaggle CPU)")
    meta = here / "kernel-metadata.json"
    if meta.is_file():
        m = json.loads(meta.read_text(encoding="utf-8"))
        assert m.get("enable_internet") is True, "enable_internet must be true (S3)"
        assert m.get("enable_gpu") is False, "enable_gpu must be false"
        srcs = m.get("dataset_sources") or []
        assert "dmitrijkv/koz3-s3-cred" in srcs, "koz3-s3-cred must be in dataset_sources"
        assert not any("w4-geo" in s for s in srcs), "w4-geo datasets must not be wired"
        print("OK kernel-metadata internet/gpu flags")
        print("dataset_sources:", srcs)
    print("DRY_RUN_OK missing_optional=", missing)
    return 0



def unpack_zip(z: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    # detect root that has inference/main.py
    for cand in [dest, *dest.iterdir()]:
        if (cand / "inference" / "main.py").is_file():
            return cand
    raise FileNotFoundError(f"inference/main.py not found under {dest} from {z}")


def patch_toml(repo: Path, *, enabled: bool, impl: str, budget_s: int = 100000) -> str:
    toml = repo / "configs" / "default.toml"
    text = toml.read_text(encoding="utf-8")
    bak = text
    text = re.sub(
        r"(?m)^(\[geophys\.pixgbm\]\s*\n(?:.*\n)*?)enabled\s*=\s*\w+",
        lambda m: re.sub(r"enabled\s*=\s*\w+", f"enabled = {'true' if enabled else 'false'}", m.group(0), count=1),
        text,
        count=1,
    )
    if re.search(r"(?m)^impl\s*=", text):
        text = re.sub(r"(?m)^impl\s*=\s*\S+", f'impl = "{impl}"', text, count=1)
    else:
        text = re.sub(
            r"(?m)^(\[geophys\.pixgbm\]\s*\n)",
            rf'\1impl = "{impl}"\n',
            text,
            count=1,
        )
    text = re.sub(r"(?m)^budget_s\s*=\s*\d+", f"budget_s = {budget_s}", text)
    text = re.sub(r"(?m)^watchdog\s*=\s*true", "watchdog = false", text)
    toml.write_text(text, encoding="utf-8")
    return bak


def vendor_geo_pixgbm(repo: Path, inputs: dict[str, Path]) -> str | None:
    dest = repo / "expds" / "geo_pixgbm.py"
    if dest.is_file():
        return sha256_file(dest)
    for base in inputs.values():
        for cand in base.rglob("geo_pixgbm.py"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, dest)
            return sha256_file(dest)
    # try archive_review paths inside repo
    for cand in repo.rglob("geo_pixgbm.py"):
        if "archive" in str(cand) or "review" in str(cand) or "v2eb268d6" in str(cand):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, dest)
            return sha256_file(dest)
    return None


def install_models(repo: Path, src: Path, rel: str) -> dict:
    dst = repo / rel
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("poly.txt", "finds.txt"):
        # search
        hits = list(src.rglob(name)) if src.is_dir() else []
        if src.is_file() and src.name == name:
            hits = [src]
        if not hits:
            # maybe src is models root
            p = src / name
            if p.is_file():
                hits = [p]
        if not hits:
            out[name] = None
            continue
        shutil.copy2(hits[0], dst / name)
        out[name] = sha256_file(dst / name)
    return out


def run_main(repo: Path, regions: Path, result: Path, log_path: Path, env_extra: dict | None = None) -> int:
    result.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "EXPDS_PIXGBM_STRICT": os.environ.get("EXPDS_PIXGBM_STRICT", "1"),
        "PIXGBM_BUDGET_S": "100000",
        "EXPDS_SOFT_CEILING_S": "0",
    })
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, str(repo / "inference" / "main.py"), str(regions), str(result)]
    log(f"run {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"cmd={' '.join(cmd)}\nts={utc_now()}\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(repo), stdout=lf, stderr=subprocess.STDOUT, env=env, text=True)
    return proc.returncode


def find_score_one(repo: Path) -> Path | None:
    for cand in (
        repo / "score" / "score_one.py",
        repo / "materials" / "compute_metrics.py",
        repo / "scripts" / "e2e_score.py",
    ):
        if cand.is_file():
            return cand
    return None


def score_pair(repo: Path, gt: Path, ref_gj: Path, cand_gj: Path, scores_out: Path) -> int:
    """Best-effort score. Prefer score/score_one.py if present."""
    score_one = find_score_one(repo)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    if score_one and score_one.name == "score_one.py":
        # Typical: score_one.py pred gt --out scores.json (API varies; try common forms)
        trials = [
            [sys.executable, str(score_one), str(ref_gj), str(cand_gj), str(gt), str(scores_out)],
            [sys.executable, str(score_one), "--ref", str(ref_gj), "--new", str(cand_gj), "--gt", str(gt), "--out", str(scores_out)],
        ]
        for cmd in trials:
            log("score try", cmd)
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            if r.returncode == 0 and scores_out.is_file():
                return 0
            (OUT / "notes" / "score_try.log").parent.mkdir(parents=True, exist_ok=True)
            with (OUT / "notes" / "score_try.log").open("a", encoding="utf-8") as f:
                f.write(f"cmd={cmd}\nec={r.returncode}\nstdout={r.stdout[:2000]}\nstderr={r.stderr[:2000]}\n\n")
        return 19
    # Fallback: if caller attached a precomputed scores.json
    for p in Path("/kaggle/input").rglob("scores.json") if Path("/kaggle/input").is_dir() else []:
        shutil.copy2(p, scores_out)
        return 0
    write_status("FAILED", reason="no_score_tool")
    return 19


def run_gate(helpers: Path, scores: Path) -> int:
    """Gate expects JobDir layout: out_ref/, out_cand|out_new/, notes/run_meta.json, scores.json."""
    gate = helpers / "bot6_r3_gate.py"
    # Normalize out_cand -> out_new for gate
    cand = OUT / "out_cand"
    newp = OUT / "out_new"
    if cand.is_dir() and not newp.exists():
        try:
            newp.symlink_to(cand)
        except OSError:
            shutil.copytree(cand, newp)
    cmd = [
        sys.executable, str(gate),
        "--job", str(OUT),
        "--scores", str(scores),
        "--tol", str(TOL),
        "--log-ref", str(OUT / "out_ref" / "main.log"),
        "--log-new", str(OUT / "out_new" / "main.log"),
    ]
    log("gate", cmd)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / "notes" / "gate_stdout.txt").write_text(r.stdout or "", encoding="utf-8")
    (OUT / "notes" / "gate_stderr.txt").write_text(r.stderr or "", encoding="utf-8")
    # Mirror gate.json
    gj = OUT / "gate.json"
    if gj.is_file():
        shutil.copy2(gj, OUT / "gate_report.json")
    return r.returncode


def build_gt(helpers: Path, repo: Path, tree: Path, gt_out: Path) -> int:
    script = helpers / "bot6_r3_build_gt.py"
    if not script.is_file():
        return 18
    region_list = tree / "_region_list.txt"
    if not region_list.is_file():
        geos = sorted(p.name for p in tree.iterdir() if p.is_dir() and p.name[:3].isdigit() and 200 <= int(p.name[:3]) <= 217)
        region_list.write_text("\n".join(geos) + "\n", encoding="utf-8")
    cmd = [sys.executable, str(script), "--repo", str(repo), "--root", str(tree), "--out", str(gt_out), "--regions-from", str(region_list)]
    log("build_gt", cmd)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / "notes" / "build_gt.log").write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8")
    return r.returncode


def main() -> int:
    if os.environ.get("PIXGBM_CR_DRY_RUN", "").strip() in ("1", "true", "yes") or (
        len(sys.argv) > 1 and sys.argv[1] in ("--syntax", "--dry-run", "syntax", "dry-run")
    ):
        return dry_run_syntax()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "notes").mkdir(parents=True, exist_ok=True)
    (OUT / "out_ref").mkdir(parents=True, exist_ok=True)
    (OUT / "out_cand").mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", phase="start")
    try:
        helpers = ensure_helpers()
        inputs = find_inputs()
        wheeled = ensure_offline_wheels()
        mode = os.environ.get("GATE_MODE", "auto").strip().lower()
        log("mode", mode, "inputs", sorted(inputs), "wheels", wheeled)

        # Rebuild geo tree
        write_status("RUNNING", phase="rebuild")
        rebuild = load_rebuild()
        report = rebuild.rebuild(dest=TREE)
        (OUT / "notes" / "rebuild_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log("rebuild", {k: report.get(k) for k in ("n_geo_dirs", "n_geo_raster_files_guess", "hollow")})
        if report.get("hollow"):
            write_status("FAILED", phase="rebuild", reason="hollow_geo_tree", exit=2)
            return 2

        # Regions dir = TREE (full rebuilt train subset)
        regions = TREE
        meta = {
            "mode": mode,
            "tol": TOL,
            "geo_pix_sha_expected": GEO_PIX_SHA,
            "ref_poly_expected": REF_POLY,
            "ref_finds_expected": REF_FINDS,
            "rebuild": {k: report.get(k) for k in ("n_geo_dirs", "n_geo_raster_files_guess", "hollow", "geo_region_dirs")},
            "docker_ec": {},
            "t_start_epoch": {},
            "incontainer_ok": {"ref": True, "new": True},
            "result_identical": None,
            "model_sha_differ": None,
        }

        ref_gj = OUT / "out_ref" / "result.geojson"
        cand_gj = OUT / "out_cand" / "result.geojson"

        if mode == "score":
            # Allow pre-dropped geojsons via env or package notes
            for envk, dest in (("REF_GEOJSON", ref_gj), ("CAND_GEOJSON", cand_gj)):
                src = os.environ.get(envk)
                if src and Path(src).is_file():
                    shutil.copy2(src, dest)
            if not ref_gj.is_file() or not cand_gj.is_file():
                write_status("FAILED", reason="score_mode_missing_geojson")
                return 4
        else:
            # Locate REF / CAND via mount-name globs (robust to /kaggle/input slug variants)
            resolved = resolve_ref_cand(inputs, mode)
            meta["resolved"] = {
                "ref": str(resolved["ref"]) if resolved["ref"] else None,
                "cand": str(resolved["cand"]) if resolved["cand"] else None,
                "ref_kind": resolved["ref_kind"],
                "cand_kind": resolved["cand_kind"],
            }
            if resolved["errors"]:
                for e in resolved["errors"]:
                    log(e)
                write_status(
                    "FAILED",
                    reason="INVALID_missing_ref_or_cand",
                    errors=resolved["errors"],
                    inputs=sorted(inputs),
                )
                return 3

            # auto: REF output + CAND zip → hybrid (only CAND runs inference)
            if mode == "auto":
                if resolved["ref_kind"] == "output" and resolved["cand_kind"] == "zip":
                    mode = "hybrid"
                elif resolved["ref_kind"] == "zip" and resolved["cand_kind"] == "zip":
                    mode = "zip"
                elif resolved["ref_kind"] == "output" and resolved["cand_kind"] == "tip":
                    mode = "hybrid"
                else:
                    write_status(
                        "FAILED",
                        reason="INVALID_auto_cannot_resolve",
                        ref_kind=resolved["ref_kind"],
                        cand_kind=resolved["cand_kind"],
                        hint="Need REF result.geojson + CAND submission zip (preferred)",
                    )
                    return 3
                meta["mode"] = mode
                log("auto resolved GATE_MODE →", mode)

            pairs = []
            if mode == "hybrid":
                # Copy REF output; run CAND zip (or tip models via unpack) only.
                shutil.copy2(resolved["ref"], ref_gj)
                (OUT / "out_ref" / "main.log").write_text(
                    f"REF=v2_output_only\nsrc={resolved['ref']}\n", encoding="utf-8"
                )
                meta["docker_ec"]["ref"] = 0
                meta["t_start_epoch"]["ref"] = int(time.time())
                if resolved["cand_kind"] == "zip":
                    cand_repo = unpack_zip(resolved["cand"], OUT / "unpack_cand")
                else:
                    write_status("FAILED", reason="INVALID_hybrid_needs_cand_zip",
                                 cand_kind=resolved["cand_kind"])
                    return 3
                pairs = [("cand", cand_repo, cand_gj)]
            elif mode == "zip":
                if resolved["ref_kind"] != "zip" or resolved["cand_kind"] != "zip":
                    write_status(
                        "FAILED",
                        reason="INVALID_zip_mode_needs_zips",
                        ref_kind=resolved["ref_kind"],
                        cand_kind=resolved["cand_kind"],
                        hint="Drop two submission_*.zip or use GATE_MODE=auto with REF output",
                    )
                    return 3
                ref_repo = unpack_zip(resolved["ref"], OUT / "unpack_ref")
                cand_repo = unpack_zip(resolved["cand"], OUT / "unpack_cand")
                pairs = [("ref", ref_repo, ref_gj), ("cand", cand_repo, cand_gj)]
            else:
                # tip mode: one code tree, swap models + impl
                code = find_named(inputs, "expds-koz3", "koz3-code", "bot6-pixgbm", "code-snap")
                if code is None:
                    here = Path(__file__).resolve().parent
                    for cand in (here.parent.parent, Path("/kaggle/working/expds-koz3")):
                        if (cand / "inference" / "main.py").is_file():
                            code = cand
                            break
                if code is None or not (Path(code) / "inference" / "main.py").is_file():
                    hit = None
                    for p in Path("/kaggle/input").rglob("inference/main.py") if Path("/kaggle/input").is_dir() else []:
                        hit = p.parent.parent
                        break
                    code = hit
                # zip can also serve as code snapshot
                if code is None and resolved["ref_kind"] == "zip":
                    code = unpack_zip(resolved["ref"], OUT / "unpack_code_from_ref")
                if code is None:
                    write_status(
                        "FAILED",
                        reason="INVALID_no_code_snapshot",
                        hint="Attach a tip code tarball/dataset with inference/main.py, or use GATE_MODE=zip with full submission zips",
                    )
                    return 3
                code = Path(code)
                ref_repo = OUT / "work_ref"
                cand_repo = OUT / "work_cand"
                if ref_repo.exists():
                    shutil.rmtree(ref_repo)
                if cand_repo.exists():
                    shutil.rmtree(cand_repo)
                shutil.copytree(code, ref_repo, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"))
                shutil.copytree(code, cand_repo, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"))

                gsha = vendor_geo_pixgbm(ref_repo, inputs)
                meta["geo_pix_sha_ref"] = gsha
                if gsha != GEO_PIX_SHA:
                    log("WARN geo_pixgbm sha", gsha, "expected", GEO_PIX_SHA)
                if gsha is None:
                    write_status(
                        "FAILED",
                        reason="INVALID_tip_mode_refuses_en",
                        hint="REF is v2 output only; use GATE_MODE=auto|score with result.geojson. "
                             "Do not re-introduce EN geo_pixgbm.py.",
                    )
                    return 3

                ref_models = find_named(inputs, "pixgbm-cr-ref", "pixgbm-ref", "pixgbm-all", "models-pixgbm", "baseline-v2")
                cand_models = find_named(inputs, "pixgbm-cr-cand", "pix-clean", "pix_clean", "models-pix-clean", "models_pix_clean", "clean-models")
                if ref_models is None:
                    ref_models = resolved["ref"] if resolved["ref_kind"] == "tip" else None
                if cand_models is None:
                    cand_models = resolved["cand"] if resolved["cand_kind"] == "tip" else None
                if ref_models is None or cand_models is None:
                    write_status(
                        "FAILED",
                        reason="INVALID_tip_mode_missing_models",
                        ref=str(ref_models),
                        cand=str(cand_models),
                    )
                    return 3
                rh = install_models(ref_repo, Path(ref_models), "models/pixgbm/all")
                ch = install_models(cand_repo, Path(cand_models), "models/pix_clean")
                meta["ref_model_sha"] = rh
                meta["cand_model_sha"] = ch
                if not rh.get("poly.txt") or not rh.get("finds.txt"):
                    write_status("FAILED", reason="INVALID_ref_models_incomplete", sha=rh)
                    return 3
                if not ch.get("poly.txt") or not ch.get("finds.txt"):
                    write_status("FAILED", reason="INVALID_cand_models_incomplete", sha=ch)
                    return 3
                patch_toml(ref_repo, enabled=True, impl="legacy")
                patch_toml(cand_repo, enabled=True, impl="clean")
                pairs = [("ref", ref_repo, ref_gj), ("cand", cand_repo, cand_gj)]

            for name, repo, dest_gj in pairs:
                write_status("RUNNING", phase=f"infer_{name}")
                t_epoch = int(time.time())
                meta["t_start_epoch"][name if name != "cand" else "new"] = t_epoch
                ec = run_main(repo, regions, dest_gj, OUT / f"out_{name}" / "main.log")
                key = "new" if name == "cand" else name
                meta["docker_ec"][key] = ec
                log(name, "exit", ec, "geojson", dest_gj.is_file())
                if not dest_gj.is_file():
                    write_status("FAILED", phase=f"infer_{name}", exit=ec)
                    return 4 if name == "ref" else 5

        # GT + score + gate
        write_status("RUNNING", phase="gt_score_gate")
        # Prefer tip/code for catalog helpers
        repo_for_tools = None
        for cand in (OUT / "unpack_ref", OUT / "work_ref", OUT / "unpack_cand", OUT / "work_cand"):
            if (cand / "expds" / "catalog.py").is_file():
                repo_for_tools = cand
                break
        if repo_for_tools is None:
            for p in Path("/kaggle/input").rglob("expds/catalog.py") if Path("/kaggle/input").is_dir() else []:
                repo_for_tools = p.parent.parent
                break
        gt_out = OUT / "gt.geojson"
        if repo_for_tools is not None and not gt_out.is_file():
            gec = build_gt(helpers, repo_for_tools, TREE, gt_out)
            if gec != 0 or not gt_out.is_file():
                log("WARN build_gt failed", gec)

        scores = OUT / "scores.json"
        if repo_for_tools is not None and gt_out.is_file():
            sec = score_pair(repo_for_tools, gt_out, ref_gj, cand_gj, scores)
        else:
            sec = 19
            # allow preattached scores
            for p in Path("/kaggle/input").rglob("scores.json") if Path("/kaggle/input").is_dir() else []:
                shutil.copy2(p, scores)
                sec = 0
                break

        if ref_gj.is_file() and cand_gj.is_file():
            hr = sha256_file(ref_gj)
            hn = sha256_file(cand_gj)
            meta["result_sha"] = {"ref": hr, "new": hn}
            meta["result_identical"] = hr == hn

        meta_path = OUT / "notes" / "run_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if sec != 0 or not scores.is_file():
            write_status("FAILED", phase="score", exit=sec, reason="scores_missing")
            return 19

        gec = run_gate(helpers, scores)
        # Map gate exit to STATUS
        if gec == 0:
            write_status("DONE", verdict="PASS", gate_exit=0, tol=TOL)
            return 0
        if gec == 9:
            write_status("DONE", verdict="FAIL_EQUIV", gate_exit=9, tol=TOL)
            return 9
        if gec in (11, 12, 13, 14):
            write_status("DONE", verdict="INVALID", gate_exit=gec, tol=TOL)
            return gec
        write_status("FAILED", verdict="GATE_ERROR", gate_exit=gec)
        return gec if gec else 1
    except Exception as e:
        write_status("FAILED", error=repr(e)[:300])
        (OUT / "notes" / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
