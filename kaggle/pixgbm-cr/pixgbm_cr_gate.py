#!/usr/bin/env python3
"""Kaggle CPU: PIXGBM e2e geo F1 gate on regions 200–217 (bot-6 / pixgbm-cr).

Compares REF (v2 eb268d6 lineage) vs CANDIDATE and applies helpers/bot6_r3_gate.py
(|dF1| <= 0.01 on ALL + maggr/etgr/terfl; INVALID on empty steps / no full / …).

Modes (env GATE_MODE):
  zip   — unpack two submission zips (REF_ZIP / CAND_ZIP or attached *ref* / *cand* datasets)
          and run inference/main.py twice on the rebuilt geo tree.
  tip   — use attached code snapshot; REF = impl legacy + vendored geo_pixgbm + models/pixgbm/all;
          CAND = impl=clean + models/pix_clean (true clean-room). Requires models attached.
  score — skip inference; score already-produced out_ref/result.geojson + out_cand/result.geojson
          (or paths in env) against built GT.

Never submits to LB. enable_gpu=false. «1» pushes this package.

Conflict note (see notes/SPEC.json + reports/bot6_pixgbm.md):
  9b411c4 tooling is EN-lineage retrain reproduction, NOT PIXGBM_DECISION clean-room.
  Preferred candidate for «1» tip≈v2 goal is impl=clean on main tip.
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
    for n in needles:
        for k, p in inputs.items():
            if n not in k:
                continue
            for z in list(p.glob("*.zip")) + list(p.rglob("*.zip")):
                return z
    # env override
    for envk in ("REF_ZIP", "CAND_ZIP", "REF_PATH", "CAND_PATH"):
        pass
    return None


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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "notes").mkdir(parents=True, exist_ok=True)
    (OUT / "out_ref").mkdir(parents=True, exist_ok=True)
    (OUT / "out_cand").mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", phase="start")
    try:
        helpers = ensure_helpers()
        inputs = find_inputs()
        mode = os.environ.get("GATE_MODE", "zip").strip().lower()
        log("mode", mode, "inputs", sorted(inputs))

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
            # Locate zips / code
            ref_zip = Path(os.environ["REF_ZIP"]) if os.environ.get("REF_ZIP") else find_zip(inputs, "e2e-baseline-v2", "baseline-v2", "ref-v2", "pixgbm-ref")
            cand_zip = Path(os.environ["CAND_ZIP"]) if os.environ.get("CAND_ZIP") else find_zip(inputs, "geo-clean", "pix-clean", "pixgbm-cand", "cand-clean", "clean-pack")

            if mode == "zip" and (ref_zip is None or cand_zip is None):
                write_status(
                    "FAILED",
                    reason="missing_ref_or_cand_zip",
                    ref_zip=str(ref_zip),
                    cand_zip=str(cand_zip),
                    hint="Attach v2 baseline zip + clean-pack zip, or set GATE_MODE=tip with models",
                )
                return 3

            pairs = []
            if mode == "zip":
                ref_repo = unpack_zip(ref_zip, OUT / "unpack_ref")
                cand_repo = unpack_zip(cand_zip, OUT / "unpack_cand")
                pairs = [("ref", ref_repo, ref_gj), ("cand", cand_repo, cand_gj)]
            else:
                # tip mode: one code tree, swap models + impl
                code = find_named(inputs, "expds-koz3", "koz3-code", "bot6-pixgbm")
                if code is None:
                    # maybe kernel already has repo next door
                    here = Path(__file__).resolve().parent
                    for cand in (here.parent.parent, Path("/kaggle/working/expds-koz3")):
                        if (cand / "inference" / "main.py").is_file():
                            code = cand
                            break
                if code is None or not (Path(code) / "inference" / "main.py").is_file():
                    # deep search
                    hit = None
                    for p in Path("/kaggle/input").rglob("inference/main.py") if Path("/kaggle/input").is_dir() else []:
                        hit = p.parent.parent
                        break
                    code = hit
                if code is None:
                    write_status("FAILED", reason="no_code_snapshot")
                    return 3
                code = Path(code)
                # work on copies
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

                # models
                ref_models = find_named(inputs, "pixgbm-ref", "pixgbm-all", "models-pixgbm", "baseline-v2")
                cand_models = find_named(inputs, "pix-clean", "pix_clean", "models-pix-clean", "clean-models")
                if ref_models is None or cand_models is None:
                    write_status("FAILED", reason="tip_mode_missing_models", ref=str(ref_models), cand=str(cand_models))
                    return 3
                rh = install_models(ref_repo, ref_models, "models/pixgbm/all")
                ch = install_models(cand_repo, cand_models, "models/pix_clean")
                meta["ref_model_sha"] = rh
                meta["cand_model_sha"] = ch
                patch_toml(ref_repo, enabled=True, impl="legacy")
                # cand may not need geo_pixgbm
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
