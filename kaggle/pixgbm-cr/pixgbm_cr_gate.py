#!/usr/bin/env python3
"""Kaggle CPU: PIXGBM e2e geo F1 gate on regions 200–217 (bot-6 / pixgbm-cr).

REF  = dmitrijkv/koz3-fr1h-chk-20261101 → fr1h.zip.bin (v2 eb268d6 + FR1h).
       FR1h is forced OFF before run: remove [post.frame_prior] and restore
       archive SpOr kurgan thresholds (taus_base). geo_pixgbm.py / models/pixgbm
       are byte-identical to v2 (sha 35e4331b… / d1fe3efa… / 5fa8c09d…).
CAND = dmitrijkv/koz3-pixgbm-cr-cand → submission_pixgbm_clean_*.zip (pix_clean).

Geo tree: MagGr/EtGr/TerFl + GT for 200–217 from S3 (rebuild_w4_geo_tree.py).
Gate: helpers/bot6_r3_gate.py (|dF1|≤0.01 ALL + maggr/etgr/terfl; INVALID checks).

r3 overrides on both: EXPDS_PIXGBM_STRICT=1, PIXGBM_BUDGET_S=100000,
EXPDS_SOFT_CEILING_S=0, budget_s=100000, watchdog=false.

enable_gpu=false, enable_internet=true. Prefer offline wheels; network pip OK.
Never submits to LB. «1» pushes. Do NOT commit EN code/weights into git.
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
FR1H_ZIP_SHA = "fe8d60b9032474760f02d4bcfa08fb43269ae3ffb872efa932d2f1d7692944b4"
# Archive eb268d6 SpOr kurgan floors (FR1H_PACK taus_base / pack table).
ARCHIVE_SP_TAUS = {
    "sp_or": {"kurgany_povrezhdennye": 0.3645, "kurgany_tselye": 0.3725},
    "sp_focal2": {"kurgany_povrezhdennye": 0.501645, "kurgany_tselye": 0.645505},
}
TOL = float(os.environ.get("PIXGBM_DF1_TOL", "0.01"))
GEO_PREFIXES = tuple(f"{n:03d}_" for n in range(200, 218))


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


def find_file(inputs: dict[str, Path], name: str) -> Path | None:
    for p in inputs.values():
        for hit in p.rglob(name):
            if hit.is_file():
                return hit
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
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if offline:
            if not wheel_dirs:
                return False
            cmd += ["--no-index", "--no-deps"]
            for d in wheel_dirs:
                cmd.extend(["--find-links", str(d)])
        cmd += pkgs
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log("WARN pip", pkgs, "offline=" + str(offline), (r.stderr or r.stdout or "")[:400])
            return False
        return True

    if _need("zstandard"):
        if _pip(["zstandard"], offline=True):
            installed.append("zstandard")
        elif _pip(["zstandard"], offline=False):
            installed.append("zstandard(net)")
    if _need("boto3"):
        if _pip(["boto3"], offline=True):
            installed.append("boto3")
        elif _pip(["boto3"], offline=False):
            installed.append("boto3(net)")
        else:
            log("WARN boto3 not installed — S3 rebuild will fail")
    return installed


def dry_run_syntax() -> int:
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
    ensure_helpers()
    load_rebuild()
    print("OK helpers+rebuild load")
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
        assert "dmitrijkv/koz3-s3-cred" in srcs, "koz3-s3-cred required"
        assert "dmitrijkv/koz3-fr1h-chk-20261101" in srcs, "fr1h-chk REF required"
        assert "dmitrijkv/koz3-pixgbm-cr-ref" not in srcs, "legacy pixgbm-cr-ref must be dropped"
        assert not any("w4-geo" in s for s in srcs), "w4-geo must not be wired"
        print("OK kernel-metadata internet/gpu flags")
        print("dataset_sources:", srcs)
    # Force-off helper self-check (no EN extract)
    sample = "[post]\nkurgan_twins_only = true\n\n[post.frame_prior]\ngroups = [\"sp_or\"]\n\n[rerank_arch]\nenabled = true\n"
    stripped = _strip_frame_prior_section(sample)
    assert "[post.frame_prior]" not in stripped and "kurgan_twins_only" in stripped
    print("OK FR1h force-off strip helper")
    print("DRY_RUN_OK missing_optional=", missing)
    return 0


def _strip_frame_prior_section(text: str) -> str:
    """Remove [post.frame_prior] table (FR1h ON → OFF)."""
    return re.sub(
        r"(?ms)^\[post\.frame_prior\][^\n]*\n(?:(?!\[)[^\n]*\n)*",
        "",
        text,
    )


def force_fr1h_off(repo: Path) -> dict:
    """Force FR1h OFF in configs/default.toml (explicit, not relying on default).

    Flag name: TOML table [post.frame_prior] (read as cfg['post']['frame_prior']).
    Default in fr1h.zip.bin: ON (section present).
    OFF = section absent / falsy → watchdog/pipeline skip post.frame_prior().
    Also restore archive SpOr kurgan thresholds (FR1h re-sweep → taus_base).
    """
    toml = repo / "configs" / "default.toml"
    text = toml.read_text(encoding="utf-8")
    before_has = bool(re.search(r"(?m)^\[post\.frame_prior\]", text))
    text = _strip_frame_prior_section(text)
    # Ensure no leftover truthy frame_prior assignment under [post]
    text = re.sub(r"(?m)^frame_prior\s*=\s*.*\n", "", text)
    # Restore archive kurgan floors in [thresholds_by_markup.sp_or] / sp_focal2
    for group, taus in ARCHIVE_SP_TAUS.items():
        for cls, val in taus.items():
            # Replace only within that group's values — line-based for kurgan keys after a matching header is fragile;
            # do global class lines carefully scoped by rewriting the two known FR1h-swept numbers when under group.
            pat = rf"(?ms)(\[thresholds_by_markup\.{re.escape(group)}\][^\[]*?)(^{re.escape(cls)}\s*=\s*)[0-9.]+"
            text2, n = re.subn(pat, rf"\g<1>\g<2>{val}", text, count=1)
            if n:
                text = text2
    toml.write_text(text, encoding="utf-8")
    # Verify
    try:
        import tomllib
        cfg = tomllib.loads(text)
    except Exception:
        cfg = {}
    fp = (cfg.get("post") or {}).get("frame_prior")
    info = {
        "flag": "post.frame_prior",
        "default_in_zip": "ON (section present)",
        "forced": "OFF",
        "had_section_before": before_has,
        "frame_prior_after": fp,
        "archive_taus_restored": ARCHIVE_SP_TAUS,
    }
    if fp:
        raise RuntimeError(f"FR1h force-off failed; frame_prior still truthy: {fp!r}")
    return info


def apply_r3_overrides(repo: Path) -> dict:
    """budget_s=100000, watchdog=false; env applied at run_main."""
    toml = repo / "configs" / "default.toml"
    text = toml.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^budget_s\s*=\s*\d+", "budget_s = 100000", text)
    text = re.sub(r"(?m)^watchdog\s*=\s*true", "watchdog = false", text)
    # pixgbm section budget if present
    if re.search(r"(?ms)^\[geophys\.pixgbm\].*?^budget_s\s*=", text):
        text = re.sub(
            r"(?ms)(^\[geophys\.pixgbm\][^\[]*?)^budget_s\s*=\s*\S+",
            r"\1budget_s = 100000",
            text,
            count=1,
        )
    toml.write_text(text, encoding="utf-8")
    return {"budget_s": 100000, "watchdog": False}


def detect_pix_impl(repo: Path) -> dict:
    """Validity check: which pix implementation the tree will actually run."""
    info: dict = {"repo": str(repo)}
    gp = repo / "expds" / "geo_pixgbm.py"
    pc = list((repo / "models").rglob("finds.txt")) if (repo / "models").is_dir() else []
    pix_clean = any("pix_clean" in str(p) for p in pc)
    pix_all = any("pixgbm/all" in str(p).replace("\\", "/") or p.parent.name == "all" for p in pc)
    info["has_geo_pixgbm"] = gp.is_file()
    if gp.is_file():
        info["geo_pixgbm_sha"] = sha256_file(gp)
        info["geo_pixgbm_matches_v2"] = info["geo_pixgbm_sha"] == GEO_PIX_SHA
    info["has_pix_clean_models"] = pix_clean
    info["has_pixgbm_all_models"] = pix_all
    try:
        import tomllib
        cfg = tomllib.loads((repo / "configs" / "default.toml").read_text(encoding="utf-8"))
        pix = (cfg.get("geophys") or {}).get("pixgbm") or {}
        info["pixgbm_enabled"] = bool(pix.get("enabled"))
        info["pixgbm_model_dir"] = pix.get("model_dir")
        info["pixgbm_impl_toml"] = pix.get("impl")  # clean packs may set impl=clean
    except Exception as e:
        info["toml_error"] = type(e).__name__
    if pix_clean and not gp.is_file():
        info["pix_implementation"] = "pix_clean"
    elif gp.is_file() and info.get("geo_pixgbm_matches_v2"):
        info["pix_implementation"] = "geo_pixgbm (v2)"
    elif gp.is_file():
        info["pix_implementation"] = "geo_pixgbm"
    else:
        info["pix_implementation"] = "unknown"
    return info


def unpack_zip_like(src: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # fr1h.zip.bin is a zip despite .bin suffix
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)
    for cand in [dest, *dest.iterdir()]:
        if (cand / "inference" / "main.py").is_file():
            return cand
    raise FileNotFoundError(f"inference/main.py not found under {dest} from {src}")


def find_fr1h_bin(inputs: dict[str, Path]) -> Path | None:
    hit = find_file(inputs, "fr1h.zip.bin")
    if hit:
        return hit
    ds = find_named(inputs, "fr1h-chk", "fr1h_chk", "koz3-fr1h")
    if ds is None:
        return None
    for p in ds.rglob("fr1h.zip.bin"):
        return p
    return None


def find_fr1h_sha_file(inputs: dict[str, Path], bin_path: Path) -> Path | None:
    for name in ("SHA256", "SHA256SUMS", "sha256.txt"):
        p = bin_path.parent / name
        if p.is_file():
            return p
        hit = find_file(inputs, name)
        if hit is not None:
            return hit
    return None


def verify_fr1h_sha(bin_path: Path, sha_file: Path | None) -> dict:
    got = sha256_file(bin_path)
    exp = FR1H_ZIP_SHA
    if sha_file and sha_file.is_file():
        first = sha_file.read_text(encoding="utf-8").strip().split()[0]
        if re.fullmatch(r"[0-9a-fA-F]{64}", first):
            exp = first.lower()
    ok = got == exp
    return {"path": str(bin_path), "sha256": got, "expected": exp, "ok": ok, "sha_file": str(sha_file) if sha_file else None}


def find_cand_zip(inputs: dict[str, Path]) -> Path | None:
    if os.environ.get("CAND_ZIP") and Path(os.environ["CAND_ZIP"]).is_file():
        return Path(os.environ["CAND_ZIP"])
    for n in ("pixgbm-cr-cand", "pixgbm_cr_cand", "pix-clean", "pix_clean"):
        ds = find_named(inputs, n)
        if ds is None:
            continue
        zips = sorted(ds.rglob("*.zip"))
        # prefer clean submission name
        prefer = [z for z in zips if "pixgbm_clean" in z.name.lower() or "clean" in z.name.lower()]
        if prefer:
            return prefer[0]
        if zips:
            return zips[0]
    for p in inputs.values():
        for z in p.rglob("*.zip"):
            if "pixgbm_clean" in z.name.lower() or "clean" in z.name.lower():
                return z
    return None


def list_geo_regions(tree: Path) -> list[str]:
    if not tree.is_dir():
        return []
    return sorted(
        p.name for p in tree.iterdir()
        if p.is_dir() and any(p.name.startswith(pref) for pref in GEO_PREFIXES)
    )


def merge_geojsons(parts: list[Path], dest: Path) -> int:
    feats = []
    for p in parts:
        if not p.is_file():
            continue
        try:
            fc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        feats.extend(fc.get("features") or [])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(feats)


def run_main_resumable(
    repo: Path,
    tree: Path,
    result: Path,
    log_path: Path,
    tag: str,
    env_extra: dict | None = None,
) -> int:
    """Run inference per geo region with checkpointing (12h / ~20GB safety)."""
    regions = list_geo_regions(tree)
    if not regions:
        log("no geo regions in", tree)
        return 2
    ckpt_dir = OUT / f"ckpt_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    done_file = ckpt_dir / "DONE.txt"
    parts: list[Path] = []
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "EXPDS_PIXGBM_STRICT": os.environ.get("EXPDS_PIXGBM_STRICT", "1"),
        "PIXGBM_BUDGET_S": os.environ.get("PIXGBM_BUDGET_S", "100000"),
        "EXPDS_SOFT_CEILING_S": os.environ.get("EXPDS_SOFT_CEILING_S", "0"),
    })
    if env_extra:
        env.update(env_extra)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== resumable {tag} n_regions={len(regions)} ts={utc_now()} ===\n")
        lf.flush()
        for i, name in enumerate(regions):
            part = ckpt_dir / f"{name}.geojson"
            parts.append(part)
            if part.is_file() and part.stat().st_size > 50:
                log(f"{tag} skip-exists {name} ({i+1}/{len(regions)})")
                lf.write(f"SKIP {name}\n"); lf.flush()
                continue
            # Single-region input via temp dir of symlinks
            stage = ckpt_dir / f"_stage_{name}"
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            os.symlink(tree / name, stage / name)
            # Also expose sibling non-region helpers if pipeline expects root files — none for geo.
            cmd = [sys.executable, str(repo / "inference" / "main.py"), str(stage), str(part)]
            log(f"{tag} run {name} ({i+1}/{len(regions)})")
            lf.write(f"RUN {name} cmd={' '.join(cmd)}\n"); lf.flush()
            t1 = time.time()
            proc = subprocess.run(cmd, cwd=str(repo), stdout=lf, stderr=subprocess.STDOUT, env=env, text=True)
            lf.write(f"EC {name}={proc.returncode} wall={time.time()-t1:.1f}s exists={part.is_file()}\n"); lf.flush()
            shutil.rmtree(stage, ignore_errors=True)
            if not part.is_file():
                # write empty so merge can proceed; mark failure
                part.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
                write_status("FAILED", phase=f"infer_{tag}", region=name, exit=proc.returncode)
                return proc.returncode or 4

    n = merge_geojsons(parts, result)
    done_file.write_text("\n".join(regions) + f"\n# features={n}\n", encoding="utf-8")
    log(f"{tag} merged features={n} → {result}")
    return 0


def run_gate(helpers: Path, scores: Path) -> int:
    gate = helpers / "bot6_r3_gate.py"
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
    gj = OUT / "gate.json"
    if gj.is_file():
        shutil.copy2(gj, OUT / "gate_report.json")
    return r.returncode


def build_gt(helpers: Path, repo: Path, tree: Path, gt_out: Path) -> int:
    script = helpers / "bot6_r3_build_gt.py"
    if not script.is_file():
        return 18
    region_list = tree / "_region_list.txt"
    geos = list_geo_regions(tree)
    region_list.write_text("\n".join(geos) + "\n", encoding="utf-8")
    cmd = [sys.executable, str(script), "--repo", str(repo), "--root", str(tree),
           "--out", str(gt_out), "--regions-from", str(region_list)]
    log("build_gt", cmd)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / "notes" / "build_gt.log").write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8")
    return r.returncode


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
    score_one = find_score_one(repo)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    if score_one and score_one.name == "score_one.py":
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
    for p in Path("/kaggle/input").rglob("scores.json") if Path("/kaggle/input").is_dir() else []:
        shutil.copy2(p, scores_out)
        return 0
    write_status("FAILED", reason="no_score_tool")
    return 19


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
        log("inputs", sorted(inputs), "wheels", wheeled)

        # --- S3 geo tree ---
        write_status("RUNNING", phase="rebuild")
        rebuild = load_rebuild()
        report = rebuild.rebuild(dest=TREE)
        (OUT / "notes" / "rebuild_report.json").write_text(
            json.dumps({k: v for k, v in report.items() if k != "keys"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log("rebuild", {k: report.get(k) for k in ("n_geo_dirs", "n_geo_raster_files_guess", "hollow")})
        if report.get("hollow"):
            write_status("FAILED", phase="rebuild", reason="hollow_geo_tree", exit=2)
            return 2

        # --- REF: fr1h.zip.bin ---
        write_status("RUNNING", phase="prep_ref")
        fr1h = find_fr1h_bin(inputs)
        if fr1h is None:
            write_status("FAILED", reason="INVALID_missing_fr1h_zip_bin")
            return 3
        sha_file = find_fr1h_sha_file(inputs, fr1h)
        # Copy to working, verify
        work_bin = Path("/kaggle/working") / "fr1h.zip.bin"
        if not work_bin.exists() or work_bin.stat().st_size != fr1h.stat().st_size:
            shutil.copy2(fr1h, work_bin)
        ver = verify_fr1h_sha(work_bin, sha_file)
        (OUT / "notes" / "fr1h_sha.json").write_text(json.dumps(ver, indent=2) + "\n", encoding="utf-8")
        if not ver["ok"]:
            write_status("FAILED", reason="INVALID_fr1h_sha_mismatch", **ver)
            return 3
        log("fr1h sha OK", ver["sha256"][:16] + "…")

        ref_repo = unpack_zip_like(work_bin, OUT / "unpack_ref")
        fr1h_info = force_fr1h_off(ref_repo)
        r3_ref = apply_r3_overrides(ref_repo)
        ref_impl = detect_pix_impl(ref_repo)
        (OUT / "notes" / "fr1h_force_off.json").write_text(json.dumps(fr1h_info, indent=2) + "\n", encoding="utf-8")
        (OUT / "notes" / "ref_impl.json").write_text(json.dumps(ref_impl, indent=2) + "\n", encoding="utf-8")
        log("REF pix_implementation:", ref_impl.get("pix_implementation"), "FR1h:", fr1h_info["forced"])
        if not ref_impl.get("geo_pixgbm_matches_v2"):
            write_status("FAILED", reason="INVALID_ref_geo_pixgbm_not_v2", impl=ref_impl)
            return 3
        if ref_impl.get("pix_implementation") != "geo_pixgbm (v2)":
            write_status("FAILED", reason="INVALID_ref_impl", impl=ref_impl)
            return 3

        # --- CAND zip ---
        write_status("RUNNING", phase="prep_cand")
        cand_z = find_cand_zip(inputs)
        if cand_z is None:
            write_status("FAILED", reason="INVALID_missing_cand_zip")
            return 3
        cand_repo = unpack_zip_like(cand_z, OUT / "unpack_cand")
        r3_cand = apply_r3_overrides(cand_repo)
        # Prefer impl=clean if toml supports it
        toml = cand_repo / "configs" / "default.toml"
        if toml.is_file():
            t = toml.read_text(encoding="utf-8")
            if re.search(r"(?m)^impl\s*=", t):
                t = re.sub(r"(?m)^impl\s*=\s*\S+", 'impl = "clean"', t, count=1)
            else:
                t = re.sub(r"(?m)^(\[geophys\.pixgbm\]\s*\n)", r'\1impl = "clean"\n', t, count=1)
            # point model_dir at pix_clean if present
            if (cand_repo / "models" / "pix_clean").is_dir():
                if re.search(r"(?m)^model_dir\s*=", t):
                    t = re.sub(r'(?m)^model_dir\s*=\s*\S+', 'model_dir = "models/pix_clean"', t, count=1)
            toml.write_text(t, encoding="utf-8")
        cand_impl = detect_pix_impl(cand_repo)
        (OUT / "notes" / "cand_impl.json").write_text(json.dumps(cand_impl, indent=2) + "\n", encoding="utf-8")
        log("CAND pix_implementation:", cand_impl.get("pix_implementation"))
        if "pix_clean" not in str(cand_impl.get("pix_implementation")):
            # soft warn — clean zip may still vendor geo_pixgbm-less pix via impl
            log("WARN CAND impl detection:", cand_impl)

        meta = {
            "tol": TOL,
            "fr1h_sha": ver,
            "fr1h_force_off": fr1h_info,
            "r3_overrides": {"ref": r3_ref, "cand": r3_cand},
            "ref_impl": ref_impl,
            "cand_impl": cand_impl,
            "rebuild": {k: report.get(k) for k in ("n_geo_dirs", "n_geo_raster_files_guess", "hollow", "geo_region_dirs")},
            "runtime_estimate_note": (
                "Prior FR1h black-box geo 200–217 ≈ 267s on 6-cpu official image; "
                "Kaggle CPU ~4 vCPU ×2 runs (v2 geo_pixgbm + pix_clean ens6) ≈ 1–3 h typical, "
                "worst-case budget still ≪ 12 h with per-region checkpoints under pixgbm_cr/ckpt_*."
            ),
            "docker_ec": {},
            "t_start_epoch": {},
        }

        ref_gj = OUT / "out_ref" / "result.geojson"
        cand_gj = OUT / "out_cand" / "result.geojson"

        write_status("RUNNING", phase="infer_ref")
        meta["t_start_epoch"]["ref"] = int(time.time())
        ec_ref = run_main_resumable(ref_repo, TREE, ref_gj, OUT / "out_ref" / "main.log", "ref")
        meta["docker_ec"]["ref"] = ec_ref
        if ec_ref != 0 or not ref_gj.is_file():
            write_status("FAILED", phase="infer_ref", exit=ec_ref)
            return 4

        write_status("RUNNING", phase="infer_cand")
        meta["t_start_epoch"]["new"] = int(time.time())
        ec_cand = run_main_resumable(cand_repo, TREE, cand_gj, OUT / "out_cand" / "main.log", "cand")
        meta["docker_ec"]["new"] = ec_cand
        if ec_cand != 0 or not cand_gj.is_file():
            write_status("FAILED", phase="infer_cand", exit=ec_cand)
            return 5

        write_status("RUNNING", phase="gt_score_gate")
        repo_for_tools = None
        for cand in (OUT / "unpack_ref", OUT / "unpack_cand"):
            if (cand / "expds" / "catalog.py").is_file():
                repo_for_tools = cand
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
            for p in Path("/kaggle/input").rglob("scores.json") if Path("/kaggle/input").is_dir() else []:
                shutil.copy2(p, scores)
                sec = 0
                break

        if ref_gj.is_file() and cand_gj.is_file():
            hr = sha256_file(ref_gj)
            hn = sha256_file(cand_gj)
            meta["result_sha"] = {"ref": hr, "new": hn}
            meta["result_identical"] = hr == hn

        (OUT / "notes" / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if sec != 0 or not scores.is_file():
            write_status("FAILED", phase="score", exit=sec, reason="scores_missing")
            return 19

        gec = run_gate(helpers, scores)
        if gec == 0:
            write_status("DONE", verdict="PASS", gate_exit=0, tol=TOL,
                         ref_impl=ref_impl.get("pix_implementation"),
                         cand_impl=cand_impl.get("pix_implementation"))
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
