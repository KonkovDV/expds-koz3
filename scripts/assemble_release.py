#!/usr/bin/env python3
"""ONE-command release assembler: main tip → submission candidate kit + checks.

Never pushes to main. Does not commit weights. Does not submit to Kaggle/LB.

Pipeline:
  1) inject v2 artifacts (scripts/v2_release_artifacts.py) with loud sha256 fail
  2) apply release profile (FR1h / sat2 / D-FINE Ae; default OFF)
  3) build kit zip via build_submission_git.py (--config effective)
  4) optional: OFF==v2 byte-for-byte compare (--v2-ref scored output or kit)
  5) compliance: kit DOC-1 clean; scored zip ≤100MB / no kit files / validator;
     runtime ≤20 min (budget_guard)

Usage (see docs/RELEASE_BUILD.md):
  python scripts/assemble_release.py --profile off
  python scripts/assemble_release.py --profile fr1h --out-dir dist/release
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOFT_CEILING_S = 1200.0  # 20 min
MAX_SUBMISSION_MB = 100.0


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True)


def step_inject(*, skip: bool, check_only: bool = False) -> int:
    if skip:
        print("SKIP inject (--skip-inject)")
        return 0
    args = [sys.executable, str(ROOT / "scripts" / "v2_release_artifacts.py")]
    if check_only:
        args.append("--check")
    r = _run(args)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        print(
            "FAIL v2_release_artifacts: sha256 mismatch or missing artifact "
            "(see scripts/v2_release_artifacts.py MANIFEST)",
            file=sys.stderr,
        )
    return r.returncode


def step_profile(name: str, out_dir: Path) -> tuple[dict, Path, int]:
    out_toml = out_dir / f"effective_{name}.toml"
    r = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "apply_release_profile.py"),
            "--profile",
            name,
            "--root",
            str(ROOT),
            "--out",
            str(out_toml),
        ]
    )
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return {}, out_toml, r.returncode
    # reload profile meta
    sys.path.insert(0, str(ROOT / "scripts"))
    from apply_release_profile import load_profile  # type: ignore

    return load_profile(name, ROOT / "configs" / "release" / "profiles"), out_toml, 0


def step_sat2_artifact(profile: dict, out_dir: Path, sat2_pack: Path | None) -> int:
    feats = profile.get("features") or {}
    if not feats.get("sat2"):
        return 0
    art = (profile.get("artifacts") or {}).get("sat2_replace_pack")
    src = sat2_pack or (Path(art) if art else None)
    if src is None:
        print(
            "FAIL sat2=ON but no pack: pass --sat2-pack PATH or set "
            "[artifacts].sat2_replace_pack in the profile",
            file=sys.stderr,
        )
        return 1
    src = Path(src)
    if not src.is_file():
        print(f"FAIL sat2 pack missing: {src}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "sat2_replace_pack.zip"
    shutil.copy2(src, dst)
    print(f"OK sat2 pack → {dst} ({dst.stat().st_size} B)")
    return 0


def step_build(label: str, effective: Path, out_dir: Path, *, skip: bool) -> tuple[Path | None, int]:
    if skip:
        print("SKIP build (--skip-build)")
        return None, 0
    # Prefer git build for reproducibility; weights from working tree via --weight if needed later.
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_submission_git.py"),
        "--repo",
        str(ROOT),
        "--ref",
        "HEAD",
        "--label",
        label,
        "--config",
        str(effective),
        "--out-dir",
        str(out_dir),
    ]
    r = _run(cmd)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        print("FAIL build_submission_git", file=sys.stderr)
        return None, r.returncode or 1
    # find newest zip in out_dir matching label
    zips = sorted(out_dir.glob(f"koz3_{label}_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        # fallback any zip
        zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        print("FAIL: no kit zip produced", file=sys.stderr)
        return None, 1
    print(f"OK kit zip → {zips[0]}")
    return zips[0], 0


def step_doc1(kit: Path | None) -> int:
    if kit is None:
        print("SKIP DOC-1 (no kit)")
        return 0
    r = _run([sys.executable, str(ROOT / "scripts" / "check_archive_clean.py"), str(kit)])
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        print("FAIL DOC-1 / archive clean", file=sys.stderr)
    return r.returncode


def step_byte_compare(v2_ref: Path | None, candidate: Path | None, *, required_for_off: bool, is_off: bool) -> int:
    if v2_ref is None:
        if required_for_off and is_off:
            print(
                "WARN OFF profile: pass --v2-ref PATH for byte-identity check "
                "(or file PC request in shared koz3/pc/QUEUE.md)",
                file=sys.stderr,
            )
        return 0
    if candidate is None:
        print("FAIL --v2-ref set but no candidate path (build skipped or --result missing)", file=sys.stderr)
        return 1
    r = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_submission_bytes.py"),
            str(v2_ref),
            str(candidate),
        ]
    )
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def step_submission_compliance(result: Path | None) -> int:
    if result is None:
        print("SKIP scored-submission compliance (no --result)")
        return 0
    # kit-member scan
    sys.path.insert(0, str(ROOT / "scripts"))
    from kit_denylist import kit_members  # type: ignore

    if result.suffix.lower() == ".zip" or zipfile.is_zipfile(result):
        with zipfile.ZipFile(result) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
        bad = kit_members(names)
        if bad:
            print("FAIL kit files inside scored submission:", file=sys.stderr)
            for b in bad[:30]:
                print(f"  {b}", file=sys.stderr)
            return 1
        mb = result.stat().st_size / 1_000_000
        if mb > MAX_SUBMISSION_MB:
            print(f"FAIL scored zip {mb:.1f} MB > {MAX_SUBMISSION_MB} MB", file=sys.stderr)
            return 1
    # validator / check_compliance
    r = _run([sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(result)])
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def step_budget(effective: Path, wall_sec: float | None, wall_file: Path | None) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "budget_guard.py"),
        "--config",
        str(effective),
        "--max-sec",
        str(SOFT_CEILING_S),
        "--require-soft-ceiling",
    ]
    if wall_sec is not None:
        cmd += ["--wall-sec", str(wall_sec)]
    if wall_file is not None:
        cmd += ["--wall-file", str(wall_file)]
    r = _run(cmd)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def write_manifest(out_dir: Path, payload: dict) -> Path:
    path = out_dir / "RELEASE_MANIFEST.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OK manifest → {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True, help="configs/release/profiles/<name>.toml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "dist" / "release")
    ap.add_argument("--skip-inject", action="store_true")
    ap.add_argument("--skip-build", action="store_true", help="materialize + checks only")
    ap.add_argument("--v2-ref", type=Path, default=None, help="reference file for OFF==v2 byte compare")
    ap.add_argument(
        "--result",
        type=Path,
        default=None,
        help="scored submission (result.geojson|zip) for ≤100MB / no-kit / validator",
    )
    ap.add_argument("--sat2-pack", type=Path, default=None)
    ap.add_argument("--wall-sec", type=float, default=None)
    ap.add_argument("--wall-file", type=Path, default=None)
    ap.add_argument("--label", default=None, help="kit label (default: rel_<profile>)")
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or f"rel_{args.profile}"
    rc = 0

    # 1 inject
    irc = step_inject(skip=args.skip_inject)
    if irc:
        return irc

    # 2 profile
    profile, effective, prc = step_profile(args.profile, out_dir)
    if prc:
        return prc
    is_off = not any((profile.get("features") or {}).values())

    # 2b sat2 artifact
    src = step_sat2_artifact(profile, out_dir, args.sat2_pack)
    if src:
        return src

    # 3 build kit
    kit, brc = step_build(label, effective, out_dir, skip=args.skip_build)
    if brc:
        return brc

    # 5a DOC-1 on kit
    drc = step_doc1(kit)
    rc = rc or drc

    # 4 byte compare: prefer --result vs --v2-ref; else kit vs --v2-ref
    cand = args.result or kit
    crc = step_byte_compare(args.v2_ref, cand, required_for_off=True, is_off=is_off)
    rc = rc or crc

    # 5b scored compliance
    src = step_submission_compliance(args.result)
    rc = rc or src

    # 5c budget
    brc = step_budget(effective, args.wall_sec, args.wall_file)
    rc = rc or brc

    tip = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_manifest(
        out_dir,
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "profile": profile.get("_name"),
            "features": profile.get("features"),
            "tip_sha": tip,
            "effective_toml": str(effective),
            "kit_zip": str(kit) if kit else None,
            "result": str(args.result) if args.result else None,
            "v2_ref": str(args.v2_ref) if args.v2_ref else None,
            "soft_ceiling_s": SOFT_CEILING_S,
            "ok": rc == 0,
        },
    )
    if rc == 0:
        print(f"PASS assemble_release profile={args.profile}")
    else:
        print(f"FAIL assemble_release profile={args.profile} rc={rc}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
