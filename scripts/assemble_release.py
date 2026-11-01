#!/usr/bin/env python3
"""ONE-command release assembler: main tip → submission candidate kit + checks.

Never pushes to main. Does not commit weights. Does not submit to Kaggle/LB.

Pipeline:
  1) inject v2 artifacts (scripts/v2_release_artifacts.py) with loud sha256 fail
  2) apply release profile (FR1h / sat2 / D-FINE Ae; default OFF)
  3) build kit zip via build_submission_git.py (--config effective)
  4) OFF==v2: --v2-ref (geojson OR submission zip) byte compare + ref sha256 in
     report; until Drive REF arrives, default OFF stand-in = inject --check
     (v2 reproduced via v2_release_artifacts.py)
  5) compliance: kit DOC-1 clean (+ kit size informational); scored SUBMISSION
     zip ≤100MB / no kit files / validator; runtime ≤20 min (budget_guard)

Usage (see docs/RELEASE_BUILD.md):
  python scripts/assemble_release.py --profile off
  python scripts/assemble_release.py --profile fr1h --out-dir dist/release
  python scripts/assemble_release.py --profile dfine_ae --ckpt PATH --sha HEX
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOFT_CEILING_S = 1200.0  # 20 min
MAX_SUBMISSION_MB = 100.0  # scored SUBMISSION zip only — NOT the kit


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    sys.path.insert(0, str(ROOT / "scripts"))
    from apply_release_profile import load_profile  # type: ignore

    return load_profile(name, ROOT / "configs" / "release" / "profiles"), out_toml, 0


def step_sat2_artifact(profile: dict, out_dir: Path, sat2_pack: Path | None) -> int:
    """Pack-alongside-kit only; no main / default.toml logic changes."""
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
    print(f"OK sat2 pack-alongside-kit → {dst} ({dst.stat().st_size} B)")
    return 0


def step_dfine_ckpt(profile: dict, ckpt: Path | None, sha: str | None) -> tuple[int, dict | None]:
    """dfine_ae profile REQUIRES --ckpt and --sha; pin after ae-dfine-gate."""
    feats = profile.get("features") or {}
    if not feats.get("dfine_ae"):
        return 0, None
    if ckpt is None or not sha:
        print(
            "FAIL dfine_ae profile REQUIRES --ckpt PATH and --sha HEX "
            "(checkpoint pin after ae-dfine-gate; refuse to assemble without both)",
            file=sys.stderr,
        )
        return 1, None
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        print(f"FAIL dfine_ae --ckpt missing: {ckpt}", file=sys.stderr)
        return 1, None
    got = sha256_file(ckpt)
    expect = sha.strip().lower()
    if got != expect:
        print(
            f"FAIL dfine_ae --sha mismatch: got={got} expected={expect} path={ckpt}",
            file=sys.stderr,
        )
        return 1, None
    meta = {"ckpt": str(ckpt), "sha256": got}
    print(f"OK dfine_ae ckpt sha256={got} path={ckpt}")
    return 0, meta


def step_build(label: str, effective: Path, out_dir: Path, *, skip: bool) -> tuple[Path | None, int]:
    if skip:
        print("SKIP build (--skip-build)")
        return None, 0
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
    zips = sorted(out_dir.glob(f"koz3_{label}_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        print("FAIL: no kit zip produced", file=sys.stderr)
        return None, 1
    kit = zips[0]
    mb = kit.stat().st_size / 1_000_000
    print(f"OK kit zip → {kit} (kit_size={mb:.1f} MB informational; 100 MB limit is SUBMISSION only)")
    return kit, 0


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


def step_byte_compare(
    v2_ref: Path | None,
    candidate: Path | None,
    *,
    is_off: bool,
    skip_inject: bool,
) -> tuple[int, dict]:
    """OFF==v2 gate.

    With --v2-ref (geojson OR submission zip): extract scored geojson, compare,
    and record sha256 of the *given* REF file.

    Without --v2-ref: until Drive canonical
    ``submission_bot7cand_v2_eb268d6.zip`` (LB 0.41238) arrives, default OFF
    stand-in is ``v2_release_artifacts.py --check`` (v2 reproduced via inject).
    """
    meta: dict = {
        "v2_ref": str(v2_ref) if v2_ref else None,
        "v2_ref_file_sha256": None,
        "v2_ref_kind": None,
        "mode": None,
    }
    if v2_ref is not None:
        v2_ref = Path(v2_ref)
        if not v2_ref.is_file():
            print(f"FAIL --v2-ref missing: {v2_ref}", file=sys.stderr)
            return 1, meta
        meta["v2_ref_file_sha256"] = sha256_file(v2_ref)
        meta["mode"] = "byte_compare"
        print(f"OK v2_ref file sha256={meta['v2_ref_file_sha256']} path={v2_ref}")
        if candidate is None:
            print(
                "FAIL --v2-ref set but no candidate path (build skipped or --result missing)",
                file=sys.stderr,
            )
            return 1, meta
        sys.path.insert(0, str(ROOT / "scripts"))
        from compare_submission_bytes import compare_paths  # type: ignore

        errs, cmeta = compare_paths(v2_ref, Path(candidate))
        meta["v2_ref_kind"] = cmeta.get("ref_kind")
        meta["ref_payload_sha256"] = cmeta.get("ref_payload_sha256")
        meta["cand_payload_sha256"] = cmeta.get("cand_payload_sha256")
        if errs:
            print("FAIL OFF≠v2 (or REF≠CAND) byte compare:", file=sys.stderr)
            for e in errs:
                print(f"  {e}", file=sys.stderr)
            return 1, meta
        print("PASS byte-identical scored payload")
        return 0, meta

    # No --v2-ref: default OFF stand-in = inject reproduction check
    if is_off:
        meta["mode"] = "inject_check_standin"
        print(
            "NOTE no --v2-ref: OFF==v2 defaults to v2_release_artifacts --check "
            "(canonical REF=submission_bot7cand_v2_eb268d6.zip LB 0.41238 on Drive/Elena — not on box yet)",
            flush=True,
        )
        if skip_inject:
            # Already injected earlier or skipped intentionally; still verify
            rc = step_inject(skip=False, check_only=True)
        else:
            # Inject already ran in step 1; re-check is cheap + documents the stand-in
            rc = step_inject(skip=False, check_only=True)
        if rc:
            print("FAIL OFF==v2 stand-in (inject --check)", file=sys.stderr)
            return rc, meta
        print("PASS OFF==v2 stand-in (v2 artifacts reproduced)")
        return 0, meta

    meta["mode"] = "skipped_non_off"
    print("SKIP OFF==v2 (non-off profile and no --v2-ref)")
    return 0, meta


def step_submission_compliance(result: Path | None, kit: Path | None = None) -> tuple[int, dict]:
    """100 MB limit applies to SUBMISSION zip only; kit size is informational."""
    info: dict = {}
    if kit is not None and kit.is_file():
        info["kit_size_mb"] = round(kit.stat().st_size / 1_000_000, 3)
        print(f"INFO kit_size={info['kit_size_mb']} MB (informational; not subject to 100 MB limit)")
    if result is None:
        print("SKIP scored-submission compliance (no --result)")
        return 0, info
    sys.path.insert(0, str(ROOT / "scripts"))
    from kit_denylist import kit_members  # type: ignore

    result = Path(result)
    if result.suffix.lower() == ".zip" or zipfile.is_zipfile(result):
        with zipfile.ZipFile(result) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
        bad = kit_members(names)
        if bad:
            print("FAIL kit files inside scored submission:", file=sys.stderr)
            for b in bad[:30]:
                print(f"  {b}", file=sys.stderr)
            return 1, info
        mb = result.stat().st_size / 1_000_000
        info["submission_size_mb"] = round(mb, 3)
        if mb > MAX_SUBMISSION_MB:
            print(
                f"FAIL scored SUBMISSION zip {mb:.1f} MB > {MAX_SUBMISSION_MB} MB "
                f"(kit size is NOT gated by this limit)",
                file=sys.stderr,
            )
            return 1, info
        print(f"OK submission_size={mb:.1f} MB ≤ {MAX_SUBMISSION_MB} MB")
    # validator / check_compliance
    r = _run([sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(result)])
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode, info


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
    ap.add_argument(
        "--v2-ref",
        type=Path,
        default=None,
        help="OFF==v2 REF: result.geojson OR submission zip (extract scored geojson); "
        "sha256 of the given file is written to RELEASE_MANIFEST. Canonical: "
        "submission_bot7cand_v2_eb268d6.zip (LB 0.41238, Drive/Elena).",
    )
    ap.add_argument(
        "--result",
        type=Path,
        default=None,
        help="scored SUBMISSION (result.geojson|zip) for ≤100MB / no-kit / validator "
        "(100 MB limit = this zip only, not the kit)",
    )
    ap.add_argument("--sat2-pack", type=Path, default=None, help="sat2 REPLACE pack (pack-alongside-kit)")
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="REQUIRED with dfine_ae: checkpoint path (pin after ae-dfine-gate)",
    )
    ap.add_argument(
        "--sha",
        default=None,
        help="REQUIRED with dfine_ae: expected sha256 hex of --ckpt",
    )
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
    feats = profile.get("features") or {}
    is_off = not any(feats.values())

    # 2a dfine_ae requires --ckpt/--sha
    drc, dfine_meta = step_dfine_ckpt(profile, args.ckpt, args.sha)
    if drc:
        return drc

    # 2b sat2 artifact (pack-alongside only)
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

    # 4 OFF==v2 byte compare / inject stand-in
    cand = args.result or kit
    crc, v2_meta = step_byte_compare(
        args.v2_ref, cand, is_off=is_off, skip_inject=args.skip_inject
    )
    rc = rc or crc

    # 5b scored compliance (100 MB = submission only)
    src, size_info = step_submission_compliance(args.result, kit)
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
            "kit_size_mb": size_info.get("kit_size_mb"),
            "result": str(args.result) if args.result else None,
            "submission_size_mb": size_info.get("submission_size_mb"),
            "max_submission_mb": MAX_SUBMISSION_MB,
            "v2_ref": v2_meta.get("v2_ref"),
            "v2_ref_file_sha256": v2_meta.get("v2_ref_file_sha256"),
            "v2_ref_kind": v2_meta.get("v2_ref_kind"),
            "off_eq_v2_mode": v2_meta.get("mode"),
            "dfine_ae_ckpt": dfine_meta,
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
