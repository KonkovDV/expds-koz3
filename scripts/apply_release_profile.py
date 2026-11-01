#!/usr/bin/env python3
"""Apply a release profile onto configs/default.toml → effective release TOML.

Features default OFF. Profile may turn on: fr1h, sat2 (sp_det2 inference + OOF pack), dfine_ae.
Uses text surgery on a copy of the base TOML so comments and unrelated sections stay intact.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = ROOT / "configs" / "release" / "profiles"

FR1H_BLOCK = """
# --- release profile: FR1h ON ---
[post.frame_prior]
groups = ["sp_or", "sp_focal", "sp_focal2"]
classes = ["kurgany_povrezhdennye", "kurgany_tselye", "arkhitektury", "object_poly", "fortifikatsii", "finds_points"]
d0 = 0.5
edge = 0.5
"""

DFINE_AE_BLOCK = """
# --- release profile: D-FINE Ae ON ([dfine_aeor]) ---
[dfine_aeor]
enabled = true
weights = { ae = ["models/dfine_ae"], or = ["models/dfine_or"] }
thresholds_file = "models/dfine_aeor/thresholds_oof.json"
"""

SAT2_BLOCK = """
# --- release profile: sat2 / sp_det2 ON (CenterNet REPLACE) ---
[sp_det2]
enabled = true
weights = [
  "models/sp_det2/sp_det2_cnxt_f0.pt",
  "models/sp_det2/sp_det2_cnxt_f1.pt",
  "models/sp_det2/sp_det2_cnxt_f2.pt",
  "models/sp_det2/sp_det2_cnxt_f3.pt",
  "models/sp_det2/sp_det2_cnxt_f4.pt",
]
drop_prefixes = ["sp_"]

[thresholds_by_markup.sp_det2]
kurgany_povrezhdennye = 0.3645
kurgany_tselye = 0.3725
gorodishcha = 1.1
arkhitektury = 1.1
fortifikatsii = 1.1
object_poly = 1.1
finds_points = 1.1
"""


def _strip_active_sp_det2(text: str) -> str:
    text = re.sub(
        r"(?m)^\[sp_det2\][^\n]*\n(?:(?!^\[)[^\n]*\n)*",
        "",
        text,
    )
    text = re.sub(
        r"(?m)^\[thresholds_by_markup\.sp_det2\][^\n]*\n(?:(?!^\[)[^\n]*\n)*",
        "",
        text,
    )
    return text


def _force_sp_det2_off(text: str) -> str:
    return _strip_active_sp_det2(text).rstrip() + "\n\n[sp_det2]\nenabled = false\n"


def _force_sp_det2_on(text: str) -> str:
    return _strip_active_sp_det2(text).rstrip() + "\n" + SAT2_BLOCK



def load_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> dict:
    path = profiles_dir / f"{name}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in profiles_dir.glob("*.toml"))
        raise FileNotFoundError(f"profile {name!r} not found in {profiles_dir}; known={known}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    feats = dict(data.get("features") or {})
    for key in ("fr1h", "sat2", "dfine_ae"):
        feats.setdefault(key, False)
    data["features"] = feats
    data["_path"] = str(path)
    data["_name"] = data.get("name") or name
    return data


def _strip_active_frame_prior(text: str) -> str:
    """Remove an active (uncommented) [post.frame_prior] table if present."""
    # Match from a line that is exactly [post.frame_prior] until next [section] or EOF.
    return re.sub(
        r"(?m)^\[post\.frame_prior\][^\n]*\n(?:(?!^\[)[^\n]*\n)*",
        "",
        text,
    )


def _force_dfine_aeor_off(text: str) -> str:
    """Ensure [dfine_aeor] enabled = false (first enabled= under that section)."""
    m = re.search(r"(?m)^\[dfine_aeor\][^\n]*\n", text)
    if not m:
        return text + "\n[dfine_aeor]\nenabled = false\n"
    start = m.end()
    # find next section
    n = re.search(r"(?m)^\[", text[start:])
    end = start + n.start() if n else len(text)
    block = text[start:end]
    block2, nsub = re.subn(
        r"(?m)^enabled\s*=\s*\S+",
        "enabled = false",
        block,
        count=1,
    )
    if nsub == 0:
        block2 = "enabled = false\n" + block
    return text[:start] + block2 + text[end:]


def _force_dfine_aeor_on(text: str, profile: dict) -> str:
    """Replace or append [dfine_aeor] with enabled profile block."""
    overlay = profile.get("dfine_aeor") or {}
    weights = overlay.get("weights") or {"ae": ["models/dfine_ae"], "or": ["models/dfine_or"]}
    thr = overlay.get("thresholds_file") or "models/dfine_aeor/thresholds_oof.json"
    # format weights inline table
    def fmt_list(xs: list) -> str:
        return "[" + ", ".join(f'"{x}"' for x in xs) + "]"

    if isinstance(weights, dict):
        wparts = []
        for k, v in weights.items():
            xs = v if isinstance(v, list) else [v]
            wparts.append(f"{k} = {fmt_list(xs)}")
        wtxt = "{ " + ", ".join(wparts) + " }"
    else:
        wtxt = fmt_list(list(weights))
    block = (
        "\n# --- release profile: D-FINE Ae ON ([dfine_aeor]) ---\n"
        "[dfine_aeor]\n"
        "enabled = true\n"
        f"weights = {wtxt}\n"
        f'thresholds_file = "{thr}"\n'
    )
    # drop existing [dfine_aeor] table (active)
    text2 = re.sub(
        r"(?m)^\[dfine_aeor\][^\n]*\n(?:(?!^\[)[^\n]*\n)*",
        "",
        text,
    )
    return text2.rstrip() + "\n" + block


def apply_profile_text(base_text: str, profile: dict) -> str:
    feats = profile.get("features") or {}
    text = base_text
    # Always start from a state with FR1h absent, sat2/sp_det2 off, dfine_aeor off; then enable as needed.
    text = _strip_active_frame_prior(text)
    text = _force_dfine_aeor_off(text)
    text = _force_sp_det2_off(text)
    if feats.get("fr1h"):
        text = text.rstrip() + "\n" + FR1H_BLOCK
    if feats.get("sat2"):
        text = _force_sp_det2_on(text)
    if feats.get("dfine_ae"):
        text = _force_dfine_aeor_on(text, profile)
    else:
        text = _force_dfine_aeor_off(text)
    return text if text.endswith("\n") else text + "\n"


def materialize(
    profile_name: str,
    *,
    root: Path = ROOT,
    base_toml: Path | None = None,
    out_toml: Path | None = None,
) -> tuple[dict, dict, Path]:
    base_path = base_toml or (root / "configs" / "default.toml")
    base_text = base_path.read_text(encoding="utf-8")
    profile = load_profile(profile_name, root / "configs" / "release" / "profiles")
    text = apply_profile_text(base_text, profile)
    out = out_toml or (root / "dist" / "release" / f"effective_{profile['_name']}.toml")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# AUTO-GENERATED by apply_release_profile.py — do not hand-edit\n"
        f"# profile={profile['_name']} base={base_path.as_posix()}\n"
        f"# features={profile.get('features')}\n\n"
    )
    out.write_text(header + text, encoding="utf-8")
    cfg = tomllib.loads(out.read_text(encoding="utf-8"))
    return profile, cfg, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--base", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    try:
        profile, cfg, out = materialize(
            args.profile, root=args.root, base_toml=args.base, out_toml=args.out
        )
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    feats = profile["features"]
    has_fr = "frame_prior" in (cfg.get("post") or {})
    df_on = bool((cfg.get("dfine_aeor") or {}).get("enabled"))
    sat_on = bool((cfg.get("sp_det2") or {}).get("enabled"))
    print(
        f"OK profile={profile['_name']} features={feats} "
        f"frame_prior={has_fr} sp_det2.enabled={sat_on} dfine_aeor.enabled={df_on} → {out}"
    )
    if not feats.get("fr1h") and has_fr:
        print("FAIL: FR1h OFF but frame_prior present", file=sys.stderr)
        return 1
    if feats.get("fr1h") and not has_fr:
        print("FAIL: FR1h ON but frame_prior missing", file=sys.stderr)
        return 1
    if bool(feats.get("dfine_ae")) != df_on:
        print("FAIL: dfine_ae feature/flag mismatch", file=sys.stderr)
        return 1
    if bool(feats.get("sat2")) != sat_on:
        print("FAIL: sat2 feature/flag mismatch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
