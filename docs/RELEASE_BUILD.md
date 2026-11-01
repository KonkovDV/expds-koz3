# Release build — one command

Assemble a submission **candidate kit** from the **current main tip** (read-only; never push/merge to main) with optional ACCEPTED feature flags.

## One command

```bash
# From a bots/bot5-* worktree at main tip (or any checkout of the tip):
python scripts/assemble_release.py --profile off
```

Other profiles:

```bash
python scripts/assemble_release.py --profile fr1h
python scripts/assemble_release.py --profile sat2 --sat2-pack /path/to/sat2_det2_oof_replace_pack.zip
python scripts/assemble_release.py --profile dfine_ae
python scripts/assemble_release.py --profile fr1h_sat2_dfine_ae --sat2-pack /path/to/pack.zip
```

Outputs land in `dist/release/` (override with `--out-dir`):

| artifact | meaning |
|----------|---------|
| `effective_<profile>.toml` | config packed as `configs/default.toml` |
| `koz3_rel_<profile>_*.zip` | kit zip (code + referenced weights) |
| `sat2_replace_pack.zip` | copy of sat2 REPLACE pack when profile has sat2 ON |
| `RELEASE_MANIFEST.json` | tip sha, features, paths, ok flag |

## What the command does

1. **Inject v2 artifacts** via `scripts/v2_release_artifacts.py` (sha256 verified; loud FAIL on mismatch/missing). Artifacts stay untracked (`.gitignore`); **never commit weights**.
2. **Apply profile** (`configs/release/profiles/<name>.toml`). Features **default OFF** unless the profile turns them on: `fr1h`, `sat2`, `dfine_ae`.
3. **Build kit** with `scripts/build_submission_git.py --config effective_*.toml`.
4. **OFF == v2 byte-for-byte** when `--v2-ref PATH` is set (scored `result.geojson`/zip preferred; else kit). Clear diff summary on failure (`scripts/compare_submission_bytes.py`).
5. **Compliance**
   - kit: DOC-1 / `check_archive_clean.py`
   - scored `--result`: ≤100 MB, **no kit files** (`scripts/kit_denylist.py` + LESSONS §5), `validate_submission.py` → `check_compliance.py`
   - runtime ≤ 20 min: `budget_guard.py` on `[time].soft_ceiling_s` (and optional `--wall-sec` / `--wall-file`)

## Profiles

See `configs/release/README.md`. Summary:

| profile | FR1h | sat2 | D-FINE Ae |
|---------|------|------|-----------|
| `off` | OFF | OFF | OFF |
| `fr1h` | ON | OFF | OFF |
| `sat2` | OFF | ON | OFF |
| `dfine_ae` | OFF | OFF | ON |
| `fr1h_sat2_dfine_ae` | ON | ON | ON |

## Useful flags

```bash
--skip-inject          # already injected / CI unit path
--skip-build           # materialize + validators only (box unit tests)
--v2-ref PATH          # reference for byte-identity
--result PATH          # scored FeatureCollection for ≤100MB / no-kit / validator
--sat2-pack PATH       # required when profile has sat2=true
--wall-sec N           # measured inference wall seconds
--out-dir DIR          # default dist/release
```

## Tests

```bash
# box (no GPU / no test volume):
python -m pytest tests/test_assemble_release.py tests/test_compare_submission_bytes.py -q
```

Negative cases covered: bad sha inject, oversize scored zip, kit file inside scored zip, OFF≠v2 byte mismatch, unknown profile, sat2 without pack.

## PC / Docker e2e

Byte-identity of **real pipeline output** on the test volume needs Docker `pantum/n7q:20261024`. File a request in `/home/box/shared/koz3/pc/QUEUE.md` (branch, sha, command, time, criterion). **Only bot 1 creates leases.**

## Constraints (release engineering)

- Read-only `main` — branch `bots/bot5-*`, draft PR, do not merge / force-push.
- Do not touch / commit `models/rerank_arch_lgbm.txt` or other weights.
- No cloud agents; no LB/Kaggle submissions from this bot (bot 1 owns the Kaggle gate).
