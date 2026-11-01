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
python scripts/assemble_release.py --profile dfine_ae --ckpt /path/to/model.safetensors --sha <hex>
python scripts/assemble_release.py --profile fr1h_sat2_dfine_ae --sat2-pack /path/to/pack.zip \
  --ckpt /path/to/model.safetensors --sha <hex>
```

Outputs land in `dist/release/` (override with `--out-dir`):

| artifact | meaning |
|----------|---------|
| `effective_<profile>.toml` | config packed as `configs/default.toml` |
| `koz3_rel_<profile>_*.zip` | **kit** zip (code + referenced weights); size is **informational** |
| `sat2_replace_pack.zip` | copy of sat2 REPLACE pack when profile has sat2 ON (pack-alongside-kit) |
| `RELEASE_MANIFEST.json` | tip sha, features, paths, `v2_ref_file_sha256`, ok flag |

## What the command does

1. **Inject v2 artifacts** via `scripts/v2_release_artifacts.py` (sha256 verified; loud FAIL on mismatch/missing). Artifacts stay untracked (`.gitignore`); **never commit weights**.
2. **Apply profile** (`configs/release/profiles/<name>.toml`). Features **default OFF** unless the profile turns them on: `fr1h`, `sat2`, `dfine_ae`.
3. **Build kit** with `scripts/build_submission_git.py --config effective_*.toml`.
4. **OFF == v2**
   - `--v2-ref PATH` accepts a scored **`result.geojson`** *or* a **submission zip** (detect by extension/content; zip → extract scored FeatureCollection). Computes **sha256 of the given REF file** and writes it to `RELEASE_MANIFEST.json` (`v2_ref_file_sha256`) plus stdout.
   - **Canonical REF:** `submission_bot7cand_v2_eb268d6.zip` (LB **0.41238**), held on **Drive with Elena** — **not on the box**. Until it arrives, **default OFF==v2** (no `--v2-ref`) is the **v2 reproduced via `scripts/v2_release_artifacts.py --check`** (inject stand-in). Clear FAIL on sha mismatch.
5. **Compliance**
   - kit: DOC-1 / `check_archive_clean.py`; **kit size reported informational only**
   - scored `--result` **SUBMISSION** zip: **≤100 MB** (this limit does **not** apply to the kit), **no kit files** (`scripts/kit_denylist.py` + LESSONS §5), `validate_submission.py` → `check_compliance.py`
   - runtime ≤ 20 min: `budget_guard.py` on `[time].soft_ceiling_s` (and optional `--wall-sec` / `--wall-file`)

## Profiles

See `configs/release/README.md`. Summary:

| profile | FR1h | sat2 | D-FINE Ae |
|---------|------|------|-----------|
| `off` | OFF | OFF | OFF |
| `fr1h` | ON | OFF | OFF |
| `sat2` | OFF | ON | OFF |
| `dfine_ae` | OFF | OFF | ON (requires `--ckpt` + `--sha`) |
| `fr1h_sat2_dfine_ae` | ON | ON | ON (requires `--ckpt` + `--sha` + `--sat2-pack`) |

### sat2

Flag **OFF by default**. sat2 is an OOF REPLACE pack — **pack-alongside-kit is sufficient**; **no logic changes on main** / `default.toml`. Provide `--sat2-pack` (or `[artifacts].sat2_replace_pack`).

### dfine_ae

**Placeholder** until ae-dfine-gate pins the checkpoint. Profile **REQUIRES** `--ckpt PATH` and `--sha HEX`; assemble **fails loudly** if either is missing or sha mismatches. Pin the release checkpoint after the gate; never commit weights.

### Historical PIXGBM LB packs

Historical PIXGBM-containing LB hold packs are **only** via **Drive / Elena** (outside this tool’s DOC-1 kits). This assembler keeps kits DOC-1 clean and does not re-pack those historical zips.

## Useful flags

```bash
--skip-inject          # already injected / CI unit path
--skip-build           # materialize + validators only (box unit tests)
--v2-ref PATH          # geojson OR submission zip; sha256 recorded in manifest
--result PATH          # scored SUBMISSION for ≤100MB / no-kit / validator
--sat2-pack PATH       # required when profile has sat2=true (pack-alongside)
--ckpt PATH --sha HEX  # required when profile has dfine_ae=true
--wall-sec N           # measured inference wall seconds
--out-dir DIR          # default dist/release
```

## Tests

```bash
# tool tests (no GPU / no test volume):
python -m pytest tests/test_assemble_release.py tests/test_compare_submission_bytes.py -q --noconftest
# full box suite (needs numpy etc.):
python -m pytest -q
```

Negatives: zip+geojson refs, missing `--ckpt`/`--sha`, submission-zip size rule (kit unlimited), bad sha inject, kit file inside scored zip, OFF≠v2, unknown profile, sat2 without pack.

## PC / Docker e2e

Byte-identity of **real pipeline output** vs canonical Drive REF needs Docker `pantum/n7q:20261024`. File a request in `/home/box/shared/koz3/pc/QUEUE.md` (branch, sha, command, time, criterion). **Only bot 1 creates leases.** Prefer `--v2-ref submission_bot7cand_v2_eb268d6.zip` once Elena’s Drive copy is on the PC.

## Constraints (release engineering)

- Read-only `main` — branch `bots/bot5-*`, draft PR, do not merge / force-push.
- Do not touch / commit `models/rerank_arch_lgbm.txt` or other weights.
- No cloud agents; no LB/Kaggle submissions from this bot (bot 1 owns the Kaggle gate).
