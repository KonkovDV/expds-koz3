# bot5_release_tool — SHORT (follow-ups PR #24)

| field | value |
|-------|-------|
| **branch** | `bots/bot5-release-tool` |
| **tip** | `f44ea2a659814c6eb47de43477c9b045f5ef3a3d` |
| **base main** | `f127db8` |
| **PR** | https://github.com/KonkovDV/expds-koz3/pull/24 (draft) |
| **date** | 01.11.2026 ~19:30 Europe/Moscow |

## Follow-ups (bot-1 decisions)

1. `--v2-ref` = geojson **or** submission zip (extract scored FC); REF file sha256 → manifest. Canonical `submission_bot7cand_v2_eb268d6.zip` (LB 0.41238, Drive/Elena). Until on box: OFF==v2 default = `v2_release_artifacts.py --check`.
2. sat2: pack-alongside-kit sufficient; flag OFF; no main logic.
3. `dfine_ae`: requires `--ckpt` + `--sha` (loud fail); pin after ae-dfine-gate.
4. Historical PIXGBM LB packs: Drive/Elena only (docs).
5. 100 MB = **SUBMISSION** zip only; kit size informational.

## Tests

| suite | result |
|-------|--------|
| tool (`test_assemble*` + `test_compare*`, `--noconftest`) | **28 passed** |
| full box `pytest -q` | **1044 passed**, 14 failed (pre-existing unrelated), 47 skipped |

## Notes (one-liners)

- QUEUE REQUEST updated to tip `f44ea2a6` + `--v2-ref` zip command.
- No main / force-push / weights / PR #21 / merge.
