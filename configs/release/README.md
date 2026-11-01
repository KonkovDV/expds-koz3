# Release profiles

Each `profiles/<name>.toml` declares which ACCEPTED features are ON.
**All features default OFF** unless the profile sets them true.

| profile | FR1h | sat2 | D-FINE Ae |
|---------|------|------|-----------|
| `off` | OFF | OFF | OFF |
| `fr1h` | ON | OFF | OFF |
| `sat2` | OFF | ON (pack-alongside-kit; no main logic) | OFF |
| `dfine_ae` | OFF | OFF | ON — **requires** `--ckpt` + `--sha` |
| `fr1h_sat2_dfine_ae` | ON | ON | ON — **requires** `--ckpt` + `--sha` + sat2 pack |

CLI: `python scripts/assemble_release.py --profile <name>`
See `docs/RELEASE_BUILD.md`.

Historical PIXGBM LB packs: Drive/Elena only (not produced by this tool).
