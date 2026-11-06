# Murder Board / Red Team — main after the 29.10.2026 integration (condensed)

Full version (shared workspace): `koz3_shared/work/MURDER_BOARD_MAIN_2026-10-29.md`. Reference submission: v2
(eb268d6 / 1b62124 lineage, LB 0.41238). No data, weights or secrets in this file.

## Top findings
1. **Clean main ≠ LB archive, silently.** `[geophys.pixgbm] enabled = true` but `expds/geo_pixgbm.py` is held out →
   pipeline warns and falls back to classic geophys (≈ −0.012, bot-10 global keep +0.011935).
   *Mitigated:* `check_submission` now FAILs when pixgbm is on and the module is not in the archive.
2. **v3 silently == v2.** A container without `transformers.DFine*` made the D-FINE phase return 0 objects
   (PC e2e invalid). *Fixed (bot-10 requirement):* `[sp_dfine] strict = true` (default) → `sp_dfine.preflight`
   before any work (DFine classes, `transformers_version = "5.21.0"` pin, weights, thresholds) raises
   `DFineUnavailable`; model-load failure is fatal too; the run protocol prints
   `sp_dfine: transformers X; models N; raw N; written N`; `check_submission` FAILs without weights /
   thresholds / vendored `transformers/models/d_fine`. `enabled = false` default unchanged.
3. **EN provenance cannot be removed from history without a rewrite (forbidden).** `expds/geo_pixgbm.py` is reachable
   from main through 9 merged branches and is present in the main *tree* as 3 copies under `analysis/2026-10-29/`
   (one byte-identical to v2, sha256 `35e4331b…`). `expds/li_decode.py` docstring cites сторонний репозиторий.
   `bots/bot8-cpu-a-clean*` (not clean-room) are still on the remote.
4. **CI had two blockers.** (a) account billing — jobs never start; (b) even with billing fixed the job would have
   failed: unpinned `torch` + timm pulled an ABI-mismatched torchvision (`operator torchvision::nms does not exist`
   at collection), and `ubuntu-latest` moves to Ubuntu 26. *(b) fixed:* `torch==2.11.0 torchvision==0.26.0`
   (same as `docker/Dockerfile`) from the CPU index, `runs-on: ubuntu-24.04`.
5. **Default-path code changed a lot; only config equivalence is proven.** Flattened `configs/default.toml`
   1b62124 → main: 0 changed, 0 removed, 8 added (all off). `expds/` diff: 26 files, +4860/−918. No golden
   byte-comparison of main's output vs the v2 archive yet.

## Ranked risks (severity × likelihood, 1–5)
| # | Risk | S×L | State |
|---|---|---|---|
| 1 | v3 silently = v2 | 25 | fixed (strict preflight + protocol + pin + static check); redo e2e |
| 2 | main build without geo_pixgbm.py | 20 | check_submission FAIL; needs release procedure |
| 3 | EN provenance (li_decode, pixgbm lineage, analysis/ copies, cpu-a-clean branches) | 15 | owner decision |
| 4 | CI blocked by billing; ~25 pixgbm/package tests skip in CI | 15 | workflow fixed; owner fixes billing |
| 5 | default code ≠ v2 at equal config | 12 | needs golden e2e |
| 6 | v3 time budget not measured on 4090 | 12 | open (bot-10 condition) |
| 7 | tracked `.pt` (15 × r18) — training-data provenance (previous-stage data ban) | 8–12 | confirm |
| 8 | strict=true crashes the platform run ⇒ no answer | 10 | gate with check_submission + docker e2e; `strict=false` fallback |

## Before the next LB upload
1. Build from main + external files (geo_pixgbm.py sha256 `35e4331b…`, pixgbm/rerank weights per
   `models/pixgbm/all/MODELS_NOT_IN_GIT.md`); `scripts/check_submission.py` → 0 FAIL.
2. Golden e2e: same input, LB-0.41238 archive vs main archive → identical per-class/per-layer counts in the run
   protocol, then identical sorted geojson. Any diff → bisect `git log --merges --first-parent 38f2879..main`.
3. Protocol must not contain "pixgbm … нет в этой сборке — классика"; for v3 it must show `sp_dfine … models ≥1, raw >0`.
4. Runtime on target GPU < `budget_s`, no soft_skipped.

## Restore pixgbm for a release build (no new EN bytes in git)
```bash
cp analysis/2026-10-29/bots/bot-4/runs/review/v2eb268d6/expds/geo_pixgbm.py expds/geo_pixgbm.py
echo "35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b  expds/geo_pixgbm.py" | sha256sum -c
# do NOT commit; expds/geo_pixgbm.py is in .gitignore
```

## Decisions for the owner
1. `li_decode`: **recommend removing from the tree** (rejected, off, EN-citing docstring; history stays).
2. `geo_pixgbm`: (a) keep out of `expds/`, inject at build + honest note in THIRD_PARTY/AI_DECLARATION, or
   (b) clean-room rewrite + re-validate the +0.012. The current hold-out hides nothing (copies in `analysis/`, history).
3. Delete/archive `bots/bot8-cpu-a-clean*` on the remote (not done: branch deletion is not allowed for agents).
4. `strict` for the final v3 upload (see risk 8).
5. Fix GitHub billing: Settings → Billing and licensing → Payment information (failed payment) and
   Budgets/Spending limit for **Actions** (raise from $0 or untick "stop usage"), then re-run the failed workflow.
6. Confirm training data of the tracked `.pt` weights.

## Branches / PRs
Open PRs: none. Not in main: `bots/bot8-cpu-a-clean` (8), `bots/bot8-cpu-a-clean-on-v2` (9) — not clean-room
(spec written from EN behaviour, oracle loads competitor repo, wheel); `speedup-2026-10-28` (1) — changes default
paths + `.whl`. All other remote branches are merged.
