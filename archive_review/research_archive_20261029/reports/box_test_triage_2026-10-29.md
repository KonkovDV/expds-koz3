# Box CPU test triage — 2026-10-29

Обновлено: 29.10 16:35 МСК (Europe/Moscow).  
Runner: `/home/box/shared/koz3/box/run_tests.full.sh` (= `run_tests.sh`).  
Clone: `/workspace/koz3-repo` · venv: `/workspace/koz3-venv` · baseline: `main@70dedf4` → **666 passed / 0 failed / 43 skipped**.

## Runner fixes (goal 1)

Backup: `run_tests.full.sh.bak_2026-10-29_1612` (+ `run_tests.sh.bak_…`; also `.bak_oneshot_broken` from an intermediate attempt).

| Change | Why |
|---|---|
| Only existing dirs among `tests/`, `centerhead/tests` passed to pytest | Old script hard-coded `centerhead/tests` → exit 4 on older branches (bot8-orsp, bot2, …) |
| Separate pytest invocations per existing path | One-shot `tests centerhead/tests` hit conftest ImportError on some trees; counts stay additive and consistent |
| Flags `-q -rfEs` (was `-rs`) | `-rs` hid `FAILED` lines; short summary now shows failures |
| Plugin `box/plugins/koz3_box_env.py` via `-p koz3_box_env` | Missing `models/pixgbm/*`, `rerank_arch_lgbm.txt`, `geo_pixgbm.py`, optional deps, CUDA → **skip**, not fail |
| Worktree: retry + remove stale `index.lock`; single-flight `.run_tests.lock` | Concurrent batch left `index.lock` → bot3-aerial stopped before pytest |
| Count parsing from last summary per path | Avoids double-counting when logs append or two runners race |

Pass-count variance (424…1338) explained: (1) older tips (~190 commits behind main, merge-base `d420ceb`) lack `centerhead/` → ~424–541 vs ~666; (2) first batch double-counted when concurrent runners / dual-path one-shot appended the same log.

## Per-branch table (after fixed runner)

| Branch | Tip | Merge-base | Paths | Passed | Failed | Skipped | Verdict |
|---|---|---|---|---:|---:|---:|---|
| main | 70dedf4 | — | tests+ch | 666 | 0 | 43 | **green** (baseline) |
| bots/bot2-e2e-v3-prep | 1ec2d7c | d420ceb | tests | 455 | 0 | 23 | **green** (was env-only) |
| bots/bot3-aerial-recall | 475bafa | d420ceb | tests | 541 | 0 | 37 | **green** (was worktree lock) |
| bots/bot3-dfine-ae-kernel | 6e42cf6 | d420ceb | tests | 541 | 0 | 37 | **green** |
| bots/bot4-large-objects | 337a27c | d420ceb | tests | 539 | 0 | 37 | **green** (was env-only) |
| bots/bot5-candidate-v4 | 50f6f27 | d420ceb | tests | 541 | 0 | 37 | **green** (was env-only) |
| bots/bot6-pixgbm-cleanroom | b40ffd4 tip; report `0632c19` | 70dedf4 | tests+ch | 669† | 0 | 44 | **green** |
| bots/bot7-honest-eval-layer | ae4c816 | 70dedf4 | tests+ch | 663 | **11** | 44 | **real regressions** |
| bots/bot8-orsp-tta | 35ca3df | 4c3e29e | tests | 424 | 0 | 7 | **green** (was exit-4 missing centerhead) |
| bots/bot8-oof-kaggle | a6f8992 | 70dedf4 | tests+ch | 671 | 0 | 43 | **green** |

† bot6 first-batch report showed duplicated summary lines (619×2 + 50×2) under the old concurrent runner; failed=0 both times. Tip `b40ffd4` is ahead of `0632c19` but only packaging/thread defaults — not re-run under RAM pressure; classify green.

### Classification key

- **(a) environment/runner artifact** — missing weights, CUDA, missing `centerhead/`, worktree lock, exit-4 path list.
- **(b) pre-existing on main / merge-base** — none observed that survive the fixed runner (main is 0 failed).
- **(c) real regression introduced by the branch** — only bot7 (below).

## bot3 worktree failure

First batch log (`batch_2026-10-29_1612.log`):

```
fatal: Unable to create '/workspace/koz3-repo/.git/worktrees/bots-bot3-aerial-recall/index.lock': File exists.
```

Cause: parallel `run_tests` processes racing on the same git worktree index. No pytest ran. Fixed by lock retries + stale-lock cleanup + single-flight advisory lock. Re-run: **541p / 0f**.

## Category (c) — bot7 only

All 11 failures share the same root cause in `expds/metric.py` inside `build_curves_reference`:

```
NameError: name 'per_site' is not defined
  File "expds/metric.py", line 745, in build_curves_reference
    if per_site and len(q_pred):
```

On this tip the function sets `keyed` / `p_sites` / `g_sites` for layer+site scopes but never assigns `per_site`. On `main@70dedf4` the same function still has `per_site = match_scope(scope) == "site"` before the loop.

| # | Test | Error |
|---|---|---|
| 1 | `tests/test_bot8_speed_cache.py::test_build_curves_equal_reference_synthetic[0-global]` | NameError `per_site` @ metric.py:745 |
| 2 | `…[0-site]` | same |
| 3 | `…[1-global]` | same |
| 4 | `…[1-site]` | same |
| 5 | `…[2-global]` | same |
| 6 | `…[2-site]` | same |
| 7 | `test_fast_match_off_equal[global]` | same |
| 8 | `test_fast_match_off_equal[site]` | same |
| 9 | `test_mutated_input_is_not_served_from_cache` | same |
| 10 | `test_corrupt_cache_file_does_not_change_curves` | same |
| 11 | `test_disabled` | same |

**Likely culprit commit:** `d1a98aec0987430bc49432d5462f0c3f33c51cbb` — `feat(metric): flag-gated per-layer matching (EXPDS_MATCH_SCOPE=layer)` (bot-5, 29.10 16:06 МСК). Refactor introduced `keyed`/`key_fn` but left the `if per_site` branch without defining `per_site`.

**Suggested fix (for bot7, not applied here):** in `build_curves_reference`, set `per_site = p_sites is not None and g_sites is not None` (as the non-reference path already does on main), or restore `per_site = match_scope(scope) == "site"` and gate layer separately.

Merge-base with main is `70dedf4` (0 failed) → these are **not** pre-existing.

## First-batch → after-fix mapping (goal 2)

| Branch | First batch | After fix | Class |
|---|---|---|---|
| bot2 | 456p/5f | 455p/0f | (a) packaging/pixgbm |
| bot3-aerial | stopped at worktree | 541p/0f | (a) index.lock race |
| bot3-dfine | (queued) | 541p/0f | green |
| bot4 | 540p/10f | 539p/0f | (a) packaging/pixgbm |
| bot5 | 536p/10f | 541p/0f | (a) packaging/pixgbm |
| bot6 | 1338p/0f (double) | green | green |
| bot7 | 1328p/22f (double+env) | 663p/11f | **(c)** NameError + former (a) |
| bot8-orsp | exit 4 centerhead | 424p/0f | (a) hardcoded path |
| bot8-oof | green | green | green |

## Artifacts

- Reports/logs: `/home/box/shared/koz3/box/results/*_<sha>.md|.log|.meta`
- Plugin: `/home/box/shared/koz3/box/plugins/koz3_box_env.py`
- This note: `/home/box/shared/koz3/reports/box_test_triage_2026-10-29.md`

No GitHub push. Tokens not printed.
