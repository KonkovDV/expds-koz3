# Бот 5 — кандидат v4

**Обновлено:** 29.10 19:50 МСК  
**Репо:** https://github.com/KonkovDV/expds-koz3

## Контракт

| поле | значение |
|---|---|
| **Цель** | Собрать кандидата v4 **без** SpOr D-FINE: v3-контент + bot-3 patches (уже в базе) + пакет gor_dense + слоты `[dfine_aeor]`; всё новое OFF; ворота PROTOCOL §5 |
| **Гипотеза** | `enabled=false` на `[sp_dfine]` восстанавливает U-Net sp_or (фаза не входит → drop_groups R не срабатывает). gor_dense package-only даст ≈+0.0154 при ON; dfine_aeor — отдельный слот |
| **Критерий приёмки** | (1) дешёвые тесты green; (2) e2e flags-OFF = v2 mock **0.3125** / sp_or **0.2429** послойно; (3) gor_dense ON: Δ>0, ни один слой не падает >**0.01**; (4) затем dfine_aeor ON — та же рамка |
| **Бюджет** | box CPU: ~0,2 ч (сборка+тесты); ПК GPU: 2× mock e2e ≤15 мин wall каждый (лимит 30) |
| **Ветка** | `bots/bot5-candidate-v4` @ `ed73c5e9d1eff859b064220930cdc356df0c85b9` (**test-only** packaging fix; pipeline tip still `2b745df`; heads `40fa599`; Or assert `999572d`) |

Worktree: `/workspace/koz3-wt/bot5-candidate-v4` (от `/workspace/koz3-repo`). **main не трогали.** `models/rerank_arch_lgbm.txt` не трогали.

## SpOr-off (выбор)

| вариант | эффект |
|---|---|
| **`[sp_dfine] enabled=false` (выбран)** | `pipeline.run` не вызывает `_sp_dfine_phase` → `drop_groups` не применяется → слой sp_or/sp_focal/sp_arch/sp_focal2 = путь v2/v3 pre-D-FINE |
| `drop_groups=[]` при `enabled=true` | D-FINE **добавляет** детекции поверх U-Net (без снятия). Нужен только если включать SpOr D-FINE без R. DEFAULTS уже `drop_groups=[]` |
| вариант R (`drop_groups` = SpOr-группы) | **запрещён для v4** до фикса локализации (bot1: soft_tp 0/13, медиана ~191 м) |

Страж: `tests/test_bot5_v4_flags.py` (`test_v4_spor_off_is_enabled_false`, `test_v4_enabled_false_skips_phase_even_if_drop_groups_r`).

Источник диагноза: `reports/bot1_v3_spor_diag.md`.

## Коммиты сборки (после tip манифеста `f414cc6`)

| # | SHA | компонент | флаг |
|---|---|---|---|
| 1 | [`d1b6241`](https://github.com/KonkovDV/expds-koz3/commit/d1b62414d1a5a732200303d979fb32f365418d1a) | v3 content minus SpOr D-FINE: комментарий + guard | `[sp_dfine] enabled=false` |
| 2 | [`3823469`](https://github.com/KonkovDV/expds-koz3/commit/382346909ebe227b3c71d1b23429e7fbec43863e) | bot-3 confirm: `3e7e7c82`←`e3c4f41` ground_metres; `26ab4187`←`e0ec0ff` dedupe_union — **уже в базе**, без re-cherry | (opt-in OFF) |
| 3 | [`90094d2`](https://github.com/KonkovDV/expds-koz3/commit/90094d27a2e28101224c3de7b7ba3e07f483302e) | gor_dense package τ=**0.824066** centre_s=**250** (expected +0.0154) | `[gor_dense] enabled=false` |
| 4 | [`50f6f27`](https://github.com/KonkovDV/expds-koz3/commit/50f6f2777f0f48670818e96491c837e4e6644106) | dfine_aeor: tip = `3ccc388` (≡ `aba432d`) + `164c9a4`/`012942a`; bare aba432d **не** cherry-pick | `[dfine_aeor] enabled=false` |
| 5 | [`19ccc9c`](https://github.com/KonkovDV/expds-koz3/commit/19ccc9cf20c2326e9a259b260e175706901a989e) | `scripts/gor_heads` + `extract_feats`: resumable retrain of missing heads | (train; flags unchanged) |
| 6 | [`4dadeab`](https://github.com/KonkovDV/expds-koz3/commit/4dadeab1788dbf9f259623a5f8afda14d6917f60) | WEIGHTS_MANIFEST_v4: Or f0–f2 sha256 (Kaggle koz3-dfine-or) | docs |
| 7 | [`999572d`](https://github.com/KonkovDV/expds-koz3/commit/999572d) | hard-FAIL assert if Or dfine_aeor dets == 0 (+ provisional Or thr) | e2e gate |
| 8 | [`9f62a96`](https://github.com/KonkovDV/expds-koz3/commit/9f62a9676b84e6963b892e6eb9f0907ad975f6ba) | gor_heads: dinov2_loader (=gor_gate/import_dfine) + preflight; Docker runner | train |

Предшествующие (манифест/флаги): `44cc910` → `7ac7458` → `37e74d5` → `f414cc6` (PC TSV sha256).

База ветки: [`cbd4804`](https://github.com/KonkovDV/expds-koz3/commit/cbd48049ceaa6b56d0e4b85c72f4d35e839b2d3c).

## Конфликты / расхождения

| источник | решение |
|---|---|
| `aba432d` vs tip (`ahead 1 / behind 20` на compare tip…aba432d) | tip **суперсет**: introduce = `3ccc388`; сохранены GROUP_ALIASES, per-window `left()`, `gdal_cache_mb`, `merge_union=false`, фаза `gor_dense`, orgate, ground_metres wiring. Cherry-pick голого `aba432d` **откатил бы** эти фиксы — не делали |
| `78af57f` (v3 tip) vs tip | tip **суперсет** v3 (+dfine_aeor, gor_dense, orgate, bot-3 patches, v4 docs/tests) |
| вариант R в закомм. enable-блоке v3 | в v4 рецепте заменён на `drop_groups=[]`; R оставлен закомментированным как антипаттерн |

## Флаги (default.toml, все OFF)

| флаг | default | роль v4 |
|---|---|---|
| `[sp_dfine]` | **false** | SpOr D-FINE выкл. (регрессия mock) |
| `[dfine_aeor]` | false | слот Ae/Or D-FINE |
| `[gor_dense]` | false | пакет hillfort; enable-рецепт: τ=0.824066, centre_s=250 |
| `[gor_gate]` | false | не ставить вместе с R |
| `[orgate]` | false | bot-6 arch gate |

## Дешёвые ворота (box, 29.10 16:17 МСК)

```
PYTHONPATH=worktree pytest -q \
  tests/test_bot5_v4_flags.py \
  tests/test_bot5_v4_base_patches.py \
  tests/test_bot5_gor_dense.py::test_default_flag_off_and_noop \
  tests/test_bot5_gor_dense.py::test_enable_block_parses_and_is_complete \
  tests/test_bot5_gor_dense.py::test_threshold_by_markup_gor_dense \
  tests/test_bot5_gor_dense.py::test_phase_order_after_sp_dfine_before_gor_gate \
  tests/test_bot7_v3_flags.py::test_default_flags_off \
  tests/test_bot7_v3_flags.py::test_run_flag_off_never_touches_sp_dfine \
  tests/test_bot7_v3_flags.py::test_drop_groups_variant_r \
  tests/test_dfine_aeor.py::test_default_off_and_names \
  tests/test_ground_metres.py \
  tests/test_union_layers.py \
  --noconftest
→ 24 passed @ 50f6f27
```

После каждого компонентного коммита: flags (+ релевантные) green.

## QUEUE ПК (max 2 open)

| # | заявка | sha | критерий |
|---|---|---|---|
| (a) | flags-OFF identity vs v2 0.3125 | `50f6f27` | послойно = v2 / v3-without-SpOr; inference ≤15 мин |
| (b) | gor_dense package ON | `50f6f27` | Δ>0, слой не падает >0.01; ≈+0.0154; затем заявка dfine_aeor ON |

Leases: `leases/pc_bot5_v4_flags_off.lease`, `leases/pc_bot5_v4_gor_dense.lease` (status QUEUED).

Старая pytest-заявка bot5 — снята/заменена этими двумя.

## Манифест весов

[`docs/WEIGHTS_MANIFEST_v4.md`](https://github.com/KonkovDV/expds-koz3/blob/bots/bot5-candidate-v4/docs/WEIGHTS_MANIFEST_v4.md) @ линия до `f414cc6` (сборка тесты/комменты не меняла хеши). PIXGBM MATCH; dfine_f0 sha256 с ПК; dinov2/gor_*/AeOr — pending Kaggle.

## EXPERIMENTS.csv

Строк **нет** — жду реальных метрик с (a)/(b).

## Следующий шаг

1. «1» запускает QUEUE (a), затем (b).  
2. После green (b) — заявка (c) `[dfine_aeor] ON`.  
3. SpOr D-FINE не включать, пока bot-2 не даст soft_tp>0.


## PC e2e — 2026-10-29 16:28 МСК (pc-job-runner)

- **GPU lane:** bot2 hotfix A/B DONE → **(a) flags-OFF RUNNING** → (b) gor_dense next.
- Worktree `C:\\Sloi\\wt_bot5_50f6f27` @ 50f6f27; out `C:\\Sloi\\pc_jobs\\bot5_v4_off_50f6f27`.
- **(b) blocker:** `dinov2-small` / `gor_v3_head.npz` / `gor_dense_head.npz` **pending Kaggle** (WEIGHTS_MANIFEST_v4); на C:\\Sloi нет.


## PC e2e (a) flags-OFF — DONE PASS — 2026-10-29 16:34 МСК

| | ALL | li | sp_or | ae |
|---|---:|---:|---:|---:|
| v2 base | 0.312509 | 0.336983 | 0.242915 | 0.095556 |
| **v4 flags-OFF** @50f6f27 | 0.312873 | 0.336983 | 0.245064 | 0.095556 |
| Δ | +0.000364 | +0.000000 | +0.002150 | +0.000000 |

- **Критерий:** ALL ≈ 0.3125; слои ≈ v2; никто не падает >0.01 → **PASS** (≡ bot2 hotfix B).
- **Wall inference:** 100.1 с (лимит 30 мин).
- **honest-10** (`scoreboard.py`, scope=**global** — `match-scope` нет в tip): base ALL≈0.339456; honest ALL≈0.297118.
- Out: `C:\Sloi\pc_jobs\bot5_v4_off_50f6f27` (STATUS=DONE, e2e_v2_off.json, notes/honest10.json).
- Worktree: `C:\Sloi\wt_bot5_50f6f27` @ `50f6f27`.

## PC e2e (b) gor_dense ON — BLOCKED — 2026-10-29 16:34 МСК

Нужны веса (WEIGHTS_MANIFEST_v4 / FOR_BOT1_GORDENSE_TIMING):

| path | md5 | статус |
|---|---|---|
| `models/dinov2-small/model.safetensors` | ea251be6… | скачан на box `/tmp/dinov2-small` (OK); CopyFromBox на PC временно отказал (unreachable / path root) |
| `models/gor_gate/gor_v3_head.npz` | acb95c52… | **нет** на C:\Sloi / box analysis |
| `models/gor_dense/gor_dense_head.npz` | e35570e1… | **нет**; `dense.npz` для re-export тоже нет |

Источник: private Kaggle `dmitrijkv/koz3-b5-gordense-timing` (на box нет `kaggle` CLI). После появления голов — включить пакет (τ=0.824066, centre_s=250), out `C:\Sloi\pc_jobs\bot5_v4_gordense_50f6f27`, критерий ΔALL>0 и ни один слой не −>0.01 vs (a).

## PC e2e (b) update — 2026-10-29 16:36 МСК

- dinov2-small **staged on PC**: `C:\Sloi\wt_bot5_50f6f27\models\dinov2-small` (model.safetensors 88249960 B, md5 ea251be6…)
- Kaggle CLI installed in koz3-venv; auth with `KAGGLE_API_TOKEN` → **HTTP 403** on dataset `dmitrijkv/koz3-b5-gordense-timing`
- Job stub: `C:\Sloi\pc_jobs\bot5_v4_gordense_50f6f27\STATUS.txt=BLOCKED`
- Lease `pc_bot5_v4_gor_dense.lease` → BLOCKED; GPU freed for bot4 LO

## TASK C — weight unblock probe — 2026-10-29 17:02 MSK

### Verdict: still BLOCKED (heads not found; Kaggle slug is a phantom)

| artifact | expected md5 | result |
|---|---|---|
| `models/dinov2-small/` | ea251be6… | **OK** already on `C:\Sloi\wt_bot5_50f6f27\models\dinov2-small` |
| `models/gor_gate/gor_v3_head.npz` (407736 B) | acb95c52956d712a16c3dd389237a691 | **MISSING** on box + PC |
| `models/gor_dense/gor_dense_head.npz` (204984 B) | e35570e10d17edac32bea8d26fa0f5f4 | **MISSING** on box + PC |

### Why Kaggle returns 403

- Documented private dataset slug: `dmitrijkv/koz3-b5-gordense-timing` (FOR_BOT1_GORDENSE_TIMING.md / `docs/bots/2026-10-29/bot-5/kaggle/gordense_timing/`).
- Owner's `kaggle datasets list -m` does **not** include this slug (only real private sets like xfer-s000…s009, koz3-s3-cred, koz3-b5-or-arch-crops, …).
- `kaggle datasets status/files/metadata dmitrijkv/koz3-b5-gordense-timing` → **HTTP 403 Forbidden**. Same 403 for other never-created slugs. Known-good `dmitrijkv/koz3-xfer-s000` returns **ready**. So 403 here = **dataset was never created / does not exist**, not wrong-owner ACL on a live dataset. (Kaggle returns 403 instead of 404 for missing private resources.)
- Companion kernel `dmitrijkv/koz3-b5-gordense-timing` also never pushed (`kernels status` → Permission denied / wrong slug; `kernels list -s gordense` → Not found). README says bot-1 must `datasets create` then `kernels push` after or_o2c — that step never happened.
- Same 403 for sibling phantom `dmitrijkv/koz3-b5-dinov2reg-gorpatch`.

### Canonical on-box sources (gone)

From FOR_BOT1 / dataset README (previous box workspace, not present now):

- `gor_v3_head.npz` ← `/workspace/koz3_shared/bot-5/analysis/gor_patch/gor_v3_head.npz`
- `gor_dense_head.npz` ← `/workspace/koz3_shared/bot-5/work/gd/root/models/gor_dense/gor_dense_head.npz` (export md5 e35570e1… in `docs/.../runs/gd/export_full.log`)
- `/workspace/koz3_shared` **does not exist** on this box; agent store has no copies; no files of sizes 407736 / 204984 anywhere under `/workspace` `/home/box` `/tmp`.

### PC search (C:\Sloi, F:\2, F:\expds3) — 29.10 ~17:00 MSK

- Exact names `gor_v3_head.npz` / `gor_dense_head.npz` / `dense.npz` (as head source): **0 hits**.
- Exact sizes 407736 / 204984: **0 hits**.
- No `koz3_shared` / `bot-5/analysis/gor_patch` trees on those drives.
- `F:\expds3\work\models` has no gor/dino heads.
- Only unrelated `.npz` (kurgan crops, pixgbm feats, numpy/scipy test data).

### Unblock path (not done this turn)

1. Recover heads from a backup of `koz3_shared/bot-5`, **or**
2. Re-export: need `analysis/gor_dense/dense.npz` (+ feats) then `scripts/gor_dense/export_head.py`; gate head from `scripts/gor_patch` fit — those inputs also missing locally.
3. Then place into `C:\Sloi\wt_bot5_50f6f27\models\gor_gate\` and `...\gor_dense\`, record sha256 here, mark QUEUE **UNBLOCKED**.

**No UNBLOCKED tag yet** — e2e (b) superseded by (b→train) in pc/QUEUE.md; unblock when train STATUS=DONE.

## TASK C follow-up — head retrain + Or manifest — 2026-10-29 17:25 МСК

### Box re-search (bot5, this turn)

- By name `*gor*head*` / exact sizes **407736** / **204984** under `/workspace`, `/home/box`, `/cursor/stores`: **0 hits**.
- `git log --all -- '*gor*head*'`: only `export_head.py` added; **no head binaries in history/LFS**.
- Docs mention: `FOR_BOT7_GOR_V3.md`, `FOR_BOT1_GORDENSE_TIMING.md`, `export_full.log` (md5 e35570e1…, centre_s=250, 204984 B), `gor_v3_taus.json` (head_md5 acb95c52…).
- **Verdict unchanged: heads not found → must retrain.**

### Training package (committed)

| item | value |
|---|---|
| commit | `19ccc9c` (`scripts/gor_heads/*`, patched `scripts/gor_patch/extract_feats.py`) |
| tip (pushed) | `4dadeab` (+ Or manifest) |
| recipe | (1) extract_feats DINOv2-S 384+768 → feats.npz; (2) fit_v3_head PCA-64+L2-logreg C=1 + Landauer 0.85/0.05 on fam10; (3) dense_feats GSDS 2+4; (4) export_head variant `all` → centre_s from CP_GRID (expect **250**); labels from vendored `fixtures/gor_dense_inputs.json` (original export_inputs dump) |
| PROTOCOL §4 | `STATUS.txt` RUNNING/DONE/FAILED, `ckpt.json` stages, idempotent restart, SEED=0, logs in WORK |
| box smoke | `smoke_train_heads.py` **PASS** — both heads exact legacy **byte sizes** 407736 / 204984; shapes/dtypes + head_p sanity OK |
| confidence | **High** on procedure (matches FOR_BOT7 / export_full.log / export_head.py). **Medium** on byte-identical legacy md5 (needs identical dinov2 feats; re-extract may drift). centre_s=250 recovery is the dense sanity metric. |
| uncertain | Exact md5 match; need full `F:\2\train-expds-3` SpOr tifs on PC; `gor_scene.py` not required (labels from fixture). |

### Resource estimate

| machine | threads | wall | peak RAM | notes |
|---|---:|---:|---:|---|
| **PC CPU (queued)** | **2** (18/20 busy) | **~20–25 мин** | ~1.7 GB | extract_v3 ~644 с + dense ~552 с + fit/export ~20 с (box logs) |
| PC CPU | 8 | ~12–15 мин | ~1.7 GB | torch threads; partially IO-bound on tif read |
| Kaggle T4 (alt) | GPU | ~5–10 мин after queue | — | quota reset **31.10 03:00 МСК**; PC preferred (data+dinov2 local; old Kaggle slug phantom) |

### QUEUE (29.10 17:27 МСК)

| slot | status | sha | notes |
|---|---|---|---|
| (a) flags-OFF | **DONE/PASS** @50f6f27 | — | mock ALL 0.3125 path; see risk below |
| (b→train) host @4dadeab | **FAILED** 17:17–17:21 | `4dadeab` | `No module named 'transformers'` on host `.venv` (lease). **SUPERSEDED** |
| (b→train) Docker | **DONE 17:58** | `9f62a96` | runner `pc/runners/run_bot5_gor_heads_docker.ps1`; out `C:\Sloi\pc_jobs\bot5_gor_heads_9f62a96`; ~12–15 мин @8 thr |
| (c) Or-only dfine_aeor | **blocked** | wait `829ea7d` | waiting on bot2 A3 (D-FINE in Docker **n_raw>0**); then rebase on 829ea7d + Docker runner. **Do not cherry-pick 829ea7d yet.** |
| (b′) gor_dense ON e2e (Docker, OFF+ON) | **QUEUED 18:05** | `40fa599` | runner `pc/runners/run_bot5_gor_dense_e2e_docker.ps1`; τ=0.824066 centre_s=250; new head md5 |

Open count vs 2-limit: **(b′) QUEUED** + blocked (c) = 2/2 ((b→train) Docker DONE 17:58). No new leases created by bot5 (existing `pc_bot5_gor_heads.lease` is pc-job-runner).

### Heads retrain approach (`9f62a96`)

1. **Docker-only** on PC — vendor `inference/lib/vendor` is manylinux/cp312; host Windows cannot load `_regex` / tokenizers `.so` (bot2 confirmed; same as A2).
2. Loader aligned with inference: `scripts/gor_heads/dinov2_loader.py` → `gor_gate.load_backbone` → `import_dfine` (transformers **5.21.0**).
3. **Preflight** in `train_heads.py`: MODEL `config.json`+`model.safetensors` + transformers import → else `STATUS=FAILED` (exit 2) before extract.
4. Runner unpacks vendor from `submission_bot7cand_v3on_0876090.zip` into worktree; image `pantum/n7q:20261024`.
5. Box smoke `smoke_train_heads.py` **PASS** (legacy byte sizes 407736 / 204984). Preflight fail-path verified (missing safetensors + no transformers → FAILED).
6. **Not** cherry-picking bot2 `829ea7d` (loud UNAVAILABLE / make_on vendor check) — A3 owns that; bot5 waits for n_raw>0 before Or-only.

**In-repo command (container):**
```bash
export REPO=/repo DATA=/data MODEL=/model WORK=/work DEVICE=cpu THREADS=8
export PYTHONPATH=/repo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/gor_heads/train_heads.py
```

### Risk — host-venv e2e / flags-OFF base

Earlier PC e2e runs on the **host `.venv`** (including the v2 base **0.3125** and bot5 **(a) flags-OFF** identity) may have executed **without** Linux-only components (vendor transformers / D-FINE). Bot2 is checking whether the base needs **recomputing in Docker**. That affects our **(a) flags-OFF identity comparison** — if Docker base ≠ 0.3125 host figure, re-score (a) in Docker before claiming identity. Track bot2 A3 / base-recompute notes before locking EXPERIMENTS.csv rows that depend on that number.

### Or-only (c) — unblocked → (c′) queued

- A3 @`829ea7d` **DONE** (`pc/results/bot2_a3_docker_829ea7d/`): n_raw=**2506**, ENV=docker pantum/n7q, EXPDS_STRICT=1.
- Bring-in: **not** rebase/force-push. Cherry-pick of `829ea7d` conflicted on tip (divergent `sp_dfine`/`pipeline`; `make_on` absent). Ported A3 essentials as normal commit **`2b745df`**: `strict_mode` / `vendor_platform_hint` / `write_unavailable_warning`, import hint, region_feats `load_error`+WARNING, pipeline phase log; tests `test_bot2_dfine_import_loud` (7 passed). Default flags stay OFF.
- Assert gate: `scripts/assert_dfine_aeor_or_nonzero.py` @`999572d` (still on tip).
- Runner: `pc/runners/run_bot5_dfine_aeor_or_docker.ps1` (JobDir `F:\Sloi_pc_jobs\bot5_dfine_aeor_or_2b745df`).
- QUEUE: previous (c)@999572d **SUPERSEDED**; (c′) QUEUED after (b′). Open: (b′)+(c′)=2/2.

**No leases created by bot5.** main / `rerank_arch_lgbm.txt` untouched.


### Update 29.10 18:05 МСК — heads DONE → (b′) e2e filed

- PC Docker train @`9f62a96` **DONE** (docker rc=0, 17:40→17:58 МСК, ~16 мин @8 thr CPU). All ckpt stages DONE incl. validate.
- Box check: `gor_v3_head.npz` 407736 B sha256 `5bd88f69…a75e7d` (md5 `487c48e4…`), `gor_dense_head.npz` 204984 B sha256 `feb1591b…5b2cb7` (md5 `d92f38f3…`) — **MATCH** SUMMARY/HEADS_SHA256; sizes = legacy; md5 ≠ legacy (re-extracted feats, same recipe+SEED=0).
- `docs/WEIGHTS_MANIFEST_v4.md` updated, commit `40fa599` on `bots/bot5-candidate-v4` (pushed; main untouched; heads stay out of git — `*.npz` gitignored).
- Filed (b′) in `pc/QUEUE.md`: mock e2e OFF + gor_dense ON, **both Docker** (flags-off base recomputed in Docker per §10), ON overrides `gate_head_md5`/`head_md5` to the new md5 (legacy md5 ≠). Criterion PROTOCOL §5: Δ>0 vs Docker OFF, no layer drop >0.01, **report Δ vs +0.0154**, wall_sec (target ≤15 / limit 30). If Δ ≪ +0.0154 → follow-up **τ recalibration** (do not retune in this run).
- Open bot5 requests (as of 18:05): (b′) + blocked (c). Updated 18:49: (b′)+(c′)=2/2 after A3 unblock. No leases created by bot5.

### Update 29.10 18:49 МСК — A3 green → (c′) Or-only Docker e2e filed

- Bot2 A3 @`829ea7d` lifted blocker: n_raw=2506 in Docker (details `pc/results/bot2_a3_docker_829ea7d/`).
- Merged A3 gate via **normal commit** `2b745df` on `bots/bot5-candidate-v4` (pushed). Conflicts on full cherry-pick/merge of bot2 chain → minimal port only. **No force-push / no rebase.**
- Box tests: full pytest **558 passed** / 26 skipped (1 flaky soft-ceiling timing failed once, passed on retry); new loud-import **7/7**; flag-off tests green. Packaging tests needed local pixgbm stubs (not committed).
- Filed (c′) in `pc/QUEUE.md` @`2b745df`: paired Docker OFF vs dfine_aeor Or-only ON + assert nonzero. Criterion: HARD FAIL if n_raw/Or dets==0; ΔALL>0; no layer drop >0.01; wall_sec ≤15/limit 30. Runs after (b′).
- Open bot5: (b′)+(c′)=2/2. No leases.

### Update 29.10 19:45 МСК — (b′) REJECTED; pytest-log fetch queued; (c′) loc plan

#### (A) (b′) gor_dense @40fa599 — DONE/REJECTED

- Docker paired e2e: **ON ≡ OFF**, ALL F1 **0.312873**, Δ=**0** (layer-identical).
- `warn_scan.txt`: gor_dense emitted **4** raw points, confs **0.179 / 0.167 / 0.228 / 0.219** → **max_raw=0.228 ≪ τ=0.824066** → **0** survived threshold → **NO-OP**.
- Verdict: **not accepted**. **`[gor_dense]` stays OFF in v4.**
- Likely cause: retrained heads (md5 `487c48e4…` / `d92f38f3…` ≠ legacy) shift score distribution; legacy τ calibrated on old heads is too high for the new score scale.
- Artifacts: `pc/results/bot5_gor_dense_e2e_40fa599/` (scores.json, warn_scan.txt; full out_* not mirrored).
- EXPERIMENTS.csv: NO-OP/REJECTED row @40fa599 (enriched with max_raw vs τ).
- **τ recalibration sketch (low priority, NO PC request):** on a box/offline pass, collect gor_dense pre-thr scores (or re-run ON with τ≈0 dump) over mock SpOr tiles with new heads; pick τ as OOF quantile / max-F1 on held-out sites so expected n_pass ≈ legacy package intent (~few high-precision hits, Δ≈+0.0154). Re-validate PROTOCOL §5 only after τ chosen — do **not** retune on the rejected e2e alone.

#### (B) PC pytest @2b745df exit 1

- Box same sha PASS (bot-1: 548 passed / 0 failed / 37 skipped).
- Searched `/home/box/shared/koz3/pc/results/` for `cpu_pytest_bot5_tip` / `pytest.log` — **not found**.
- Queued CPU fetch: `pc/runners/run_bot5_fetch_pytest_log.ps1` → `pc/results/bot5_pytest_log_2b745df/` (pytest.log + python -V + pip freeze). No lease. Open slots: **(c′)+fetch = 2/2**. No code fix until log triage.

#### (C) (c′) Or localization analysis (prepared, not run on PC)

Bot2 FYI (`reports/bot2_dfine_a3_localization.md`): A3 SpOr D-FINE — 13 above-thr dets all FP, median dist_any ≈ **388 m**, thr-sweep ≈ 0 TP (domain/model, not CRS).

**Plan when (c′) lands:**
1. Confirm `assert_dfine_aeor_or_nonzero.py` (HARD FAIL if Or n_raw/dets==0).
2. Run `scripts/analyze_dfine_aeor_or_localization.py` (shared path below) on `out_on/result.geojson` + GT (+ `dump_on` if present) → `analysis/summary.json`: n_or, score max/p50, **median dist_any_m**, within 2/50/200 m, soft_tp@2m.
3. If median ≫ match radius / soft_tp@2m≈0 (A3-like), treat as localization failure even if count>0; still require PROTOCOL §5 ΔALL>0.
4. No existing bot2 reusable CLI for this (A3 analysis was offline one-shot → `analysis/summary.json`); this script encodes the same checklist.

**Script path:** `/home/box/shared/koz3/scripts/analyze_dfine_aeor_or_localization.py`

**Open bot5 (superseded by 19:50 update):** was (c′)+fetch; fetch CANCELLED; open=(c′) 1/2.

### Update 29.10 19:50 МСК — PC pytest packaging fix (test-only tip)

- Bot1 fetched PC pytest.log: **1 failed / 43 passed** — `tests/test_dfine_aeor.py::test_not_packaged_when_off` → `SystemExit` from `scripts/build_submission.py:105` `модель не найдена: models/pixgbm/all/poly.txt` (weights not in git; clean PC worktree).
- **Fix ONLY in tests** @ `ed73c5e` on `bots/bot5-candidate-v4` (pushed, no force-push / no main):
  - `tests/conftest.py`: `allow_missing_submission_models(monkeypatch)` — Path.exists treats missing `models/*` as present so `collect()` packaging asserts still run.
  - Applied to packaging callers: `test_dfine_aeor`, `test_bot6_orgate`, `test_bot5_gor_dense` (off), `test_bot7_v3_flags` (train_only + flag_modules), `test_open_hooks`, `test_bot7_candidate_v2` collect-pix; `test_pix_models_shipped` → `pytest.skip` if weights absent.
- **Verify (box):**
  - pixgbm weights hidden: packaging/pix tests → **8 passed, 1 skipped**.
  - full suite with models present → **559 passed, 26 skipped**.
- **Pipeline unchanged** → (c′) Or-only Docker e2e still valid @ **`2b745df`**. New tip `ed73c5e` is **test-only**.
- QUEUE: `run_bot5_fetch_pytest_log` **CANCELLED** (bot1 got log). Open bot5: **(c′) only = 1/2**. `models/rerank_arch_lgbm.txt` / main untouched.
