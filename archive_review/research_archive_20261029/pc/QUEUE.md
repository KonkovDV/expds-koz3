# Очередь ПК (ведёт «1»)

## READY [bot-4] Fix C `dfine_ae_gor_area_tau` mock+honest-10 Docker — 29.10.2026 ~20:41 МСК

- **СТАТУС: READY-TO-LAUNCH — GPU Docker** (mock OFF+ON; honest-10 if regions; CPU score/assert)
- **от:** bot-4
- **ветка / sha:** `bots/bot4-large-objects` @ **`1a0e779`** (Fix C code `17e35a5` + runner `e9d187a`…`fc18035` + report)
- **что:** изоляция Fix C поверх Fix A path. **Оба плеча** `[dfine_aeor] enabled=true markups=["ae"]` + thresholds_oof (gor=1.01). **ON** += `[large_objects] enabled=true` + **только** `dfine_ae_gor_area_tau=true` (τ_large=0.20, ae+gorodishcha+box_area≥5000). Остальные LO sub = false. li не трогаем.
- **диагноз:** `reports/ae_ortho_diagnosis.md` §3 + OOF Ae 43/52@0.05 / 21/52@0.3; gor τ=1.01 OFF; ae_fort≈64 м. split_large OFF / min_area_px=3 — не culprits. Полный LO mock −0.016 — не включать.
- **OOF schedule:** τ=0.20 → gor TP1 FP397 (all-ae worst-case); 0.10 TP9 FP6305; 0.30 TP0 FP32. Выбран 0.20.
- **runner:** `pc/runners/run_bot4_fix_c_docker.ps1` + `launch_bot4_fix_c_schtasks.ps1` + `bot3_ae_weights_preflight.py` + `scripts/assert_dfine_aeor_ae_nonzero.py`
- **JobDir / wt:** `C:\Sloi\pc_jobs\bot4_fix_c` / `C:\Sloi\pc_jobs\wt_bot4_fix_c` (**C:** only)
- **weights:** `C:\Sloi\pc_jobs\weights\dfine_ae_ft` (sha prefix `3e619878`) + `models/dfine_aeor/thresholds_oof.json`; Docker full-load preflight; HARD FAIL assert Ae dets==0
- **запуск (schtasks):**
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File pc\runners\launch_bot4_fix_c_schtasks.ps1
  ```
- **ожидаемое время:** ~mock 2×(5–15 мин) + optional honest-10; лимит 60 мин
- **критерий (PROTOCOL §5):** ΔALL_F1(ON−OFF) **≥ 0** и ни один слой не падает >0.01; Ae dets>0 на обоих плечах. Δ=0 при dets>0 на mock = «нет mid-conf large ae gor» (не silent miss) — смотреть honest-10 / log `dfine_aeor n_det`.
- **артефакты:** `out_off/`, `out_on/`, `e2e_off_vs_on.json`, optional `scores_honest.json`, SUMMARY, STATUS, heartbeat; копия `pc/results/bot4_fix_c/`
- **не делать:** не host `.venv` inference; не main; не LB; не включать coarse/полный LO; флаг в default.toml остаётся OFF; **не писать JobDir/wt на F:**
- **слоты bot-4:** заявка #1 (единственная открытая); ablation DONE; OOF post-tuning SUPERSEDED.

## READY [bot3] Fix A+B ae-ortho dfine_aeor+ae_recall_relax Docker e2e — 29.10 ~20:40 MSK
- **STATUS: READY-TO-LAUNCH — GPU Docker** (mock OFF+ON; honest-10 if regions; CPU score/assert)
- **from:** bot-3
- **branch / SHA:** `bots/bot3-ae-ortho-recall` / **`1e7073afd29a503e2811fe5a9a9e0e52cb92643e`**
- **diagnosis:** `reports/ae_ortho_diagnosis.md` — eb268d6 ae R≈0.013 by construction (fort-mask@0.998 only; no D-FINE). Fix A expected +0.02–0.05 LB; Fix B small.
- **what:** PROTOCOL §10 only `pantum/n7q:20261024`. Flags default OFF in TOML; runner injects ON:
  - **A** `[dfine_aeor] enabled=true` markups=["ae"] weights models/dfine_ae + thresholds_file models/dfine_aeor/thresholds_oof.json (f4pack **249b86fb**)
  - **B** `[ae_recall_relax] enabled=true` (ae_fort→0.995 + optional li_ae/fusion)
- **paths:**
  - JobDir `C:\Sloi\pc_jobs\bot3_ae_ortho_ab_f4`
  - Wt `C:\Sloi\pc_jobs\wt_bot3_ae_ortho_ab_f4`
  - Weights `C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4\` (box `/workspace/kaggle/out/dfine-ae-ft-f4pack/`)
  - Vendor `C:\Sloi\pc_jobs\_vendor_zips\submission_bot7cand_v3on_0876090.zip`
- **PASS:** ENV=docker line; docker full-load preflight (249b86fb); assert Ae n_det>0 ON; scores_mock ON ae≥OFF ae and ALL not −>0.01; honest-10 if regions.
- **runtime:** ~20–40 min mock GPU.
- **runner:** `pc/runners/run_bot3_ae_ortho_ab_docker.ps1 -Variant f4`
- **launch:** `pc/runners/launch_bot3_ae_ortho_ab_f4_schtasks.ps1` (task `koz3_bot3_ae_ortho_ab_f4`)
- **lease:** `leases/pc_bot3_ae_ortho_ab_f4.lease`
- **report:** `reports/bot3_ae_ortho_recall.md`
- **do not:** host .venv inference; LB; merge main; use aeor-eval tip for this gate (wrong code).


## [bot2] 29.10 18:52 МСК — A4 D-FINE agreement_m=50 (Docker)
- **СТАТУС: INVALID 20:35 — wall~28s exit=-1 stale geojson (LESSONS); do NOT use numbers; NOT requeued (bot2 sat-recall analysis)**
- **СТАТУС: CANCELLED 19:45 — лог снят ботом 1 напрямую, причина: нет models/pixgbm/all/poly.txt в worktree**
- **от:** bot2
- **ветка:** `bots/bot2-e2e-v3-prep`
- **коммит:** `c3d912c1a7fb2a551d2a4d37446f8d840e138d71`
- **что:** mock e2e A4 = D-FINE ON + `drop_groups=[]` + **`agreement_m=50`** в Docker `pantum/n7q:20261024` (НЕ host venv). Гипотеза A3: 13 pass_thr — далёкие FP (median ~388 м, все ≥70 м от U-Net); agreement отсекает их → ожидаемо ALL/sp_or ≈ B (0.31287 / 0.245). Не claim «D-FINE помогает».
- **критерий PASS:**
  1. ENV=docker:pantum/n7q:20261024 в STATUS;
  2. mock ALL ≥ **0.31287** и sp_or ≥ **0.245−0.01** (=≥0.235); ae/li без drop >0.01 vs B;
  3. stages: n_raw≈2506 (модели грузятся); после agreement ожидаемо n добавленных sp_dfine в GeoJSON ≈0 (или ≪13);
  4. SCORE vs B geojson; STATUS DONE.
- **ожидаемое время:** ≤15–20 мин GPU wall (A3 ~15 мин).
- **runner / команда:** `pc/runners/run_bot2_a4_dfine_agreement_docker.ps1` (модель A3 docker import — overlay `expds/sp_dfine.py`+`pipeline.py`, toml enabled=true drop_groups=[] agreement_m=50)
  ```
  # JobDir=F:\Sloi_pc_jobs\bot2_a4_dfine_agreement
  # Wt=F:\Sloi_wt\wt_bot2_a4 @ c3d912c1a7fb2a551d2a4d37446f8d840e138d71
  # ZipV3=F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v3on_0876090.zip
  # Regions=F:\Sloi_pc_jobs_archive\v3e2e\regions
  # Image=pantum/n7q:20261024
  # docker run --gpus all -e EXPDS_DFINE_DUMP=/data/output/dump_A \
  #   -v zip_A:/data/zip:ro -v regions:/data/regions:ro -v JobDir:/data/output \
  #   -w /data/zip pantum/n7q:20261024 \
  #   python inference/main.py /data/regions /data/output/out_A/result.geojson
  # score vs B=F:\Sloi_pc_jobs_archive\bot2_v3_hotfix_1ec2d7c\out_B\result.geojson
  powershell -File pc/runners/run_bot2_a4_dfine_agreement_docker.ps1
  ```
- **отчёт:** `reports/bot2_dfine_a3_localization.md`
- **не делать:** не host `.venv` inference; не LB; не main; не PR; не включать agreement_m в default.toml навсегда (остаётся 0 в коде; только toml патч раннера).
- **lease:** bot-1 выдаёт; GPU-полоса.

## [bot7] QUEUED CPU — compare_candidates_v2docker fix @69a9a85 — 29.10 19:46 МСК
- **СТАТУС: DONE 19:50 (bot1; sha в раннере исправлен) — pc/results/bot7_compare_69a9a85.md** (было: QUEUED, повтор после FAIL `compare_candidates_v2docker`: не было `--out`; теперь `--out` optional + явный в команде; колонка `env` по PROTOCOL §10)
- **от:** bot7
- **ветка:** `bots/bot7-compare-candidates` (DRAFT PR #8, base `bots/bot7-honest-eval-layer` / stacked on PR #2)
- **коммит:** `69a9a85c07c0f346a6021825381761a2c24986f1` (было b6f835d; fix --out/--env)
- **что:** прогон `scripts/compare_candidates.py` по кандидатам mock e2e: v2 (docker out_mock), v3, hotfix B, v4 OFF, bot4 OFF/ON → compare.md/csv + rule PASS/FAIL + li↔ae + **env** column. Baseline V2 = docker `bot2_v2_base_docker` out_mock (не host).
- **критерий:** скрипт завершается 0; таблица ALL/li/sp_or/ae × global/layer + **env**; Δ vs v2; правило «слой не падает >0.01»; EXPERIMENTS с `env=…`; артефакты в `%KOZ3_OUT%\compare` (+ копия в shared `pc/results/`).
- **ожидаемое время:** ≤10–15 мин CPU @ ≤4 thr (без inference; только metric matching).
- **env (PROTOCOL §10 / bot2_runtime_env):** v2=docker, v3=docker, hotfix_B=host, v4_off=host, bot4_off=host, bot4_on=host.
- **предыдущий FAIL:** filler `bot7_compare_candidates_v2docker` (V2=out_mock) упал: `error: the following arguments are required: --out`. Исправлено в 69a9a85 (`--out` optional → `%KOZ3_OUT%\compare` / `./compare_out`) + команда ниже передаёт `--out` явно.
- **предыдущий DONE:** @b6f835d 18:22 → `pc/results/bot7_compare_candidates_b6f835d/` (scoring ENV=docker; числа: v3 FAIL sp_or; bot4_on FAIL li; hotfix/v4/bot4_off PASS).
- **команда (Windows) — исправленная, с явным --out и --env:**
  ```
  set KOZ3_OUT=F:\Sloi_pc_jobs\bot7_compare_candidates_69a9a85
  mkdir %KOZ3_OUT%
  echo RUNNING > %KOZ3_OUT%\STATUS.txt
  git fetch origin bots/bot7-compare-candidates && git checkout bots/bot7-compare-candidates && git reset --hard 69a9a85

  rem GT: archive v3e2e (C: reclaim) / fallback
  set GT=F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson
  if not exist %GT% set GT=F:\expds3\mock_val

  rem V2 baseline = docker out_mock (v2docker); остальные — archive/host paths
  set V2=C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6\out_mock
  if not exist %V2% set V2=F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v2
  set V3=F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v3
  set HOTFIX=F:\Sloi_pc_jobs_archive\bot2_v3_hotfix_1ec2d7c
  set V4OFF=C:\Sloi\pc_jobs\bot5_v4_off_50f6f27
  if not exist %V4OFF% set V4OFF=F:\Sloi_pc_jobs_archive\bot5_v4_off_50f6f27
  set B4OFF=F:\Sloi_pc_jobs_archive\bot4_lo_mock_337a27c\out_off
  set B4ON=F:\Sloi_pc_jobs_archive\bot4_lo_mock_337a27c\out_on

  python scripts/compare_candidates.py ^
    --gt %GT% --v2 %V2% --out %KOZ3_OUT%\compare --workers 4 ^
    --labels v2,v3,hotfix_B,v4_off,bot4_off,bot4_on ^
    --env docker,docker,host,host,host,host ^
    --branch bots/bot7-compare-candidates --sha 69a9a85 ^
    --experiments-row --experiments-file %KOZ3_OUT%\compare\EXPERIMENTS_rows.csv ^
    %V2% %V3% %HOTFIX% %V4OFF% %B4OFF% %B4ON%

  rem find_geojson сам возьмёт out_B/result.geojson из hotfix при передаче корня HOTFIX
  echo DONE >> %KOZ3_OUT%\STATUS.txt
  ```
- **lease:** `leases/pc_bot7_compare_candidates.lease` (bot1: QUEUED → RUNNING)
- **не делать:** не inference/docker GPU; не main; не LB; не submissions; не force-push чужих веток.




## [bot2] 29.10 17:35 МСК — v2 base в docker
- **СТАТУС: CANCELLED 19:45 — лог снят ботом 1 напрямую, причина: нет models/pixgbm/all/poly.txt в worktree**
- **от:** bot2
- **ветка / sha базы:** zip `submission_bot7cand_v2_eb268d6.zip` (= дерево кандидата v2 **`eb268d6e4adfbed32670cabeef15a709649b8246`**); runner/отчёт с ветки `bots/bot2-e2e-v3-prep` @ `829ea7d` (только шаблон docker-раннера, **без** overlay sp_dfine ON)
- **что:** пересчёт **gate baseline** mock ALL/по слоям + **honest-10** в Linux Docker `pantum/n7q:20261024` (тот же образ/монтирования, что A3). Цель: канон ворот в той же среде, что LB и A3; сравнить с 0.3125 / 0.3129; поймать SP_DFINE_*/accel UNAVAILABLE.
- **зачем:** kit `v3e2e` 0.3125 был docker (BOARD), но hotfix A/B / A2 / часть flags-OFF шли на host `.venv` (ALL 0.312873; D-FINE vendor manylinux не грузится). Отчёт: `reports/bot2_runtime_env.md`.
- **критерий:**
  1. `main.log` среда Linux / образ pantum; **нет** `SP_DFINE_UNAVAILABLE` (у v2 sp_dfine OFF — если есть WARN, зафиксировать);
  2. mock: ALL + li/sp_or/ae vs **0.312509 / 0.336983 / 0.242915 / 0.095556** (допуск ε≤0.001 на ALL, слой не хуже >0.01 без объяснения);
  3. honest-10: ALL (+ по слою если scorer умеет) записан; сравнить с PROTOCOL 0.3129 / LB 0.41238 (не обязан совпасть — другой GT);
  4. STATUS DONE + geojson + scores.json в `%KOZ3_OUT%`.
- **ожидаемое время:** mock ≤10–15 мин GPU (v2 ~83 с inference в BOARD + score); honest-10 ≤25–40 мин wall (оценка; как bot7/v3e2e honest). Итого слот ≤60 мин.
- **команда / runner:** модель `pc/runners/run_bot2_a3_dfine_docker_import.ps1` → `pc/runners/run_bot2_v2_base_docker.ps1`
  ```
  set KOZ3_OUT=C:/Sloi/pc_jobs/bot2_v2_base_docker_eb268d6
  rem Image=pantum/n7q:20261024
  rem ZipV2=C:\Sloi\pc_jobs\v3e2e\kit\submission_bot7cand_v2_eb268d6.zip
  rem Regions=C:\Sloi\pc_jobs\v3e2e\regions
  rem docker run --gpus all --network none -v zip_work:/app -v regions:/data -v out:/out
  rem   pantum/n7q:20261024 python inference/main.py /data /out/result.geojson
  rem score mock vs gt; honest-10 через tools_v3e2e / scoreboard --honest 10 (как kit)
  rem сохранить main.log (rg SP_DFINE|UNAVAILABLE|ускоритель|среда) + scores
  ```
- **артефакты:** `%KOZ3_OUT%/out_mock/result.geojson`, `scores_mock.json`, `out_honest10/`, `main.log`, `STATUS.txt`, `heartbeat.log`.
- **не делать:** не включать sp_dfine/gor_gate; не overlay 829ea7d в zip v2; не host `.venv` для inference; не LB; не main.

## [bot2] 29.10 16:41 МСК — A2 D-FINE dump+stages (почему A≡B)
- **СТАТУС: DONE** — ImportError / n_raw=0 on PC path (A≡B); stages in `pc/results/bot2_a2_dfine_stages_fb66c89/`. Follow-up → A3 Docker @829ea7d.
- **от:** bot2
- **ветка:** `bots/bot2-e2e-v3-prep`
- **коммит:** `fb66c899a6b534f6bbec06c7496736c3eaa4c3ac` (1ec2d7c + stages dump + geo CRS guard)
- **что:** один mock e2e **A2** = A (enabled=true, drop_groups=[]) с `EXPDS_DFINE_DUMP` + **stages.json** (raw→thr→in_state). Цель: где обнуляется D-FINE (A≡B). B уже известен; не повторять B.
- **критерий:** в `%KOZ3_OUT%/dump_A/stages.json` есть totals; по `n_raw_boxes` / `n_would_pass_thr` / `n_sp_dfine_in_state` / `skip_reason` видно стадию. Ожидание: in_state=0 или raw=0 (гипотеза b). wall ~2–3 мин GPU.
- **ожидаемое время:** ≤10 мин (checkout + overlay + 1× main + score vs B geojson если есть).
- **команда:**
  ```
  set KOZ3_OUT=C:/Sloi/pc_jobs/bot2_a2_dfine_stages_fb66c89
  mkdir %KOZ3_OUT%
  echo RUNNING > %KOZ3_OUT%/STATUS.txt
  git fetch origin bots/bot2-e2e-v3-prep && git checkout bots/bot2-e2e-v3-prep && git reset --hard fb66c89

  rem zip base = v3on (есть weights); overlay pipeline.py + sp_dfine.py из worktree
  rem Patch-Toml A: enabled=true, drop_groups=[]
  set EXPDS_DFINE_DUMP=%KOZ3_OUT%/dump_A2
  rem python inference/main.py <Regions> %KOZ3_OUT%/out_A2/result.geojson
  rem сохранить main_A2.log + dump_A/stages.json + stages.jsonl + boxes*.json
  rem e2e_score A2 vs out_B из bot2_v3_hotfix_1ec2d7c (ожидание ≡ если снова 0 dets)

  rem runner: можно скопировать run_bot2_v3_hotfix_1ec2d7c.ps1 → только A, Sha=fb66c89, JobDir=...
  echo DONE >> %KOZ3_OUT%/STATUS.txt
  ```
- **runner:** `pc/runners/run_bot2_a2_dfine_stages_fb66c89.ps1` (A + dump; B reuse hotfix)
- **артефакты:** `dump_A/stages.json`, `main_A.log` (строки SP_DFINE_UNAVAILABLE / n_det=).

## [bot2] 29.10 16:14 МСК — mock e2e v3 hotfix A/B
- **СТАТУС: DONE (A≡B ALL 0.312873 / sp_or 0.245064; B → v4). Follow-up A2 dump+stages @ fb66c89.**

- **от:** bot2
- **ветка:** `bots/bot2-e2e-v3-prep`
- **коммит:** `1ec2d7cec6c66066a2ef88cae9080cbdf7e163df` (база `78af57f` + safety valve `5614d0d` + harvest C `daeeed9` + dump `8eacb68` + hotfix docs `1ec2d7c`)
- **что:** два варианта mock e2e (docker, GTX 1650) vs v2 baseline ALL 0.3125 / sp_or 0.2429. Диагноз bot1: R снял −46 v2; 13 D-FINE — FP (CRS OK, ~191 м до GT).
- **команда:**
  ```
  set KOZ3_OUT=C:/Sloi/pc_jobs/bot2_v3_hotfix_1ec2d7c
  mkdir %KOZ3_OUT%
  echo RUNNING > %KOZ3_OUT%/STATUS.txt
  git fetch origin bots/bot2-e2e-v3-prep && git checkout bots/bot2-e2e-v3-prep && git reset --hard 1ec2d7c

  rem === A: D-FINE ON, drop_groups=[] (make_on MODE=C / default) ===
  rem Собрать zip: python analysis/2026-10-29/bots/bot-7/work6/on/make_on.py <zip_off> %KOZ3_OUT%/v3on_C.zip C
  rem   (B7_REPO=worktree этой ветки; веса dfine_f0 как у v3on_0876090)
  set EXPDS_DFINE_DUMP=%KOZ3_OUT%/dump_A
  rem docker e2e mock как у v3e2e (тот же образ/монтирования F:\expds3\mock_val → out_A)
  rem сохранить submission.geojson + main.log → %KOZ3_OUT%/out_A/

  rem === B: sp_dfine enabled=false (zip flags-off той же ветки / без make_on C) ===
  rem docker e2e mock → %KOZ3_OUT%/out_B/
  rem EXPDS_DFINE_DUMP не нужен

  echo DONE >> %KOZ3_OUT%/STATUS.txt
  ```
  Точный docker-раннер — как у `pc_jobs/v3e2e` (полоса «1» подставляет). Resumable: STATUS.txt + out_A/out_B.
- **ожидаемое время:** ≤15 мин GPU × 2 (каждый ~3–5 мин wall как 143 с у v3on).
- **критерий прохождения:** для принятого варианта — ALL F1 ≥ 0.3125−0.01 (=0.3025) и ни один слой не падает >0.01 vs v2; sp_or ≈ 0.2429 (±0.01). Предпочтение A; если A роняет ALL >0.01 из‑за FP D-FINE — принять B. Записать обе строки в EXPERIMENTS.csv.
- **не делать:** не MODE=R / не drop_groups R; не порт 0037c0e; не LB; не main; не force-push чужих веток.



## [bot5] 29.10 16:17 МСК — (a) mock e2e v4 flags-OFF identity vs v2 base

- **от:** bot5
- **ветка:** `bots/bot5-candidate-v4`
- **коммит:** `50f6f2777f0f48670818e96491c837e4e6644106`
- **что:** e2e mock при **всех новых флагах OFF** (`[sp_dfine]`/`[dfine_aeor]`/`[gor_dense]`/`[gor_gate]`/`[orgate]` = false). Должен совпасть послойно с v2 / v3-without-SpOr (база mock ALL **0.3125**, sp_or **0.2429**). SpOr D-FINE намеренно OFF (bot1_v3_spor_diag).
- **команда:**
  ```
  set KOZ3_OUT=C:/Sloi/pc_jobs/bot5_v4_flags_off_50f6f27
  mkdir %KOZ3_OUT%
  echo RUNNING > %KOZ3_OUT%/STATUS.txt
  git fetch origin bots/bot5-candidate-v4 && git checkout bots/bot5-candidate-v4 && git reset --hard 50f6f27
  rem docker e2e mock как у pc_jobs/v3e2e (образ pantum/n7q:20261024; F:\expds3\mock_val → %KOZ3_OUT%/out/)
  rem конфиг: configs/default.toml БЕЗ правок (все новые флаги уже false)
  rem сохранить submission.geojson + main.log + wall-clock (inference time)
  echo DONE >> %KOZ3_OUT%/STATUS.txt
  ```
  Точный docker-раннер подставляет полоса «1». Resumable: STATUS.txt.
- **ожидаемое время:** ≤15 мин wall на объёме mock (лимит 30); целевой inference ≤15 мин.
- **критерий прохождения:** ALL F1 = v2 base **0.3125** (±ε слоя); послойно = v2 / v3-without-SpOr (sp_or ≈ 0.2429); **ни один слой не падает >0.01**; записать wall-clock inference. EXPERIMENTS.csv — после метрик (бот 5).
- **не делать:** не включать sp_dfine/dfine_aeor/gor_*; не вариант R; не main; не трогать `models/rerank_arch_lgbm.txt`.


## [bot5] 29.10 17:20 МСК — (b→train) reproduce gor_v3_head + gor_dense_head — **FAILED 17:17–17:21 МСК** (SUPERSEDED)
- **СТАТУС: FAILED / SUPERSEDED** — host `.venv` `No module named 'transformers'` (lease `pc_bot5_gor_heads.lease`). Do not rerun host command. Replaced by Docker (b→train) @9f62a96 below.


- **от:** bot5
- **заменяет:** прежний (b) mock e2e gor_dense ON @50f6f27 — **BLOCKED** (heads missing; TASK C). e2e (b) **re-queues after heads exist** (same τ=0.824066, centre_s=250, Δ≈+0.0154). Limit 2: this slot = train, not a third request. (a) flags-OFF already DONE/PASS.
- **ветка:** `bots/bot5-candidate-v4`
- **коммит:** `4dadeab1788dbf9f259623a5f8afda14d6917f60` (train scripts @ `19ccc9c`; tip includes Or manifest)
- **что:** воспроизвести обе головы DINOv2-S: `gor_v3_head.npz` (PCA-64+logreg, legacy md5 acb95c52… / 407736 B) + `gor_dense_head.npz` (variant all, centre_s=250, legacy md5 e35570e1… / 204984 B). Рецепт `scripts/gor_heads/train_heads.py` (STATUS/ckpt resumable, SEED=0). RULES: только KOZ3 SpOr; dinov2 Apache-2.0; без Yandex/Google/Bing/Esri / КОЗ №2.
- **ресурсы:** **2 CPU threads** (на ПК сейчас ~18/20 busy), peak RSS ~1.7 GB. Оценка wall: **~20–25 мин @ 2 thr** (extract_v3 ~644 с + dense_feats ~552 с + fit/export ~20 с; из box logs). @ 8 thr: ~12–15 мин (частично IO). Альтернатива Kaggle GPU после reset **31.10 03:00 МСК**: extract ~1–2 мин×2 → всего ~5–10 мин GPU, но PC лучше (данные + dinov2 уже на месте; Kaggle slug gordense-timing — phantom).
- **порядок GPU полосы:** после hotfix A/B бота 2; затем этот train; затем (когда heads готовы) бывший (b) gor_dense ON; flags-OFF (a) уже PASS.
- **команда:**
  ```
  set KOZ3_OUT=C:/Sloi/pc_jobs/bot5_gor_heads_4dadeab
  mkdir %KOZ3_OUT%
  echo RUNNING > %KOZ3_OUT%/STATUS.txt
  git fetch origin bots/bot5-candidate-v4 && git checkout bots/bot5-candidate-v4 && git reset --hard 4dadeab
  set DATA=F:\2\train-expds-3
  set MODEL=C:\Sloi\wt_bot5_50f6f27\models\dinov2-small
  set WORK=%KOZ3_OUT%
  set DEVICE=cpu
  set THREADS=2
  set OMP_NUM_THREADS=2
  set PYTHONPATH=.
  python scripts/gor_heads/train_heads.py
  rem resumable: ckpt.json stages; on crash re-run same command
  rem outputs also copied to models/gor_gate/ + models/gor_dense/
  ```
- **ожидаемое время:** 20–25 мин @ 2 thr (лимит 45); при освобождении ядер можно THREADS=8 (~12–15 мин).
- **критерий прохождения:** `STATUS.txt` → DONE; оба `.npz` в `%WORK%/heads/`; `validate_heads.py` ok (shapes feat_dim 1536/768, W 64, w 65, head_p∈[0,1]); dense export summary `centre_s=250` (или documented); SUMMARY.json с sha256+md5. Byte-identical legacy md5 — желательно, не обязательно (feats re-extract). Скопировать heads на `C:\Sloi\...\models\gor_gate\` и `models\gor_dense\`.
- **выходы + хеши:** `%WORK%/heads/gor_v3_head.npz`, `gor_dense_head.npz`, `SUMMARY.json`, `ckpt.json`, `train_heads.log`, `STATUS.txt`. Записать sha256/md5 в SUMMARY (бот 5 подхватит в манифест).
- **после DONE:** bot5 re-queues (b′) mock e2e gor_dense ON with new head sha256 in config.
- **не делать:** не main; не EXPERIMENTS; не трогать `models/rerank_arch_lgbm.txt`; не создавать lease-файлы (только bot1); не Kaggle push.


## [bot5] 29.10 17:27 МСК — (b→train) gor_heads retrain **in Docker** @9f62a96 (BLOCKER for gor_dense e2e)

- **от:** bot5
- **СТАТУС: DONE 17:58 МСК** (docker rc=0; heads sha256 verified on box, sizes 407736/204984; manifest @40fa599). Replaced by (b′) e2e below.
- **заменяет:** 17:20 (b→train) host command — **FAILED**. e2e (b′) still re-queues after heads DONE.
- **ветка:** `bots/bot5-candidate-v4`
- **коммит:** `9f62a9676b84e6963b892e6eb9f0907ad975f6ba` (dinov2_loader + preflight; tip includes Or assert `999572d`)
- **что:** те же головы `gor_v3_head.npz` + `gor_dense_head.npz`, но запуск **только в Docker** `pantum/n7q:20261024` с `inference/lib/vendor` из v3on zip (manylinux/cp312). extract/dense → `gor_gate.load_backbone` / `import_dfine` (transformers 5.21.0). Preflight → STATUS FAILED если нет MODEL файлов или import.
- **ресурсы:** CPU в контейнере, **THREADS=8** (~12–15 мин) или 2 если ПК busy (~20–25 мин); peak RSS ~1.7 GB. GPU не обязателен.
- **порядок:** после bot2 A3 docker lane если GPU пересекается; иначе CPU parallel OK. Не ждать A3 n_raw для **train**.
- **runner:** `pc/runners/run_bot5_gor_heads_docker.ps1` (JobDir `C:\Sloi\pc_jobs\bot5_gor_heads_9f62a96`)
- **команда (in-repo / in-container):**
  ```
  rem host: powershell -File pc\runners\run_bot5_gor_heads_docker.ps1
  rem (runner sets Sha=9f62a96…, unpacks vendor from submission_bot7cand_v3on_0876090.zip, docker run)
  rem equivalent inner:
  set REPO=/repo
  set DATA=/data
  set MODEL=/model
  set WORK=/work
  set DEVICE=cpu
  set THREADS=8
  set OMP_NUM_THREADS=8
  set PYTHONPATH=/repo
  set HF_HUB_OFFLINE=1
  set TRANSFORMERS_OFFLINE=1
  python scripts/gor_heads/train_heads.py
  ```
- **ожидаемое время:** 12–15 мин @ 8 thr (лимит 45).
- **критерий прохождения:** `STATUS.txt` → DONE; оба `.npz` в `%WORK%/heads/`; validate ok; centre_s=250 (или documented); SUMMARY.json sha256+md5. Скопировать в `models/gor_gate/` + `models/gor_dense/`.
- **выходы:** `%WORK%/heads/gor_v3_head.npz`, `gor_dense_head.npz`, `SUMMARY.json`, `ckpt.json`, `train_heads.log`, `STATUS.txt`, `train_heads.docker.log`.
- **после DONE:** bot5 re-queues (b′) mock e2e gor_dense ON.
- **не делать:** не host `.venv` python; не cherry-pick `829ea7d`; не main; не EXPERIMENTS; не lease-файлы; не Kaggle.

## [bot5] 29.10 18:05 МСК — (b′) mock e2e gor_dense ON vs flags-OFF v4 — **both in Docker**
- **СТАТУС: DONE/REJECTED 19:26 МСК** — ON ≡ OFF, ALL F1 **0.312873**, Δ=**0**. gor_dense raw **4** (conf 0.179/0.167/0.228/0.219), **max_raw=0.228 ≪ τ=0.824066** → 0 passed thr → **NO-OP, not accepted**. gor_dense stays OFF in v4. Likely cause: retrained heads (md5≠legacy) shift score dist so legacy τ too high. Results: `pc/results/bot5_gor_dense_e2e_40fa599/`. No τ-recal PC request (low priority; sketch only in report).
- **от:** bot5
- **ветка:** `bots/bot5-candidate-v4`
- **коммит:** `40fa5997a046c8b08fbb65af0a5f043dd0fd75af` (tip; = 9f62a96 + manifest with retrained heads)
- **что:** два mock e2e в `pantum/n7q:20261024` (PROTOCOL §10: host `.venv` результаты для гейта не валидны): **OFF** = `configs/default.toml` без правок (пересчёт flags-off базы v4 в Docker; прежний (a) 0.312873 был host-venv) и **ON** = `[gor_dense] enabled=true`, centre_s=250, τ `[thresholds_by_markup.gor_dense] gorodishcha=0.824066`, головы из `pc_jobs\bot5_gor_heads_9f62a96\heads` с **новыми md5** (`gate_head_md5=487c48e4…`, `head_md5=d92f38f3…`; legacy md5 в DEFAULTS не совпадут). Остальные новые флаги OFF (sp_dfine/dfine_aeor/gor_gate/orgate).
- **runner:** `pc/runners/run_bot5_gor_dense_e2e_docker.ps1` (JobDir `C:\Sloi\pc_jobs\bot5_gor_dense_e2e_40fa599`; worktree @sha, robocopy недостающих models из `C:\Sloi\wt_bot5_50f6f27\models` (pixgbm, dinov2-small), vendor из v3on zip, OFF→ON, toml UTF-8 без BOM, restore; scoring host python как у bot2 v2 base; resumable по result.geojson). Пути — проверь под фактический layout ПК (F:\Sloi_wt как у heads runner).
- **команда:** `powershell -File pc\runners\run_bot5_gor_dense_e2e_docker.ps1`
- **ожидаемое время:** OFF ~2–3 мин + ON ~3–6 мин GPU (gor_dense DINOv2 на SpOr) + score ~1 мин → **≤15 мин** (target), **лимит 30**.
- **критерий (PROTOCOL §5):** ΔALL_F1(ON − OFF, оба Docker) **> 0**; **ни один слой не падает >0.01** vs OFF; записать **inference wall_sec** обоих (`out_*/wall_sec.txt`); main.log без Traceback / md5 mismatch по gor_dense (новые md5 уже в override); OFF Docker vs host 0.312873 — зафиксировать расхождение. **Отчёт Δ vs legacy expectation +0.0154** (пакет на legacy heads; md5 ≠ → +0.0154 не гарантирован). Если Δ **существенно ниже** +0.0154 → follow-up: **τ recalibration** (и при необходимости centre_s) на новых heads; τ в этом прогоне не крутить. EXPERIMENTS.csv — после метрик (bot5).
- **артефакты:** `out_off/`, `out_on/` (result.geojson, main.log, default.toml, wall_sec.txt), `scores.json`, `heads_sha256.txt`, `warn_scan.txt`, `STATUS.txt`, `heartbeat.log`; копию в `pc/results/bot5_gor_dense_e2e_40fa599/`.
- **не делать:** не host `.venv` для inference; не включать sp_dfine/dfine_aeor/gor_gate; не main; не LB; не трогать `models/rerank_arch_lgbm.txt`; lease — только bot1.

## [bot5] 29.10 17:27 МСК — (c) dfine_aeor **Or-only** e2e — **blocked**

- **от:** bot5
- **СТАТУС: SUPERSEDED 18:49 МСК** — A3 @829ea7d DONE (n_raw=2506 in Docker). Replaced by (c′) Or-only Docker e2e @`2b745df` below (loud UNAVAILABLE port; no rebase/force-push).
- **ветка:** `bots/bot5-candidate-v4` (will rebase/cherry after A3 proves import)
- **коммит (prepared, not for launch yet):** `999572d` (+ assert Or n_raw>0) / tip `9f62a96`
- **что:** mock e2e `[dfine_aeor] enabled=true`, `markups=["or"]`, Or weights f0–f2 from `pc/weights/dfine_or/` / manifest; **hard FAIL** if Or dets==0 (`scripts/assert_dfine_aeor_or_nonzero.py`). SpOr D-FINE stays OFF.
- **зависимость:** bot2 A3 @`829ea7d` must show **n_raw>0** in Docker (vendor import OK). Do **not** cherry-pick 829ea7d onto bot5 until then. After green A3: rebase/adapt loud-UNAVAILABLE as needed + Docker runner (model on `run_bot2_a3_dfine_docker_import.ps1`).
- **порядок GPU:** after bot8; after A3 evidence.
- **критерий (when unblocked):** assert pass (n_raw/count>0); ΔALL≥0 vs flags-off (±ε); no layer drop >0.01; wall ≤15 мин.
- **не делать:** не запускать сейчас; не host-venv e2e; не Ae until FT; не main.



## [bot5] 29.10 18:49 МСК — (c′) dfine_aeor **Or-only** mock e2e vs flags-OFF — **both in Docker**
- **СТАТУС: WITHDRAWN 20:45 МСК — bot5** (frees GPU). Reason: bot6 honest Or multiclass eval `reports/bot6_ortho_multiclass.md` — provisional wF1≈0.036, nested non-arch **0 official TP**, arch@0.5 ΔLB ≈0…+0.001 within noise → **gain not proven**; dfine_aeor Or-only **excluded from v5** (flag stays OFF/absent). Do **not** run `run_bot5_dfine_aeor_or_docker.ps1`. Prior BLOCKED 19:49 missing_dfine_or_weights kept as history; tip note `ed73c5e` unchanged. bot5 open slots after withdraw: **0/2** (until v5 e2e filed).
- **от:** bot5
- **ветка:** `bots/bot5-candidate-v4`
- **коммит:** `2b745df167d315de3a377ca112d1283454ab1953` (tip; = 40fa599 + A3 loud-UNAVAILABLE / EXPDS_STRICT / vendor hint port from 829ea7d; **no rebase/force-push**)
- **что:** paired mock e2e in `pantum/n7q:20261024` (PROTOCOL §10): **OFF** = `configs/default.toml` untouched (все новые флаги OFF, SpOr OFF); **ON** = `[dfine_aeor] enabled=true`, `markups=["or"]`, `weights.or = models/dfine_or/dfine_f{0,1,2}`, `thresholds_file = models/dfine_aeor/thresholds_or_oof_provisional.json` (Ae slot off — Ae weights pending). `EXPDS_STRICT=1`. После ON: `scripts/assert_dfine_aeor_or_nonzero.py` (**HARD FAIL** if n_raw==0 or Or detections==0 — not 'no change').
- **зависимость / порядок GPU:** after (b′) gor_dense e2e @40fa599 (queued next after v2 base). A3 evidence: `pc/results/bot2_a3_docker_829ea7d/` n_raw=2506.
- **runner:** `pc/runners/run_bot5_dfine_aeor_or_docker.ps1` (JobDir `F:\Sloi_pc_jobs\bot5_dfine_aeor_or_2b745df`; wt `F:\Sloi_wt\wt_bot5_dfine_aeor_or`; models RO fill + optional `C:\Sloi\pc_weights\dfine_or`; vendor from v3on zip; resumable STATUS.txt + result.geojson). Modeled on A3 + gor_dense Docker runners.
- **команда:** `powershell -File pc\runners\run_bot5_dfine_aeor_or_docker.ps1`
- **ожидаемое время:** OFF ~2–3 мин + ON ~5–10 мин GPU (Or D-FINE) + assert/score ~1 мин → **≤15 мин** (target), **лимит 30**.
- **критерий (PROTOCOL §5):** assert Or n_raw/dets **> 0** (HARD FAIL otherwise); ΔALL_F1(ON − OFF, оба Docker) **> 0**; **ни один слой не падает >0.01** vs OFF; записать **inference wall_sec** обоих (`out_*/wall_sec.txt`); main.log без DFINE_AEOR_UNAVAILABLE / ImportError. EXPERIMENTS.csv — после метрик (bot5).
- **артефакты:** `out_off/`, `out_on/`, `dump_on/`, `assert_or_nonzero.log`, `scores.json`, `warn_scan.txt`, `STATUS.txt`, `heartbeat.log`; копию в `pc/results/bot5_dfine_aeor_or_2b745df/`.
- **заменяет:** 17:27 (c) blocked @999572d — **SUPERSEDED**.
- **не делать:** не host `.venv` inference; не включать sp_dfine/gor_dense/gor_gate/Ae; не main; не LB; не трогать `models/rerank_arch_lgbm.txt`; lease — только bot1; не rebase/force-push.


## [bot5] 29.10 19:45 МСК — fetch PC pytest.log @2b745df (CPU)
- **СТАТУС: CANCELLED** — bot1 fetched the PC pytest.log; open requests now **(c′) only = 1/2**. Runner not needed.
- **от:** bot5
- **ветка:** `bots/bot5-candidate-v4`
- **коммит:** `2b745df167d315de3a377ca112d1283454ab1953`
- **что:** PC pytest @2b745df **exit 1** (box same sha PASS 548+/0). Copy `C:\Sloi\pc_jobs\cpu_pytest_bot5_tip\pytest.log` (+ `python -V`, `pip freeze`) into `pc/results/bot5_pytest_log_2b745df/` so box can triage env-vs-code. **No code change until log read.**
- **runner:** `pc/runners/run_bot5_fetch_pytest_log.ps1`
- **команда:** `powershell -File pc\runners\run_bot5_fetch_pytest_log.ps1`
- **ожидаемое время:** ~1 мин CPU (copy + env dump).
- **критерий:** `STATUS.txt` DONE; `pytest.log` + `python_version.txt` + `pip_freeze.txt` present under JobDir and mirrored to shared results.
- **не делать:** не перезапускать полный pytest; не lease (bot1); не main; не force-push.


## [bot2] 2026-10-29 — дамп D-FINE raw/3857 vs GT mock (диагностика) → СНЯТО: закрыто bot1_v3_spor_diag (не CRS; localization)
- **Кто:** бот 2
- **Ветка:** `bots/bot7-candidate-v3-nomodels` (чтение; zip уже прогнан: `submission_bot7cand_v3on_0876090`)
- **SHA:** `78af57f5ed440caf666eb78e8de647e8edb98723` (tip кандидата; код пути D-FINE совпадает с zip по линии e460b49+)
- **Зачем:** развести CRS-сдвиг vs доменные FP (см. `reports/bot2_v3_regression.md` §4–6). Без этого патч CRS vs порогов — гадание.
- **Ожидаемое время:** 10–20 мин GPU+CPU (повторный короткий прогон mock **не обязателен**, если сохранились выход GeoJSON и лог v3on; иначе один docker-прогон ~3–5 мин как 143 с)
- **Критерий приёмки заявки:** в `C:\Sloi\pc_jobs\bot2_dfine_dump_2026-10-29\` лежат файлы ниже и краткий `SUMMARY.txt` с 5+ примерами pred/GT (координаты 3857, класс, conf, dist_m или IoU). Заявка = диагностика, не приёмка фикса.

### Что вернуть
1. `UTM.json` каждого участка mock (`F:\expds3\mock_val\...`) + из лога строка CRS сцены D-FINE (если есть).
2. `pred_sp_dfine_3857.geojson` — объекты с `threshold_group=sp_dfine` **до** глобального порога, если возможно; иначе пост-порог (13 шт.) + отдельно сырые центры.
3. `gt_mock_3857.geojson` — эталон mock в EPSG:3857.
4. `SUMMARY.txt` — таблица: для ≥5 pred с наивысшим conf: class, conf, centroid_x/y_3857, nearest_gt_same_class dist_m, nearest_gt_any_class dist_m; плюс min/median dist по всем 13; факт «0 TP @ 2 м» подтвердить/опровергнуть.
5. По желанию: 3 сырых бокса в пикселях окна и те же после `boxes_to_ellipses`.

### Команда (PowerShell + docker, GTX 1650; пути как у существующих pc_jobs)

Если выход v3on ещё на диске — **только разбор** (без GPU):

```powershell
$Out = "C:\Sloi\pc_jobs\bot2_dfine_dump_2026-10-29"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
# Подставьте фактический путь выдачи прогона 14:48:
$Pred = "C:\Sloi\pc_jobs\<run_v3on_0876090>\submission.geojson"   # или output.geojson
$GT   = "F:\expds3\mock_val"   # корень mock с разметкой / либо готовый gt.geojson если есть
$Code = @"
import json, math
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import shape, mapping, Point
from shapely import wkt
pred_path = Path(r"$Pred")
out = Path(r"$Out")
fc = json.loads(pred_path.read_text(encoding="utf-8"))
dfine = []
for f in fc["features"]:
    p = f.get("properties") or {}
    g = p.get("threshold_group") or p.get("markup_type")
    if g == "sp_dfine" or (p.get("markup_type")=="sp_or" and "dfine" in str(p).lower()):
        dfine.append(f)
# fallback: все sp_or после v3 (их 13)
if not dfine:
    dfine = [f for f in fc["features"] if (f.get("properties") or {}).get("markup_type")=="sp_or"]
(out/"pred_sp_or_or_dfine.geojson").write_text(json.dumps({"type":"FeatureCollection","features":dfine,"crs":{"type":"name","properties":{"name":"EPSG:3857"}}}, ensure_ascii=False, indent=2), encoding="utf-8")
print("n_pred", len(dfine))
# GT: собрать все *.geojson разметки mock (кроме границ/шум), перевести в 3857 если нужно
root = Path(r"$GT")
gt_feats = []
to3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
# эвристика: если координаты уже |x|>1e6 — считаем 3857
for p in root.rglob("*.geojson"):
    low = p.name.lower()
    if "границ" in low or low.endswith("_шум.geojson") or "utm.json" in low: continue
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    for f in data.get("features") or []:
        geom = shape(f["geometry"]) if f.get("geometry") else None
        if geom is None or geom.is_empty: continue
        xs = list(geom.bounds)
        if max(abs(xs[0]), abs(xs[2])) < 1e6:  # похоже на lon/lat или UTM малый — пробуем через свойства CRS
            # оставляем как есть; метрика официальная сама перепроецирует — для дампа лучше 3857 вручную
            pass
        props = dict(f.get("properties") or {})
        props["source_file"] = str(p.relative_to(root))
        gt_feats.append({"type":"Feature","properties":props,"geometry":mapping(geom)})
(out/"gt_mock_raw.geojson").write_text(json.dumps({"type":"FeatureCollection","features":gt_feats}, ensure_ascii=False, indent=2), encoding="utf-8")
print("n_gt", len(gt_feats))
# nearest distances (предполагаем pred уже 3857)
lines = ["conf,class,x,y,dist_same_m,dist_any_m"]
def cent(g):
    c = shape(g).centroid
    return c.x, c.y
preds = []
for f in dfine:
    x,y = cent(f["geometry"]); preds.append((float((f["properties"] or {}).get("confidence") or 0), (f["properties"] or {}).get("class_name"), x, y, f))
preds.sort(reverse=True)
gts = []
for f in gt_feats:
    try:
        x,y = cent(f["geometry"])
    except Exception:
        continue
    cls = (f["properties"] or {}).get("class_name") or (f["properties"] or {}).get("class")
    gts.append((cls, x, y))
def mindist(x,y, cls=None):
    best = 1e18
    for c,gx,gy in gts:
        if cls is not None and c != cls: continue
        d = math.hypot(x-gx, y-gy)
        if d < best: best = d
    return best
for conf,cls,x,y,_ in preds[:20]:
    ds = mindist(x,y,cls); da = mindist(x,y,None)
    lines.append(f"{conf},{cls},{x:.2f},{y:.2f},{ds:.2f},{da:.2f}")
dists = [mindist(x,y,cls) for conf,cls,x,y,_ in preds]
lines.append(f"n={len(preds)} min_same={min(dists) if dists else None} median_same={sorted(dists)[len(dists)//2] if dists else None}")
(out/"SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")
print("wrote", out)
"@
$Code | Set-Content -Encoding UTF8 "$Out\dump_overlap.py"
# Запуск в том же docker-образе, что e2e (pyproj/shapely есть), монтируя Out и Pred:
# docker run --rm -v C:\Sloi\pc_jobs:C:\Sloi\pc_jobs -v F:\expds3:F:\expds3 pantum/n7q:20261024 python C:/Sloi/pc_jobs/bot2_dfine_dump_2026-10-29/dump_overlap.py
python "$Out\dump_overlap.py"
Copy-Item -Force (Get-ChildItem -Recurse "$GT" -Filter UTM.json | ForEach-Object { $_.FullName }) -Destination "$Out\" -ErrorAction SilentlyContinue
Get-ChildItem "$Out"
```

Дополнительно скопировать кусок лога прогона со строками `sp_dfine:` (crs/utm, n_det, drop_groups −N).

**Не делать:** правки репо, новый полный train, сдачу на LB.

## bot7 — honest-eval по слою
- **статус:** DONE (ae4c816 gate 16:23 МСК; global=0.297118 layer=0.276497 Δ=−0.020621). Tip теперь `aa3694f`.
- **от:** bot7
- **ветка:** `bots/bot7-honest-eval-layer`
- **коммит:** `ae4c816` (PR draft https://github.com/KonkovDV/expds-koz3/pull/2)
- **что:** сверка flag-gated `--match-scope layer` vs default global на стеке v2 eb268d6 (LB 0.41238); unit уже green на box.
- **команды:**
  ```
  git fetch origin && git checkout bots/bot7-honest-eval-layer && git reset --hard ae4c816
  /workspace/koz3-venv/bin/pytest -q tests/test_bot7_layer_scope.py tests/test_bot8_scope_default.py tests/test_bot8_site_scope.py -k "not real_neighbour and not site_fast_collect" --tb=line
  # global (старое поведение; должно совпасть с прежним honest-10)
  python scripts/scoreboard.py --work F:/expds3/work --train F:/expds3/train-expds-3 \
      --metric materials/compute_metrics.py --honest 10 --out notes/bot7_honest_global_ae4c816.json
  # layer (диагностика)
  python scripts/scoreboard.py --work F:/expds3/work --train F:/expds3/train-expds-3 \
      --metric materials/compute_metrics.py --honest 10 --match-scope layer \
      --out notes/bot7_honest_layer_ae4c816.json
  ```
  Если predictions/стек eb268d6 лежат иначе — тот же `--work`/`--train`, что давали honest для v2; не трогать пороги/конфиг сдачи.
- **ожидаемое время:** pytest ~1 мин; scoreboard×2 ~10–30 мин CPU (без GPU).
- **критерий прохождения:** (1) pytest green; (2) global honest-10 байт/число = прежнему прогону (режим unchanged); (3) записать layer ALL_F1 и Δ(layer−global) ≤ 0; сравнить layer vs LB **0.41238** (ожидание: layer не объясняет gap вверх). Строка в EXPERIMENTS.csv.
- **не делать:** не merge PR; не трогать main; не включать layer в сдачу; не force-push чужих веток.


## bot7 — Δ decomposition dump (diff_match_scope)
- **статус:** DONE 2026-10-29 17:00 МСК (PC) / анализ box 17:15; honest g=0.297118 l=0.276497 Δ=−0.020621; TP 4991→4674 cross=647 rematch≈330; артефакты `pc/results/bot7_diff_scope_aa3694f/`.
- **от:** bot7
- **ветка:** `bots/bot7-honest-eval-layer`
- **коммит:** `aa3694f` (на базе f161c93; PR draft https://github.com/KonkovDV/expds-koz3/pull/2)
- **что:** dump cross-layer TP + per-class/layer Δ для разложения honest-10 Δ(layer−global)=−0.020621; `--honest 10` должен воспроизвести gate числа.
- **команды:**
  ```
  git fetch origin && git checkout bots/bot7-honest-eval-layer && git reset --hard aa3694f
  /workspace/koz3-venv/bin/pytest -q tests/test_bot7_diff_match_scope.py tests/test_bot7_layer_scope.py --tb=line
  python scripts/diff_match_scope.py --work F:/expds3/work --train F:/2/train-expds-3 \
      --metric materials/compute_metrics.py --honest 10 \
      --out C:/Sloi/pc_jobs/bot7_diff_scope_aa3694f
  ```
  Тот же `--work`/`--train`, что давали honest-10 global=0.297118 / layer=0.276497. Не трогать пороги/сдачу.
- **ожидаемое время:** pytest ~30 с; dump ~10–30 мин CPU (без GPU; стек как scoreboard).
- **критерий прохождения:** (1) pytest green; (2) summary.json honest_global.all ≈ 0.297118 (±0.0005) и honest_layer.all ≈ 0.276497 (±0.0005); (3) записать n_cross_layer_tp_global, top cross_by_pred_gt_layer, top cross_by_class, delta_by_class.csv → EXPERIMENTS + скопировать summary.json/cross_matches.csv/delta_by_class.csv/tp_by_class_layers.csv в `C:\Sloi\pc_jobs\bot7_diff_scope_aa3694f` (и при возможности в shared notes).
- **не делать:** не merge PR; не трогать main; не менять сдачу; не force-push чужих веток; не сабмитить LB.

## [bot-3] aerial recall flags-off + optional ae_recall_relax — 29.10.2026 ~16:06 МСК
- **от:** bot-3
- **Что:** (A) pytest/flags-off на ветке; (B) mock e2e identity flags-off vs база v4; (C опционально) honest-10 на work-стеке с `[ae_recall_relax] enabled=true` vs false.
- **Ветка:** `bots/bot3-aerial-recall`
- **коммит:** `475bafad60504cd6058366f522fd9cd178a93b6a` (база `cbd48049…` + raster_fix cherry-pick + ae_recall_relax + eval_aerial_honest10)
- **Команды:**
  ```
  # A — без GPU
  pytest -q tests/test_bot3_ae_recall_relax.py tests/test_bot6_raster_fix.py tests/test_dfine_aeor.py -q --tb=line

  # B — mock e2e flags-off (все новые флаги OFF; ожидание: как v4/cbd4804 без включений)
  # (типовой docker/worktree C:\Sloi\wt_* по регламенту «1»; конфиг default.toml без правок)

  # C — honest-10 (если есть work-стек предсказаний)
  python scripts/eval_aerial_honest10.py --work F:/expds3/work --train F:/2/train-expds-3 --out C:/Sloi/pc_jobs/bot3_aerial_honest10.json
  ```
- **Ожидаемое время:** A ~1–3 мин; B ~2–3 мин mock без D-FINE; C ~5–15 мин CPU на готовом стеке.
- **Критерий:** A green; B число объектов/F1 mock совпадает с flags-off базой v4; C при relax ON — honest accept или явный отрицательный результат в EXPERIMENTS.csv (ae Δ задокументирован).
- **Не делать:** не включать dfine_aeor без весов; не трогать main; не сдавать на LB; GPU не обязателен для A/C.


## DONE/SUPERSEDED [bot-4] large_objects unit+flag-smoke на mock — 29.10.2026 17:08 МСК (заменено ablation @5097c4a)
- Результат: OFF vs v2 Δ+0.00036 ok; ON vs OFF Δ−0.01629 FAIL; артефакты `pc/results/bot4_lo_mock_337a27c/`; причина coarse_hillfort +3 FP li/gorodishcha. Новая заявка — ablation per sub-flag.

## WAS [bot-4] large_objects unit+flag-smoke на mock (docker GPU) — 29.10.2026 ~16:15 МСК
- **Что:** e2e mock с `[large_objects] enabled=false` (регрессия vs cbd4804) и отдельно `enabled=true` (smoke). Resumable: STATUS.txt + лог в папке задачи; повтор безопасен.
- **Ветка / sha:** `bots/bot4-large-objects` @ `337a27c` (после push).
- **Команда (пример, полоса «1» правит под существующий docker-раннер):**
  ```
  mkdir -p C:/Sloi/pc_jobs/bot4_lo_mock && cd C:/Sloi/pc_jobs/bot4_lo_mock
  echo RUNNING $(date -Iseconds) > STATUS.txt
  # 1) flag OFF — ожидание байт-в-байт / F1 как cbd4804 на том же mock
  # 2) flag ON  — smoke, смотреть count gorodishcha/fort и F1 по слоям
  # docker run ... --branch bots/bot4-large-objects --config large_objects.enabled=true \
  #   --input F:/expds3/mock_val --out C:/Sloi/pc_jobs/bot4_lo_mock/out_on
  echo DONE $(date -Iseconds) >> STATUS.txt
  ```
- **Ожидаемое время:** ≤15 мин GPU (GTX 1650 4 ГБ) × 2 прогона.
- **Критерий:** OFF — Δ ALL_F1 к cbd4804 на mock ∈ [−0.002; +0.002]; ON — ни один слой не падает >0.01 vs OFF; STATUS.txt=DONE.
- **Не трогать:** main, чужие ветки, LB.

## SUPERSEDED [bot-4] OOF post-tuning large_objects (CPU) — 29.10.2026 ~16:15 МСК
- **СТАТУС: SUPERSEDED** Fix C (`dfine_ae_gor_area_tau`, абсолютный τ_large по OOF) заменяет tau_relax sweep. Слот освобождён.
## WAS [bot-4] OOF post-tuning large_objects (CPU) — 29.10.2026 ~16:15 МСК
- **Что:** подбор `tau_relax` / порогов `gor_from_fort`/`gor_coarse_dem` по готовым OOF-предсказаниям (без GPU-инференса). Resumable: чекпоинт лучших τ в `best_tau.json`, STATUS.txt, идемпотентный перезапуск.
- **Ветка / sha:** `bots/bot4-large-objects` @ `337a27c`.
- **Команда (черновик для «1»):**
  ```
  mkdir -p C:/Sloi/pc_jobs/bot4_lo_oof
  echo RUNNING $(date -Iseconds) > C:/Sloi/pc_jobs/bot4_lo_oof/STATUS.txt
  # python scripts/oof_transform.py ... --flag large_objects --sweep tau_relax=0.05:0.25:0.05
  #   --out C:/Sloi/pc_jobs/bot4_lo_oof  (точный скрипт уточнит «1» по наличию OOF)
  echo DONE $(date -Iseconds) >> C:/Sloi/pc_jobs/bot4_lo_oof/STATUS.txt
  ```
- **Ожидаемое время:** ≤2 ч CPU (≤12 потоков).
- **Критерий:** honest-10 (или proxy OOF) Δ>0 без провала слоя >0.01; артефакт `best_tau.json` + лог.
- **Не трогать:** GPU-очередь; обучение.

## [bot8] compliance on SUBMISSION output (CPU) — 29.10.2026 ~20:35 МСК

- **от:** bot8
- **ветка:** `bots/bot8-tta-lb`
- **коммит:** `f6fe20a41b73e406c3a5ebd62134966e8c0bda6f` (base `eb268d6` + TTA/MS + kit-guard)
- **runner:** `pc/runners/run_bot8_compliance.ps1` — JobDir/Wt on **C:\Sloi\pc_jobs\...**; ASCII-only; schtasks; Test-Path-only pytest (no centerhead)
- **target:** `$env:KOZ3_SUBMISSION` = path to **submission output** (`result.geojson` or LB zip ≤100 MB). Defaults try `bot2_v2_base_docker_eb268d6\out_mock\result.geojson`. **NEVER kit zip** (v3on / models/.pt) — runner + `check_compliance` kit-guard FAIL exit 2 (LESSONS §5).
- **запуск (schtasks):**
  ```powershell
  $env:KOZ3_SHA='f6fe20a41b73e406c3a5ebd62134966e8c0bda6f'
  $env:KOZ3_SUBMISSION='C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6\out_mock\result.geojson'
  powershell -NoProfile -ExecutionPolicy Bypass -File pc\runners\launch_bot8_compliance_schtasks.ps1
  # Task=koz3_bot8_compliance JobDir=C:\Sloi\pc_jobs\bot8_compliance_f6fe20a
  ```
- **ожидаемое время:** ≤3 мин CPU
- **критерий (PROTOCOL §5):** pytest PASS (test_check_compliance + test_tta); check_compliance PASS on submission output; kit path → FAIL loud
- **артефакты:** STATUS.txt, pytest.log, compliance_submission.log, compliance_target.txt, heartbeat.log
- **не делать:** not kit zip; not centerhead/tests; not gate e2e from host

## [bot8] TTA/MS per-layer Docker gate (ae→or→sp_or) — 29.10.2026 ~20:35 МСК

- **от:** bot8
- **ветка:** `bots/bot8-tta-lb`
- **коммит:** `f6fe20a41b73e406c3a5ebd62134966e8c0bda6f`
- **draft PR:** https://github.com/KonkovDV/expds-koz3/pull/9
- **runner:** `pc/runners/run_bot8_tta_ms_docker.ps1` — image **`pantum/n7q:20261024`**; JobDir/Wt **C:\Sloi\pc_jobs\...**; ASCII; weights preflight size>0 + `torch.load` in Docker (`scripts/torch_preflight.py`); variants baseline / tta_prio(ae=4,or=4) / tta_all(+sp_or=4) / ms_prio / tta_ms_prio; optional honest-10; **compliance on out_tta_ms_prio/result.geojson** (submission output, never kit); wall_summary.csv
- **запуск (schtasks):**
  ```powershell
  $env:KOZ3_SHA='f6fe20a41b73e406c3a5ebd62134966e8c0bda6f'
  powershell -NoProfile -ExecutionPolicy Bypass -File pc\runners\launch_bot8_tta_ms_schtasks.ps1
  # Task=koz3_bot8_tta_lb JobDir=C:\Sloi\pc_jobs\bot8_tta_lb_docker_f6fe20a
  ```
- **ожидаемое время (est. vs LB base 3m48s):** mock tta_prio ~5–8 min; tta_all ~10–14; ms_prio ~6–10; tta_ms_prio ~12–18 (≤20 min LB budget); +honest-10 ≤1.5–2 h GPU. Resumable via result.geojson.
- **критерий (PROTOCOL §5):** Δ>0 ALL_F1 vs v2 mock **0.3125** / LB **0.41238**; no layer drop >0.01; compliance PASS on submission geojson; STATUS has `ENV=docker:pantum/n7q:20261024`
- **артефакты:** out_{baseline,tta_prio,tta_all,ms_prio,tta_ms_prio}/, wall_summary.csv, scores.json, compliance_tta_ms_prio.log, notes/torch_preflight.log, STATUS.txt, heartbeat.log
- **не делать:** host venv gate numbers; no LB submit; no main; no force-push; no kit as compliance target
- **честно про ae R=0.013:** на eb268d6 ae = fort-only U-Net τ≈0.998; TTA не создаст курганы — нужен bot3 пороги/D-FINE Ae для ×5 recall


## [bot8] TTA on bot3 f4pack (AFTER first bot8 gate) — 29.10.2026 ~20:50 МСК
- **СТАТУС: QUEUED — GPU Docker; запускать ПОСЛЕ** первого gate bot8 (`bots/bot8-tta-lb` @ `f6fe20a` compliance+TTA/MS)
- **от:** bot8
- **ветка / SHA:** `bots/bot8-tta-on-bot3` / **`82ab8528ef6cb7f252907e0ea5772370f245ce53`** (base `bots/bot3-dfine-aeor-eval` @ `2f46f20`)
- **draft PR:** https://github.com/KonkovDV/expds-koz3/pull/10
- **что:** сравнение bot3 candidate Ae f4pack flags-OFF (new TTA) vs +TTA ae vs +TTA/MS prio; mock + honest-10; wall; compliance на submission `out_tta_ms_prio/result.geojson`
- **image:** `pantum/n7q:20261024`
- **weights:** f4pack `C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4\` (sha prefix `249b86fb`); preflight host+docker
- **JobDir / Wt:** `C:\Sloi\pc_jobs\bot8_tta_on_bot3_82ab852` / `C:\Sloi\pc_jobs\wt_bot8_tta_on_bot3`
- **runner:** `pc/runners/run_bot8_tta_on_bot3_docker.ps1`
- **launch:**
  ```
  $env:KOZ3_SHA='82ab8528ef6cb7f252907e0ea5772370f245ce53'
  powershell -NoProfile -ExecutionPolicy Bypass -File pc\runners\launch_bot8_tta_on_bot3_schtasks.ps1
  # Task=koz3_bot8_tta_on_bot3 JobDir=C:\Sloi\pc_jobs\bot8_tta_on_bot3_82ab852
  ```
- **порядок:** **AFTER** `[bot8] TTA/MS per-layer Docker gate` @ f6fe20a и `[bot8] compliance` @ f6fe20a
- **отчёт:** `reports/bot8_tta_on_bot3.md` (EXPECTED LB = ESTIMATE)
- **не делать:** LB submit; force-push; менять bot3 defaults; EXPERIMENTS без PC цифр

## SUPERSEDED [bot8] TTA/MS host@993c2cc — 29.10.2026 ~16:10 МСК
- **причина:** host venv (PROTOCOL §10); C:/F: path churn; pytest exit 4 on centerhead/tests. Replaced by Docker+schtasks below.

## SUPERSEDED [bot8] compliance/docker runners @81b1d8d — 29.10 ~19:45 МСК
- **причина:** F: JobDir (disk full); non-ASCII in .ps1 (PS5 ANSI ParserError); no weights preflight; no schtasks. Replaced below.

## SUPERSEDED [bot8] compliance pytest @82a167c — validated kit zip (LESSONS §5). Replaced below @f6fe20a.

## OLD [bot8] compliance pytest (CPU filler) — 29.10.2026 ~19:55 МСК

- **от:** bot8
- **ветка:** `bots/bot8-orsp-tta`
- **коммит:** `82a167c7939d926b6da87228678a5adc51cdbdf1` (tip `73b1fee` pins this; has `scripts/torch_preflight.py` + ASCII runners)
- **runner:** `pc/runners/run_bot8_compliance.ps1` — JobDir/Wt on **C:\Sloi\pc_jobs\...**; inputs READ from F: ok; Test-Path-only pytest paths (no centerhead)
- **запуск (schtasks, session-detached):**
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File pc\runners\launch_bot8_compliance_schtasks.ps1
  # Task=koz3_bot8_compliance JobDir=C:\Sloi\pc_jobs\bot8_compliance_82a167c
  ```
- **ожидаемое время:** ≤3 мин CPU (14 tests)
- **критерий (PROTOCOL §5):** unit pytest PASS (14); optional check_compliance on archive zip — not gate e2e
- **артефакты:** STATUS.txt, pytest.log, pytest_exit.txt, heartbeat.log
- **не делать:** not centerhead/tests; not gate e2e numbers from host

## SUPERSEDED [bot8] TTA/MS Docker @82a167c (sp_or-only, branch orsp-tta). Replaced below @f6fe20a per-layer from eb268d6.

## OLD [bot8] TTA/MS mock + honest-10 Docker — 29.10.2026 ~19:55 МСК

- **от:** bot8
- **ветка:** `bots/bot8-orsp-tta`
- **коммит:** `82a167c7939d926b6da87228678a5adc51cdbdf1`
- **runner:** `pc/runners/run_bot8_tta_ms_docker.ps1` — image `pantum/n7q:20261024`; JobDir/Wt on **C:\Sloi\pc_jobs\...**; regions/gt/tools READ from F:; ASCII-only; weights preflight (exists, size>0, `torch.load` **inside Docker** via `scripts/torch_preflight.py`) before variants; baseline/tta/ms/tta_ms + optional honest-10; compliance on out_tta_ms inside Docker; scores.json
- **запуск (schtasks, session-detached):**
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File pc\runners\launch_bot8_tta_ms_schtasks.ps1
  # Task=koz3_bot8_tta_ms JobDir=C:\Sloi\pc_jobs\bot8_tta_ms_docker_82a167c
  ```
- **ожидаемое время:** mock ~3–12 min/variant x4 ≤40 min; +honest-10 ≤1.5–2 h GPU. STATUS+heartbeat; skip if result.geojson exists.
- **критерий (PROTOCOL §5):** Δ>0 ALL_F1 vs v2 mock **0.3125** / LB **0.41238**; no layer drop >0.01; compliance PASS on tta_ms; STATUS has `ENV=docker:pantum/n7q:20261024`
- **артефакты:** out_{baseline,tta,ms,tta_ms}/, scores.json, compliance_tta_ms.log, notes/torch_preflight.log, STATUS.txt, heartbeat.log
- **не делать:** host venv for gate numbers; no sp_dfine; no main; no force-push

## DONE bot2 hotfix A/B — 2026-10-29 16:28 МСК
- A≡B ALL=0.312873 sp_or=0.245064 vs v2 0.312509; PASS
- out: `C:\\Sloi\\pc_jobs\\bot2_v3_hotfix_1ec2d7c`


## DONE [bot5] (a) flags-OFF @50f6f27 — 2026-10-29 16:34 МСК
- ALL=0.312873 li=0.336983 sp_or=0.245064 ae=0.095556 vs v2 0.312509; wall=100.1s; PASS
- out: `C:\\Sloi\\pc_jobs\\bot5_v4_off_50f6f27`

## SUPERSEDED [bot5] (b) gor_dense ON @50f6f27 — replaced 17:20 MSK by (b→train) head retrain @4dadeab
## FAILED [bot5] (b→train) host @4dadeab — 29.10 17:17–17:21 МСК — No module named transformers; superseded by Docker @9f62a96
## DONE [bot5] (b→train) Docker @9f62a96 — 29.10 17:58 МСК — heads in pc/results/bot5_gor_heads_9f62a96/; (b′) e2e filed 18:05
## DONE/REJECTED [bot5] (b′) gor_dense @40fa599 — 29.10 19:26 МСК — ON≡OFF Δ=0; max_raw=0.228≪τ; stays OFF
## CANCELLED (bot1 прочитал лог напрямую) [bot5] fetch pytest.log @2b745df — 29.10 19:45 МСК — CPU ~1 мин; runner run_bot5_fetch_pytest_log.ps1
## WITHDRAWN 20:45 МСК [bot5] (c′) Or-only Docker e2e @2b745df — bot6 honest eval ΔLB≤0 / wF1 0.036 (reports/bot6_ortho_multiclass.md); excluded from v5; do not run
- Heads still missing; train request now in open queue (slot of former b). e2e (b) re-queues after DONE.
- See open entry «(b→train) reproduce gor_v3_head + gor_dense_head».
- GPU lane → next: bot4 LO @337a27c

## DONE [bot-4] large_objects mock OFF→ON @337a27c — 2026-10-29 17:08 МСК
- STATUS=DONE; gate FAILED: ON vs OFF ALL Δ=−0.01629 (|Δ|>0.01); li Δ=−0.01538; ae/sp_or Δ=0
- counts: li/gorodishcha 9→12 (TP1 FP8→11); coarse=3 fort→gor=1
- out: `C:\\Sloi\\pc_jobs\\bot4_lo_mock_337a27c` / `pc/results/bot4_lo_mock_337a27c/`
- superseded by ablation request @5097c4a

## WAS-RUNNING [bot-4] large_objects mock OFF→ON @337a27c — 2026-10-29 16:38 МСК
- 16:38 FAILED_OFF: default.toml записан с BOM (PS5 Set-Content utf8) → TOMLDecodeError; 16:50 runner исправлен (UTF-8 без BOM), toml восстановлен, перезапуск schtasks koz3_bot4_lo
- out: `C:\\Sloi\\pc_jobs\\bot4_lo_mock_337a27c` (STATUS+heartbeat); wt `wt_bot4_lo_337a27c`
- after bot5 (a) DONE / (b) BLOCKED (Kaggle 403)

## [bot-6] PIXGBM cleanroom retrain CPU — 29.10.2026 ~16:44 МСК
- **СТАТУС: DONE 18:37 МСК — FAIL metrics (D-FINE ON worse); analysis in reports/bot2_dfine_a3_localization.md**
- **от:** bot-6
- **ветка:** `bots/bot6-pixgbm-cleanroom`
- **коммит:** `d173a756453d8d9b498dc692782c994a1b68109d` (база `b40ffd4` + Windows `rss_mb` без `import resource` + fail если нет models + PIXGBM_NUM_THREADS)
- **что:** clean-room переобучение PIXGBM (`--enable-retrain`) на полном локальном дереве `F:\2\train-expds-3` (geo 200–217 с растрами+разметкой). Kaggle v6 @b40ffd4: wall=8.3s, 0 моделей — s001–s009 без geo-rasters.
- **предпосылки:**
  - Python ≥3.12; `pip install "lightgbm==4.7.0"` (+ из `requirements.txt`: numpy scipy rasterio shapely … как у обычного PC env KOZ3)
  - файл `expds/geo_pixgbm.py` в worktree (gitignore; эталон sha256 `35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b` — скопировать из `analysis/2026-10-29/bots/bot-4/runs/review/v2eb268d6/expds/geo_pixgbm.py` если нет)
  - данные: `F:\2\train-expds-3` с растрами MagGr/EtGr/TerFl и ObjectPoly/FindsPoints для 200_–217_
- **команда (PowerShell, resumable STATUS.txt):**
  ```powershell
  $ErrorActionPreference = "Stop"
  $env:KOZ3_OUT = "C:\Sloi\pc_jobs\bot6_pixgbm"
  $env:PIXGBM_NUM_THREADS = "12"
  New-Item -ItemType Directory -Force -Path $env:KOZ3_OUT | Out-Null
  if (Test-Path "$env:KOZ3_OUT\STATUS.txt") {
    $st = Get-Content "$env:KOZ3_OUT\STATUS.txt" -Raw
    if ($st -match '^DONE') { Write-Host "already DONE — skip"; exit 0 }
  }
  Set-Content -Path "$env:KOZ3_OUT\STATUS.txt" -Value "RUNNING $(Get-Date -Format o)"
  Set-Location C:\Sloi\wt_bot6_pixgbm   # или существующий wt этой ветки
  git fetch origin bots/bot6-pixgbm-cleanroom
  git checkout bots/bot6-pixgbm-cleanroom
  git reset --hard d173a75
  # geo_pixgbm.py если отсутствует:
  if (-not (Test-Path "expds\geo_pixgbm.py")) {
    Copy-Item "analysis\2026-10-29\bots\bot-4\runs\review\v2eb268d6\expds\geo_pixgbm.py" "expds\geo_pixgbm.py"
  }
  python -m pip install -q "lightgbm==4.7.0"
  python scripts\pixgbm_retrain_cleanroom.py `
    --enable-retrain `
    --root "F:\2\train-expds-3" `
    --out "$env:KOZ3_OUT" `
    --rounds 300 `
    --num-threads 12 `
    --metric-tol 1e-4 `
    --expected-hashes "models\pixgbm\all\MODELS_NOT_IN_GIT.md" `
    2>&1 | Tee-Object -FilePath "$env:KOZ3_OUT\train.log"
  $rc = $LASTEXITCODE
  if ($rc -eq 0 -and (Test-Path "$env:KOZ3_OUT\models\all\poly.txt") -and (Test-Path "$env:KOZ3_OUT\models\all\finds.txt")) {
    Get-FileHash "$env:KOZ3_OUT\models\all\poly.txt","$env:KOZ3_OUT\models\all\finds.txt" -Algorithm SHA256 |
      Out-File "$env:KOZ3_OUT\SHA256SUMS.txt"
    Set-Content -Path "$env:KOZ3_OUT\STATUS.txt" -Value "DONE $(Get-Date -Format o) rc=$rc"
  } else {
    Set-Content -Path "$env:KOZ3_OUT\STATUS.txt" -Value "FAILED $(Get-Date -Format o) rc=$rc"
    exit $rc
  }
  ```
- **альтернатива WSL2 / Docker (если native Win нет rasterio/GDAL):**
  ```bash
  # WSL, те же пути через /mnt/f и /mnt/c
  export PIXGBM_NUM_THREADS=12
  export KOZ3_OUT=/mnt/c/Sloi/pc_jobs/bot6_pixgbm
  mkdir -p "$KOZ3_OUT"
  echo "RUNNING $(date -Iseconds)" > "$KOZ3_OUT/STATUS.txt"
  cd /mnt/c/Sloi/wt_bot6_pixgbm && git fetch origin bots/bot6-pixgbm-cleanroom && git reset --hard d173a75
  python3 -m pip install -q 'lightgbm==4.7.0'
  python3 scripts/pixgbm_retrain_cleanroom.py \
    --enable-retrain \
    --root /mnt/f/2/train-expds-3 \
    --out "$KOZ3_OUT" \
    --rounds 300 --num-threads 12 --metric-tol 1e-4 \
    --expected-hashes models/pixgbm/all/MODELS_NOT_IN_GIT.md \
    2>&1 | tee "$KOZ3_OUT/train.log"
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$KOZ3_OUT/models/all/poly.txt" ]; then
    (cd "$KOZ3_OUT/models/all" && sha256sum poly.txt finds.txt) > "$KOZ3_OUT/SHA256SUMS.txt"
    echo "DONE $(date -Iseconds) rc=$rc" > "$KOZ3_OUT/STATUS.txt"
  else
    echo "FAILED $(date -Iseconds) rc=$rc" > "$KOZ3_OUT/STATUS.txt"; exit $rc
  fi
  ```
  Docker: тот же python+deps образ, `-v F:/2/train-expds-3:/data:ro -v C:/Sloi/pc_jobs/bot6_pixgbm:/out`, `--root /data --out /out`.
- **env:** `PIXGBM_NUM_THREADS=12` (скрипт читает, если `--num-threads` 0; здесь явно `--num-threads 12`); флаг **`--enable-retrain` обязателен** (иначе exit 2).
- **ожидаемое время / RAM (оценка):** extract ~2–10 мин + train LOFO+all ~3–15 мин → **wall 15–60 мин** на 12 потоках; peak RSS **~1–2 ГБ** (ист. extract maxrss≈0.6–1.0 ГБ @ bot-3). Оценка, не замер на этом ПК.
- **resume:** STATUS.txt RUNNING/DONE/FAILED. При DONE — skip. При FAILED/обрыве (ПК выкл) — повтор той же команды: extract/train идемпотентно перезаписывают `feats/*.npz` и `models/**`; чекпоинт = наличие `models/all/{poly,finds}.txt` + `retrain_log.json`.
- **вернуть:** `STATUS.txt`, `retrain_log.json` (JSON verdict), `models/all/poly.txt`, `models/all/finds.txt`, `SHA256SUMS.txt`, `train.log`, `feats/extract_log.json`, `models/train_info.json`.
- **критерий прохождения:** `retrain_log.json` → `status=ok` (sha256 == MODELS_NOT_IN_GIT) **или** `compare.metric_fallback.status=ok` (max|Δp|≤1e-4 vs эталон, если эталонные .txt лежат в `models/pixgbm/all/`); оба файла моделей на месте; STATUS=DONE. Пустые модели / `train_produced_no_models` = FAIL.
- **не делать:** не merge PR #5; не трогать main; не перезапускать Kaggle-ядро; не включать `[geophys.pixgbm]` в default.toml.


## READY-TO-LAUNCH [bot7] diff_scope @aa3694f — 29.10 16:52 МСК
- runner `pc/runners/run_bot7_diff_scope_aa3694f.ps1` (CPU 4 потока, resumable); не запущен — PC Shell spawn EPERM. Запуск: скопировать в C:\Sloi\pc_jobs\bot7_diff_scope_aa3694f\run.ps1 и schtasks.

## RUNNING GPU [bot2] A2 dump+stages @fb66c89 — 2026-10-29 16:51 МСК
- old task `koz3_bot2_hotfix` **deleted**; job `C:\\Sloi\\pc_jobs\\bot2_a2_dfine_stages_fb66c89`
- wt `wt_bot2_fb66c89`; after bot5 a DONE / b BLOCKED; before bot4

## RUNNING CPU [bot-6] PIXGBM @d173a75 — 2026-10-29 16:51 МСК
- wt `C:\\Sloi\\wt_bot6_d173a75` (NOT b40ffd4); out `C:\\Sloi\\pc_jobs\\bot6_pixgbm`; threads=12
- geo_pixgbm sha 35e4331b… OK; lightgbm 4.7.0

## RUNNING/RETRY CPU [bot7] diff_scope @aa3694f — 2026-10-29 16:51 МСК
- pytest initially FAILED — diagnosing; threads<=6; wt `wt_bot7_honest_layer` reset to aa3694f only

## DONE [bot-4] large_objects mock ablation per sub-flag @5097c4a — 29.10.2026 20:19 МСК
- **СТАТУС: DONE 20:19 МСК** — GPU Docker ablation (PROTOCOL §10). Результаты `pc/results/bot4_lo_ablate_5097c4a/`.
- **итог:** area_tau/protect/shape **PASS Δ0**; fort_to_gor/coarse/all_except_coarse **FAIL ΔALL≈−0.006**. Crash note: all_except_coarse exit−1 при валидном geojson (docker/host race). → переход к Fix C (не полный LO).
- **WAS-RUNNING detail below preserved historically:**
## WAS-RUNNING [bot-4] large_objects mock ablation per sub-flag @5097c4a — 29.10.2026 ~17:36 МСК (upd Docker/schtasks/C-wt/preflight)
- **СТАТУС: WAS RUNNING 19:52 → DONE 20:19** — GPU Docker ablation (PROTOCOL §10). Host `.venv` только scoring + torch preflight.
- **от:** bot-4
- **ветка / sha:** `bots/bot4-large-objects` @ **`fc5a0f9`** (`fc5a0f930516453a3a0d5b4eb0580cfaae6f35d4`; wt+JobDir on C:; torch preflight; код флагов `5097c4a`)
- **что:** 7 mock e2e в `pantum/n7q:20261024`: **off** + **on_area_tau** / **on_protect_merge** / **on_shape_cores** / **on_fort_to_gor** / **on_coarse_hillfort** / **on_all_except_coarse**. Master `[large_objects] enabled` + ровно указанные sub-flags; остальные sub = false. Resumable по `out_*/result.geojson` + STATUS.txt.
- **runner:** `pc/runners/run_bot4_lo_ablate_docker.ps1` + `bot4_lo_ablate_summary.py` + `bot4_lo_ablate_torch_preflight.py` (ASCII; без inline python)
- **JobDir / wt:** `C:\Sloi\pc_jobs\bot4_lo_ablate_5097c4a` / `C:\Sloi\pc_jobs\wt_bot4_lo_ablate_5097c4a` (**оба на C:** — F: ~3.5 GB free)
- **preflight:** drop empty weight stubs → robocopy /IS /IT → required assets size>0 → all `.pt/.pth/.safetensors` size>0 → host `torch.load` via `bot4_lo_ablate_torch_preflight.py` (STATUS FAILED on any miss)
- **запуск (schtasks, session-detached; как koz3_bot5_gordense):**
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File pc\runners\launch_bot4_lo_ablate_schtasks.ps1
  ```
- **ожидаемое время:** ~2–4 мин/вариант ×7 + score ≤5 → target ≤40 мин, лимит 60. Heartbeat каждые 20 с на docker run.
- **критерий (PROTOCOL §5):** per-variant ΔALL_F1(ON−OFF, оба Docker) **≥ 0** и **ни один слой не падает >0.01**; SUMMARY + `summary_layers.csv` / `summary_classes.csv`. Ожидание: coarse_hillfort alone ≈ FAIL (Δ≈−0.016); all_except_coarse / одиночные без coarse — кандидаты PASS. EXPERIMENTS.csv после метрик (bot-4).
- **артефакты:** `out_*/` (result.geojson, main.log, default.toml, wall_sec.txt, toml_snip.txt), `e2e_off_vs_*.json`, SUMMARY.txt, warn_scan.txt, STATUS.txt, heartbeat.log; копия в `pc/results/bot4_lo_ablate_5097c4a/`.
- **не делать:** не host `.venv` inference; не main; не LB; флаг в default.toml остаётся OFF; lease — только bot1; **не писать JobDir/wt на F:**.
- **слоты bot-4:** заявка #2; #1 = OOF post-tuning (открыта); e2e @337a27c — DONE/SUPERSEDED.

## FAILED [bot-6] PIXGBM @3f61ea8 — 29.10 17:34 МСК (SUPERSEDED → b0add08)
- retrain.err: `TypeError: pick_rasters() takes 1 positional argument but 4 were given` at scripts/pixgbm.py:148 cmd_extract.
- **Причина:** 80d50b88 перевёл вызовы на 4-arg `pick_rasters(files, rule, mt, crs)` (hp_energy), а cleanroom vendored `expds/geo_pixgbm.py` @sha256 `35e4331b…` — по-прежнему 1-arg как @441bb71. Не регрессия Cyrillic/--match-reference.
- **Фикс:** `b0add08e194853a351d520df19cfb9810c4db983` — три call-site → 1-arg; e2e `tests/test_pixgbm_cmd_extract_e2e.py`. Логи краша: C:\Sloi\pc_jobs\bot6_pixgbm_3f61ea8\retrain.err.

## DONE FAIL [bot-6] PIXGBM @b0add08 — 29.10 18:24 МСК (sha/Δp; → path b e2e-metric)
- **от:** bot-6
- **ветка:** `bots/bot6-pixgbm-cleanroom`
- **коммит:** `b0add08e194853a351d520df19cfb9810c4db983` (3f61ea8 + pick_rasters 1-arg fix)
- **runner:** `pc/runners/run_bot6_pixgbm_b0add08.ps1`
- **JobDir / wt:** `C:\Sloi\pc_jobs\bot6_pixgbm_b0add08` / `C:\Sloi\wt_bot6_b0add08`
- **threads:** 2; **`--match-reference`** ON (seed=0 bagging как эталон); иначе как @3f61ea8
- **geo:** sha256 `35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b`
- **данные:** `F:\2\train-expds-3`
- **команда (ядро, то же что runner):**
  ```powershell
  $env:KOZ3_OUT = "C:\Sloi\pc_jobs\bot6_pixgbm_b0add08"
  $env:PIXGBM_NUM_THREADS = "2"
  python scripts\pixgbm_retrain_cleanroom.py `
    --enable-retrain `
    --match-reference `
    --root "F:\2\train-expds-3" `
    --out $env:KOZ3_OUT `
    --rounds 300 `
    --num-threads 2 `
    --metric-tol 1e-4 `
    --expected-hashes "models\pixgbm\all\MODELS_NOT_IN_GIT.md"
  ```
- **критерий:** STATUS=DONE + (sha256 == MODELS_NOT_IN_GIT **или** metric_fallback≤1e-4); models/all/{poly,finds}.txt
- **не делать:** не merge PR #5; не трогать main; не включать `[geophys.pixgbm]` в default.toml
- **результат:** FAIL sha mismatch; max|Δp|≈0.446 poly / 0.566 finds; features+300 trees OK. EXPERIMENTS 18:24 (bot-1). Диагноз → path **(b)** ниже.
## INVALID (false-equal) [bot-6] PIXGBM e2e-metric REF vs NEW @b0add08 — 29.10 18:40 МСК → см. r2 ниже
- **СТАТУС: DONE 19:43 — NEW≡REF Δ0 (gate exit=9 = баг парсера)** (mock e2e ×2; CPU score)
- **от:** bot-6
- **ветка / SHA:** `bots/bot6-pixgbm-cleanroom` / **`b0add08e194853a351d520df19cfb9810c4db983`** (без нового коммита; path **b**)
- **почему не retrain:** byte/sha match эталона **недостижим** (код признаков =441bb71, матрица n/pos/feature_infos ≠ REF; нет снимка feats 28.10). См. `reports/bot6_pixgbm.md`.
- **что:** два mock e2e в `pantum/n7q:20261024` одного пайплайна S (`[geophys.pixgbm] enabled=true, replace=true`; прочие новые флаги OFF):
  1. **REF** — эталонные веса `poly.txt` sha256 `d1fe3efa…` / `finds.txt` `5fa8c09d…` (источник: `F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v2\models\pixgbm\all` или wt `models/pixgbm/all` после fill).
  2. **NEW** — `C:\Sloi\pc_jobs\bot6_pixgbm_b0add08\models\all\{poly,finds}.txt` (sha `c0c1c7de…` / `c338126b…`).
- **набор:** mock `F:\expds3\mock_val` → regions как у v3e2e/bot5 (`F:\Sloi_pc_jobs_archive\v3e2e\regions`); GT `…\out_v3fix2\gt.geojson`.
- **критерий PASS:**
  - Δ = F1_NEW − F1_REF на mock **ALL ≥ −0.01**;
  - **ни один слой** (li / sp_or / ae / …) **не падает > 0.01** vs REF (PROTOCOL);
  - STATUS=DONE; scores.json; оба result.geojson на месте.
  - (фаза 2, после mock PASS) honest-10: тот же Δ≥−0.01 и no-drop >0.01 — отдельный слот, не блокирует mock.
- **ожидаемое время:** 2× mock ≤15–20 мин GPU wall + score ≤5 мин; лимит слота 45 мин.
- **runner:** `pc/runners/run_bot6_pixgbm_e2e_metric_b0add08.ps1`
- **JobDir:** `C:\Sloi\pc_jobs\bot6_pixgbm_e2e_metric_b0add08`
- **команда (ядро; runner делает fill + toml + docker ×2 + score):**
  ```powershell
  $Sha = 'b0add08e194853a351d520df19cfb9810c4db983'
  $Job = 'C:\Sloi\pc_jobs\bot6_pixgbm_e2e_metric_b0add08'
  $Wt  = 'C:\Sloi\wt_bot6_e2e_metric_b0add08'
  $Image = 'pantum/n7q:20261024'
  $Regions = 'F:\Sloi_pc_jobs_archive\v3e2e\regions'
  $RefModels = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v2\models\pixgbm\all'
  $NewModels = 'C:\Sloi\pc_jobs\bot6_pixgbm_b0add08\models\all'
  $ModelsFill = 'C:\Sloi\wt_bot5_50f6f27\models'   # dinov2 etc.; pixgbm overwritten per variant
  $Gt = 'F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson'
  # wt @ $Sha; robocopy ModelsFill; для REF/NEW: Copy-Item poly/finds в $Wt\models\pixgbm\all
  # toml: [geophys.pixgbm] enabled=true replace=true; остальные новые флаги OFF
  # docker run --gpus all --network none -v ${Wt}:/data/zip:ro -v ${Regions}:/data/regions:ro \
  #   -v ${Job}:/data/output -w /data/zip $Image \
  #   python inference/main.py /data/regions /data/output/out_ref/result.geojson
  # то же → out_new/; score e2e_score.py --a ref=… --b new=… --out $Job\scores.json
  # PASS iff ΔALL>=-0.01 and no layer drop >0.01
  ```
- **не делать:** новый full retrain; не merge PR #5; не включать pixgbm в default.toml навсегда; **не** дублировать EXPERIMENTS retrain-строку (уже есть FAIL 18:24); EXPERIMENTS только для этого e2e-слота после DONE.
- **lease:** bot-1 выдаёт; GPU-полоса.

## CANCELLED 20:17 (orchestrator: mock regions have no maggr/etgr/terfl → PIXGBM structurally no-op; do NOT run r2; bot6 queues r3 REF vs NEW on train geo 200-217 later) [bot-6] PIXGBM e2e-metric r2 REF vs NEW @b0add08 — 29.10 19:55 МСК
- **СТАТУС: CANCELLED 20:17 — do not run `run_bot6_pixgbm_e2e_metric_b0add08_r2.ps1`; reason: mock has no maggr/etgr/terfl rasters so PIXGBM cannot run; r3 (geo 200-217) later; before r3 copy to C: keep ≥8 GB free on C:**
- **от:** bot-6 · **ветка / SHA:** `bots/bot6-pixgbm-cleanroom` / `b0add08e194853a351d520df19cfb9810c4db983` (коммит тот же; меняется только runner)
- **почему r2:** r1 (ALL 0.312509 = 0.312509, Δ 0) — **ложное равенство**. `expds/geo_pixgbm.py` в `.gitignore` и не отслеживается в git → в `git worktree` его нет → `pipeline.py` ловит ImportError («pixgbm включён, но expds/geo_pixgbm.py нет в этой сборке — классика») → оба прогона чистая классика, модели не читались вовсе. Проверь в r1 `out_ref\main.log`/`main.err` — эта строка должна быть.
- **что меняет runner r2:** (1) кладёт `expds/geo_pixgbm.py` из отслеживаемой копии `analysis/2026-10-29/bots/bot-4/runs/review/v2eb268d6/expds/geo_pixgbm.py` с проверкой sha `35e4331b…` (= v2 eb268d6); (2) внутри контейнера пишет `sha256sum` poly/finds/geo_pixgbm в `out_<v>\incontainer_sha.txt`, FAIL exit=10 при несовпадении; (3) FAIL exit=11, если в логе нет строк `pixgbm …: backend=… ступени {'full'…}` или есть «нет в этой сборке»/«pixgbm пропущен»; (4) пишет, побайтно ли совпали result.geojson; (5) гейт читает формат `e2e_score.py` (`rows`, region=ALL/markup) — баг `could_not_parse_delta_ALL` исправлен.
- **runner:** `pc/runners/run_bot6_pixgbm_e2e_metric_b0add08_r2.ps1` · **JobDir:** `C:\Sloi\pc_jobs\bot6_pixgbm_e2e_metric_b0add08_r2` · wt тот же `C:\Sloi\wt_bot6_e2e_metric_b0add08`
- **критерий:** PASS iff ΔF1 ALL ≥ −0.01 и ни один слой не падает > 0.01, STATUS=DONE. Ожидаю ненулевую Δ (max|Δp| 0.566).
- **вернуть:** STATUS.txt, gate.json, scores.json, heartbeat.log, out_*/incontainer_sha.txt, out_*/pix_lines.txt → `pc/results/bot6_pixgbm_e2e_metric_b0add08_r2/`.
- **lease:** bot-1; GPU-полоса.

## READY-TO-LAUNCH [bot-6] PIXGBM geo r3 REF vs NEW @13082a3e — 29.10 ~20:25 МСК
- **СТАТУС: READY-TO-LAUNCH — GPU Docker** (train geo 200–217; NOT mock; r2 CANCELLED)
- **от:** bot-6
- **ветка / SHA:** `bots/bot6-pixgbm-cleanroom` / **`13082a3e1f14be1590593f9fb887641cccb0484c`** (STRICT: ERROR + `EXPDS_PIXGBM_STRICT=1` raises if `expds/geo_pixgbm.py` absent; module NOT committed)
- **что:** paired e2e PIXGBM **S** REF vs NEW on KOZ-3 **train** geo regions **200–217** (maggr/etgr/terfl). Vendor `expds/geo_pixgbm.py` sha `35e4331b…` from `analysis/.../v2eb268d6/`. Gate: validity (sha/log/mtime/docker/identical-geojson) + **PASS iff |ΔF1|≤0.01** on ALL **and** maggr/etgr/terfl.
- **диск (обязательно):** C: ~18 GB free; F: 2.6 GB — **ничего не писать на F:**. Копировать на C: **только** geo-растры maggr/etgr/terfl + boundaries/noise/UTM.json (без LAS/aerial/annotations). Runner считает и логирует размер **до** копирования; **abort exit=15** если plan > **~10 GB**.
- **ожидаемый размер копии:** ~**3.5 GB** (оценка по pack geo200 ≈3.45 GB / 328 files из `reports/bot1_kaggle_unblock.md`; kaggle `*_manifest.json` **не** содержат регионов 200–217). Runner пишет `notes/planned_copy_gb.txt` + `notes/copy_plan.json`. Если plan >>3.5 GB — **«1» проверяет** plan и free space до длинного GPU.
- **paths (C: JobDir/Wt/regions copy; F: read-only train):**
  - JobDir `C:\Sloi\pc_jobs\bot6_pixgbm_geo_r3`
  - Wt `C:\Sloi\wt_bot6_pixgbm_geo_r3` @13082a3e
  - Regions dst `C:\Sloi\pc_jobs\bot6_pixgbm_geo_r3\regions` (selective copy)
  - Train root **RO** `F:\2\train-expds-3`
  - REF models `F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\zip_v2\models\pixgbm\all` (poly `d1fe3efa…` / finds `5fa8c09d…`)
  - NEW models `C:\Sloi\pc_jobs\bot6_pixgbm_b0add08\models\all`
- **overrides:** docker `-e EXPDS_PIXGBM_STRICT=1 -e PIXGBM_BUDGET_S=100000 -e EXPDS_SOFT_CEILING_S=0`; job toml `[time] budget_s=100000 watchdog=false`; `[geophys.pixgbm] enabled=true replace=true budget_s=100000`
- **runner:** `pc/runners/run_bot6_pixgbm_geo_r3.ps1`
- **launch:** `pc/runners/launch_bot6_pixgbm_geo_r3_schtasks.ps1` (task `koz3_bot6_pixgbm_geo_r3`) — ASCII, schtasks, **no python -c**
- **helpers:** `bot6_r3_select_copy.py` (exit 15 if >max-gb), `bot6_r3_build_gt.py` (geo markup GT only), `bot6_r3_gate.py` (exits 9/10/11/12/13/14/19)
- **PASS:** STATUS=DONE; gate.json pass; |ΔF1|≤0.01 ALL+maggr+etgr+terfl; incontainer_sha OK; pix lines with `full` step; result.geojson not byte-identical while models differ (else INVALID 14)
- **вернуть:** STATUS.txt, gate.json, scores.json, heartbeat.log, notes/copy_plan.json, notes/planned_copy_gb.txt, notes/run_meta.json, out_*/incontainer_sha.txt, gt.geojson → `pc/results/bot6_pixgbm_geo_r3/`
- **не делать:** не mock regions; не писать на F:; не копировать целые папки регионов с LAS/aerial; не коммитить `geo_pixgbm.py`; не LB/main
- **lease:** bot-1; GPU-полоса после текущих READY bot3 если пересекается

## RETRY [bot2] A3 @829ea7d — 29.10 18:05 МСК
- 17:32 падение не из-за кода: zip/regions/tools перенесены в F:\Sloi_pc_jobs_archive, раннер смотрел в старый путь. Раннеры перенаправлены, A3 перезапущен.

## READY [bot3] Ae-ft FINAL `[dfine_aeor]` Docker e2e — 29.10 ~20:00 MSK
- **STATUS: READY-TO-LAUNCH — GPU Docker** (mock e2e OFF+ON; honest-10 if regions; CPU score/assert)
- **from:** bot-3
- **branch / SHA:** `bots/bot3-dfine-aeor-eval` / **`2f46f20a05e0e4c4a1998016db33fe23e22e3404`**
- **what:** PROTOCOL section 10 — only `pantum/n7q:20261024`. Mock OFF vs Ae-only `[dfine_aeor]` ON with Ae-ft **final** weights (`sha256` prefix **`3e619878`**) + thresholds_oof.json. Host size/sha + Docker **full safetensors load** preflight (hard FAIL). Hard FAIL if n_raw/Ae dets == 0. `[dfine_aeor]` stays **false** in default.toml (not LB).
- **paths (C: only for JobDir/Wt/weights; F: read-only inputs):**
  - JobDir `C:\Sloi\pc_jobs\bot3_dfine_aeor_ae_ft`
  - Wt `C:\Sloi\pc_jobs\wt_bot3_dfine_aeor_ae_ft`
  - Weights `C:\Sloi\pc_jobs\weights\dfine_ae_ft\` (bot-1 copies; need `config.json`, `model.safetensors`, `thresholds_oof.json`)
- **weights source (box):** `/workspace/kaggle/out/dfine-ae-ft/dfine_ae/` + `models/dfine_aeor/thresholds_oof.json`
- **mock:** `F:\Sloi_pc_jobs_archive\v3e2e\regions` + GT
- **PASS:** STATUS=DONE + `ENV=docker:pantum/n7q:20261024`; docker_preflight PASS; assert_ae_nonzero PASS (**n_raw>0**); scores_mock.json with ae delta ON-OFF; honest-10 if regions (no layer drop >0.01)
- **runtime:** 2x mock ~20-30 min GPU + score; slot 90 min
- **runner:** `pc/runners/run_bot3_dfine_aeor_ae_ft_docker.ps1 -Variant final`
- **launch:** `pc/runners/launch_bot3_dfine_aeor_ae_ft_schtasks.ps1` (task `koz3_bot3_dfine_aeor_ae_ft`)
- **lease:** `leases/pc_bot3_dfine_aeor_ae_ft.lease`
- **do not:** merge main; enable aeor in default; write JobDir to F:; host .venv for gate numbers


## READY [bot3] Ae-ft FOLD4 `[dfine_aeor]` Docker e2e (2nd run) — 29.10 ~20:00 MSK
- **STATUS: READY-TO-LAUNCH — GPU Docker** (after or parallel-queued after FINAL; own JobDir)
- **from:** bot-3
- **branch / SHA:** `bots/bot3-dfine-aeor-eval` / **`2f46f20a05e0e4c4a1998016db33fe23e22e3404`**
- **what:** Same gate as FINAL but weights = FT **fold4** (full 2000 iters, sha prefix **`249b86fb`**). Prefer this over 1-iter final for packaging quality.
- **paths:**
  - JobDir `C:\Sloi\pc_jobs\bot3_dfine_aeor_ae_ft_f4`
  - Wt `C:\Sloi\pc_jobs\wt_bot3_dfine_aeor_ae_ft_f4`
  - Weights `C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4\`
- **weights source (box pack):** `/workspace/kaggle/out/dfine-ae-ft-f4pack/` (HF layout + thresholds; sha256 `249b86fb9713cc1cb234cfbf11d239414092ef1af8178f434df2a38edfa94efa`)
  - bot-1 copy entire pack -> `C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4\` (need `config.json`, `model.safetensors`, `thresholds_oof.json` at that root)
- **PASS:** same as FINAL (n_raw>0, docker full-load preflight, scores_mock); compare ae delta vs FINAL run
- **runtime:** same ~20-30 min mock
- **runner:** `pc/runners/run_bot3_dfine_aeor_ae_ft_docker.ps1 -Variant f4`
- **launch:** `pc/runners/launch_bot3_dfine_aeor_ae_ft_f4_schtasks.ps1` (task `koz3_bot3_dfine_aeor_ae_ft_f4`)
- **lease:** `leases/pc_bot3_dfine_aeor_ae_ft_f4.lease`
- **do not:** confuse with final sha 3e619878; skip docker preflight

## READY [bot2] sat_thr_relax scale=0.85 OFF vs ON @491c63e — 29.10 ~20:40 MSK
- **STATUS: READY-TO-LAUNCH — GPU Docker** (mock e2e; after current READY bot3/bot6 if GPU contended)
- **from:** bot-2 · **Rule P** satellite recall
- **branch / SHA:** `bots/bot2-sat-recall` / **`491c63ec246a30949c21cd2c7675ba1ef75a0a34`** (base **eb268d6**)
- **what:** overlay v2 zip with sat_thr_relax code; run **OFF** then **ON** (`enabled=true`, `scale=0.85` on sp_or/sp_focal/sp_focal2/sp_arch). Score off↔on and vs v2ref. Flag stays OFF in default.toml.
- **hypothesis:** lower thr admits U-Net peaks; hit-rate >8.9% break-even → sat type F1↑; expected **ΔLB +0.003…+0.010** (full R→0.4 band +0.008…+0.025).
- **paths (C: JobDir/Wt; F: RO kit):**
  - JobDir `C:\Sloi\pc_jobs\bot2_sat_thr_relax_491c63e`
  - Wt `C:\Sloi\wt_bot2_sat_thr_relax_491c63e` @491c63e
  - Zip `F:\Sloi_pc_jobs_archive\v3e2e\kit\submission_bot7cand_v2_eb268d6.zip`
  - Regions `F:\Sloi_pc_jobs_archive\v3e2e\regions`
- **image:** `pantum/n7q:20261024` only
- **runner:** `pc/runners/run_bot2_sat_thr_relax_491c63e.ps1`
- **validity:** wipe stale geojson; DONE iff docker exit==0 **and** result.geojson mtime ≥ container start and size>50 (LESSONS §1-2). Wall must be minutes not seconds.
- **PASS:** STATUS=DONE; OFF ≈ v2 ALL 0.3125 / sp_or 0.2429 (±0.01); ON: no layer drop >0.01 **or** ΔALL>0 with layer drop ≤0.01; report sp_or Δ and n_pred OFF/ON.
- **runtime:** 2× mock ~20–40 min GPU
- **do not:** requeue A4; enable sat_thr_relax in default submission; main/PR/LB submit; write JobDir to F:
- **lease:** bot-1; GPU-полоса
- **report:** `reports/bot2_sat_recall.md`

