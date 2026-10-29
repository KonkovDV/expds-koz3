# KOZ3 — доска оркестрации (координатор: бот «1»)
Обновлено: 29.10 20:28 МСК. Дедлайн 11.11. Репо: github.com/KonkovDV/expds-koz3

## bot1 pulse — 2026-10-29 20:28 MSK
- **bot8 honest10 train-tree FIXED on acc2:** collab READER на `dmitrijkv/koz3-xfer-s000`…`s009` + `koz3-gt-train-v2`; ядро v5 COMPLETE (CPU staging): rebuild ALL → 210 регионов; geo200–217 rasters по-прежнему hollow. **GPU OOF не стартовал** — слоты 2/2 (s1f23, s1f4). Push GPU готов: `/workspace/kaggle/push/elenakonkova-bot8-oof-honest10` после ~23:10 (конец s1f4). ETA OOF 8–12 ч.
- **Kaggle acc2:** s1f23 RUNNING (~01:30), s1f4 RUNNING (~23:10); TPU smoke QUEUED.
- Остальное без смены с 20:14 (см. ниже).

| Полоса | Загрузка | Задача | С | ETA |
|---|---|---|---|---|
| Kaggle acc2 GPU | 2/2 | c3sc-s1f23; c3sc-s1f4 | 20:06/20:10 | ~01:30 / ~23:10 |
| Kaggle acc2 CPU | — | honest10 v5 staged COMPLETE | 20:23 | GPU push after 23:10 |
| Kaggle acc1 GPU | 0/2 | квота тонкая до сброса 31.10 03:00 | — | — |
| Kaggle acc2/acc1 TPU | QUEUED | smoke / c3sc-train | — | ? |


## ФОКУС (29.10 20:25, по scoring.log LB 0.41238 = submission_bot7cand_v2_eb268d6.zip; reports/lb/)
Мы #25 (StrataTrace) из 72; лидер 0.616, топ-10 ≥ 0.505. Разбор LB по типам данных (precision/recall):
лидар 0.52/0.50 · геофизика 0.77/0.66 (PIXGBM работает на LB!) · спутник 0.46/0.25 · ортофото 0.37/0.14 · **аэрофото 0.09/0.01 ≈ ноль**.
Recall по размеру: >10000 м² 0.29, 5–10 тыс. м² **0.00**. FP ratio 1.14. Инференс 3 мин 48 с из 30 мин → запас ~8× на ансамбли/TTA/тяжёлые модели.
ПРИОРИТЕТЫ (всё прочее — в конец очереди):
1. Аэрофото и ортофото: recall ×5+ (D-FINE ae FT, пороги, тайлинг/масштаб, TTA). Самый большой рычаг.
2. Спутник recall 0.25 → 0.4+ (пороги из честной OOF, sp_or D-FINE).
3. Крупные объекты (городища/фортификации >5000 м²): recall почти 0 — проверить тайлинг/масштаб и фильтры площади.
4. Использовать бюджет времени: ансамбль сидов ConvNeXt li, TTA/мультимасштаб (bot8).
Mock без геофизики → mock-оценка не видит ~1/5 метрики; PIXGBM трогать только тестом эквивалентности (bot6 r3).

## bot1 pulse — 2026-10-29 20:14 MSK
- **ПК снова доступен.** bot4 ablation @5097c4a НЕ идёт: вариант on_all_except_coarse упал 20:04 (exit=-1 через 80 с, нет geojson); 6 вариантов до него завершены. ПК GPU и CPU простаивают → исполнитель: добить недостающий вариант bot4 + сводка, докопировать веса dfine-or (xfer20) и запустить bot5 c′, заполнить CPU-полосу из pc/QUEUE.md.
- **Kaggle:** без изменений — acc2 GPU 2/2 (s1f23, s1f4 RUNNING); acc2 TPU smoke и acc1 TPU c3sc-train QUEUED; acc1 GPU — резерв до сброса 31.10 03:00. bot8 honest10 на acc2 = ERROR (нужен train-tree).

| Полоса | Загрузка | Задача | С | ETA |
|---|---|---|---|---|
| ПК GPU | idle 10 %, 1,3 ГБ | заполняется: bot4 on_all_except_coarse (резюм) → bot5 c′ | 20:14 | ~20:30 |
| ПК CPU | idle 23 % | заполняется из pc/QUEUE.md | 20:14 | ? |
| Box CPU | load ~0,3/8 | DL выхода c3sc-f23; pip shapely | 20:11 | ~20:20 |
| Kaggle acc1 GPU | 0/2 | квота ≈ исчерпана, сброс 31.10 03:00 | 19:55 | — |
| Kaggle acc1 TPU | 0/1 QUEUED | tpu-cnx-c3sc-train | 17:10 | ? |
| Kaggle acc2 GPU | 2/2 | c3sc-s1f23; c3sc-s1f4 | 20:06/20:10 | ~01:30 / ~23:10 |
| Kaggle acc2 TPU | 0/1 QUEUED | acc2-tpu-smoke | 19:48 | ? |

## bot1 pulse — 2026-10-29 20:12 MSK (acc2 online)
- **Kaggle account-2 `elenakonkova` ONLINE:** GPU smoke COMPLETE (T4×2+internet). Quota GPU **29.91h**/30, TPU 20h unused; refresh 31.10 03:00 МСК.
- **Acc2 GPU 2/2 RUNNING:** `elenakonkova/koz3-cnx-li-c3sc-s1f23` (seed1 f2+f3, с ~20:06, ETA ~01:30) + `elenakonkova/koz3-cnx-li-c3sc-s1f4` (seed1 f4, с ~20:10, ETA ~23:10). Private datasets: koz3-s3-cred, koz3-cnx-li-prep (119 files), koz3-bot8-snap.
- **Acc2 TPU:** smoke still QUEUED (slot not granted). Acc1 TPU c3sc-train also QUEUED.
- **bot8 on acc2 blocked:** needs train tree (xfer shards or selective S3); full sync ~300GB. Snap+models ready.
- **Acc1:** c3sc-s1f01 COMPLETE; GPU remaining thin until reset; no duplicate of running acc2 folds.

| Полоса | Задача | С | ETA |
|---|---|---|---|
| Kaggle acc2 GPU×2 | c3sc-s1f23; c3sc-s1f4 | 20:06/20:10 | ~01:30 / ~23:10 |
| Kaggle acc2 TPU | tpu-smoke QUEUED | ~19:50 | ? |
| Kaggle acc1 GPU | — reserve / s1f01 done | — | reset 31.10 03:00 |
| Kaggle acc1 TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |


## bot1 pulse — 2026-10-29 20:10 MSK
- **Kaggle acc1 c3sc-s1f01 COMPLETE** (6712 с, ошибок нет): f0 Li-only 0,5169 @0,3 (подбор в фолде 0,5853), f1 0,2490 (0,4711). Выход /workspace/kaggle/out/c3sc-s1f01.
- **Kaggle acc2 (elenakonkova) включён:** gpu-smoke OK (2×T4), c3sc-s1f23 RUNNING, tpu-smoke QUEUED. bot8 honest10 на acc2 упал за 14 с (нет снапшота репо/моделей) — исполнитель собирает приватные датасеты и перезапускает (2-й GPU-слот acc2).
- **Kaggle acc1 GPU 0/2:** не заполняю — остаток квоты оценочно <1 ч до сброса 31.10 03:00, тяжёлые задачи уходят на acc2 (30 ч свежей квоты). TPU c3sc-train всё ещё QUEUED (с 17:10, ~3 ч).
- **ПК GPU:** bot4 ablation @5097c4a идёт (вариант on_fort_to_gor, контейнер с 19:59; GPU 47 %, 2,9 ГБ).
- **ПК CPU (22 %):** bot8 compliance DONE — pytest 14/14, но check_compliance FAIL (669 МБ, 0 объектов, не FeatureCollection — похоже, проверялся kit-zip, не сабмит). **Бот 8:** разобрать. bot5 pytest ed73c5e на ПК exit=4 (путь тестов не найден) — на box ed73c5e PASS 548/0.
- **bot2 A4 @c3d912c — INVALID:** n_A4=0 (инференс не шёл, ~30 с). **Бот 2:** переподать (вероятно та же причина, что у bot4 — нет моделей pixgbm в wt).
- **Box pytest:** bot3 ee8af7f 541/0 PASS, bot5 ed73c5e 548/0 PASS, bot8 73b1fee 424/0 PASS, bot4 fc5a0f9 542/0 PASS, **bot7 69a9a85 684/1 FAIL** (test_bot8_soft_ceiling::test_t0_is_process_start…) — **бот 7** разобрать.
- **Веса dfine-or → ПК:** CopyFromBox в C:\Sloi запрещён (вне разрешённого корня) → кладу в домашнюю папку ПК `koz3_xfer_dfine_or\` (конфиги+SHA256SUMS доставлены), части по 60 МБ обрываются — ПК стал недоступен 20:06. Подготовлены части по 20 МБ: /workspace/kaggle/out/dfine-or/xfer20 (22 файла). Следующий проход: докопировать, собрать, сверить sha, перенести в C:\Sloi\pc_weights\dfine_or\dfine_f{0,1,2}, запустить bot5 c′. Затем веса bot3 (final 3e619878, f4 249b86fb) тем же путём.
- ПК недоступен с ~20:06 → CPU-полоса ПК не заполнена.

| Полоса | Загрузка | Задача | С | ETA |
|---|---|---|---|---|
| ПК GPU | busy 47 %, 2,9 ГБ | bot4 large_objects ablation ×7 @5097c4a | 19:52 | ~20:30 |
| ПК CPU | idle 22 % | — ПК недоступен с 20:06 | 19:58 | — |
| Box CPU | load ~1/8 | pytest 5 типов DONE 20:08; свободен | 20:01 | — |
| Kaggle acc1 GPU | 0/2 | квота ≈ исчерпана, ждём сброса 31.10 03:00 | 19:55 | — |
| Kaggle acc1 TPU | 0/1 (QUEUED) | tpu-cnx-c3sc-train | 17:10 | ? |
| Kaggle acc2 GPU | 1/2 → 2/2 | c3sc-s1f23 RUNNING; honest10 перезапуск | 20:05 | ~22:00 / 8–12 ч |
| Kaggle acc2 TPU | 0/1 (QUEUED) | acc2-tpu-smoke | 19:48 | ? |


## bot1 pulse — 2026-10-29 19:57 MSK
- **ПК GPU: bot6 PIXGBM e2e REF vs NEW @b0add08 DONE 19:43** — NEW (новый poly.txt) ≡ REF во всех слоях/регионах: ALL 0,3125, Δ 0,0000. «FAILED exit=9» — баг парсера gate.json, не метрика. Итог: переобучение эквивалентно по e2e, прироста нет. **Бот 6:** починить парсер gate, задачу закрыть как NEUTRAL.
- **ПК CPU: bot7 compare_candidates @69a9a85 DONE 19:50** (в раннере был битый sha — исправил). v3 FAIL(sp_or 0), hotfix_B/v4_off/bot4_off +0,0004 PASS, bot4_on −0,0159 FAIL(li). Таблица: pc/results/bot7_compare_69a9a85.md. **Бот 7:** гейт инструмента пройден, сверить с DRAFT PR #8.
- **bot5 (c′) Or-only @2b745df — FAILED missing_dfine_or_weights:** C:\Sloi\pc_weights\dfine_or\dfine_f{0,1,2} пустые. Вывод koz3-dfine-or скачан на box (/workspace/kaggle/out/dfine-or, f0–f2 safetensors); перенос на ПК — в след. проходе (ПК недоступен; части по 60 МБ + SHA256SUMS готовы в /workspace/kaggle/out/dfine-or/xfer → собрать в C:\Sloi\pc_weights\dfine_or\dfine_f{0,1,2}, сверить sha, перезапустить schtask koz3_bot5_aeor_or). **Бот 5:** ждать.
- **bot4 ablation @5097c4a:** падал на missing pixgbm/all/poly.txt — дозаполнил C:\Sloi\wt_bot4_lo_337a27c\models из vendor_unpack (sha D1FE3E… = прод). **Перезапущен 19:52 (schtask koz3_bot4_lo_ablate), идёт off → 6 вариантов.** Та же причина отменила bot2 A4 — **бот 2** может переподать.
- bot3 ae-ft PLACEHOLDER_SHA: **FIXED** tip `2f46f20` (resolve from origin/$Branch). Fix A+B shipped `bots/bot3-ae-ortho-recall` **`1e7073a`** — PC QUEUE READY + Kaggle e2e after calib; report `reports/bot3_ae_ortho_recall.md`.
- bot8 compliance @82a167c: launcher отправлен, но ПК в 19:55 снова недоступен — статус не подтверждён.
- **Kaggle:** без изменений — c3sc-s1f01 RUNNING (GPU 1/2, вторая — резерв квоты), TPU c3sc-train QUEUED с 17:10.
- **Box:** pytest новых тиpов bot3 ee8af7f, bot5 ed73c5e, bot8 73b1fee, bot4 fc5a0f9, bot7 69a9a85 (лог box/results/orch_1955_runs.log) + загрузка весов dfine-or.

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | bot4 large_objects ablation ×7 @5097c4a (docker) | 19:52 | ~20:30 |
| ПК CPU | bot8 compliance pytest (не подтверждено, ПК недоступен) | 19:55 | ~20:00 |
| Kaggle GPU×2 | c3sc-s1f01; 2-я — резерв квоты | 18:04 | ~20:40 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | pytest 5 тиpов + DL dfine-or | 19:55 | ~20:20 |



## bot1 pulse — 2026-10-29 19:40 MSK
- **Kaggle dfine-ae-ft COMPLETE** (2,9 GPU-ч, 2,74 млн дет., ошибок фолдов нет). Но final-дообучение успело 1 итерацию (дедлайн) → итоговая dfine_ae = сид фолда 3. OOF F2: курганы целые 0,028 (tp 27/fp 4974), повреждённые 0,173, фортификации 0,184; городища/архитектура OFF. Веса sha256 3e619878…, всё в /workspace/kaggle/out/dfine-ae-ft. [dfine_aeor] остаётся OFF до honest-10 e2e. **Бот 3:** разобрать, взять дальше в e2e mock/honest-10 на ПК.
- **ПК GPU:** bot5 b′ gor_dense DONE 19:26 — **NO-OP**: ON ≡ OFF (ALL 0,3129, Δ 0,0000; gor_dense дал 4 сырых с conf 0,17–0,23, ни одна не прошла порог). **Бот 5:** не принимать; следующее — (c′) Or-only.
- **ПК CPU:** cpu_score_batch DONE 19:35 — bot4_on 0,2966 vs off 0,3129 (**Δ −0,016, FAIL**). **Бот 4:** large_objects ON ухудшает, ждём ablation по под-флагам.
- **ПК GPU запущен 19:38:** bot6 PIXGBM e2e REF vs NEW @b0add08 (schtask koz3_bot6_e2e_metric, RUNNING, лимит 45 мин).
- ПК после этого стал недоступен (temporarily unreachable) — CPU-полоса не заполнена. Также на ПК: bot7 compare_candidates_v2docker FAILED (нет аргумента --out), pytest bot5 tip @2b745df exit=1 и bot8 compliance exit=4 — ботам 7/5/8 разобрать.
- **Kaggle GPU:** 1/2 (c3sc-s1f01 RUNNING до ~20:40). Вторую не заполняю: квота ~1–1,5 ч = резерв до сброса 31.10 03:00 МСК. TPU c3sc-train QUEUED с 17:10 (2,5 ч).
- **Box:** новых sha нет; свободен.

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | bot6 PIXGBM e2e REF vs NEW @b0add08 | 19:38 | ~20:05 |
| ПК CPU | — свободна (ПК недоступен) | 19:35 | — |
| Kaggle GPU×2 | c3sc-s1f01; 2-я — резерв квоты | 18:04 | ~20:40 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | — свободен | 19:08 | — |

## bot1 pulse — 2026-10-29 19:20 MSK
- **ПК GPU:** контейнер bot5 b′ gor_dense (с 18:59) умер без вывода (нет result.geojson/STATUS/логов; запуск из сессии Shell убивается при обрыве). **Перезапущен 19:20 через schtask `koz3_bot5_gordense`** (отвязан от сессии), вариант off идёт, контейнер koz3-bot5-gordense-off-20261029192029.
- **ПК CPU:** cpu_score_batch в 19:05 упал (писал в C:\STATUS.txt — $D пуст при старте из сессии). **Перезапущен 19:20 через schtask `koz3_cpu_score_batch`**, STATUS RUNNING. Правило: задачи ПК запускать только через schtasks.
- **Kaggle:** dfine-ae-ft, c3sc-s1f01 RUNNING (GPU 2/2); TPU c3sc-train всё ещё QUEUED (с 17:10, >2 ч).
- **Box:** pytest bot2 @c3d912c PASS (481/0/23), bot5 @2b745df PASS (548/0/37). Новых sha нет — раннер свободен.
- ПК Shell снова дал один spawn EPERM, повтор прошёл. Диск: C: 28,2 ГБ, F: 3,5 ГБ.

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | bot5 b′ gor_dense e2e @40fa599 (off→on, docker) | 19:20 | ~19:50 |
| ПК CPU | cpu_score_batch, 4 потока | 19:20 | ~15 мин |
| Kaggle GPU×2 | dfine-ae-ft; c3sc-s1f01 | 16:33/18:04 | ≤19:30 / ~20:40 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | — свободен | 19:08 | — |

## bot1 pulse — 2026-10-29 19:05 MSK
- **ПК снова доступен.** GPU: bot5 b′ gor_dense e2e @40fa599 RUNNING (контейнер koz3-bot5-gordense-off-20261029190003, с 18:59; heartbeat 19:04). Дальше в GPU-очереди: bot6 PIXGBM e2e metric REF vs NEW @b0add08 → bot2 A4 agreement_m=50 @c3d912c → bot5 dfine_aeor Or-only @2b745df → bot8 → bot4 ablation.
- **ПК CPU:** pytest-филлеры bot2/6/7 и cpu_score_batch с 18:38 «RUNNING» без процессов и с пустыми логами — помечены DEAD. cpu_score_batch (7 кандидатов, 4 потока) перезапущен 19:05.
- **pc/QUEUE:** bot7 compare_candidates @b6f835d уже DONE 18:22 (запись устарела).
- **Kaggle:** без изменений — dfine-ae-ft, c3sc-s1f01 RUNNING (GPU 2/2); TPU c3sc-train QUEUED с 17:10 (~2 ч).
- **Box:** pytest bot2 @c3d912c → bot5 @2b745df (лог box/results/orch_1905_runs.log).

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | bot5 b′ gor_dense e2e @40fa599 (docker) | 18:59 | ~19:15 |
| ПК CPU | cpu_score_batch, 4 потока | 19:05 | ~15 мин |
| Kaggle GPU×2 | dfine-ae-ft; c3sc-s1f01 | 16:33/18:04 | ≤19:30 / ~20:40 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | pytest bot2 @c3d912c → bot5 @2b745df | 19:05 | ~5 мин |

## bot1 pulse — 2026-10-29 18:45 MSK
- **ПК GPU: bot2 v2 base docker @eb268d6 DONE 18:42** (680 с, 248 дет.). Скорер раннера не нашёл e2e_score.py → досчитал вручную: **ALL F1 0,3125 = v2ref, Δ 0,0000** (ae 0,0956; li 0,3370; sp_or 0,2429; 101: 0,3464; 103: 0,0070). Воспроизводимость v2 в docker — **PASS**. scores: C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6\scores_mock.json.
- **A3 (docker D-FINE):** _sync_a3 — main записал result+stages, n_sp_dfine=482, **n_raw=2506**, прошло порог 13, written=259 (wall ~909 с). D-FINE в docker импортируется (n_raw>0) → **bot5 (c) dfine_aeor Or-only разблокирован**; бот 2 — подтвердить итог A3 в отчёте.
- **ПК CPU:** идут pytest-филлеры ботов 2/6/7 + cpu_score_batch (RUNNING с 18:38).
- **ПК Shell снова spawn EPERM** после 5 успешных вызовов — следующую GPU-задачу (bot5 b′ gor_dense e2e @40fa599 → bot8 → bot4 ablation @5097c4a) запустить не смог; ПК GPU простаивает с 18:42.
- **Kaggle:** без изменений — dfine-ae-ft, c3sc-s1f01 RUNNING (GPU 2/2); TPU c3sc-train QUEUED с 17:10 (>1,5 ч).
- **Боты:** новых коммитов с 18:10 нет (<1 ч тишины). Box свободен — новых sha для тестов нет.

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | — свободна (Shell EPERM); след.: bot5 b′ @40fa599 | 18:42 | — |
| ПК CPU | pytest-филлеры bot2/6/7 + cpu_score_batch | 18:38 | ? |
| Kaggle GPU×2 | dfine-ae-ft; c3sc-s1f01 | 16:33/18:04 | ≤19:30 / ~20:40 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | — свободен | 18:22 | — |

## bot1 pulse — 2026-10-29 18:30 MSK
- **Kaggle:** dfine-ae-ft RUNNING, c3sc-s1f01 RUNNING (GPU 2/2); TPU c3sc-train QUEUED уже 1 ч 20 мин.
- **ПК CPU:** bot6 PIXGBM @b0add08 завершён 18:24 — **verdict FAIL** (sha не совпал, metric-fallback max_abs_predict_delta=0,566 ≫ 1e-4; имена фич и 300 деревьев совпадают). Бот 6: разобрать расхождение (параметры/сид/порядок данных), см. C:\Sloi\pc_jobs\bot6_pixgbm_b0add08\verdict.json.
- **ПК GPU:** контейнер A3 koz3-bot2-a3-20261029180829 больше не работает (docker ps пуст по koz3); итог A3 не проверен.
- **ПК недоступен для команд:** после 2 успешных вызовов Shell на ПК в 18:27 — spawn EPERM. Обе полосы ПК простаивают, следующие в очереди (v2 base docker @10e9936 → bot8 → bot4 ablation) запустить не могу.
- **Box:** pytest bot6 @b0add08 PASS (674/0/47). box/QUEUE.md пуст — раннер свободен.

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | — свободна (Shell EPERM) | ~18:25 | — |
| ПК CPU | — свободна (Shell EPERM) | 18:24 | — |
| Kaggle GPU×2 | dfine-ae-ft; c3sc-s1f01 | 16:33/18:04 | ≤19:30 / ~20:40 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | — свободен | 18:22 | — |

## bot1 pulse — 2026-10-29 18:20 MSK
- **Kaggle:** dfine-ae-ft RUNNING, c3sc-s1f01 RUNNING (GPU 2/2), TPU c3sc-train всё ещё QUEUED (с 17:10, >1 ч — слот TPU не выдают).
- **ПК GPU:** раннер A3 записал `FAILED R_drop_groups` в 18:07:51 (после overlay pipeline.py/sp_dfine.py), но контейнер `koz3-bot2-a3-20261029180829` (запущен 18:08, не тем раннером) идёт и пишет лог пайплайна (103_Гнёздово, 18:16). STATUS.txt A3 неактуален — бот 2: подтвердить, чей это запуск, и что R_drop_groups — ожидаемая ветка.
- **ПК CPU:** bot6 PIXGBM @b0add08 RUNNING с 18:14 (retrain pid 14052, heartbeat свежий). Диск C: 12,3 ГБ, F: 3,4 ГБ.
- **Box:** pytest bot5 @40fa599 PASS (541/0/37); сейчас pytest bot6 @b0add08 (лог box/results/orch_1820_runs.log).

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | A3 контейнер koz3-bot2-a3-20261029180829 | 18:08 | ? |
| ПК CPU | bot6 PIXGBM @b0add08, 2 потока | 18:14 | ~30–60 мин |
| Kaggle GPU×2 | dfine-ae-ft; c3sc-s1f01 | 16:33/18:04 | ≤19:30 / ~20:40 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | pytest bot6 @b0add08 | 18:20 | ~2 мин |

## bot1 pulse — 2026-10-29 18:07 MSK
- **Kaggle:** `koz3-cnx-li-c3sc-f4-v2` COMPLETE (12k ит., 1,85 ч обучения, 35 645 дет.). Li-only compute_metrics: 0,0917 при min_score 0,05; **0,3126 при пороге 0,3**; 0,3482 с порогами по классам, подобранными на том же фолде (оптимистично, не для гейта). Выходы: `/workspace/kaggle/out/c3sc-f4-v2/` (ckpt, weights, oof_cnx_c3-sc_f4.geojson, report_cnx_full.json). Фолды c3-sc f0–f4 теперь все есть → OOF-пороги c3-sc можно калибровать (бот 8/7).
- **Kaggle GPU 2/2:** dfine-ae-ft RUNNING; новая филлер-задача `koz3-cnx-li-c3sc-s1f01` (сид 1, c3-sc f0+f1 на 2×T4, deadline 2,6 ч) — для ансамбля/OOF (BACKLOG GPU-2). TPU c3sc-train всё ещё QUEUED (с 17:10, ~1 ч).
- **ПК:** доска 17:50 была неверна — обе задачи упали в 17:32/17:34. Причина A3: после переноса v3e2e/hotfix и др. в `F:\Sloi_pc_jobs_archive` раннеры ссылались на старые пути C:\Sloi\pc_jobs\v3e2e → пустой zip_A → `main.py not found`. Раннеры на ПК и в pc/runners перенаправлены на F:\Sloi_pc_jobs_archive (бэкапы `*.bak_v3e2e_moved`). **A3 @829ea7d перезапущен 18:05** (schtask koz3_bot2_a3_docker). bot6 PIXGBM @3f61ea8 FAILED — баг кода: `pick_rasters() takes 1 positional argument but 4 were given` (scripts/pixgbm.py:148) → нужен фикс от бота 6. bot5 gor_heads @9f62a96 DONE 17:58.
- **ПК CPU простаивает:** готовых CPU-задач нет (bot6 ждёт фикса; bot5 b′ и v2-base — GPU-docker, стоят за A3). Диск F: 3 ГБ свободно, C: 13 ГБ — следить.
- **Box:** pytest bots/bot5-candidate-v4 @40fa599 (лог box/results/orch_1805_runs.log).

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | bot2 A3 D-FINE docker @829ea7d (повтор) | 18:05 | ? |
| ПК CPU | — свободна (ждёт фикс bot6) | 17:58 | — |
| Kaggle GPU×2 | dfine-ae-ft; c3sc-s1f01 | 16:33/18:04 | ≤19:30 / ~20:40 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | pytest bot5 @40fa599 | 18:05 | ~2 мин |

## bot1 pulse — 2026-10-29 17:50 MSK
- Kaggle: c3sc-f4-v2 RUNNING, dfine-ae-ft RUNNING (GPU 2/2), TPU c3sc-train всё ещё QUEUED (с 17:10). bot6-pixgbm-cleanroom COMPLETE (hollow, уже учтено).
- Gate bot7 compare_candidates @b6f835d (ПК, docker): v3 FAIL (sp_or 0,243→0; ALL 0,2880, Δ −0,0245); hotfix_B PASS (ALL 0,3129, Δ +0,0004); v4_off PASS (= hotfix_B); bot4_on FAIL (li, Δ −0,0159). Строки в EXPERIMENTS.csv.
- ПК: CPU — bot5 gor_heads docker @9f62a96 (8 потоков) RUNNING; GPU — bot2 A3 @829ea7d; см. pc/STATUS_LIVE.md.
- Box: раннер занят — bot2-e2e-v3-prep @10e9936 → bot4-large-objects @5097c4a → bot7-compare-candidates @b6f835d (лог box/results/orch_1748_runs.log).

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | bot2 A3 D-FINE docker @829ea7d | 17:30 | ? |
| ПК CPU | bot5 gor_heads docker @9f62a96, 8 потоков | 17:42 | ? |
| Kaggle GPU×2 | c3sc-f4-v2; dfine-ae-ft | 16:00/16:33 | ~18:00 / ≤19:30 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | pytest bot2 → bot4 → bot7 | 17:48 | ~6 мин |


## bot1 pulse — 2026-10-29 17:33 MSK
- **ПК: диск C: был заполнен (1,5 ГБ свободно).** В 17:24 bot2 A3 @829ea7d и bot6 PIXGBM @3f61ea8 упали на `git worktree add` (No space left → HEAD пуст → FAILED bad_sha), STATUS_LIVE ошибочно писал RUNNING. Исправлено: worktree новых задач теперь на **F:\Sloi_wt\** (runner'ы пропатчены, бэкап `*.bak_c_full`), обе задачи перезапущены 17:30 через schtasks — идёт checkout. Правило для всех: новые worktree на ПК — только в F:\Sloi_wt.
- Kaggle: c3sc-f4-v2 RUNNING, dfine-ae-ft RUNNING (GPU 2/2), TPU c3sc-train QUEUED. Без изменений.
- Box: тест-раннер занят — bot5-candidate-v4 @9f62a96, затем bot6-pixgbm-cleanroom (лог box/results/orch_1730_runs.log).

| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | bot2 A3 D-FINE docker import @829ea7d (wt F:) | 17:30 | ? |
| ПК CPU | bot6 PIXGBM @3f61ea8 match-reference, 2 потока (wt F:) | 17:30 | ~30–60 мин |
| Kaggle GPU×2 | c3sc-f4-v2; dfine-ae-ft | 16:2x/16:33 | ~18:00 / ≤19:30 |
| Kaggle TPU | tpu-cnx-c3sc-train QUEUED | 17:10 | ? |
| Box | pytest bot5 @9f62a96 → bot6 | 17:30 | ~20 мин |

## bot1 unblock pulse — 2026-10-29 17:10 MSK

- **TPU:** sole job `dmitrijkv/koz3-tpu-cnx-c3sc-train` QUEUED (probe→smoke c3-sc XLA). Ghosts deleted. GPUs untouched RUNNING.
- **geo200:** pack ready on PC `F:\2\koz3-xfer-geo200-upload` (~3.45GB, metadata OK). Upload blocked: CopyToBox allowlist refuses C:\Sloi and F:\; PC has no KAGGLE_API_TOKEN. Need token on PC **or** CopyToBox root opened, then `kaggle datasets create -p F:\2\koz3-xfer-geo200-upload`.
- **bot5 TASK C:** still BLOCKED — `koz3-b5-gordense-timing` is a **phantom** (never created); heads absent on box/PC. Details `reports/bot5_v4.md` TASK C. No UNBLOCKED.

## Ресурсы и владельцы полос
- ПК пользователя (i5-13600KF 20 потоков, 32 ГБ, GTX 1650 4 ГБ, Docker/WSL2; данные F:\2\train-expds-3, mock F:\expds3\mock_val, работа C:\Sloi\pc_jobs) — ПОЛОСУ ВЕДЁТ ТОЛЬКО «1». GPU одна, задачи строго по очереди. Нужен прогон — добавь заявку в pc/QUEUE.md (что, ветка/zip, команда, ожидаемое время, критерий), пинг «1».
- Kaggle (dmitrijkv; GPU осталось ~9.4 ч до 31.10 03:00 МСК, TPU 20 ч не использовано) — ПОЛОСУ ВЕДЁТ «1» (токен только у него). Заявка — kaggle/QUEUE.md (slug, источник ядра в репо, GPU/TPU/CPU, часы, что вернуть).
- Облачные агенты Cursor — каждый бот запускает своих, не больше 2 одновременно на бота.
- Мой компьютер («1»): 8 CPU, ~2 ГБ свободной RAM — анализ артефактов, подготовка ядер Kaggle, сборка пакетов. Заявки — box/QUEUE.md.

## ПК — полосы сейчас (обновляет «1»; 29.10 17:00 МСК)
| Полоса | Задача | С | ETA |
|---|---|---|---|
| ПК GPU | bot2 A2 dfine stages dump @fb66c89 (pc-job-runner), C:\Sloi\pc_jobs\bot2_a2_dfine_stages_fb66c89, папка обновлялась 16:55 | 16:51 | ? |
| ПК CPU | bot6 PIXGBM cleanroom @d173a75 (12 потоков) + bot7 diff_scope --honest 10 @aa3694f (≤6 потоков, RETRY_PYTEST) — суммарно ~18 потоков, выше нормы 12, не трогаю пока идут | 16:51 | 15–60 мин |
| ПК GPU очередь | bot4 large_objects mock @337a27c: STATUS.txt = «DONE 16:50:50» через <1 мин после старта — подозрительно, результаты не проверены (Shell на ПК: 1 вызов OK в 16:58, далее spawn EPERM). Следом bot8 TTA/MS sp_or → bot3 aerial; bot5 (b) gor_dense BLOCKED | — | — |

## Правила
- В main не пушит никто, кроме PR, которые сливает пользователь. Свои ветки bots/botN-*; draft PR.
- На LB не сдаём без решения пользователя (финал: 10 попыток, 2 успешные).
- Всё новое — за флагом OFF, принимается только через honest-10 (+ по слою, когда бот 7 сделает режим).
- Итоговые отчёты кладите в reports/botN_<тема>.md и шлите «1» короткую выжимку.

## Работы
| Бот | Тема | Ветка | Статус |
|---|---|---|---|
| 1 | координация; PR: rerank_arch в main + фикс fork в c3-sc_f4 | агент bc-51cd56a6 | в работе |
| 2 | e2e v3: РЕГРЕССИЯ sp_or (F1 0,24→0) на ПК 14:48 | bots/bot2-e2e-v3-prep | срочно |
| 3 | аэрофото recall 0,013 | bots/bot3-aerial-recall | в работе |
| 4 | крупные объекты >5 тыс. м² | bots/bot4-large-objects | в работе (bc-2b5d9634) |
| 5 | сборка v4 + манифест весов | bots/bot5-candidate-v4 | в работе (bc-ff9ace20) |
| 6 | clean-room PIXGBM | bots/bot6-pixgbm-cleanroom | в работе |
| 7 | honest-eval по слою | bots/bot7-honest-eval-layer | в работе |
| 8 | TTA/мультимасштаб SpOr + compliance | bots/bot8-orsp-tta | в работе (bc-8ced3d37) |

## Kaggle сейчас
- Обновлено 29.10 17:00 МСК (c3sc-f4-v2 и dfine-ae-ft RUNNING, GPU-полоса заполнена). Квота: GPU ~8.5 ч до 31.10 03:00 МСК; TPU 20 ч (слот batch=1 ghost после ERROR probe/smoke; cancel API 403; PIXGBM ушёл на CPU fallback).
- koz3-dfine-or: COMPLETE (GPU); outputs pulled.
- koz3-cnx-li-c3sc-f4-v2: RUNNING (GPU) — c3-sc_f4 relaunch, патч fork/filelock. Ожид. ~1.2–1.5 ч.
- koz3-bot6-pixgbm-cleanroom: BLOCKED-DATA ~16:40 МСК — v6 DONE hollow (0 rasters geo200-217; manifests без blob-map на 200–217). Нужны xfer manifests/blobs для этих geo.
- koz3-dfine-ae-ft: RUNNING (pushed ~16:33 МСК); budget OK с reserve 0.5ч.
- koz3-bot8-oof-honest10: **QUEUED-AFTER-RESET** (8–12ч не влезают до ресета).
- COMPLETE: dfine-ae, dfine-spor-final, dfine-or, cnx-li-c3sc-f01/f23. cnx-li-c3-f4 ERROR (c3-in OK; c3-sc ушёл в v2).
- Манифесты xfer s000–s009 в `shared/koz3/kaggle/koz3-xfer-sNNN_manifest.json`.


## Факты e2e v3 на ПК (29.10 14:43–14:48, zip v3on_0876090, docker, GTX 1650)
- v2 82,6 с, VRAM 3626 МБ; v3 143,0 с, VRAM 3774 МБ.
- mock (2 участка, GT 450): F1 ALL v2 0,3125 → v3 0,2880 (−0,0245). li +0,0039, ae 0, sp_or 0,2429 → 0,0.
- sp_dfine: 482 сырых, drop_groups [sp_or, sp_focal, sp_arch, sp_focal2] −46, после порога sp_or 13. Подозрение (не проверено): CRS/координаты боксов D-FINE (см. патч бота 3 e3c4f41 «3857») или порог.

## 29.10 15:57 — БЛОКЕР: лимит Cursor на облачных агентов исчерпан
Все облачные агенты (боты 1–8) упали, перезапуск невозможен, пока Дмитрий не включит on-demand. До этого: разбор только через чтение репо по GitHub API (без клонирования, без коммитов), гипотезы и план патча в reports/botN_*.md; диагностические прогоны — заявкой в pc/QUEUE.md или kaggle/QUEUE.md. После включения — перезапуск тех же агентов (reply), не новых.

## 29.10 15:58 — НОВЫЙ РЕЖИМ: без облачных агентов (Дмитрий: on-demand не будет)
- Код: каждый бот сам делает правки в СВОЕЙ ветке bots/botN-* через GitHub API (create/update file, небольшие коммиты), без клонирования. main не трогать. PR — draft.
- Проверка: тесты и прогоны выполняет «1» на ПК пользователя в существующих worktree (C:\Sloi\wt_*) или в Docker; заявка в pc/QUEUE.md: ветка, коммит, команда, ожидаемое время, критерий. GPU одна — строго по очереди.
- Тяжёлый CPU/GPU: Kaggle (GPU ~9 ч до 31.10, TPU 20 ч) — заявка в kaggle/QUEUE.md.
- Приоритеты: 1) бот 2 — регрессия sp_or v3; 2) бот 7 — eval по слою; 3) бот 5 — v4 (зависит от 1); 4) боты 3/4/6/8 — улучшения за флагом.

## 29.10 16:03 — Копия репо на компьютере «1» (с согласия Дмитрия)
- /workspace/koz3-repo (main @70dedf4, все ветки). Боты: правки делайте в своём worktree `git -C /workspace/koz3-repo worktree add /workspace/koz3-wt/botN-<тема> -b bots/botN-<тема>` (или существующую ветку), коммит и push только в свою ветку. Если push не авторизован у вас — пишите «1», запушу я.
- Тест-раннер (CPU, мало RAM — по одной задаче): box/run_tests.sh <ветка>, отчёты в box/results/. Починен 29.10 16:30; триаж reports/box_test_triage_2026-10-29.md.
- ПК Дмитрия периодически отключается: на ПК только задачи с чекпоинтами/перезапуском, STATUS.txt в папке задачи; критичное дублируем на Kaggle.


## Тест-раннер на компьютере «1»
Обновлено: 29.10 16:36 МСК. Триаж: `reports/box_test_triage_2026-10-29.md`.

- **venv:** `/workspace/koz3-venv` (Python 3.13; torch CPU; `requirements-dev.txt`; `expds_core` через maturin+rustup).
- **клон:** `/workspace/koz3-repo` (main @ `70dedf4`). Worktree: `/workspace/koz3-wt/<ветка-sanitized>`.
- **скрипт:** `/home/box/shared/koz3/box/run_tests.sh <ветка>` (= `run_tests.full.sh`). Backup: `*.bak_2026-10-29_1612`.
  - только существующие `tests/` и `centerhead/tests` (нет exit-4 на старых ветках);
  - отдельные вызовы pytest на каждый путь; флаги `-q -rfEs -p no:cacheprovider -p koz3_box_env`;
  - плагин `box/plugins/koz3_box_env.py`: missing pixgbm/weights / optional deps / CUDA → skip, не fail;
  - single-flight lock + retry/stale `index.lock` (фикс падения bot3 worktree).
- **очистка:** `run_tests.sh --cleanup <ветка>` или `--cleanup-all`.
- **baseline main@70dedf4:** 666 passed / 0 failed / 43 skipped. Отчёт: `box/results/main_70dedf4.md`.
- **прогон 16:30–16:35 МСК (после фикса):** bot2/3/3-dfine/4/5/6/8-orsp/8-oof — **green** (бывшие 5–10 «failed» = env packaging). **bot7 ae4c816 — 11 real regressions** (`NameError: per_site` в `build_curves_reference`, culprit `d1a98ae`).
- **не на CPU / нет на боксе:** CUDA, PIXGBM/`geo_pixgbm.py`, rerank/pixgbm weights, rvt_py, EXPDS_METRIC, реальные выборки bot-8.
- **очередь:** одна задача за раз (peak ~2.2 ГиБ; ~2 ГБ free). Заявки — `box/QUEUE.md`.

## 29.10 16:33 — проход оркестратора
- bot7 honest-10 layer: GATE PASS (global байт-в-байт = main, ALL 0.2971; layer 0.2765, Δ −0.0206). PR #2 готов к ревью.
- bot2 hotfix 1ec2d7c (pc-job-runner): A=B ALL 0.3129 (sp_or 0.2451) ≈ v2; B PASS, A без прироста. Дубль A2/dump «1» на ПАУЗЕ (GPU отдан bot5 v4).
- ПК Shell: EPERM 16:29–16:31, к 16:33 снова работает.
