# BRANCHES: инвентаризация KonkovDV/expds-koz3 (29.10.2026 22:50 MSK)
Только рекомендации. **Ничего не удалено и не переписано.** Удаление или слияние — решение Елены/бота 1.
Всего удалённых веток 104 (90 `bots/*`). За 28.10 обновлено 40, за 29.10 — 64. Открытые PR: #2–#11 (10 шт.). Открытых issues нет.

## Актуальные (новая схема, потоки W0–W5)
| Ветка | SHA | Автор | Что | Статус |
|---|---|---|---|---|
| bots/orch-protocol | (этот коммит) | coord | протокол, аренды, расписание, STATE | единственный источник orch |
| bots/w0-rust-speed | 697409e | bot-1-w0 | Rust: потоковый conf-фильтр GeoJSON, orjson-загрузка, GC-пауза в build_curves (e2e honest-10 ×2.5, фильтр ×7.8) | готово к ревью W4 |
| bots/w4-gate | 22a32c3 | W4 gate bot | PREREG PIXGBM, проверка de1edc9 | активна |
| bots/w5-rerank | 8a76da4 | bot-W5-rerank | реранкер V1–V4 + nested gate, PREREG, ядро dino-emb (не запущено) | активна, поток W5 (п.7) |
| (нет) bots/w1-*, w2-*, w3-* | — | — | W1–W3 пока работают вне git (orch/bot-WN, /workspace/koz3/wN) | **завести ветки и worktree** |

## Флаги
1. **Чужие пуши.** bot-8 с 19:49 до 20:42 запушил в чужие ветки: bot2-sat-recall, bot3-ae-ortho-recall, bot3-dfine-aeor-eval, bot3-dfine-ae-kernel, bot5-candidate-v4, bot6-pixgbm-cleanroom, bot7-honest-eval-layer, bot7-compare-candidates, bot2-e2e-v3-prep, bot3-aerial-recall (bot-5). Нарушает правило «одна ветка — один владелец». Теперь это ловит check_conflicts.
2. **Дубли.** `orch` (7641174, PR #11 → main) дублирует `bots/orch-protocol`. Рекомендация: закрыть PR #11 без слияния и оставить orch-protocol. bot10-honest-bootstrap ≡ bot10-tools (a839da2); bot10-bot8site-check ≡ bot8-reliability-ro (8343483). `origin` (ветка) ≡ main (70dedf4); `master` — устаревший корень.
3. **Параллельные кандидаты.** v2/v3/v3fix/v4-nomodels (bot7-candidate-*), bot5-gor-dense-v3/v4, bot6-crsfix/orgate/aeor-detect, bot3-v4-patches — 4 линии кандидата без одной головы. Кандидат v3 соберёт W4 из main + принятые флаги. Остальные — в архив (тег `archive/<имя>`), не удалять до 14.11.
4. **Отрицательные результаты** (REVIEW_SOTA §2.4) — оставить как ссылки, в сборку не брать: bot2-morph*, bot2-adaf, bot2-li-real, bot2-bigobj, bot2-li-half / bot8-li-half*, bot3-geosnap (до перепроверки п.5), bot5-gor-dense-* (gor_dense NO-OP), bot5-nopretrain.
5. **Риск лицензии/EN.** fix/remove-li-decode-provenance — li_decode удалён (линия EN). bots/bot6-pixgbm-cleanroom и PIXGBM: сборка из main ≠ архив v2. Паритет — п.1 у W4. Спецификация чистой комнаты — `docs/PIXGBM_SPEC_CLEANROOM.md` в этой ветке.
6. **PR в чужие базы**: #10 bot8-tta-on-bot3 → bots/bot3-dfine-aeor-eval, #8 bot7-compare-candidates → bots/bot7-honest-eval-layer. Целевые ветки уже без владельцев. Рекомендация: перебазировать на main или закрыть.
7. Разовые интеграционные ветки 29.10 (integrate/, harvest/, fix/murder-board, compliance/tr-approved, research/competitors, claude/audit, archive/research-results-20261029): документация. Нужное перенести в docs/ через PR W4, ветки оставить.

## Открытые PR
| # | head → base | создан (UTC) | заголовок |
|---|---|---|---|
| 11 | orch → main | 2026-10-29T18:19 | orch: доска задач, Kaggle-диспетчер, роли; rerank_arch off; фикс kurgan-subclass |
| 10 | bots/bot8-tta-on-bot3 → bots/bot3-dfine-aeor-eval | 2026-10-29T17:42 | bot8: TTA/MS on bot3 dfine_aeor Ae f4pack (flags OFF) |
| 9 | bots/bot8-tta-lb → main | 2026-10-29T17:33 | bot-8: TTA/MS per-layer (ae/or/sp_or) + kit-zip REJECT compliance (from eb268d6) |
| 8 | bots/bot7-compare-candidates → bots/bot7-honest-eval-layer | 2026-10-29T14:40 | bot7: compare_candidates gate (mock e2e ALL/li/sp_or/ae + layer-drop rule) |
| 7 | bots/bot8-oof-kaggle → main | 2026-10-29T13:20 | bot-8: GPU-5 OOF honest-10 Kaggle kernel (no push/run) |
| 6 | bots/bot3-dfine-ae-kernel → main | 2026-10-29T13:14 | bot-3: Kaggle kernel koz3-dfine-ae-ft (GPU-3 D-FINE Ae resume) |
| 5 | bots/bot6-pixgbm-cleanroom → main | 2026-10-29T13:10 | bot-6: PIXGBM clean-room retrain + Kaggle TPU-VM-as-CPU kernel + provenance docs |
| 4 | bots/bot8-orsp-tta → main | 2026-10-29T13:08 | bot-8: TTA/MS sp_or (flags OFF) + check_compliance финального zip |
| 3 | bots/bot3-aerial-recall → main | 2026-10-29T13:07 | draft(bot-3): aerial recall flags OFF (ae_recall_relax, raster_fix; dfine_aeor already off |
| 2 | bots/bot7-honest-eval-layer → main | 2026-10-29T13:07 | honest-eval: per-layer matching mode (flag, default off) |

## Все ветки (по дате коммита, MSK)
| Ветка | SHA | Дата | Автор | Сообщение |
|---|---|---|---|---|
| bots/w5-rerank | 8a76da4 | 29.10 22:33 | bot-W5-rerank | W5 rerank: policies V1-V4 + nested gate, PREREG, dino-emb kernel (not pushed), p |
| bots/w0-rust-speed | 697409e | 29.10 22:29 | bot-1-w0 | w0: fastjson (orjson load, Rust streaming conf-filter), GC pause in build_curves |
| bots/orch-protocol | bbebfe3 | 29.10 22:24 | Konkov Dmitrij | docs: PIXGBM clean-room functional spec (author: bot 1 spec team; implementer bo |
| bots/w4-gate | 22a32c3 | 29.10 22:21 | W4 gate bot | W4: PREREG PIXGBM (P-retrain vs P-clean), проверка de1edc9 при вык |
| orch | 7641174 | 29.10 21:26 | KonkovDV | orch: README — данные S3 на CPU-боксе (без кредов) |
| archive/research-results-20261029 | 43d2513 | 29.10 20:46 | koz3-bot1 | Plan to deadline 11.11 |
| bots/bot8-tta-on-bot3 | 82ab852 | 29.10 20:42 | bot-8 | pc: pin runner KOZ3_SHA tip ef2bd88 |
| bots/bot4-large-objects | 1a0e779 | 29.10 20:42 | bot-4 | bot-4: report tip sha 62f7fb1 |
| bots/bot3-ae-ortho-recall | ffa9178 | 29.10 20:39 | bot-8 | bot-3 report: fix tip markdown backticks |
| bots/bot2-sat-recall | 491c63e | 29.10 20:38 | bot-8 | feat(sat): [sat_thr_relax] scale sp_* thresholds (default OFF) |
| bots/bot8-tta-lb | f6fe20a | 29.10 20:31 | bot-8 | bot-8: kit-zip REJECT в check_compliance (LESSONS §5) |
| bots/bot7-honest-eval-layer | 42b4bf8 | 29.10 20:28 | bot-8 | fix(bot7): loosen TimeGuard left() vs run_elapsed() assert under suite load |
| bots/bot7-compare-candidates | 5e43510 | 29.10 20:28 | bot-8 | fix(bot7): loosen TimeGuard left() vs run_elapsed() assert under suite load |
| bots/bot3-dfine-aeor-eval | 2f46f20 | 29.10 20:28 | bot-8 | pc(bot3): resolve runner Sha from origin tip (no PLACEHOLDER) |
| bots/bot6-pixgbm-cleanroom | 13082a3 | 29.10 19:54 | bot-8 | fix(bot-6): pixgbm enabled without expds/geo_pixgbm.py -> ERROR log (+ opt-in EX |
| bots/bot5-candidate-v4 | ed73c5e | 29.10 19:50 | bot-8 | test: packaging collect() without gitignored pixgbm weights |
| bots/bot8-orsp-tta | 73b1fee | 29.10 19:49 | bot-8 | chore(bot8): pin runners to tip 82a167c for PC JobDir |
| bots/bot2-e2e-v3-prep | c3d912c | 29.10 18:52 | bot-8 | docs(bot2): pin A3 localization report tip to f9ad045 |
| bots/bot8-oof-kaggle | a6f8992 | 29.10 16:20 | bot-8 | bot-8: Kaggle kernel GPU-5 OOF honest-10 (resumable, CPU smoke) |
| bots/bot3-dfine-ae-kernel | 6e42cf6 | 29.10 16:13 | bot-8 | docs(bot3): README for koz3-dfine-ae-ft kernel (env, outputs, GPU budget) |
| bots/bot3-aerial-recall | 475bafa | 29.10 16:06 | bot-5 | scripts(bot3): eval_aerial_honest10 wrapper around scoreboard --honest 10 |
| origin | 70dedf4 | 29.10 15:35 | grok-bot | docs: final bot reports sync 29.10 15:35 MSK (stop of work) |
| main | 70dedf4 | 29.10 15:35 | grok-bot | docs: final bot reports sync 29.10 15:35 MSK (stop of work) |
| bots/bot5-gordense-timing | fedaeac | 29.10 14:48 | bot-5 | gor_dense: T4 timing kernel (kaggle/gordense_timing, not run), FOR_BOT1_GORDENSE |
| bots/bot7-candidate-v4-nomodels | cbd4804 | 29.10 14:36 | bot-5 | gor_dense на v4 (f3526cf): порядок sp_dfine → dfine_aeor → gor_dens |
| bots/bot5-gor-dense-v4 | 86a60c5 | 29.10 14:11 | bot-5 | gor_dense на v4 (f3526cf): порядок sp_dfine → dfine_aeor → gor_dens |
| bots/bot7-candidate-v3fix-nomodels | 0037c0e | 29.10 14:05 | bot-7 | crs: geographic UTM.json (008_Герите EPSG:4326) -> UTM zone by region cent |
| bots/bot6-crsfix-nomodels | 634bcdd | 29.10 14:05 | bot-6 | bot-6 v4: check_submission — вердикт смоука dfine_aeor в report( |
| bots/bot6-orgate-nomodels | 424520c | 29.10 13:58 | bot-6 | bot-6 v4: check_submission — вердикт смоука dfine_aeor в report( |
| bots/bot5-gor-dense-v3-nomodels | e792f70 | 29.10 13:40 | bot-5 | gor_dense: фаза ДО gor_gate (после sp_dfine), GOR_DENSE_UNAVAILABLE � |
| bots/bot7-candidate-v3-nomodels | 78af57f | 29.10 13:08 | bot7 | sp_dfine: robust to image deps without version metadata (tqdm found=None in pant |
| bots/bot6-aeor-detect-nomodels | 3686592 | 29.10 12:52 | bot-6 | bot-6 AEOR fixes: merge key isolates dfine_ae/dfine_or (union off), separate dfi |
| bots/bot3-v4-patches-nomodels | 1327e88 | 29.10 12:32 | bot-3 | geo_pixgbm: правило U (dedupe_union) сравнивает только т |
| bots/bot5-dinov2reg-nodata | 6002660 | 29.10 12:32 | bot-5 | crop_verify: --dump-crops / --from-crops / --apply-p (crops on CPU box, ViT prob |
| bots/bot5-gor-dense-loc-nodata | 4e815b2 | 29.10 12:32 | bot-5 | gor_dense: prereg confirmation seeds 5-9 (prereg_confirm.py) |
| feat/aeor-detect-2026-10-29 | 6c2511c | 29.10 12:05 | bot-1 StrataTrace | feat(aeor_dfine): D-FINE всех классов на Ae/Or за флагом (O |
| bots/bot2-bigobj | 17c04e1 | 29.10 11:56 | bot-2 | bigobj: результаты honest-10 (все варианты отклонен |
| compliance/tr-approved-2026-10-29 | bfa903c | 29.10 11:22 | grok-bot | docs: правила утв. ТР, сверка 29.10, план ПКК (Прил |
| bots/bot3-opt | 8c86700 | 29.10 11:21 | bot-3 | geo_pixgbm: opt-in C2 ([geophys.pixgbm] ensemble_dirs — среднее P мо� |
| bots/bot2-adaf | 0ce6ffa | 29.10 11:17 | bot-2 | adaf: eval re-run on bot-10 work11/fam0 (work/fam deleted), kernel stub + README |
| research/competitors-2026-10-29 | 7084f00 | 29.10 11:14 | grok-bot | docs: COMPETITORS_2026-10-29 v2 — свежие клоны, дыры LB v2, и� |
| bots/bot3-etgr-union | e0ec0ff | 29.10 11:13 | bot-3 | geo_pixgbm: правило U (dedupe_union) сравнивает только т |
| harvest/2026-10-29 | 953f5a1 | 29.10 11:07 | Grok Bot | Harvest speedup-2026-10-28: clean CPU speedups behind [speed] flags (default OFF |
| fix/remove-li-decode-provenance | 4c34802 | 29.10 10:37 | grok-bot | Remove li_decode (EN lineage); weights provenance; pixgbm decision; branch archi |
| fix/murder-board-2026-10-29 | ad4dbb5 | 29.10 10:11 | grok-bot | Murder Board 29.10: sp_dfine strict preflight (no silent v3->v2), transformers p |
| bots/bot2-morph-fast | 0e1aa3b | 29.10 09:59 | bot-2 | morph_fast --ringw (задача 14): 4 оконных признака коль |
| integrate/2026-10-29 | 90e8b29 | 29.10 09:49 | grok-bot | docs/MERGE_2026-10-29.md: provenance flags (li_decode idea-level, bench kernel,  |
| bots/bot8-li-half | e449736 | 29.10 09:42 | bot-8 | bot-8: [li_half] self_xclass_nms_m — NMS между своими кругам� |
| bots/bot2-li-half | 92448b3 | 29.10 09:42 | bot-2 | li_half_box: вторичный вариант xnms12 (NMS 12 м внутри н |
| bots/bot8-li-half-pre0002 | 1d904f9 | 29.10 09:31 | bot-8 | bot-8: li_half/cnx dedupe cross-class for kurgans vs lidar, σ anchors include a |
| bots/bot6-orsp | abae911 | 29.10 08:57 | bot-6 | ORSP v3 (по bot-10 REPORT10 §3): пороги sp в [post.sp_score_fusion.thr |
| bots/bot2-morph | e918a3b | 29.10 08:40 | bot-2 | task10: coord alignment (068/069 UTM53->52), merge with within-region rz feature |
| bots/bot8-cnx | 1aecea9 | 29.10 08:22 | bot-8 | check_submission: обходы сканера cnx (алиасы subprocess/os/im |
| claude/audit-2026-10-29 | d4edf2f | 29.10 08:14 | Claude | audit 29.10 (ч. 2): корень проблемы — калибровка по |
| bots/bot3-audit | e3c4f41 | 29.10 08:10 | bot-3 | geophys: opt-in метры на земле для EPSG:3857 (шаг × cos φ ц |
| bots/bot2-li-real | 361a9b5 | 29.10 08:08 | bot-2 | task 9: li-decode on real li maps — results (all variants rejected vs legacy), |
| bots/bot3-backbone | 49d0200 | 29.10 08:04 | bot-3 | backbone: замороженные признаки ResNet18/DINOv2-S → PCA-32 |
| bots/bot8-softceil-pixgbm | 17c2e2c | 29.10 07:27 | bot-8 | bot-8: тест сторожа — float через math.isclose (19.99999999999 |
| bots/bot5-gor-patch-kaggle | 82058b8 | 29.10 07:25 | bot-5 | gor_v3: final DINOv2-S patch-gate head (fit) + apply to candidate GeoJSON |
| bots/bot5-rfdetr-coco | c43cb08 | 29.10 07:24 | bot-5 | RF-DETR: O365 — основной M (O365→COCO), вариант B rfdetr-b +  |
| bots/bot7-candidate-v2-nomodels | 1b62124 | 29.10 07:23 | grok-bot | bot-7: кандидат v2 eb268d6 без весов PIXGBM (models/pixgbm/all/* |
| bots/bot5-nopretrain | 43d0163 | 29.10 07:11 | bot-5 | det_train_hf: --from-scratch (ОТМЕНЕНО: веса разрешены; н� |
| bots/bot3-geosnap | 7a5ee57 | 29.10 02:24 | bot-3 | geo_snap: poly_mode=peaks (opt-in) — точка в каждом локальн |
| bots/bot3-pixgbm | 76f7317 | 29.10 02:24 | bot-3 | PIXGBM: потоки/пул по sched_getaffinity (фолбэк cpu_count, по |
| bots/bot8-soft-ceiling | 26e12ae | 28.10 23:41 | bot-8 | bot-8: мягкий потолок 1200 с (EXPDS_SOFT_CEILING_S): TimeGuard.t0/ |
| bots/bot8-cpu-a-clean-on-v2 | 18b2f66 | 28.10 23:37 | bot-8 | bot-8: build_submission_git кладёт inference/lib/expds_fast*.so из за� |
| bots/bot8-cpu-a-clean | 95bc0a6 | 28.10 23:37 | bot-8 | lidar_a: ground.rasterise через Rust expds_fast (las_grid_fast, numpy-фо� |
| speedup-2026-10-28 | 6602bb2 | 28.10 23:29 | KonkovDV | speedup 2026-10-28: ускорение CPU-части (байт-в-байт с 1 |
| bots/bot6-optics | 62145d6 | 28.10 23:23 | bot-6 | bot6: ae raster fix behind [raster_fix] flags (reproject_3857, int16_identity, z |
| bots/bot6-twins-only | 78add0e | 28.10 23:00 | bot-6 | post.kurgan_cross_modal_twins: один cos φ на участок вместо  |
| bots/bot6-geo | 67f84bd | 28.10 22:58 | bot-6 | post: [post] geo_convex_hull (выкл. по умолчанию) — выпук� |
| bots/bot6-localize | 6b58d99 | 28.10 22:23 | bot-6 | bot-6: LOCALIZE — микрокруги фортификаций 0,5 м (вык |
| bots/bot2-centerhead | 788202a | 28.10 22:12 | bot-2 | centerhead: CenterNet kurgan head (targets, focal loss, smp wrapper, decode/NMS, |
| bots/bot8-merged | 2f530a8 | 28.10 22:11 | bot-8 | Merge branch 'bot8-reliability' into bot8-merged |
| bots/bot5-rfdetr-kaggle | b355e2c | 28.10 22:10 | bot-5 | rfdetr_train: assert_no_meta before .to(device); pin transformers==5.21.0 etc. ( |
| bots/bot5-dfine-kaggle | 3027dc7 | 28.10 22:09 | bot-5 | det_train_hf: pin transformers==5.21.0 (requirements_dfine.txt, notebook, README |
| bots/bot3-geophys | 72ceb28 | 28.10 22:07 | bot-3 | audit01: [geophys.<mt>] dedupe_maps=false отключает склейку на |
| bots/bot7-candidate-measure | 1b0c294 | 28.10 21:51 | bot-7 | bot-7: scripts/pkg_fam.py — посемейный стек пакета (stack6 |
| bots/bot7-candidate | 4c3e29e | 28.10 21:49 | bot-7 | bot-7: kurgan_twins_only (bot-6) — в конфиге кандидата вык |
| bots/bot2-li-decode | 78588f6 | 28.10 21:36 | bot-2 | li-decode: point decoding of lidar-net maps behind [li] decode="peaks" (default  |
| bots/bot8-rust-metric | d073b45 | 28.10 21:30 | bot-8 | bot-8: scope=site только явно и с предупреждением в  |
| bots/bot8-reliability | 2901cb0 | 28.10 21:30 | bot-8 | bot-8: дедлайн внутри цикла окон снимков: част� |
| bots/bot8-reliability-on-bot7 | 5135517 | 28.10 21:30 | bot-8 | bot-8: дедлайн внутри цикла окон снимков: част� |
| bots/bot8-reliability-on-bot7-candidate | fd7dd50 | 28.10 21:30 | bot-8 | bot-8: дедлайн внутри цикла окон снимков: част� |
| bots/bot2-merge-check | 81c396f | 28.10 21:26 | bot-2 | E3: separate no-lidar thresholds (nested/plain), global matching eval; no pipeli |
| bots/bot7-candidate-v3pre | 8a6ce89 | 28.10 21:23 | bot-7 | bot-7: safe_finish — sota_stack6 (multipole + filter_finds_points) и в за� |
| bots/bot3-audit01 | c315a9b | 28.10 21:08 | bot-3 | audit01: [geophys.<mt>] dedupe_maps=false отключает склейку на |
| bots/bot7-candidate-v2-old-fe0e570 | fe0e570 | 28.10 21:03 | bot-7 | bot-7: кандидат — включить [rerank_arch]; модель models/ |
| bots/bot7-candidate-v1 | 913e2be | 28.10 20:57 | bot-7 | bot-7: флаг [rerank_arch] (реранк архитектуры LightGBM, /wo |
| bots/bot7-bundle-measure | de4833f | 28.10 20:40 | bot-7 | bot-7: scripts/micro_fam.py — посемейный стек → micro_circles  |
| bots/bot2-fusion-calib | c356bc8 | 28.10 20:36 | bot-2 | E2: pre-declared nl-only / no-li-fusion calibration+WBF variants, per-region mat |
| bots/bot9-kaggle-queue | 194a691 | 28.10 20:23 | bot-9 | s4_rerank: honest per-family rerank of li kurgan candidates with s4 fam10 crop-v |
| bots/bot10-site-scope-provisional | 68710a9 | 28.10 20:19 | bot-10 | ВРЕМЕННО: EXPDS_MATCH_SCOPE=site в match_collected (только зам |
| bots/bot10-unseen-keep | 4409c5b | 28.10 20:13 | bot-10 | honest: --unseen/--honest-unseen keep |
| bots/bot7-bundle | 42ada8d | 28.10 20:10 | bot-7 | bot-7: пакет микрокругов — архитектура 0,25 м (bot |
| bots/bot7-geom-opt | 43a341a | 28.10 19:59 | bot-7 | geom_opt: dedup within markup option, RECOMMENDED_* configs |
| bots/bot10-honest-bootstrap | a839da2 | 28.10 19:49 | bot-10 | bot-10: посемейные кривые, кластеры семей, шум � |
| bots/bot10-tools | a839da2 | 28.10 19:49 | bot-10 | bot-10: посемейные кривые, кластеры семей, шум � |
| bots/bot2-kurgan-morph | 1bfcb9a | 28.10 19:48 | bot-2 | kurgan subclass: morphometric LOFO classifier + OOF relabel oracle (bot-2) |
| bots/bot10-bot8site-check | 8343483 | 28.10 19:36 | bot-8 | bot-8: [verify] weights от корня архива, не от cwd |
| bots/bot8-reliability-ro | 8343483 | 28.10 19:36 | bot-8 | bot-8: [verify] weights от корня архива, не от cwd |
| bots/bot4-gorfort | ab8ad26 | 28.10 19:35 | bot-4 | bot-4: арбитр fort→gor и грубый проход городищ (ст |
| bots/bot5-arch-microcircle | 4dcad89 | 28.10 19:33 | bot-5 | arch: [post.micro_circles], oof_transform micro, tiles --shrink-classes (bot-5) |
| master | 24bd916 | 28.10 18:07 | KonkovDV | Record completion of all 17 59-LAS models across both 2xT4 GPU sessions (koz3-gp |
