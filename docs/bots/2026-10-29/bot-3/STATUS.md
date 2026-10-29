# bot-3 STATUS (geophys)
- 2026-10-28 19:22:02 МСК: COMMON.md прочитан; клон repo/ ветка bot3-geophys от 24bd916.
- Данные: в /workspace/data/train-expds-3 пока только *.geojson/UTM.json (555 geojson, 210 json), растров геофизики НЕТ (find -type f | sed ext | uniq -c).

## 2026-10-28 19:37 MSK
- GT-статистика: gt_stats_geophys.json (sim/gt_stats.py). Главное: у Гнёздова 59 % находок имеют соседа ближе 2 м → NMS радиусом 2 м (peak_radius_m) сохраняет ≤66 % при случайном порядке силы.
- Нашёл в repo уже готовый post.multipole_object_poly (коммит 01cc769), заявленные выигрыши в notes/sota_stack6_2026-10-28.json — константы там подобраны на всём трейне, проверяю.
- Патч в repo (ветка bot3-geophys): GeoParams.split/split_h/pole_circle_m, geophys.split_merged (h-максимумы + watershed_ift, только scipy), label_poles, split_spec_params; geo_tune.py --splits/--pole-circle. tests/test_geo_split.py 8/8, соседние geo-тесты 32/32.
- Идёт симуляция sim/sim_poly.py (sigma 0.25/0.5/1.0 × шум 0/0.15), лог sim_run.log.

## 2026-10-28 20:23 MSK — ГОТОВО
- Все симуляции завершены (sim_run.log: ALLDONE). Сводки: sim_poly_summary.json, sim_analyze_out.txt, sim_poly_summary_patchcode.json.
- Код патча, σ0,5 n0,15: object_poly F1 0,7477 → 0,784 (LOFO). multipole_post 0,7008 (хуже базы).
- REPORT.md написан. Растров геофизики на боксе по-прежнему нет (только LiDAR/аэро tif): оценка — симуляция из GT, на ПК нужна проверка geo_tune + scoreboard --honest 10.

## 2026-10-28 21:30 MSK — audit01 ИЗМЕРЕН, object_poly — в работе
- Ветка bot3-audit01 (worktree bot-3/repo_audit01, c315a9b), diff bot-3/audit01.diff.
- A2 (maps=all, dedupe_maps=false): honest-10 +0,0039 (drop/keep, global/site), П1–П4 ✓, П5 — только пакетом. Детали: GEO_GAP_REPORT.md раздел A, geo_gap/a01/eval_all.out.
- band_mode в [geophys.maggr] игнорируется (A3m == A2 побайтно); нужен [geophys.maggr.finds_points].
- 200 и 212(auto) не измерены (OOM при 2 ГБ) — команды в отчёте.
- Разбор FN object_poly: geo_gap/gap_poly.json; геометрия стека G1–G3: geo_gap/eval_geom.out (≤ +0,0014, не проходят П1/П2).
- Идут растровые R1–R3 (split dt:0.5 / pole_circle 0.05) на 201–217: geo_gap/run_r.sh, 1 поток.

## 2026-10-28 22:05 МСК — ehimen (находки MagGr по рецепту EN)
- Отчёт: bot-3/ehimen/EHIMEN_REPORT.md, вердикт: EHIMEN_VERDICT.md, diff: bot-3/ehimen/ehimen_port.diff (ветка bot3-ehimen, 88cc077, без push).
- «0,72» EN = leave-one-region-out. У нашего порта под LORO F1 находок 0,688 (+0,0135), под leave-family-out Δ итога от −0,0026 до +0,0037, ни один вариант не проходит П2 (work/eval_*.out). Лучший — 0,05 м/px: +0,0037, P 0,625.
- Аудит 01 (A3): +0,0060, П1–П4 ✔ — лучше порта. В сдачу порт не брать.
- D8-аугментации в train.py:246–252/:332–336 (flips=True для maggr, tiles.py:635) ломают диполь N−S, но только у сети maggr, а её в сдаче нет.
- census_finds.zip — посылка-перепись на LB (только на ПК), на метки не влияет; не отправлялась.

## 2026-10-28 22:20 MSK — GEO_GAP_REPORT.md готов (разделы A–E)
- Лучший пакет: A2 (finds, maps=all без склейки) + R3 (pole_circle_m=0.05): honest-10 global drop +0,004037 / keep +0,006048, site +0,004077 / +0,006117; П1–П4 ✓, P(Δ>0) 0,869 (geo_gap/eval_pkg.out).
- Якоря по рецепту EhimenNathan (H1): keep +0,0021, drop −0,0002, П2 ✗ (geo_gap/eval_anchor.out). layer_copy R4 −0,016, split R2 −0,0029 (geo_gap/r/eval_r34.out, eval_r2.out). R1 — OOM на 211.
- Фильтр EtGr > 2 м: 0 из 76 tif на трейне.

## 2026-10-28 23:05 МСК — PIXGBM готов
- Отчёт: bot-3/PIXGBM_REPORT.md. Ветка bot3-pixgbm (b1ea251, без push), diff: bot-3/pixgbm/pixgbm.diff.
- S (pix заменяет классику): honest-10 +0,0110 / +0,0119 / +0,0099 / +0,0141, P 0,95, П1–П4 ✔ во всех 4 режимах, Δ ≥ MDE (pixgbm/honest_lofo.json). Выигрыш в основном от находок (Sf +0,0091…+0,0098). U и AU: П2 ✘.
- Адресно (site keep, S): object_poly A 37/153 (Гн 10), D 91/406 (Гн 87), +482 FP; finds A 255/624, B 365/803, C 261/436; Гнёздово finds ΔTP −1 (pixgbm/targeted_lofo.json).
- OOF для bot-6: pixgbm/oof/ (21 файл, READY_<участок>, READY_ALL, README.md).
- 200 и 212 цвет_2 измерены. Пик RSS 1,0 ГБ.

## 28.10 ~23:35 МСК — H и бюджет PIXGBM
- Вариант H (S вне Гнёздово, U в Гнёздово) **отклонён**: H vs stack в режимах drop не проходит П2 (P = 0.737 / 0.827); в drop H уступает S. Подробности — `PIXGBM_REPORT.md` («Дополнение … вариант H») и `pixgbm/honest_lofo_H.json`. Код: ветка bot3-pixgbm, коммит 7cdace5, `union_name_re`; `pixgbm.diff` обновлён.
- Бюджет: `pixgbm/BUDGET_PIXGBM.md`. [изм] 7 регионов: 420 с при 1 потоке, 275 с при 2, пик 706 МБ, predict = 73 %. [оц] Площадка при 2 потоках: +258–431 с.

## 2026-10-29 00:20 МСК — PIXGBM: ускорение, срок фазы, пороги v2
- Коммит bot3-pixgbm 5deb94a (без push), pixgbm/pixgbm.diff обновлён. lleaves: predict ×6,75/×5,46, wall ×2,62/×1,83 (+пул procs2 ×2,97), кандидаты побайтно = lightgbm. BUDGET_PIXGBM.md §8.
- Тест малого бюджета: budget 1 с / нет моделей / исключение → выход побайтно = классике, GeoJSON валиден (bench2/tiny).
- Пороги S для models/all: pixgbm/thresholds_v2.toml (maggr_pix finds 0,93 poly 0,686; etgr_pix 0; terfl_pix 1,01) — пул LOFO OOF 0,347059 против 0,303975 без *_pix порогов; стек 0,336103 (thr/thr_v2b.json, thr_v2c.json).

## 2026-10-29 02:00 МСК — PIXGBM: состояние
- Готово: bot3-pixgbm 5deb94a (без push): lleaves, пул признаков, срок T0+300 с, лестница фолбэка; совместим с bot8-soft-ceiling 26e12ae. Пороги v2: pixgbm/thresholds_v2.toml (оценка 0,347 оптимистична — подбор и оценка на одном OOF).
- Мешает: ничего. Жду тестов бота 8 на слитом дереве и замера S+stack6 от бота 10.
- /tmp/pix (825 МБ, моя) — удаление не прошло автопроверку.
- Следующий шаг: следующая задача по bot-4/GAP.md.

## 2026-10-29 01:58 МСК — НАЧАЛО: geo_snap (GAP.md §5 на базе S)
- Задача: пересчитать FN-диагностику finds (кандидат в 2–5 м) и object_poly (рядом, но не прошла геометрия) на честном LOFO-выходе S; если разрыв жив — заявить ≤3 варианта (F1/F2/P1) и мерить honest-10 (4 режима, П1–П5). Ветка bot3-geosnap от bot3-pixgbm 5deb94a. Папка bot-3/geo_snap/.

## 2026-10-29 02:17 МСК — КОНЕЦ: geo_snap — варианты отклонены, S без изменений
- finds: разрыв 2–5 м под S в основном исчез (123 FN против 809 у A, изотропно, сдвиг TP ~6 см, потолок +0,0054) → F1/F2 не мерил.
- object_poly: 514 FN «рядом, не проходит геометрию» — слитые компоненты (304/369 содержат ≥5 GT). P1 (пики r=1 м) / P2 (r в фолде) против S: keep −0,0038 / −0,0007, drop +0,0004…+0,0021, П2 и П4 нигде → отклонены.
- Ветка bot3-geosnap ee20b87 (opt-in poly_mode="peaks", по умолчанию "pole"), diff geo_snap/geosnap.diff; отчёт geo_snap/GEO_SNAP_REPORT.md; числа honest_snap.json, diag_S.json, diag2_S.json.

## 2026-10-29 02:32 МСК — follow-up merge-теста bot-8 (потоки, зависимости, упаковка S)
- bot3-pixgbm **76f7317** (поверх 5deb94a, без push): потоки/пул по sched_getaffinity (фолбэк cpu_count, ≤8, env EXPDS_PIXGBM_THREADS→EXPDS_THREADS), procs=min(nt,4,ядра), _rate по procs, process_t0 без psutil задокументирован, model_dir от корня решения; tests/test_geo_pixgbm_threads.py. Полный tests/: 1 failed (coco, как в базе bot-8), 321 passed, 9 skipped. bot3-geosnap перебазирован → 7a5ee57 (было ee20b87), geo_snap/geosnap.diff обновлён.
- Нет в образе: psutil, lleaves, llvmlite, tl2cgen, treelite. Для S (lightgbm 4.7.0 в lock) и lleaves не нужны; tl2cgen+treelite — только backend=treelite. Патч: pixgbm/requirements_pixgbm.patch (не применён).
- Упаковка S: pixgbm/config_v2_S.toml (opt-in), модели poly.txt+finds.txt 4 233 917 Б, в zip +1 919 839 Б; патч упаковщика pixgbm/packer_pixgbm.patch (bot-8/build_submission.py:64–65, scripts/build_submission.py:40–45). Отчёт pixgbm/PACKAGING_S.md; pixgbm/pixgbm.diff (1ed01ed..76f7317) и pixgbm_76f7317.diff (5deb94a..76f7317).

## 2026-10-29 02:30 МСК — S включён в v2
- Бот 1: PIXGBM S включён; материалы (76f7317, default_S.toml, packer_pixgbm.patch) переданы боту 7.
- Бот 7 (07:13): 5deb94a/76f7317 взяты в bot7-candidate-v2 (bb0e0d8, ecb92ec), check_submission PASS; пороги *_pix — bot-9 thr_pix_full, выбор за ботом 10. Риск: пул до 4 процессов ≈+0,35 ГБ каждый, не мерено.

## 2026-10-29 07:13 МСК — новый пункт от бота 1
- Ограничения по токенам сняты; веса ImageNet/COCO/DINOv2 разрешены. Задача: проверить предобученный бэкбон для геофизики (LOFO), приёмка у бота 10.
## 2026-10-29 07:15 МСК — НАЧАЛО: backbone (предобученный бэкбон для геофизики против S)
- Задача от бота 1: честно проверить, даёт ли замороженный ImageNet/DINOv2-бэкбон прирост поверх PIXGBM S (honest-10, 4 режима, П1–П5); плюс замер RSS пула признаков procs 1/2/4. Папка bot-3/backbone/, ветка bot3-backbone от bot3-pixgbm.


### 29.10 07:45 МСК — backbone, промежуточно
- bbextract готов: backbone/bbfeats (92 растра, B1+B2, 1,5 ГБ). Выборка пикселей воспроизведена 1:1 (n/yp/yf всех 92 = pixgbm/feats; X побитно на 10 растрах) — backbone/bbextract.log. Пик RSS 1,17 ГБ (backbone/bbextract.rss). Бэкбон-прогон по выборкам: B1 38 с, B2 132 с (bbfeats/extract_log.json).
- Контроль B0 (переобучение без бэкбона): деревья = pixgbm/models (сравнение текстов моделей без строк параметров).
- Первый train B1 убит сторожем RSS (>1,95 ГБ, old/train_B1.rss) — переписан fit_bb без копий, перезапуск.

## 29.10 08:14 МСК — аудит геофизики (ветка bot3-audit, worktree audit/wt), промежуточно
- Насыщение (audit/out/sat_probe_ext.json, sat_probe_summary.txt): 644/3478 (18,5 %) эталонных finds MagGr — в ядрах насыщения ≤60 м² (в S они «нет данных»); 200 Судьбищи — 192/216 (89 %). object_poly: MagGr 34/252, EtGr/TerFl 0.
- Утверждение «на 211 серая без сигнала, цветные с сигналом» — артефакт notes/geo_probe_train.json: потолок серой считан против всех 409 finds участка (95 серой + 314 ключа цвет_2). По своим 95 меткам серая AUC 0,747 > цвет 0,698; цвет_2 PIXGBM и так обрабатывает отдельно.
- Коммиты: 459bed5 (ядра насыщения, порт аудита), 3d110b0 (признак sat), 80d50b8 (выбор копии hp_energy), e3c4f41 (метры на земле для EPSG:3857). Всё opt-in, по умолчанию поведение S.
- CRS трейна (audit/crs_inventory.json): все карты UTM (32636; MagGr 200/207 — 32637), EPSG:3857 нет.
- Запущено: SAT → SATPICK → PICK (extract/train/predict LOFO), затем вложенный LOFO калибровки (C0/C1/C2) и проверка побайтности CRS.

### backbone — итог (29.10.2026 ~08:58 МСК)
- Обучены B1 (ResNet18) и B2 (DINOv2-S), PCA-32 в фолде; пики RSS 1,26 и 1,27 ГБ. LOFO predict S+B1+B2 (пик 1,31 ГБ); контроль: S совпал с pixgbm/pix_lofo.geojson (26107, identical).
- honest-10 против S: B1 от −0,0036 до −0,0079, B2 от −0,0045 до −0,0170, P(>0)=0,000 во всех 4 режимах → **оба отклонены** (backbone/eval_bb.out, honest_bb.json).
- RSS procs 1/2/4 (204, 1,2 Мпикс): 492/1013/1505 МБ, ≈+0,25 ГБ на ребёнка (backbone/rss_procs.out).
- Код: bot3-backbone 49d0200 (локально), diff backbone/backbone.diff. Отчёт: backbone/BACKBONE_REPORT.md. /tmp/bb3 удалён.
