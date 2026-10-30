# CLEANROOM_READLOG — bot 5 (W4), исполнитель clean-room PIXGBM
Правило: ни один из файлов expds/geo_pixgbm.py (и копии), expds/geo_finds_gbm.py, tests/test_geo_pixgbm_*, ветки bots/bot3-*, репо EhimenNathan/EN, docs/research/* (разбор EN), models/pixgbm/*, w4pix/out* — не открывался. Поиски (rg/grep) запускаются с исключением этих путей.
Время MSK | файл/URL | зачем

29.10 22:27 | /workspace/koz3/orch/BRIEF_COMMON.md | правила
29.10 22:27 | /workspace/koz3/orch/PROTOCOL.md | правила
29.10 22:27 | /workspace/koz3/orch/bot-W4/STATUS.md | свой статус
29.10 22:27 | /workspace/koz3/orch/LEASES.json | аренды
29.10 22:28 | git show bbebfe3 --stat; bbebfe3:docs/PIXGBM_SPEC_CLEANROOM.md | спецификация

## Чтения после спецификации (все — не-PIXGBM код интерфейса/данных; тело geo_pixgbm.py не открывалось)
- orch/BRIEF_COMMON.md, orch/PROTOCOL.md, orch/LEASES.json, orch/bot-W4/STATUS.md (правила)
- w4clean (ветка от bots/w4-gate 22a32c3): expds/pipeline.py стр. 33–140 (TimeGuard), 195–206 (load_config), 279–289 (region_crs), 300–400 (место вызова [geophys.pixgbm] — только код вызова)
- configs/default.toml стр. 46–166 (секции thresholds; видны комментарии со значениями порогов старой линии — НЕ используются), 270–310 ([geophys.pixgbm] ключи), 352–360, 405
- scripts/check_submission.py (rg-попадания по pix/model_dir и стр. 255–320: PIX_CLASS_MODEL, проверка poly.txt/finds.txt)
- expds/geophys.py (полностью — классическая геофизика, не PIXGBM), expds/catalog.py 20–48, expds/gt.py 55–180 + список функций, expds/crs.py 50–62 + список функций, expds/raster.py 143–218 + список функций, expds/accel.py 1–30 + список функций
- materials/compute_metrics.py 60–140, 263–290, 621–900 (официальная метрика)
- expds/metric.py 73–75, 215–230, 295–310, 538–640 + список функций; expds/naming.py site_family
- scripts/scoreboard.py 1–80, 205–272; scripts/geo_predict.py 1–60; expds/candidates.py family_folds (rg); scripts/rerank_oof.py family_of_group (попутный вывод rg); scripts/oof_full.py (rg, без попаданий)
- expds/post.py — только строки rg по threshold_group
- requirements*.txt (grep lightgbm/skimage), s3_keys.tsv, local_ctx/s3_train.env (только имена переменных; значения не выводились)
- Данные: ds/gt/gt_train_v2.geojson (структура свойств), S3 train регионы 200–217 → /workspace/koz3/w4geo/train (без .las/Li/La), листинги каталогов, метаданные растров (rasterio)
- rg-поиски выполнялись с исключениями запрещённых путей; `rg -l geo_pixgbm` — только имена файлов, содержимое не выводилось
- Внешние URL: нет (документация numpy/scipy/lightgbm — из знаний, без открытия сторонних реализаций)

## Дополнение (29.10.2026, после реализации модуля; время МСК)
Прочитано дополнительно (ни один файл не из запретного списка):
- scripts/geo_predict.py стр. 35–125 (apply_override, цикл по участкам, запись) — точка вызова geophys_feats.
- expds/raster.py стр. 68–105 (mask_to_polygons, circle_3857) — общие утилиты.
- expds/pipeline.py стр. 33–80 (TimeGuard), 279–392 (region_crs, geophys_feats, блок вызова pixgbm — только вызов, без тела geo_pixgbm), список `^def` по файлу.
- expds/geophys.py стр. 53–100, 169–200, 601–650 (маски, select_rasters, dedupe_points).
- expds/post.py merge_within_markup стр. 99–111.
- expds/catalog.py стр. 1–80 и список определений (Region, scan_root).
- expds/gt.py read_annotation (сигнатура), build_gt (сигнатура).
- expds/metric.py стр. 97–111, 215–300, 555–700 (build_curves, honest_counts, score_counts, paired_bootstrap_counts).
- expds/constants.py CLASS_WEIGHTS; expds/naming.py site_family.
- configs/v3_on.toml: заголовок; раздел [geophys.pixgbm] — только ключи/значения через tomllib (enabled, model_dir, classes, replace, union_name_re, maps, t_pix=0.5, backend, budget_s, guard_reserve_s, procs); комментарии стр. 22–36, 155–169, 284–291 (видны пороги старой линии *_pix — НЕ использованы, мои пороги подбираются заново).
- /workspace/koz3/ds/gt/gt_train_v2.geojson (эталон, только подсчёт по участкам/классам).
- docs/PREREG_w4_pix_clean.md (свой), docs/PIXGBM_SPEC_CLEANROOM.md (повторно), orch/LEASES.json.
- Метаданные растров и имена файлов разметки 200–217 (свой /workspace/koz3/w4geo/survey.py, /tmp/map.py).
- Внешние URL: не открывались.
Свои новые файлы: expds/pix_clean.py, scripts/pix_clean_train.py, scripts/pix_clean_eval.py, tests/test_pix_clean_contract.py.

## 29.10, после перезагрузки коробки (MSK)
- scripts/make_mock_val.py (заголовок), tests/test_platform_robustness.py стр. ~255–300 (интерфейс mock_val/таймгарда).
- scripts/geo_predict.py main (повторно: post.finalize/merge_within_markup/writer, threshold_group/source).
- expds/pipeline.py: свой рефакторинг блока слияния → pix_merge (только вызов модуля, тело geo_pixgbm не открывалось).
- expds/metric.py CLI (sweep-groups, apply_group_thresholds).
- /workspace/koz3/orch/lease.sh (использование), lease.sh list, BRIEF_COMMON.md (строки про Kaggle/аренды).
- Пример kernel-metadata.json ядра W3 (только формат полей).
- /workspace/koz3/w4geo/fetch.py (свой).
- Golden v2 (только числа от бота 1): synth finds≈222, 210 object_poly≈10; classic 72 / 0.
- НЕ открывались: expds/geo_pixgbm.py (в т.ч. в make_ds.sh — файл удаляется по имени без чтения), models/pixgbm/*, w4pix/*, analysis/**, bots/bot3-*, EN-репозиторий, ds/bundle/expds_code, ds/snap/expds.
- Внешние URL: не открывались.

## 29.10 (after reboot, cont.) — organizer materials (allowed by bot 1)
- /workspace/koz3/org_baselines/ : README, 01_baseline_maggr_anomalies (extracted to w4geo/org_01_*.py), 02_baseline_etgr_anomalies (w4geo/org_02_*.py), utils.py
- orch/REVIEW_ORG_GEO_BASELINES.md (bots/orch-protocol b5d3485); orch/QUEUE.json (GEO-ORG1, GEO-ORG2 items)
- /workspace/koz3/local_ctx/COMPETITION_RULES.md, local_ctx/treeC__Sloi.csv (file listing only)
- ТР PDF: https://upgreat.one/upload/iblock/477/477560736561f633fc4e1bd983635da5/PROEKT_Tekhnicheskii_reglament_KOZ_3_Ekspeditsiya_DS.pdf (п. 8.5.3, 13.6)
- bots/w4-gate: docs/AI_DECLARATION.md (journal), docs/THIRD_PARTY.md (head) — non-PIXGBM docs
- repo (interface only): configs/default.toml [geophys.pixgbm] section, expds/pipeline.py geophys_feats/pix_merge
- public: matplotlib jet colormap segment points (as reproduced in expds/pix_clean._jet_lut)
- GEO-ORG3 (материалы организаторов/координатора, разрешены): origin/bots/orch-protocol af5d080 REVIEW_ORG_GEO_BASELINES.md §6; orch/geo_org_repro/run_etgr.py, make_c.py; /workspace/koz3/org_repro/02_baseline_etgr_anomalies.py (ячейки 1,2,4,6,14,17,20,22); /workspace/koz3/org_baselines/notebooks/utils.py (load_polys, rasterize_polys, largest_part, is_hit, platform_counts)
- интерфейс (не PIXGBM): expds/catalog.py (boundaries), expds/geophys.py load_mask, scripts/pix_clean_train.py gt_files_for, scripts/pix_clean_oof.py compose
- GEO-ORG3 сверка: /workspace/koz3/org_repro/out/R_etgr.json (итоги координатора)

## 30.10.2026 — pytest-триаж, DOC-1, ai_declaration (bot 5 / W4)
- origin/bots/orch-protocol:orch/QUEUE.json (записи DOC-1, P-v4-integrate) — координатор
- scripts/build_submission.py (w4-pix-clean, main 70dedf4, bots/bot5-candidate-v4 ed73c5e) — интерфейс сборки, не PIXGBM
- configs/default.toml секция [geophys.pixgbm] (те же три ветки)
- expds/pipeline.py строки 348–400 (свой код W4 и место вызова)
- tests/test_bot7_candidate_v2.py, tests/test_bot7_v3_flags.py (270–347), tests/test_open_hooks.py (35–40), tests/test_bot8_check_submission.py (строки с models/pixgbm), tests/test_audit_fixes.py, tests/test_tta.py, tests/test_gpu_safeguards.py, tests/test_coco_roundtrip.py (импорты), v4: tests/conftest.py (строки 105–134 о заглушках весов) — тесты, не тело geo_pixgbm.py
- docs/ai_declaration.md на origin/bots/w0-docs e78f286; свой docs/ai_declaration.md на bots/w4-gate (1264cf2)
- Список имён файлов (ls expds, rg -l "models/pixgbm" с исключением запрещённых путей) — содержимое запрещённых файлов не открывалось; тесты, импортирующие geo_pixgbm (test_bot7_candidate_v2, test_bot5_v4_base_patches), только запускались
- GitHub API: создан draft PR https://github.com/KonkovDV/expds-koz3/pull/12

### 30.10 01:10–01:40 MSK (модальностный гейт, кэш 200)
- orch/bot-W4/runs/independent_regate_ORG3.json (origin/bots/orch-protocol 38a60a2) — материал координатора (результаты, не код PIXGBM)
- /workspace/koz3/regate5/regate.log, /workspace/koz3/regate5/run_regate.sh — материал координатора (лог/скрипт ре-гейта)
- expds/accel.py (box_mean, стр. 319–336) — не-PIXGBM код; изменён: поблочный по строкам, побайтно тот же результат
- scripts/pix_clean_oof.py (cmd_classic), scripts/pix_clean_eval.py, expds/metric.py (paired_bootstrap_counts, honest_counts) — свой/не-PIXGBM код
- сравнение cmp /workspace/koz3/regate5/oof/classic/200_Судьбищи_2.pkl (кэш классики, не PIXGBM) — побайтно совпал
- 30.10 01:30–01:55: scripts/build_submission_git.py, scripts/check_submission.py (запуск), expds/pipeline.py (geophys-блок, output_floors), configs/default.toml — не-PIXGBM код; архив v2-clean распакован в /tmp/v2c (expds без geo_pixgbm.py) только для смоука
- 30.10 01:45–02:00: orch/lease.sh, orch/lease.py (grep), orch/pool_kaggle.sh, orch/enqueue.py, orch/watcher.py (grep), orch/PROTOCOL.md (grep), expds/post.py (output_floors), expds/pix_org_et.py (своё, thr), scripts/pix_clean_eval.py, scripts/pix_modality_gate.py (свои), scripts/check_submission.py (usage), configs/v2_clean.toml, docs/ai_declaration.md (w0-docs); вывод eval/C18et_orget_s0, C19etonly_orget_s0. PIXGBM-запретные пути не открывались.
- 30.10 08:2x MSK: w4geo/kaggle/out_v2clean_check/*; /tmp/v2c2 (own archive 26ad1ba); w4clean docker/Dockerfile, requirements.txt, requirements.lock; orch/enqueue.py, orch/lease.sh, orch/LEASES.json; w1/d3/runs/ae_final_prereg_wsa1.json, v8_prereg.json, ae_final_prereg.log (head); orch/bot-W1/work/d3_gate.py (docstring, prereg(), evaluate()); W1 inputs only hashed/consumed by d3_gate (oof_dfine_ae_flipwbf/noflip, oof_dfine_or_noflip, stack_arch, gt_train_v2, table.pkl, compute_metrics.py). No EN/pixgbm/bot3 files.

## 30.10 09:05–09:40 MSK — DOC-1 guard + push (bot5-w4)
- EN denylist: sha256 only, NO viewing — `git log --all --raw` path-limited to EN names (geo_pixgbm*.py, geo_finds_gbm*.py, pixgbm*.py, */models/pixgbm/*), then `git cat-file blob <oid> | sha256sum`; box files via `find … -exec sha256sum`. Contents never printed/opened.
- Read (own/allowed): w4clean expds/pipeline.py (my edit area), configs/v2_clean.toml, configs/default.toml (comments), scripts/build_submission*.py, scripts/check_archive_clean.py; w4/repo configs/*.toml (grep pixgbm keys), scripts/build_submission_git.py (include roots); orch/bot-W1/work/d3_gate.py (W1 V7 regate); orch/enqueue.py -h, orch/pool_kaggle.sh header.
- Archive diff old 15d59e9f vs new 545ef32a: only via `diff` of unzipped configs/default.toml + expds/pipeline.py (both shipped, clean files).
- rg -l (filenames only) for EN patterns over w4geo scripts/kaggle/orch-bot-W4 before commit.
- NOT opened: expds/geo_pixgbm.py (any branch), pix_legacy target, models/pixgbm/*, analysis/**, bot3 branches, EN repo.
