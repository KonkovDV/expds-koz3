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

## 2026-10-30 iteration
Read: bot-9 requirements-dfine.txt; scripts/build_submission.py; w4-gate expds/sp_dfine.py, pipeline.py, post.py; orch/bot-W1/work/d3_gate.py; QUEUE.json; MASTER_PLAN (grep); spec bbebfe3; 56cdf39:docs/REPRO_V2.md; eb268d6 greps of expds/post.py, geophys, pipeline, watchdog (banned files excluded); PyPI wheel URLs (transformers 5.21.0, tokenizers 0.23.3, regex 2026.9.29, typer 0.27.3, rich 15.0.0, markdown-it-py 4.2.0, mdurl 0.1.2, pygments 2.21.0, shellingham 1.5.4, annotated-doc 0.0.5); Kaggle output pull koz3-dfine-spor-full-rest (via pool, dmitrijkv).
INCIDENT (incidental exposure): `git grep -n psutil 22a32c3 -- expds` printed expds/geo_pixgbm.py lines 741–747 (psutil thread-limit lines). Not used for any code. From now on always exclude banned paths with :(exclude) pathspecs.

## 30.10.2026 15:56–16:20 MSK (iteration: v3 Kaggle check / w4-gate commits / compgate)
- orch/QUEUE.json (P-W4-compgate), docs MASTER_PLAN (M_rp line), REVIEW_SOTA (§2.1 public composition, #12, D-FINE fallback warning).
- orch/bot-W3/prereg/C5B_JOINT_PREREG.md, orch/bot-W3/code/make_joint.py (M_rp clean definition, disk()).
- orch/bot-W1/work/d3_gate.py (honest/Stack/TableEngine/merge), wt/w1/expds/metric.py (ClassCurve, score_counts, paired_bootstrap_counts, threshold_group).
- orch/bot-W4/work/v3_gate.py / v3_gate_lib.py (own code).
- PyPI wheel downloads (pypi.org / files.pythonhosted.org): transformers 5.21.0 stack for cp313 (list in w4geo/kaggle/SHA256SUMS_wheels_v3.txt).
- origin/bots/w4-gate 6 foreign commits: ONLY `git log` metadata (author, time, message) + `--stat`/name counts. Contents NOT opened.
- NOT read: analysis/** (M_rp source c4e18a9 on bots/w5-rerank — banned, not opened), all banned paths from the brief.

## 01.11.2026 19:54– MSK (bot5-W4)
Read: orch/bot-W4/STATUS.md, work/j2_regate.py/log, docs/PREREG_w4_regate_W1J2.md (ours); w4geo/kaggle/{koz3-w4-v3onchk, out_v3on_check, ds-v3on-check/check_submission.py §DFINE (lines 740–860, 1116–1215)}; orch/pool_kaggle.sh, orch/lease.sh, COORD_LOG greps; w0/noground/full_cls.py (S3 access pattern grep, secrets masked); s3_keys.tsv (103 rows); wt/w5 kaggle/2026-10-28/dfine_spor/full_rest kernel-metadata + script S3 grep (secrets masked); origin/bots/bot4-li-post log + b61e3f6 stat + EXPERIMENTS.csv grep (bot4, not banned). URL: HEAD https://s3.ru-7.storage.selcloud.ru/train-expds-3/<103 SpOr> (403).
Kaggle: kernels status/output aaaa0000004/koz3-w4-v3onchk (v2), dmitrijkv/koz3-w4-ref103 (own); metadata only (no content) for elenakonkova/koz3-w4-li-post-gate and koz3-s3-index.

## 01.11 20:21–… MSK — v4 (w5_cloudfeat)
- git: `git show --stat 4293458` и `--stat` f76f06a/1b457f3/7d92e91 (только имена файлов); `git ls-tree --name-only origin/main` с фильтром запретных путей (имена). Содержимое analysis/** из main НЕ открывалось (все файлы RR1s-cf/cloudfeat в main лежат под analysis/2026-10-30/bots/bot-W1/cf/ → запрет).
- W1, бокс (собственный код W1, не EN; предварительно grep -l на pixgbm/EhimenNathan/geo_finds_gbm/w4pix — 0 файлов): /workspace/koz3/w1/cf/kernel/koz3-w5-cloudfeat.py, kernel/kernel-metadata.json, nested2.py, common.py, build_x4.py, head feats.py/feats2.py/prep.py, ws_rr_results_w5cf.json (начало), w5data/work/X2_names.json, w1/cf/w5d/work/X4_names.json; листинги каталогов w1/cf, w5d. Случайно: `cat w5d/* | head` вывел строку stack_arch.geojson (данные) в файл вывода инструмента — не читался.
- Репо (свои/разрешённые): expds/watchdog.py (finish/safe_finish), expds/pipeline.py (run_region, output_floors, лидар-проход run, optical_feats), expds/post.py (apply_thresholds/threshold_fn), grep gor_gate.py, configs/v3_on.toml, configs/default.toml, scripts/build_submission_git.py -h; свои work/j2_regate.py, v3_gate_lib.py.

## 01.11 20:59–… MSK — v4 nocnx (модуль W1)
- До чтения: `git diff --stat f127db8 origin/bots/w1-cfdeploy` (9 файлов: docs/W1_W5CF_INFER_HANDOFF.md, expds/cloudfeat.py, expds/w5cf_infer.py, tests/test_cloudfeat.py, tests/test_w5cf_infer.py, tools/cloudfeat_parity.py, tools/rr1s_cf/train_final.py, tools/w5cf_infer_bench.py, tools/w5cf_infer_parity.py); grep диффа на pixgbm/EhimenNathan/geo_finds_gbm/w4pix/bot3/pc/tools/docs/research — совпадений по коду нет (только текстовое упоминание происхождения analysis/2026-10-30/w5_rerank в handoff, файлы оттуда не открывались).
- Прочитано: handoff, expds/w5cf_infer.py (начало, plug-in API), grep import/def по cloudfeat.py, w5cf_infer.py; tools/w5cf_infer_parity.py (шапка, аргументы); cherry-pick -x b62f59e в bots/w4-v3on.
- Бокс W1: /workspace/koz3/w1/cf/inferpar/run.sh, run.log (пути LAS), listing w1/cf, w1/cf/parity; /workspace/koz3/w2/gaudit/ds, w2/gaudit/las/190_Муравлево, w2/las (листинги). Свои: v4_fit.py → v4nc_fit.py, v4nc_archgate.py; scripts/build_submission_git.py (50–175), check_archive_clean.py (100–129), inference/main.py (1–60), kaggle/koz3-w4-v3onchk/run.py (grep pip).

## 01.11 21:52–… MSK — v3 regex (вендор под все ABI, V3_DEGRADED)
- Прочитано: expds/sp_dfine.py (import_dfine, _purge, region_feats), expds/gor_gate.py, expds/pipeline.py (run, _sp_dfine_phase, _with_scene_drel), expds/watchdog.py (86–96, ~181), scripts/check_submission.py (537–560, 815–852, 1096–1200), expds/w5_cloudfeat.py (55–110), expds/w5cf_infer.py (45–80), scripts/build_submission_git.py (61–67), tests/test_w5cf_infer.py, tests/test_w4_w5_cloudfeat.py; вендор v3/vendor/regex (_regex_core.py, __init__.py) и transformers/utils/auto_docstring.py, dependency_versions_check.py (grep regex); колёса regex 2026.9.29 cp311–cp314 с PyPI.
- Офиц. окружение: docker/Dockerfile (1–30), requirements.lock (grep), /workspace/koz3/docs_org: README.md, RULES_ORG_*.md, faq.txt, KOZ3_consult2_answers.txt, KOZ3_consult2_slides.txt, TR_KOZ3_approved.txt, templates/README.md (grep python/образ); git grep pantum по docs/ (с исключениями clean room).
- Kaggle: свои kaggle/koz3-w4-v3onchk/run.py, out_v3on_check3/*.log; orch/lease.sh list. Запрещённые пути не открывались.
- 22:25–22:45 MSK: w4geo/kaggle/out_v3onfix/check_submission.log (grep FAIL); expds/sp_dfine.py (свой код); requirements.lock; vendor transformers-5.21.0.dist-info/METADATA + dependency_versions_table.py (grep); METADATA колёс huggingface_hub 2.2.0/httpx2/httpcore2/anyio (PyPI); docs REGEX_VENDOR.md (свой)
- 23:40–23:43 MSK: out_v3onfix2/STATUS, out_v2clean_check4/*, koz3-w4-v2cleanchk2/run.py, koz3-w4-v3onchk-fix/run.py (свои); v2clean zip: unzip -l, configs/default.toml, metadata.json
