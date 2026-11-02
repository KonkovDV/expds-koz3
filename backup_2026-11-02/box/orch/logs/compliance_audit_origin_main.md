# compliance audit origin/main @ 70dedf4 — 29.10 23:07 MSK
## 1. Следы EN (EhimenNathan/EXpedition/maratgaliulin) и geo_pixgbm
origin/main:.github/workflows/ci.yml:30:        # tests needing data, weights or expds/geo_pixgbm.py are skipped with a reason (-rs)
origin/main:.gitignore:74:expds/geo_pixgbm.py
origin/main:CHANGELOG.md:15:  - Исключения перечислены в `docs/bots/2026-10-29/README.md`: секреты, код EhimenNathan, данные, веса, > 5 МБ.
origin/main:CHANGELOG.md:26:- **`li_decode` удалён из `main` (29.10, bot 1).** Докстринг `expds/li_decode.py` ссылался на «recipe studied in EhimenNathan-EXpedition» — �
origin/main:analysis/2026-10-28/bots/bot-3/pixgbm/bench/bench.py:9:from expds import geo_pixgbm as G, geophys
origin/main:analysis/2026-10-28/bots/bot-3/pixgbm/bench/run_bench.sh:4:export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib
origin/main:analysis/2026-10-28/bots/bot-3/pixgbm/run.sh:7:export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib OMP_NUM_THREADS=2
origin/main:analysis/2026-10-28/bots/bot-8/bench/bench_tile.py:7:Архитектуры повторяют EhimenNathan-EXpedition:
origin/main:analysis/2026-10-28/bots/bot-8/bench/svf_bench.py:2:# (same op pattern as EhimenNathan koz3/relief.py:33-53: n_dir directions, k = 1..K step max(1,K//10)).
origin/main:analysis/2026-10-28/bots/bot-9/bench_t4/bench_tile.py:7:Архитектуры повторяют EhimenNathan-EXpedition:
origin/main:analysis/2026-10-29/bots/bot-3/audit/crs_inventory.py:1:"""Item 3: CRS / разрешение всех геофизических растров трейна (как их видит geo_pi
origin/main:analysis/2026-10-29/bots/bot-3/audit/crs_inventory.py:7:from expds import geo_pixgbm as G
origin/main:analysis/2026-10-29/bots/bot-3/audit/run_a.sh:7:export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
origin/main:analysis/2026-10-29/bots/bot-3/audit/sat_probe_ext.py:11:from expds import geo_pixgbm as G, geophys
origin/main:analysis/2026-10-29/bots/bot-3/backbone/rss_procs.py:9:from expds import geo_pixgbm as G, geophys
origin/main:analysis/2026-10-29/bots/bot-3/backbone/run.sh:8:export PYTHONPATH=$W/pylib:/workspace/koz3_shared/bot-3/ehimen/pylib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=/tmp
origin/main:analysis/2026-10-29/bots/bot-3/geo_snap/run.sh:6:export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib OMP_NUM_THREADS=2
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/bench2/bench2.py:1:"""Замер PIXGBM end-to-end через geo_pixgbm.region_feats (bot-3).
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/bench2/bench2.py:10:from expds import geo_pixgbm as G
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/bench2/exact.py:8:from expds import geo_pixgbm as G, geophys
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/bench2/mpx.py:6:from expds import geo_pixgbm as G, geophys
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/bench2/runw2.sh:4:export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/bench2/tiny/exc.py:4:from expds import geo_pixgbm as G, watchdog, writer
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/bench2/tiny/tiny.py:5:from expds import geo_pixgbm as G, writer, watchdog
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/config_v2_S.toml:13:model_dir = "models/pixgbm/all"   # относительный: от cwd, иначе от корня решения (geo_pixgbm.re
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/fastpred/micro.py:6:from expds import geo_pixgbm as G, geophys
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/fastpred/runw.sh:4:export PYTHONPATH=/workspace/koz3_shared/bot-3/ehimen/pylib
origin/main:analysis/2026-10-29/bots/bot-3/pixgbm/packtest/default_S.toml:242:model_dir = "models/pixgbm/all"   # относительный: от cwd, иначе от корня решения (geo_p
origin/main:analysis/2026-10-29/bots/bot-4/runs/review/pixgbm428/expds/geo_pixgbm.py:3:Чтение карты — по мотивам expds/geo_finds_gbm.py (ветка bot3-ehimen, своя реал
origin/main:analysis/2026-10-29/bots/bot-4/runs/review/pixgbm428/expds/pipeline.py:355:            from . import geo_pixgbm
origin/main:analysis/2026-10-29/bots/bot-4/runs/review/pixgbm428/expds/pipeline.py:358:            new = [f for f in geo_pixgbm.region_feats(reg, cfg, pix.get("model_dir", "models/pixgbm/all"),
origin/main:analysis/2026-10-29/bots/bot-4/runs/review/pixgbm428/expds/pipeline.py:367:                new = geo_pixgbm.dedupe_union(feats, new)
origin/main:analysis/2026-10-29/bots/bot-4/runs/review/v2eb268d6/expds/geo_pixgbm.py:3:Чтение карты — по мотивам expds/geo_finds_gbm.py (ветка bot3-ehimen, своя реал
origin/main:analysis/2026-10-29/bots/bot-4/runs/review/v2eb268d6/expds/geo_pixgbm.py:700:    psutil не обязателен (в requirements.lock образа его нет): ImportError → _T_IMPO
origin/main:analysis/2026-10-29/bots/bot-4/runs/review/v2eb268d6/expds/pipeline.py:352:            from . import geo_pixgbm
origin/main:analysis/2026-10-29/bots/bot-4/runs/review/v2eb268d6/expds/pipeline.py:360:            new = [f for f in geo_pixgbm.region_feats(reg, cfg, md, guard=guard, status=pst)
origin/main:analysis/2026-10-29/bots/bot-4/runs/review/v2eb268d6/expds/pipeline.py:368:                new = geo_pixgbm.dedupe_union(feats, new)
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_fix.py:519:#: bot-3 PIXGBM (expds/geo_pixgbm.py): класс → файл модели в model_dir и методы, на которых
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_fix.py:528:    """(model_dir, нужные .txt, файлы бэкенда) для [geophys.pixgbm] — умолчания как в pip
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_fix.py:562:                                 f"geo_pixgbm.load_models): {lost}")
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_fix.py:850:#: try/except с фолбэком: psutil — geo_pixgbm.process_t0 (иначе момент импорта), tl2cgen/lleaves
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_merged.py:363:#: bot-3 PIXGBM (expds/geo_pixgbm.py): класс → файл модели в model_dir и методы, на котор�
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_merged.py:372:    """(model_dir, нужные .txt, файлы бэкенда) для [geophys.pixgbm] — умолчания как в 
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_merged.py:406:                                 f"geo_pixgbm.load_models): {lost}")
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_merged.py:664:#: try/except с фолбэком: psutil — geo_pixgbm.process_t0 (иначе момент импорта), tl2cgen/llea
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_root.py:253:#: bot-3 PIXGBM (expds/geo_pixgbm.py): класс → файл модели в model_dir и методы, на которы�
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_root.py:262:    """(model_dir, нужные .txt, файлы бэкенда) для [geophys.pixgbm] — умолчания как в pi
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_root.py:296:                                 f"geo_pixgbm.load_models): {lost}")
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_root.py:651:#: try/except с фолбэком: psutil — geo_pixgbm.process_t0 (иначе момент импорта), tl2cgen/lleave
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_rootfix.py:254:#: bot-3 PIXGBM (expds/geo_pixgbm.py): класс → файл модели в model_dir и методы, на котор
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_rootfix.py:263:    """(model_dir, нужные .txt, файлы бэкенда) для [geophys.pixgbm] — умолчания как в
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_rootfix.py:297:                                 f"geo_pixgbm.load_models): {lost}")
origin/main:analysis/2026-10-29/bots/bot-4/runs/review_cnx/t/cs_rootfix.py:835:#: try/except с фолбэком: psutil — geo_pixgbm.process_t0 (иначе момент импорта), tl2cgen/lle
origin/main:analysis/2026-10-29/bots/bot-7/runs/gor/test_scene_check.py:7:DS='/workspace/koz3_shared/competitors/yakirill99-ds-expedition/vendor/organizers/data_structure.md'
origin/main:analysis/2026-10-29/bots/bot-7/work5/pc/tools/v2/expds/geo_pixgbm.py:3:Чтение карты — по мотивам expds/geo_finds_gbm.py (ветка bot3-ehimen, своя реализ
origin/main:analysis/2026-10-29/bots/bot-7/work5/pc/tools/v2/expds/pipeline.py:242:            from . import geo_pixgbm
origin/main:analysis/2026-10-29/bots/bot-7/work5/pc/tools/v2/expds/pipeline.py:260:                new = [f for f in geo_pixgbm.region_feats(reg, cfg, md, stop=stop) if f["class_name"] in cls]
origin/main:analysis/2026-10-29/bots/bot-7/work5/pc/tools/v2/expds/pipeline.py:270:                new = geo_pixgbm.dedupe_union(feats, new)
origin/main:analysis/2026-10-29/bots/bot-8/bench/calc_7_13.py:1:"""§7.13: full package total = base + EhimenNathan port net (b-cut, clean (a)) + PIXGBM, vs working ceiling 1200 s.
origin/main:analysis/2026-10-29/bots/bot-8/check_submission.py:538:#: bot-3 PIXGBM (expds/geo_pixgbm.py): класс → файл модели в model_dir и методы, на которых он с
## 2. Запрещённые зависимости (ultralytics/YOLO/SAM3/DINOv3)
## 3. Данные в дереве инференса (tif/las/laz/geojson/gpkg)
## 4. Обязательные файлы ПКК
ok  metadata.json
ok  LICENSE
ok  README.md
НЕТ ai_declaration.md
ok  requirements.lock
## 5. Размер inference/ + models/ (байт)
856293505
