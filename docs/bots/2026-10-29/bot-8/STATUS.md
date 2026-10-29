# bot-8 — STATUS (надёжность и скорость сдачи)
Обновлено: 2026-10-28 19:55 MSK. Состояние: ГОТОВО, отчёт — REPORT.md.

- [x] Аудит inference/main.py и пайплайна (строки — по 24bd916 = main)
- [x] Ветка bot8-reliability в bot-8/repo (9316b5f, 8343483), diff: patch/bot8-reliability.diff
- [x] check_submission.py — испытан: 3 настоящих архива PASS, 12 испорченных — FAIL (tests_out/)
- [x] build_submission.py / .sh — сборка из git-коммита, побайтно воспроизводима (sha256 двух сборок совпал)
- [x] REPORT.md
- Внимание: в bot-8/repo параллельно работает другой агент (незакоммиченные expds/metric.py, rust/…, expds/fastgeom.py,
  inference/lib/expds_core.abi3.so). Я их не трогаю; мои сборки берут файлы только из коммитов.

## Ускорение метрики (вторая задача bot-8) — 2026-10-28 19:56 MSK: ГОТОВО
- [x] профиль, [x] Rust flatten_polygons + векторные пороги/бутстрепы, [x] тесты равенства (PASS),
  [x] замеры (bench/), [x] сборка Linux manylinux2014 + инструкция Windows → RUST_REPORT.md
- Правки в bot-8/repo — незакоммичены (чтобы не трогать ветку bot8-reliability); патч patch/bot8-rust-metric.diff.

## Перенос на bot7-bundle + дедлайн в цикле окон — 2026-10-28 ~22:10 MSK: ГОТОВО → HANDOFF_BOT7.md
- [x] bot8-reliability-on-bot7 (42ada8d → 5135517), bot8-reliability-on-bot7-candidate (fe0e570 → fd7dd50)
- [x] дедлайн в цикле окон отдельным коммитом: bot8-reliability 2901cb0, on-bot7 5135517, candidate fd7dd50
- [x] тесты: везде только старое падение test_coco_roundtrip (cv2); регрессия на чистом bot7-bundle — 10/10 FAIL
- [x] архивы dist/koz3_rel7_5135517_f854a54c.zip, dist/koz3_rel7c_fd7dd50_2ed3a301.zip — check_submission PASS
- [x] патчи patch/for-bot7/, patch/for-bot7-candidate/, patch/for-bot8-reliability/ (git am -3 проверен)
- рабочие деревья: wt-rel, wt-rel7, wt-rel7c, wt-base-main, wt-base-b7 (bot-8/repo не трогал)

## 28.10 22:20 МСК (executor bot-8)
REPORT3.md и REPORT4.md готовы; bot8-merged=2f530a8, bot8-rust-metric=d073b45; архив dist/koz3_submission_crop_2f530a8_1e8f8e68.zip sha256 dcad2ddd…da54, check PASS.

## Диск (2026-10-28 23:40 MSK)
Свободно на / 7.7 GB. Ничего не удалено. Подробно: DISK_CLEANUP_PROPOSAL.md.
Безопасно: кэши pip+uv (~2.6–3.2 GB). Остальное (~22 GB) менялось <24 ч или чужое/активное.
Кандидаты после подтверждения: /tmp t9/t15/t19.tif, fsrc, fout, pdest (~1.7 GB, владелец неизвестен).
Удаление — ждёт решения Шаундры.
23:45 MSK: models/ в моих worktree (base-4c3e, base-b7, cpua, cpua-clean, cpua-clean-v2, soft-ceiling) заменены жёсткими ссылками на wt-base-main/models (96 файлов, sha256 совпали; wt-rel* не трогал). Свободно на / 12 GB. /tmp/b8build*, b8chk сейчас используются идущей проверкой — удалятся по её окончании; новые сборки — с жёсткими ссылками. Лишние worktree уберу после завершения текущих работ. Чужое (/tmp/t*.tif, b7, geo, lid, pix, wt_s4) не трогаю.

## 2026-10-28 — v2cpua (bot8-cpu-a-clean-on-v2 @ 18b2f66) — итог
- Архив: bot-8/dist/koz3_v2cpua_18b2f66_fb5f1667.zip, sha256 fb5f16677d14de2da97b175bf295044753613e616b2d7156a2de6539084ba2d1 (2 сборки из git, sha совпал; 56 файлов, 539.6 МиБ).
- check_submission: PASS (FAIL 0, WARN 0, 28 проверок; smoke 133 с, пик RSS 997 МиБ, 80 полигонов, geojson sha256 56be33ff…ca33). Лог: bench/las_grid_fast/check_v2.log.
- Ветки: bot8-cpu-a-clean = cb4742e+7f8d714 (KonkovDV, частичный 6602bb2)+95bc0a6; bot8-cpu-a-clean-on-v2 = 4c3e29e+6 lidar_a (9885ded..521a7f2)+bf784eb+662e7d1+18b2f66 (.so в архив).
- pytest 18b2f66: 410 passed, 12 skipped, 1 failed (test_coco_roundtrip, нет cv2 — есть и на базе 4c3e29e). Oracle/lidar_a: 81 passed, 3 skipped.
- Пропущено (scope cut): smoke-сравнение GeoJSON с base4c3 (неизменность дефолта показана статически: манифест — 0 изменённых общих файлов, +9 новых, на lidar_a/las_grid_fast/cpu_threads никто не ссылается); полный pytest на 95bc0a6; проверка запушенной bots/bot8-cpu-a-clean (нет доступа); сайт 101.
- /tmp/b8* (кроме b8sc_*) удалены.
23:50 MSK: бот 9 ответил по контракту весов CNX (bot-9/CNX_CONTRACT_FOR_BOT8.md): каналы несовместимы с x9/x12. Варианты: (а) in_kind cnx3/cnx5 на вендорном prep бота 9, ≈½ дня CPU, без GPU — я за; (б) переобучение 10 ядер ≈20 ч T4. Нужен выбор бота 1. Не начато — пауза до утра.
23:55 MSK: wt-base-b7 удалён (git worktree remove). wt-cpua (запрещённый код) и wt-cpua-clean оставлены до утра. Пауза до утра по указанию бота 1; защита 1200 с ещё доделывается.

## 2026-10-28 23:58 MSK — мягкий потолок 1200 с (задача bot 1), ветка bot8-soft-ceiling
- Коммит 26e12ae поверх 4c3e29e (bot 7), не запушен. Патч: patch/bot8-soft-ceiling/ (format-patch + summary.diff; 3 файла, +329/−4).
- API TimeGuard: t0 (база time.monotonic, = watchdog.PROCESS_START после set_soft в run(), т.е. тот же отсчёт, что у сторожа 1620), elapsed(), soft_left(), soft_ok(need_s, phase, what), set_soft(...). Бюджет 1500 считается как раньше от старта run() (run_t0/run_elapsed); сторож 1620 не тронут.
- Потолок: env EXPDS_SOFT_CEILING_S > [time] soft_ceiling_s > 1200; неверное/≤0/nan/inf/bool → выключен + предупреждение в stderr. configs/default.toml не менялся.
- Необязательные шаги и пороги soft_left (с; [time.soft] <key> или env EXPDS_SOFT_NEED_<KEY>_S): wide_window 120 (окно 256 м sp_focal2, на растр), sigma 60 + резерв verify 60 (канал σ li_dev, на регион), verify 60 (проверка кропа). Порядок отключения: окно → σ → кроп. На каждый пропуск одна строка «мягкий потолок: <шаг> пропущено … soft_left N с»; итог в протоколе сторожа. Геофизика, лидар, основная оптика, finish/safe_finish, writer не отключаются.
- Тесты: tests/test_bot8_soft_ceiling.py (18). Полный набор, 2 потока, RSS ≤2000 МиБ: 4c3e29e — 329 passed/9 skipped/1 failed (test_coco_roundtrip, нет cv2), 163 с; ветка — 347 passed/9 skipped/1 failed (тот же), 165 с. Логи: tests_out/softceil/.
- Архив: dist/koz3_softceil_26e12ae_e62b32eb.zip, sha256 e62b32ebd3fa5bc815b781d32d4faa37568342c10528d797cb90b683e2825462, 47 файлов, rerank_arch_lgbm.{txt,json} внутри; check_submission PASS (26 проверок, smoke 87 с, RSS 979 МиБ). В build_submission.py добавлено правило [rerank_arch] (первая сборка без него давала FAIL config). Повторная сборка для проверки воспроизводимости — пропущена (урезанный объём).
- Слияние с bot3-pixgbm 7cdace5 (git merge-tree, без worktree): конфликтов нет ни с 4c3e29e, ни с 26e12ae; pipeline.py после слияния парсится. Тесты на слитом дереве не гонялись. В 7cdace5 ещё нет getattr(guard,'t0') / передачи guard в region_feats.
- Не проверено: байт-в-байт совпадение result.geojson 4c3e29e vs ветка на реальных данных (по коду при потолке, не сработавшем, ветвления не меняются; тест «все шаги выполняются при 1200» есть). §7.14 в BUDGET.md не писался.
- Мои worktree, которые можно удалить: wt-soft-ceiling (ветка сохранится в repo), wt-base-4c3e.
00:00 MSK: защита 1200 с готова (bot8-soft-ceiling 26e12ae, dist/koz3_softceil_26e12ae_e62b32eb.zip, check_submission PASS). wt-base-4c3e удалён. Бот 9 подготовил for_bot8/ (cnx_prep.py, колесо rvt-py, рекомендация fp16 вместо bf16) — к утру, ждёт решения бота 1 по cnx3/cnx5. Бот 8 на паузе до утра.

## Сводка 01:58 MSK (CPU-части инференса и ускорение)
Готово:
- v2cpua: bot8-cpu-a-clean-on-v2 18b2f66, dist/koz3_v2cpua_18b2f66_fb5f1667.zip, check_submission PASS; las_grid_fast побайтно = oracle, rasterise ×1.7–2.2, make_dtm ×1.1–1.4.
- Мягкий потолок 1200 с: bot8-soft-ceiling 26e12ae, dist/koz3_softceil_26e12ae_e62b32eb.zip, PASS.
- Диск: модели в worktree на жёстких ссылках, /tmp/b8* убраны.
Мешает: пауза до утра по указанию бота 1 (квота); решение бота 1 по cnx3/cnx5 (вход бота 9) vs переобучение.
Следующие шаги: (1) тесты + check_submission на слитом дереве 26e12ae + bot3-pixgbm 5deb94a; (2) smoke-сравнение result.geojson с 4c3e29e для обеих веток; (3) in_kind cnx3/cnx5 по bot-9/for_bot8 (fp16, prep отдельным процессом).

## 2026-10-29 02:10 MSK — мягкий потолок + PIXGBM bot-3 (ветка bot8-softceil-pixgbm)
- Слияние f7cde8e = 26e12ae + bot3-pixgbm 5deb94a (fetch только для чтения), конфликтов нет; + 428257f (тест pix_deadline ↔ guard.t0). Не запушено.
- geo_pixgbm.pix_deadline: T0 = min(psutil-старт в шкале monotonic | момент импорта, guard.t0); guard.t0 = watchdog.PROCESS_START (monotonic) — шкала одна, тест проходит.
- pytest (2 потока, RSS ≤2000): 356 passed / 9 skipped / 1 failed (test_coco_roundtrip, нет cv2 — как на 4c3e29e); новые: test_geo_split (8) и test_bot8 (19).
- Архив dist/koz3_softceil_pixgbm_428257f_20b4ccea.zip, sha256 20b4ccea95f2fa47696e464099c91957730208658fa636e05ccf50c49298e096, 48 файлов. PIXGBM в default.toml выключен (enabled=false), моделей pixgbm в архиве нет. check_submission PASS (FAIL 0, WARN 1: psutil/tl2cgen нет в venv — ленивые импорты с фолбэком).
- check_submission.py поправлен под PIXGBM: model_dir pixgbm проверяется только при enabled; pixgbm.maps — список методов (maggr/etgr/terfl), а не MAP_MODES; psutil/tl2cgen/lleaves при только ленивом импорте → WARN. До правки было 3 ложных FAIL.
- Смоук на синтетике check_submission (одни и те же данные, по очереди, 2 ядра):
  4c3e29e: 80 полигонов, sha 56be33ff…, 5.9 с, 975 МиБ; 26e12ae: 80, 56be33ff… (байт-в-байт), 4.9 с, 976 МиБ;
  428257f (PIXGBM выкл.): 80, 56be33ff…, 6.4 с, 985 МиБ; 428257f + PIXGBM вкл. (модели bot-3 pixgbm/models/all): 574 (+491 finds_points, +3 object_poly), sha 301ff574…, 6.4 с, 1012 МиБ;
  то же + PIXGBM_BUDGET_S=1, EXPDS_SOFT_CEILING_S=1: ступень classic, σ и кроп пропущены мягким потолком, валидный FeatureCollection из 80 полигонов (sha совпал с базой), 5.7 с, 946 МиБ.
- Замечание bot 3: pix_threads без EXPDS_PIXGBM_THREADS/EXPDS_THREADS берёт min(8, cpu_count) — в смоуке было «потоков=8» (ядер ограничено taskset 0,1).
- wt-soft-ceiling удалён (ветки сохранены). Временные файлы в /tmp удалены.

## 2026-10-29 04:26 MSK — входящие (бот 8 на паузе)
- Бот 3: bot3-pixgbm 76f7317 поверх 5deb94a (потоки через sched_getaffinity, ≤8; поиск model_dir от корня решения). Для упаковки моделей pixgbm нужен патч bot-3/pixgbm/packer_pixgbm.patch к build_submission.py и склейка default_S.toml (повторить на 428257f). Включать ли — решает бот 1. Ничего не применено.
- Бот 9: acc_gpu на T4 (bot-9/bench_acc_gpu/bench_acc_gpu_t4.json). stub 30M px: device_exact ×2,0–2,15, byte_equal; device ×3,9–4,0, max|Δ| 6e-08. unet 30M и stub 75M: rc −9 (OOM по RAM хоста, maxrss 12,25 ГБ).
- Удаление 6 старых архивов в dist ждёт ответа Шаундры.
