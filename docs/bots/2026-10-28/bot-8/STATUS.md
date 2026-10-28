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
