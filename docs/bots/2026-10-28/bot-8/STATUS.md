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
