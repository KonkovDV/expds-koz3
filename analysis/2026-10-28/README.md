# Код анализа 28.10.2026

- `bots/bot-N/` — скрипты ботов 2–10 (`.py .sh .ps1 .rs .toml`), структура каталогов сохранена (`work`→`runs`). Многие скрипты запускаются через `exec(open(common).read())` или как ячейки, поэтому линтер видит «неопределённые» имена. Отчёты и результаты — `docs/bots/2026-10-28/`.
- `rerank/` — реранкер кандидатов: `prep.py`, `feats*.py`, `nested*.py` (вложенный LOFam), `sub.py`, отчёт `REPORT_rerank.md`, табло JSON. Стек (`stack_arch.geojson`, 1,1 ГБ) и `gt_train_v2.geojson` не коммитятся; их SHA256 — в `rerank/manifest.json`.
- `oof59_pc/` — честный замер li_r18_hard2_59 на ПК (`fuse_r18.py`, `pipe_r18.cmd`, `results_r18.txt`, `report.json`): −0,025 к базе.
- `speedup/` — профилирование прогона (`prof_run.py`, сводка `prof/base1.*`).

Rust-метрика (`rust/expds_core/src/geo.rs`, `lib.rs`, `expds/fastgeom.py`, тесты) лежит патчем: `patches/2026-10-28/bot-8/bot8-rust-metric.diff`. Каталог `target/` не коммитится.
