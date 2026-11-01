# bot7 — compare_candidates (gate tool)

## Контракт
- **Цель:** стандартный gate-скрипт `scripts/compare_candidates.py`: N папок mock e2e → таблица F1 ALL/li/sp_or/ae (global+layer), Δ vs v2, правило «слой не падает >0.01», счётчик li↔ae kurgany_* cross-TP; md+csv; опционально строка EXPERIMENTS.csv.
- **Гипотеза:** единый CPU-инструмент сравнения кандидатов (v2/v3/hotfix/v4/bot4…) без пересчёта matching вручную — reuse `expds.metric` scope из PR #2.
- **Критерий приёмки:** unit-тесты (PASS/FAIL правила, Δ, li↔ae, md/csv) зелёные; полный pytest suite без новых падений; DRAFT PR `bots/bot7-compare-candidates` → base `bots/bot7-honest-eval-layer` (stacked on PR #2); PC CPU ≤4 threads заявка в QUEUE с точной Win-командой.
- **Бюджет:** box CPU (тесты); PC CPU ≤4 thr, wall ~5–15 мин на сравнение уже готовых geojson (без inference).
- **Ветка:** `bots/bot7-compare-candidates` @ worktree `/workspace/koz3-wt/bot7-compare-candidates`, база `origin/bots/bot7-honest-eval-layer` (aa3694f).
- **Не делать:** main, LB, submissions.

## Статус — fix @69a9a85 (QUEUED re-run v2docker)
- **tip:** `69a9a85` (fix: optional `--out` + `env` column PROTOCOL §10). Было `b6f835d`.
- **PR (DRAFT):** https://github.com/KonkovDV/expds-koz3/pull/8 — base `bots/bot7-honest-eval-layer` (stacked on PR #2)
- **Файлы:** `scripts/compare_candidates.py`, `tests/test_bot7_compare_candidates.py`
- **Suite box @69a9a85:** tests/ **636 passed / 42 skipped**; centerhead/tests **50 passed / 1 skipped**; **итого 686 passed / 0 failed / 43 skipped**. Новых падений нет (+3 теста: resolve_out, env column/CLI, parse_candidate_spec).
- **QUEUE:** `pc/QUEUE.md` → `[bot7] QUEUED CPU — compare_candidates_v2docker fix @69a9a85`
- **lease:** `leases/pc_bot7_compare_candidates.lease` (QUEUED)
- **PC @b6f835d DONE 18:22:** `pc/results/bot7_compare_candidates_b6f835d/` (ENV=docker scoring); EXPERIMENTS appended. Filler `compare_candidates_v2docker` (V2=out_mock) **FAILED** — не было `--out`.

## Фикс 29.10 19:46 МСК
1. `--out` optional: default `%KOZ3_OUT%\compare` если `KOZ3_OUT` задан, иначе `./compare_out`; печатает `out=<resolved>`.
2. Колонка `env` (docker/host/unknown) в compare.md / compare.csv / EXPERIMENTS notes (`env=…`). CLI: `--env docker,host,…` или `label=path:env` / `path:env` (default unknown).
3. Env из `bot2_runtime_env` / annotated table: v2=docker, v3=docker, hotfix_B=host, v4_off=host, bot4_off=host, bot4_on=host.

## Исправленная команда (явный --out + --env)
```
set KOZ3_OUT=F:\Sloi_pc_jobs\bot7_compare_candidates_69a9a85
mkdir %KOZ3_OUT%
echo RUNNING > %KOZ3_OUT%\STATUS.txt
git fetch origin bots/bot7-compare-candidates && git checkout bots/bot7-compare-candidates && git reset --hard 69a9a85

set GT=F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\gt.geojson
if not exist %GT% set GT=F:\expds3\mock_val

set V2=C:\Sloi\pc_jobs\bot2_v2_base_docker_eb268d6\out_mock
if not exist %V2% set V2=F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v2
set V3=F:\Sloi_pc_jobs_archive\v3e2e\out_v3fix2\res_v3
set HOTFIX=F:\Sloi_pc_jobs_archive\bot2_v3_hotfix_1ec2d7c
set V4OFF=C:\Sloi\pc_jobs\bot5_v4_off_50f6f27
if not exist %V4OFF% set V4OFF=F:\Sloi_pc_jobs_archive\bot5_v4_off_50f6f27
set B4OFF=F:\Sloi_pc_jobs_archive\bot4_lo_mock_337a27c\out_off
set B4ON=F:\Sloi_pc_jobs_archive\bot4_lo_mock_337a27c\out_on

python scripts/compare_candidates.py ^
  --gt %GT% --v2 %V2% --out %KOZ3_OUT%\compare --workers 4 ^
  --labels v2,v3,hotfix_B,v4_off,bot4_off,bot4_on ^
  --env docker,docker,host,host,host,host ^
  --branch bots/bot7-compare-candidates --sha 69a9a85 ^
  --experiments-row --experiments-file %KOZ3_OUT%\compare\EXPERIMENTS_rows.csv ^
  %V2% %V3% %HOTFIX% %V4OFF% %B4OFF% %B4ON%

echo DONE >> %KOZ3_OUT%\STATUS.txt
```

## Следующий шаг
bot1: запустить CPU-прогон по исправленной команде; скопировать `%KOZ3_OUT%\compare\` → `pc/results/bot7_compare_candidates_69a9a85/` и дописать EXPERIMENTS.
