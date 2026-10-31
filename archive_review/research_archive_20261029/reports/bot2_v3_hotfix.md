# bot2 v3 hotfix

## PC e2e hotfix A/B @ 1ec2d7c — 2026-10-29 16:28 МСК (pc-job-runner)

Worktree `C:\Sloi\wt_bot2_1ec2d7c`; zip_work from out_v3fix2 + overlay pipeline/sp_dfine.

| вариант | ALL | li | sp_or | ae | ΔALL vs v2 | wall |
|---|---:|---:|---:|---:|---:|---:|
| v2 base | 0.312509 | 0.336983 | 0.242915 | 0.095556 | 0 | — |
| **A** drop_groups=[] enabled=true | 0.312873 | 0.336983 | 0.245064 | 0.095556 | +0.000364 | ~120 с |
| **B** enabled=false | 0.312873 | 0.336983 | 0.245064 | 0.095556 | +0.000364 | ~100 с |

**Вывод:** A ≡ B (δ=0 по всем слоям). Оба ≈ v2 (±ε). SpOr восстановлен (~0.245 vs v2 0.243). Вариант B = путь v4. Артефакты: `C:\Sloi\pc_jobs\bot2_v3_hotfix_1ec2d7c\`.

## Почему A ≡ B (bot2 @ fb66c89, 16:41 МСК)

Exact δ=0 по всем слоям **несовместим** с попаданием D-FINE FP в GeoJSON (синтетический тест: +13 pred → precision↓).
Вердикт: **D-FINE не попал в scored output** на том прогоне (load/thr/UNAVAILABLE/exception).
Не «скорер игнорирует 191 м FP». Класс `kurgany_povrezhdennye` в sp_or scored set — да.
Дамп stages на PC A2 (`EXPDS_DFINE_DUMP` + `stages.json`) подтвердит стадию обнуления.
См. `reports/bot2_dfine_localization.md`.
