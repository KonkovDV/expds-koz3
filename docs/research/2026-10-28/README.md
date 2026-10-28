# Исследования 28.10.2026 — указатель

| Файл | Что внутри |
|---|---|
| [`AUDIT.md`](AUDIT.md) | аудит репозитория на HEAD 24bd916: баги, риски сдачи |
| [`COMMON.md`](COMMON.md) | общие правила для ботов 2–10 (приёмка, честное табло, запреты) |
| [`EXPERIMENTS_SHARED.md`](EXPERIMENTS_SHARED.md) | построчный журнал замеров дня (ведёт бот 10); сводка — [`docs/EXPERIMENTS.md`](../../EXPERIMENTS.md) |
| [`OSINT.md`](OSINT.md) | OSINT по конкурентам КОЗ №3 |
| [`OSINT_DEEP.md`](OSINT_DEEP.md) | сплошной проход по GitHub, Docker Hub и GHCR |
| [`OSINT_TEAMS.md`](OSINT_TEAMS.md) | соперники КОЗ1 → КОЗ2 → КОЗ3; таблицы — `osint_tables/` |
| [`COMPETITORS_DEEP.md`](COMPETITORS_DEEP.md) | разбор EhimenNathan/EXpedition, maratgaliulin/EXpedition и связанных репо: идеи P0–P2 |
| [`DETECTORS_SOTA.md`](DETECTORS_SOTA.md) | лучшие детекторы на конец 2026 г. под КОЗ №3: геофизика → спутник → декодинг лидара, Ae/Or, ансамбль, лицензии (только публичные источники) |
| `audit/*.diff` | три правки по итогам аудита (геофизика `dedupe_maps`, якоря курганов, запасной путь `limit_vertices`); `git apply --check` на 24bd916 проходит |
| [`speedup/SPEEDUP.md`](speedup/SPEEDUP.md), [`speedup/FINDINGS_EARLY.md`](speedup/FINDINGS_EARLY.md) | ускорение пайплайна: замеры и ранние находки |
| `audit/runs/` | скрипты и JSON симуляции слоёв (etgr, recall по слоям) |

Здесь только наши описания. Чужой код, клоны, слои образов и найденные в чужих репо ключи S3 в репозиторий не переносились.
