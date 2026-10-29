# bot-4: крупные объекты (>5k м²)

## Контракт
- **Цель:** поднять recall городищ/фортификаций площадью >5 000 м² (LB: R 5–10k = 0, R >10k = 0,294; OOF misses: 113 below_τ, 82 no_cand, 6 shape, 3 consumed) без регрессии мелких слоёв.
- **Гипотеза:** area-aware ослабление калиброванных τ + защита крупного при склейке + ядра только у крупных gor + fort→gor + грубый hillfort закроют below_τ / no_cand / consumed, не повторяя отвергнутый bot-2 F1 (отдельные группы под sweep) и F2 (ядра глобально).
- **Критерий приёмки:** honest-10 ALL_F1 Δ>0 к базе ветки (cbd4804 / v4) и ни один слой (li / sp_or / ae) не падает >0,01; mock e2e не хуже v2 0,3125 более чем на 0,01 до честного замера. Принято только при Δ>0 без провала слоя >0,01 (PROTOCOL §5). База сравнения: mock v2 0,3125 / LB 0,41238.
- **Бюджет:** GPU e2e mock ≤15 мин (GTX 1650 4 ГБ, docker); CPU OOF post-tuning ≤2 ч (20 потоков, без GPU). Обучение не требуется.
- **Ветка:** `bots/bot4-large-objects` (база `cbd4804`).

## Что сделано (код)
Флаг `[large_objects] enabled = false` (по умолчанию OFF → поведение = cbd4804).

| Коммит | Суть |
|---|---|
| `fd27ff3` | flag+config: модуль + конфиг (enabled=false) |
| `f2cd571` | area τ: area-aware пороги в finish/safe_finish |
| `2853dd9` | NMS/dedup: protect-склейка крупных |
| `831365f` | candidates: fort→gor, coarse hillfort, ядра крупных gor |
| `77a61d0` | report: reports/bot4_large_objects.md + unit-тесты (6) |
| `08f83d7` | report: bot-4: отчёт reports/bot4_large_objects.md + unit-тесты |

## Тесты
- `tests/test_bot4_large_objects.py` — 6/6 passed (локальный venv: shapely+pytest; box/run_tests.sh ещё не готов).
- `python -m py_compile` на large_objects/watchdog/pipeline — ok.
- **Не прогнано:** полный pytest репо, e2e mock, honest-10, LB. Заявки — в `pc/QUEUE.md`.

## Следующий пункт BACKLOG
«4: крупные объекты → постобработка слияния по OOF» (BACKLOG.md, Боты 2–8). После принятия флага — подбор `tau_relax` / порогов `gor_from_fort`/`gor_coarse_dem` по OOF на CPU ПК.

## Риски / untested
- Грубый hillfort на реальных LAS не гонялся (только unit).
- Interact с `gor_dense` / `orgate` / `rerank_arch` при совместном enabled не измерен.
- Пороги новых групп стартовые (0,20 / 0,30) — ждут OOF-тюнинг.
- `_aeor_finish` не трогали (D-FINE Ae/Or отдельно) — крупные ae-gor через fort_to_gor только в основной цепочке.
