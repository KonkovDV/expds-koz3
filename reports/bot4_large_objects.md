# bot-4: крупные объекты (>5k м²)

## Контракт
- **Цель:** поднять recall городищ/фортификаций площадью >5 000 м² (LB: R 5–10k = 0, R >10k = 0,294; OOF misses: 113 below_τ, 82 no_cand, 6 shape, 3 consumed) без регрессии мелких слоёв.
- **Гипотеза:** area-aware ослабление калиброванных τ + защита крупного при склейке + ядра только у крупных gor + fort→gor + грубый hillfort закроют below_τ / no_cand / consumed, не повторяя отвергнутый bot-2 F1 (отдельные группы под sweep) и F2 (ядра глобально).
- **Критерий приёмки:** honest-10 ALL_F1 Δ>0 к базе ветки (cbd4804 / v4) и ни один слой (li / sp_or / ae) не падает >0,01; mock e2e не хуже v2 0,3125 более чем на 0,01 до честного замера. Принято только при Δ>0 без провала слоя >0,01 (PROTOCOL §5). База сравнения: mock v2 0,3125 / LB 0,41238.
- **Бюджет:** GPU e2e mock ≤15 мин (GTX 1650 4 ГБ, docker); CPU OOF post-tuning ≤2 ч (20 потоков, без GPU). Обучение не требуется.
- **Ветка:** `bots/bot4-large-objects` (база `cbd4804`).

## Что сделано (код)
Мастер `[large_objects] enabled = false` + независимые sub-flags (все false по умолчанию): `area_tau`, `protect_merge`, `shape_cores`, `fort_to_gor`, `coarse_hillfort`. Ablation = enabled=true + один sub-flag.

| Коммит | Суть |
|---|---|
| `fd27ff3` | flag+config: модуль + конфиг (enabled=false) |
| `f2cd571` | area τ: area-aware пороги в finish/safe_finish |
| `2853dd9` | NMS/dedup: protect-склейка крупных |
| `831365f` | candidates: fort→gor, coarse hillfort, ядра крупных gor |
| `77a61d0` | report: reports/bot4_large_objects.md + unit-тесты (6) |
| `337a27c` | docs: таблица коммитов в отчёте |
| `f1d33ab` | sub-flags независимо (все OFF) + fort overlap skip + stricter coarse |
| `8f74d26` | unit-тесты sub-flags / fort skip (9) |
| `f08bb5f` | отчёт mock 337a27c FAIL + план ablation |

## Mock e2e @ 337a27c (артефакты `pc/results/bot4_lo_mock_337a27c/`)
- OFF vs v2: ALL Δ **+0.00036** (ok).
- ON vs OFF: ALL **0.31287 → 0.29658**, Δ **−0.01629** — gate FAILED (|Δ|>0.01).
- По слоям: li Δ **−0.01538**; ae Δ **0**; sp_or Δ **0**.
- Класс: только `li/gorodishcha` n 9→12; TP 1→1, FP 8→11 (+3), FN 0→0; F1 0.333→0.267.
- Логи ON: coarse hillfort = 3, fort→gor = 1. Net +3 gor = coarse FP (TP без прироста). fort→gor не драйвер Δ; shape_cores сжал площади без сдвига TP/FP; area_τ/protect без вклада в counts.
- **Причина:** `coarse_hillfort` (+3 FP li/gorodishcha).
- Вердикт EXPERIMENTS: negative (−0.01629).

## Фикс после FAIL
- Sub-flags все false; каждый механизм toggleable отдельно.
- fort_to_gor: keep-original-label; skip при пересечении с gor; `fort_to_gor_min_conf=0.5`; scale 0.8; τ группы 0.35.
- coarse: max_per_dem=1, min_enclosure=0.35, min_rel_height=1.8, flat/scarp ужесточены; τ группы 0.55.
- Следующий шаг: mock e2e ablation per sub-flag (QUEUE).

## Тесты
- `tests/test_bot4_large_objects.py` — 9/9 passed (koz3-venv).
- `python -m py_compile` на large_objects/watchdog/pipeline — ok.
- **Не прогнано:** полный pytest репо, honest-10, LB.

## Следующий пункт BACKLOG
«4: крупные объекты → постобработка слияния по OOF» (BACKLOG.md). После принятия флага — подбор `tau_relax` / порогов по OOF на CPU ПК. Пока — ablation mock по sub-flags.

## Риски / untested
- Грубый hillfort на реальных LAS не гонялся после ужесточения параметров.
- Interact с `gor_dense` / `orgate` / `rerank_arch` при совместном enabled не измерен.
- `_aeor_finish` не трогали (D-FINE Ae/Or отдельно).
