# bot-4: крупные объекты (>5k м²)

## Контракт
- **Цель:** поднять recall городищ/фортификаций площадью >5 000 м² (LB: R 5–10k = 0, R >10k = 0,294; OOF misses: 113 below_τ, 82 no_cand, 6 shape, 3 consumed) без регрессии мелких слоёв.
- **Гипотеза:** area-aware ослабление калиброванных τ + защита крупного при склейке + ядра только у крупных gor + fort→gor + грубый hillfort закроют below_τ / no_cand / consumed, не повторяя отвергнутый bot-2 F1 (отдельные группы под sweep) и F2 (ядра глобально).
- **Критерий приёмки:** honest-10 ALL_F1 Δ>0 к базе ветки (cbd4804 / v4) и ни один слой (li / sp_or / ae) не падает >0,01; mock e2e не хуже v2 0,3125 более чем на 0,01 до честного замера. Принято только при Δ>0 без провала слоя >0,01 (PROTOCOL §5). База сравнения: mock v2 0,3125 / LB 0,41238.
- **Бюджет:** GPU e2e mock ≤15 мин (GTX 1650 4 ГБ, docker); CPU OOF post-tuning ≤2 ч (20 потоков, без GPU). Обучение не требуется.
- **Ветка:** `bots/bot4-large-objects` (база `cbd4804`).

## Что сделано (код)
Мастер `[large_objects] enabled = false` + независимые sub-flags (все false по умолчанию): `area_tau`, `protect_merge`, `shape_cores`, `fort_to_gor`, `coarse_hillfort`, **`dfine_ae_gor_area_tau` (Fix C)**. Ablation = enabled=true + один sub-flag.

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
| `5097c4a` / `fc5a0f9` | sub-flags + Docker ablation runner (C: JobDir/wt, torch preflight) |
| `17e35a5` | **Fix C:** `dfine_ae_gor_area_tau` + stash `box_area_m2` (ae+gorodishcha) |
| `e9d187a`…`fc18035` | Docker runner Fix C (Ae D-FINE оба плеча) + launch schtasks |

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
- `tests/test_bot4_large_objects.py` — 14/14 passed (koz3-venv; +Fix C / box_area / cluster_circles).
- `python -m py_compile` на large_objects/watchdog/pipeline — ok.
- **Не прогнано:** полный pytest репо, honest-10, LB.

## Ablation mock @ 5097c4a / fc5a0f9 (`pc/results/bot4_lo_ablate_5097c4a/`)

DONE 29.10.2026 **20:19 МСК** (docker `pantum/n7q:20261024`).

| variant | ΔALL vs off | verdict |
|---|---|---|
| on_area_tau | 0.0 | PASS |
| on_protect_merge | 0.0 | PASS |
| on_shape_cores | 0.0 | PASS |
| on_fort_to_gor | ≈−0.006 | FAIL |
| on_coarse_hillfort | ≈−0.006 | FAIL |
| on_all_except_coarse | ≈−0.006 | FAIL |

**Crash note:** вариант `all_except_coarse` иногда exit=−1 / wall≈100 с при уже записанном `result.geojson` (~82 с, written=247) — гонка docker/host kill vs запись; geojson валиден, score прошёл. Не баг Patch-LoToml.

**Вывод:** полный LO / coarse / fort_to_gor на mock вредны; area_tau/protect/shape без Ae D-FINE → Δ=0 (не стреляют). Переход к **Fix C** (координатор `ae_ortho_diagnosis` §3).

## Fix C — `dfine_ae_gor_area_tau` (ae + gorodishcha)

**Диагноз (координатор + bot6, проверено):** max-size filter / split_large OFF / min_area_px=3 — **не** причина. Ae fort-окно ≈64 м; D-FINE Ae 640@0.5 м ≈320 м. OOF Ae D-FINE крупные GT: **43/52 @τ=0.05**, **21/52 @τ=0.3**. gorodishcha калиброван τ=**1.01 (OFF)**. Полный LO mock −0.016 — не включать.

**Механика:** абсолютный `dfine_ae_gor_tau_large` (default **0.20**) когда одновременно:
- `class_name=gorodishcha`, `markup_type=ae`, `threshold_group∈{dfine_ae,ae_dfine}`,
- `box_area_m2 ≥ min_area_m2` (5000) — площадь **бокса** D-FINE, не круга r=4 (~50 м²).
- **li не трогаем.**

Stash: `boxes_to_ellipses` → `box_area_m2`; `cluster_circles` сохраняет **max** по кластеру.

### OOF τ schedule (ae_oof_analysis / per_threshold, all-ae gor)

| τ | gor TP | gor FP | gor R | large>5k same-cls | gor>5k |
|---|---:|---:|---:|---|---|
| 0.05 | 9 | 19231 | 0.47 | 43/52 | 8/16 |
| 0.10 | 9 | 6305 | 0.47 | 43/52 | 8/16 |
| **0.20** | **1** | **397** | 0.05 | 32/52 | 1/16 |
| 0.30 | 0 | 32 | 0.00 | 21/52 | 0/16 |

Выбран **τ_large=0.20**: единственная ступень с ненулевым gor TP при умеренном FP (area-gate ещё срежет FP). 0.10 слишком шумный; 0.30 — 0 TP gor. Ожидание LB **+0.005…0.02** поверх Fix A (Ae D-FINE в сдаче).

### QUEUE

Docker mock+honest-10: оба плеча `[dfine_aeor] ae ON`; ON += Fix C only. Runner `pc/runners/run_bot4_fix_c_docker.ps1`, launch `launch_bot4_fix_c_schtasks.ps1`, JobDir `C:\Sloi\pc_jobs\bot4_fix_c`, tip **`62f7fb1`**.

## Следующий пункт BACKLOG

Дождаться Fix C mock+honest Δ; при PASS — кандидат в стек с Fix A (не полный LO). OOF post-tuning tau_relax — SUPERSEDED Fix C.
