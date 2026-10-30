# bot7 — offline eval as on LB (scoring.log breakdown)

**Дата:** 29.10.2026 ~20:35 МСК  
**Бот:** bot7  
**Ветка:** `bots/bot7-lb-breakdown` (stacked on `bots/bot7-compare-candidates` @ `5e43510`)  
**Worktree:** `/workspace/koz3-wt/bot7-lb-breakdown`  
**PR base:** `bots/bot7-compare-candidates` (DRAFT; never main)

## Контракт (PROTOCOL §1)

| | |
|---|---|
| **Цель** | Офлайн-табло печатает ту же разбивку, что LB `scoring.log` (P/R по типам данных, recall по корзинам площади, soft recall, FP ratio, center deviation), diff-able построчно с `reports/lb/scoring_v2_eb268d6_0.41238.log`; machine-readable json/csv; прогон v2 mock + honest-10 на ПК → side-by-side vs LB. |
| **Гипотеза** | Числа в `F1Calculator.auxiliary` (`materials/compute_metrics.py`) — те же, что печатает площадка; нужен только formatter + флаг, без смены final_score. |
| **Критерий приёмки** | (1) `--lb-breakdown` на scoreboard/compare не меняет default stdout/JSON; (2) synthetic tests на soft_recall / fp_ratio / center_dev / type P/R / bucket edges зелёные; (3) текст лога парсится `lb_counts.parse` и содержит 5 типов + 6 корзин; (4) QUEUE CPU ≤4 thr: mock+honest-10 → `side_by_side.md`; (5) suite 0 новых FAIL. |
| **Бюджет** | box CPU тесты ≤15 мин; ПК CPU ≤4 thr, mock+honest metric matching ≤40 мин (без GPU inference — reuse готовых geojson). |
| **Не делать** | main, LB upload, submissions. |

## FIRST: test_bot8_soft_ceiling

| | |
|---|---|
| **Где падал** | Оба бот7-ветки: `bots/bot7-compare-candidates`, `bots/bot7-honest-eval-layer` (один и тот же тест). Под полной suite нагрузкой, не в одиночном прогоне. |
| **Причина** | Не scope bot7. `TimeGuard.left()` и `run_elapsed()` делают **два** вызова `clock()`; под нагрузкой suite Δ семплов ~1.02e-6 > assert `1e-6`. `expds/pipeline.py` TimeGuard == `origin/main` (diff 0). |
| **Фикс** | Допуск `1e-3` с комментарием. |
| **SHA** | compare-candidates **`5e43510`**; honest-eval-layer **`42b4bf8`**. |

## MAIN: что сделано

- `scripts/lb_breakdown.py` — extract/format/write `.log`/`.json`/`.csv`; `side_by_side()`.
- `scripts/lb_breakdown_table.py` — CLI LB vs mock vs honest-10.
- `scripts/scoreboard.py` — `--lb-breakdown DIR` / `--lb-archive` (default output unchanged).
- `scripts/compare_candidates.py` — `--lb-breakdown` / `--lb-log` → `out/lb_breakdown/`.
- `tests/test_lb_breakdown.py` — 8 synthetic cases (bucket edges, soft recall hit/miss, fp_ratio, center_dev, type map, format headers, write+table, LB log needles).

## Определения метрик (из compute_metrics.py)

| Метрика | Формула | Источник |
|---|---|---|
| soft_recall | `soft_hits / (n_gt + 1e-6)`; soft_hit = pred∩gt area>0 (точка GT → круг r_tol) | `_count_soft_recall` |
| center_deviation | mean ‖c_pred − c_gt‖₂ по TP-парам (м, UTM) | `_calculate_center_deviation` |
| fp_ratio | `Σfp / (Σtp + 1e-6)` | auxiliary |
| type precision | `tp / (tp + 0.5·fp + ε)` | `_build_by_data_type` |
| type recall | `tp / (tp + fn + ε)` | то же |
| size recall | `tp/gt` по корзине площади GT-полигона; точки вне корзин | `_build_size_categories` |
| SIZE_BINS | `(0,100], (100,500], (500,1000], (1000,5000], (5000,10000], (10000,+∞)` м² | Config.SIZE_BINS |
| layer→type | li/la→лидар; ae/german_ae→аэрофото; sp_or/spor→спутник; or→ортофото; maggr/etgr/geogr/terfl→геофизика | DATA_TYPE_GROUPS |

### Неопределённости

1. Подписи корзин «≤ N м²» для корзин 2+ — фактически `(prev, N]`; как в логе площадки.
2. Пустая корзина (`gt=0`) → `recall=None` в JSON; в тексте печатаем `0.000000` (LB при gt>0/tp=0 тоже 0).
3. Сноска геофизики в LB: `(maggr, etgr, terfl)` — в коде ещё `geogr`; в логе не упомянут.
4. Имя архива / время инференса offline нет — только если переданы `--archive` / `--inference-time`.
5. Классы без TP-пар в блоке «Контур по классам» опускаются (как LB 0.41238); `finds_points` всегда «— (контур не применим)».
6. Abs. числа mock/honest ≠ публичный LB (другая выборка) — side-by-side сравнивает **доли/ранги**, не сырые vs LB.

## PC QUEUE

См. `pc/QUEUE.md` запись `[bot7] QUEUED CPU — lb_breakdown v2 mock+honest-10`.  
Lease: `leases/pc_bot7_lb_breakdown.lease`.  
Критерий: exit 0; `scoring_mock.log` + `scoring_honest10.log` + `side_by_side.md`; STATUS по exit code.

## Suite / SHA

- soft_ceiling fix: `5e43510` (compare-candidates), `42b4bf8` (honest-eval-layer)
- lb-breakdown tip: _(заполняется после push)_
- tests/test_lb_breakdown.py: 8 passed
- related (soft_ceiling + lb_* + layer scope): 40 passed, 1 skipped
