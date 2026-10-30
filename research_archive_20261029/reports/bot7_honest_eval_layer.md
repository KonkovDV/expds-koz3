# bot7 — honest-eval по слою (per-layer matching)

## Контракт задачи

| Поле | Значение |
|------|----------|
| **Цель** | Добавить flag-gated режим `EXPDS_MATCH_SCOPE=layer` / `--match-scope layer`: жадное сопоставление pred↔gt только внутри одного нормализованного `markup_type` (микро-F1). Default OFF (`global`) — поведение байт-идентично старому. |
| **Гипотеза** | Cross-layer matching **не** объясняет разрыв honest (~+0.011) ↔ LB `0.41238` (eb268d6). Layer-режим скорее снизит или оставит honest; разрыв = train-OOF vs public val. |
| **Критерий приёмки** | (1) pytest `tests/test_bot7_layer_scope.py` + `test_bot8_scope_default` + synthetic site-scope green; (2) default/global байт-идентичен; (3) на ПК: `scoreboard --honest 10` без флага = старое число; с `--match-scope layer` — отчёт числа vs LB 0.41238; Δ(layer−global) ≤ 0 (не поднимает). |
| **Бюджет** | box: ≤30 мин CPU (тесты+PR); ПК: 1 прогон scoreboard honest-10 ×2 (global+layer) ~10–30 мин CPU, без GPU/обучения. |
| **Ветка** | `bots/bot7-honest-eval-layer` (worktree `/workspace/koz3-wt/bot7-honest-eval-layer`) |

## BACKLOG (не стартовать)

Из `/home/box/shared/koz3/BACKLOG.md`:
- §бот5 п.5: OOF-инференс всех моделей на honest-10 для калибровки порогов по слою.
- §бот7/следующее п.1: honest-10 scoreboard по слою (после бота 7) для v2/v3/v4.

## Дизайн (кратко)

### 1. Текущие пути матчинга

- `expds/metric.py`: `site_key` → `match_scope` (`global`\|`site`) → `_collect`/`_collect_impl` (+`sites_out`) → `build_curves` / `build_curves_reference` → `match_collected` (фильтр same-key).
- `scripts/scoreboard.py`: вызывает `build_curves` без scope → env `EXPDS_MATCH_SCOPE` или default global.
- Официальный эталон: `materials/compute_metrics.py` `F1Calculator.calculate_f1` = global pool.

### 2. Новый флаг (default OFF)

- `layer_key(props)`: aliases la→li, german_ae→ae, spor→sp_or; maggr/etgr/terfl/geogr отдельно; пусто→`?`.
- `match_scope` принимает `layer`; предупреждение в stderr как у site.
- `_collect*` получают `key_fn` (site_key\|layer_key); `match_collected` без изменений (same-key).
- scoreboard: `--match-scope {global,site,layer}` → пишет `EXPDS_MATCH_SCOPE`.
- Микро-F1: сумма TP/FP/FN по слоям.

### 3. LB / гипотеза

- LB final_score eb268d6 ≈ **0.4123779189**.
- Honest на train-OOF выше public; layer matching не должен закрывать gap вверх.

### 4. Тест-план

- unit: aliases, cross-layer block, only-foreign, missing `?`, micro-sum, default==global, env switch.
- regression: `test_bot8_scope_default`, synthetic `test_bot8_site_scope`.
- ПК: honest-10 global vs layer на стеке eb268d6 / predictions.

### 5. Вопрос организаторам (RU)

«Официальная метрика площадки при сопоставлении pred↔gt учитывает `markup_type` (слой данных), или все объекты одного class_name матчатся в одном глобальном пуле независимо от слоя?»

### 6. Статус реализации

- **Ветка:** `bots/bot7-honest-eval-layer`
- **SHA:** `f161c93` (fix NameError per_site in build_curves_reference @ 44f6f9f; regress test @ f161c93)
- **PR:** https://github.com/KonkovDV/expds-koz3/pull/2 (draft)
- **Локальные ворота (box):** `tests/test_bot7_layer_scope.py` + `test_bot8_scope_default` + synthetic site → **12+6 passed** (`/workspace/koz3-venv`).
- **Очередь ПК:** см. `pc/QUEUE.md` секция `## bot7 — honest-eval по слою`.

### Изменённые файлы

- `expds/metric.py` — `layer_key`, scope=layer, `key_fn` в collect/build_curves
- `scripts/scoreboard.py` — `--match-scope`
- `tests/test_bot7_layer_scope.py` — unit
- `docs/MATCH_SCOPE_LAYER.md` — краткая шпаргалка


## Очередь (текст)

См. `/home/box/shared/koz3/pc/QUEUE.md` → `## bot7 — honest-eval по слою` (ветка `bots/bot7-honest-eval-layer`, sha `ae4c816`, pytest + scoreboard honest-10 global vs `--match-scope layer`, критерий: old unchanged; layer ≤ global; vs LB 0.41238).

## Итог box (29.10 ~16:10 МСК)

- PR draft: https://github.com/KonkovDV/expds-koz3/pull/2
- head: `ae4c816`
- файлы: `expds/metric.py`, `scripts/scoreboard.py`, `tests/test_bot7_layer_scope.py`, `docs/MATCH_SCOPE_LAYER.md`
- pytest box: 12+6 passed (`koz3-venv`)
- ПК-прогон в очереди; EXPERIMENTS.csv — строка pending


## Hotfix NameError `per_site` (до Δ-разложения)

- **Причина (1 строка):** в `d1a98ae` `build_curves_reference` заменил `per_site = …` на `keyed`, но оставил `if per_site` → NameError @ metric.py:745.
- **Фикс:** `if keyed` (фильтр site|layer); default/global по-прежнему без фильтра (= main).
- **Tip:** `f161c93` · full suite box: **678 passed / 0 failed / 44 skipped** (main@70dedf4: 666/0/43; +тесты слоя/diff).
- Δ-разложение **resumed** → см. §«Δ −0.021 разложение» ниже (tip `aa3694f`).


## Δ −0.021 разложение (контракт follow-up)

| Поле | Значение |
|------|----------|
| **Цель** | Разложить Δ(layer−global)=**−0.020621** (honest-10) по слою и классу; перечислить lost cross-type matches; ответить, оптимистичен ли global vs LB **0.41238**. |
| **Гипотезы** | **H1:** LB matching ≡ global pool (тип — вторичная атрибуция) → global **не** оптимистичен относительно правила LB. **H2:** Δ −0.021 = ценность cross-layer TP на train-OOF (+ greedy reassignment). **H3:** абсолютный gap 0.297↔0.412 = другой GT/split, не matching. |
| **Критерий** | (1)–(3) **выполнены** 2026-10-29 17:15 МСК (PC dump + box-анализ). Статус follow-up: **DONE**. |
| **Бюджет** | box: анализ+script ≤30 мин; ПК: 1 dump ~10–30 мин CPU, без GPU. |
| **Ветка / tip** | `bots/bot7-honest-eval-layer` @ `aa3694f` (база f161c93) |
| **Не делать** | merge/main; LB submit; менять сдачу. |

### Сопоставимые метрики

| Метрика | Значение | Сопоставимо с |
|---------|----------|---------------|
| honest-10 **global** (ae4c816) | **0.297118** | layer / base 70dedf4 global (byte-identical) |
| honest-10 **layer** | **0.276497** | global на том же стеке |
| Δ(layer−global) | **−0.020621** | — |
| 34 новых уч. global→layer | 0.294770 → 0.256976 (**−0.0378**) | тот же honest-10 |
| ae/optical note | **+0.0061** (layer↑) | greedy reassignment при запрете cross |
| LB final_score eb268d6 | **0.4123779189** | **не** напрямую с honest-10 (другой GT/public val) |
| LB per-type P/R (вторичные) | li 0.516/0.504; ae 0.091/0.013; sat 0.456/0.255; ortho 0.368/0.140; geo 0.769/0.656 | только secondary; matching = global |

### Выводы сейчас (без PC dump пар)

1. **Matching LB ≡ global.** `materials/compute_metrics.py`: жадный матч в общем пуле по `class_name`; затем TP/FN атрибутируются по `markup_type` **GT**, FP — по pred (`data_type_group`). Запрета same-layer нет → **`--match-scope layer` строже LB**, не ближе.
2. **Global не оптимистичен vs правило LB.** Он *совпадает* с правилом площадки. Оптимистичность «global vs LB score» путать нельзя: 0.297 vs 0.412 — **разный эталон/сплит** (train-OOF honest-10 vs public val), не артефакт matching.
3. **Δ −0.021** — цена запрета cross-layer TP на том же стеке (+ возможный выигрыш подгрупп через переназначение; ae +0.0061 уже в EXPERIMENTS). Точная матрица pred_layer→gt_layer и Δ по классам — **только после PC dump**.
4. **Диагностика готова:** `scripts/diff_match_scope.py` (+ `--honest 10` @ `aa3694f`) пишет `summary.json`, `cross_matches.csv`, `delta_by_class.csv`, `tp_by_class_layers.csv`.

### Очередь ПК (Δ dump)

См. `/home/box/shared/koz3/pc/QUEUE.md` → `## bot7 — Δ decomposition dump` · sha `aa3694f` · lease `leases/pc_bot7_diff_scope.lease` · out `C:\Sloi\pc_jobs\bot7_diff_scope_aa3694f`.

**Что запускает bot/executor «1»:** checkout `aa3694f` → pytest bot7 diff/layer → `diff_match_scope.py --work F:/expds3/work --train F:/2/train-expds-3 --honest 10 --out …` → сверить honest ALL с 0.297118/0.276497 → записать n_cross + top cross_by_* + delta_by_class в EXPERIMENTS.

### Файлы (tip aa3694f)

- `expds/metric.py` — layer_key, scope=layer, `build_curves_with_pairs`
- `scripts/scoreboard.py` — `--match-scope`
- `scripts/diff_match_scope.py` — dump + `--honest K`
- `tests/test_bot7_layer_scope.py`, `tests/test_bot7_diff_match_scope.py`
- `docs/MATCH_SCOPE_LAYER.md`

### Box gates

- tip `aa3694f`: full suite **628+50 = 678 passed / 0 failed / 44 skipped** (как f161c93).
- PR draft: https://github.com/KonkovDV/expds-koz3/pull/2


## Δ −0.021 — финальные выводы (после PC dump)

**Статус контракта follow-up:** **DONE** (критерии: gate числа воспроизведены; dump пар есть; EXPERIMENTS+QUEUE+lease закрыты; рекомендации сформулированы).

**Артефакты:** `/home/box/shared/koz3/pc/results/bot7_diff_scope_aa3694f/` · tip `aa3694f` · PC STATUS DONE 16:58 МСК · анализ box 17:15 МСК.

### Ключевые числа (honest-10, воспроизведены)

| | global | layer | Δ |
|--|-------:|------:|--:|
| ALL | **0.297118** | **0.276497** | **−0.020621** |
| 34 новых уч. | 0.294770 | 0.256976 | −0.037794 |
| optical | 0.128215 | 0.134270 | **+0.006054** |
| TP (tau=0 pairs) | 4991 | 4674 | net **−317** |
| cross-layer TP в global | **647** | — | rematch within-layer ≈ **330** (= 647−317) |

Проверка rematch по классам: kurgany_povr 366 cross / net−217 → rematch 149; kurgany_ts 232/61→171; object_poly 39/32→7; gorodishcha 10/7→3; сумма rematch **330**. При запрете cross greedy часто находит same-layer пару на тот же GT (или другой pred) — поэтому net TP-падение < числа cross.

### Матрица cross pred_layer → gt_layer

| pred → gt | n | доля |
|-----------|--:|-----:|
| **li → ae** | **367** | 56.7% |
| **ae → li** | **183** | 28.3% |
| sp_or → ae | 32 | 4.9% |
| etgr → maggr | 15 | 2.3% |
| sp_or → li | 14 | 2.2% |
| li → sp_or | 11 | 1.7% |
| terfl → maggr | 8 | 1.2% |
| прочие geo↔geo / li→or | 17 | 2.6% |

**li↔ae = 550 / 647 (85%).** Остальное: sp_or↔{ae,li} (~57) и geo↔geo (39).

### По классу (cross / net ΔTP)

| class | cross | net ΔTP (g−l) | rematch |
|-------|------:|--------------:|--------:|
| **kurgany_povrezhdennye** | **366** | **−217** | 149 |
| **kurgany_tselye** | **232** | **−61** | 171 |
| object_poly | 39 | −32 | 7 |
| gorodishcha | 10 | −7 | 3 |
| finds_points / fortifikatsii / arkhitektury | 0 | 0 | 0 |

Топ class×направление: `kurgany_povrezhdennye li→ae` **274**; `kurgany_tselye ae→li` **143**; `kurgany_tselye li→ae` **89**; `kurgany_povrezhdennye ae→li` **40**.

### IoU (quality) и «дубликаты vs чужой слой»

- quality cross: median **1.00**, mean 0.92; **74.3% ≥ 0.9**; min 0.51 (нет пар <0.5).
- same `region_name`: **90.4%** (li↔ae: 94.7%).
- high-IoU≥0.9 ∧ same-region: **442 / 647 (68%)** — геометрически это почти наверняка **один физический объект**, размеченный/детектируемый в двух типах данных.
- Асимметрия conf: **li→ae** median conf **0.10** (47% <0.05); **ae→li** median conf **1.0**. Много cross — слабый lidar-pred, «укравший» ae-GT в глобальном жадном матче; при honest-порогах часть из них отсекается, часть rematch'ится на same-layer.

**Интерпретация:** доминирует не «случайное попадание в чужой объект», а **cross-type покрытие одного кургана** (dual markup / dual detector). finds_points полностью same-layer (maggr) — geo-clean.

### Драйверы Δ −0.0206 (honest)

Абсолютный soft-F1 @ tau=0 даёт лишь Δ −0.0054 (FP-океан давит F1); honest-пороги делают TP-потерю заметнее. Ранг по **net ΔTP × CLASS_WEIGHT**:

1. **kurgany_povrezhdennye** (−217 TP, w=30) — главный драйвер  
2. **kurgany_tselye** (−61 TP, w=20)  
3. **object_poly** (−32 TP, w=10; geo↔geo)  

Optical **растёт** при layer (+0.006) — greedy reassignment иногда выгоднее подгруппе без лидара; ALL всё равно падает.

### Гипотезы (итог)

| ID | Утверждение | Вердикт |
|----|-------------|---------|
| **H1** | LB matching ≡ global pool; тип — вторичная атрибуция | **подтверждена** (код + поведение Δ) |
| **H2** | Δ −0.021 = ценность cross-layer TP (+ rematch) | **подтверждена** (647 cross, 330 rematch, net −317) |
| **H3** | abs gap 0.297↔0.412 ≠ matching optimism | **подтверждена** (global = правило LB; другой GT/split) |

**Вывод для гейта:** `match_scope=global` остаётся **основным** honest-гейтом (как LB). `layer` — только диагностика / стресс-тест fusion; в сдачу не включать.

### Рекомендации другим ботам (2–3)

1. **Гейт:** продолжать принимать/отклонять ветки по **honest-10 global**; layer-Δ ≤ 0 считать нормой, не регрессией сдачи. Не оптимизировать под layer-score.
2. **Смотреть пару li↔ae на kurgany_*** (85% cross): fusion/NMS/пороги lidar↔aerial на курганах — место, где global «дарит» TP за счёт чужого markup_type. Имеет смысл логировать, есть ли same-layer pred на тот же GT, прежде чем резать lidar-хвост conf.
3. **Cross-layer dedup предиктов перед сабмитом — осторожно.** Может снизить FP (два предикта на один объект), но **риск для LB**: удаление единственного pred, который матчится к GT другого слоя, прямо режет score при global matching. Безопаснее: within-class NMS по геометрии **с выбором max-conf независимо от слоя** (уже близко к тому, что делает global match) + проверка на holdout; отдельный A/B, не включать вслепую.

### Closure

- EXPERIMENTS: `diff_match_scope_honest10_dump` → **PASS_reproduced**; `delta_decomp_final_pc` → **DONE**
- QUEUE `## bot7 — Δ decomposition dump` → **DONE**; lease `pc_bot7_diff_scope.lease` → **DONE/released**
- Код не менялся после `aa3694f`; main/LB submit не трогались.
