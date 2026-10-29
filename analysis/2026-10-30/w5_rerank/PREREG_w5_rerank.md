# PREREG W5-rerank (29.10.2026 23:00 MSK, до замера на стеке)

Данные: stack_arch.geojson (sha b412e299…, OOF всех членов стека), GT gt_train_v2 (sha d4bc1a3b…).
Гейт (PROTOCOL п.4): honest-10 по семьям, база A = 0.314615 (V0 обязан её воспроизвести, иначе прогон недействителен).
Приём: Δ ≥ +0.003, P(B>A) ≥ 0.9 (paired bootstrap по семьям, 2000), Δ на 34 новых LAS ≥ −0.001, Δ на optical ≥ −0.001.
Пороги — per (класс, threshold_group) на train-фолдах (honest_counts); параметр политики — вложенно (inner honest на 9 фолдах).
Итоговое число варианта = строка `*_nested`; строки с фиксированным параметром — справочные.

| id | вариант | сетка | ожидание |
|---|---|---|---|
| V0 | база | — | 0.314615 |
| V1 | min-k: top-k на (участок × класс) всегда выдаётся | k ∈ {1,2,3} | ≈0 / small + на finds |
| V2 | kurgan hedge: копия кургана в другой подкласс, score×α, своя группа `~hedge` | α ∈ {0.25,0.5,0.75} | + на povr/tselye |
| V3 | cross-layer copy: копия кандидата, score×β, группа `~copy` (в сабмите — markup_type другого слоя участка) | β ∈ {0.5,0.8,1.0} | +0.005…0.02 (REVIEW_SOTA #4) |
| V4 | микро-круги r=0.75 м: (0) в центре сильного (gq≥0.9) кандидата, (1) в середине при расхождении центров 2–8 м с другим членом; score×β, группы `~mc0/~mc1` | β ∈ {0.3,0.6,0.9} | + на мелких курганах |
| V5 | LGBM-реранкер (nested2.py, label=first-hit, по семействам классов) | rounds 300 | прежний итог: все классы −0.0005 P0.50; только arch +0.0097 P0.966 |

Правило безубыточности для доп. объекта: p > F/(4−F); здесь оно реализуется сдвигом порога группы на train-фолдах.
Смоук на прокси (D-FINE ae OOF, 30 участков, 1 член) — только проверка кода, в гейт не идёт: smoke_proxy_*.json.

## Addendum 2026-10-29 23:20 MSK (declared before any measurement on these variants)
- **V3 redefinition** (already in code before the real-stack run): cross-layer copy emits the `markup_type` of a layer
  that the stack actually emits at that site (not a fixed one). Flag for W4: copies across data types may conflict with
  "между типами данных НЕ склеивать" — W4 compliance check required before any use.
- **V5x** = V5 (nested LGBM reranker, label `first`) with 5 extra features from W2 lidar ConvNeXt c3-sc honest 5-fold
  OOF, two seeds (`/workspace/koz3/w2/oof/merged/oof_cnx_c3-sc_s{0,1}.geojson`): `cnx_s0`, `cnx_s1` (max conf of an
  intersecting det of the same class family), `cnx_mean`, `cnx_agree` (both seeds >0), `cnx_samecls` (exact subclass);
  NaN outside the 58 cnx-covered sites. Gated vs A and vs V5 (same nested protocol, same folds).
- **V6c** = stack + seed-fused cnx candidates as a new threshold_group `li_cnx`: s0-anchored, conf=(c0+max intersecting
  c1 same family)/2, s1-only dets conf=c1/2. Honest-10 thresholds on the merged stack, paired bootstrap vs A.
- Gate unchanged: Δ≥0.003, P(Δ>0)≥0.9, no regression (≥−0.001) on new34 and optical. Acceptance only by W4.
