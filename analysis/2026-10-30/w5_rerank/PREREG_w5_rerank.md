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

## Addendum 2026-10-30 01:20 MSK — WS-RR1 / WS-RR2 / micro-rp resweep (declared before any measurement; script ws_rr.py)
Scores from nested_w5cnx.npz (V5x). Pairs: micro_circles anchor=rp (w0-seams 5171c35, w0_disk_pairs.pkl `rp`) for
gorodishcha/arkhitektury/object_poly/fortifikatsii in ALL variants M, RR1, RR2 (kurgans/finds pairs = stack, as A).
- **M** = A + rp pairs, thresholds re-swept honest-10 (QUEUE w0s-resweep-micro-rp).
- **RR1** = M + V5x only on mk=li rows: S = α·r̃ + (1−α)·r_b, r_b = percentile of base conf among li rows of the class,
  r̃ = rr order mapped onto the r_b quantiles of the pool rows (non-pool li rows: r_b; other layers: base conf untouched).
  α ∈ {0.3, 0.6, 1.0}. Then isotonic p(hit|S) per class×layer(mk) fit on train regions (≥5 hits, else group τ from sweep);
  keep iff p > F_c/(4−F_c), F_c = train-sweep F1 of class c (break-even); groups unseen in train dropped as before.
- **RR2** = RR1 scores/calibration + per-(site, class) top-K, K = round(λ·Σp) over that site-class, λ ∈ {1, 1.5, 2}
  (count-based selection instead of τ; calibration-less classes keep τ).
- Nested: for outer fold k, α (RR1) and (α, λ) (RR2) chosen by inner honest score over the 9 other folds (train scores
  inner[k], calibration+τ refit per inner fold), applied to k with outer scores. Reference rows: RR1 with sweep τ (no iso).
- Gate vs A=0.314615: Δ≥+0.003, P(Δ>0)≥0.9 (paired family bootstrap 2000), per-class ΔF1 floors −0.002 kurgany_povr/
  tselye/gorodishcha, −0.005 others (classes with <5 GT excluded); new34/optical split reported. Primary = RR2 if it
  passes, else RR1, else M; Bonferroni ×3 noted (P≥0.967).
