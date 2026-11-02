# W4 Q0 — диагностика минуса new34 (PREREG, 02.11 13:40 MSK)

Задача: orch/COMPETITOR_GAPS_2026-11-02.md §3 Q0 (d2fe1de). Только диагностика, гейта нет, решений не принимает.

Входы (готовые production replay W1, не пересчитываются):
- SATREC: /workspace/koz3/w1/satrec/out/out_{base,R1,C1}.geojson
- SATREC2: /workspace/koz3/w1/satrec2/out/out_{base,A1}.geojson
- SATDET трек 2: /workspace/koz3/w1/satdet/out/out_{base,P1,P2}.geojson
- GT gt_train_v2.geojson; срез new34 = T["new"] из orch/bot-W1/work/v3prod/t_taus.json;
  сцены sp_or = docs/bots/2026-10-29/bot-10/work11/frame_boxes.json (EPSG:3857).

Метод (фиксирован до расчёта):
1. Сопоставление — expds.metric.build_curves(scope=global), счёт — score_counts (как в s3_score W1).
   Санити: балл new34 base/вариант совпадает с s3-логами W1 до 1e-6 (по счётчикам GT из t_taus counts_B0).
2. Единица = сцена: (участок, бокс sp_or, содержащий representative_point объекта; при нескольких — ближе к центру);
   объекты вне боксов — «#nosat». Размер сцены = W·H в пикселях 0,5 м (метры 3857 × cos φ), Мпикс.
3. Вклад единицы u: swap-in a_u = S(B, u←V) − S(B); leave-out l_u = S(V) − S(V, u←B); отчёт — среднее (a_u+l_u)/2.
   То же по классам (вес·ΔF1/знаменатель) и по корзинам размера {<1, 1–2, 2–4, 4–8, ≥8 Мпикс, nosat}.
4. ΔTP/ΔFP/ΔFN по единице и классу.
Выход: q0_new34.json + Q0_NEW34.md в этой папке.
