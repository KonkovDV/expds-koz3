# PREREG W2 [li_cnx] — ConvNeXt lidar (зарегистрировано 29.10 22:10 MSK, ДО измерений)
Гейт для всех: honest-10 (materials/compute_metrics.py, scoreboard --honest 10, paired k10), база A=0.314615,
принятие только при Δ>=+0.003 и P(A>B)>=0.9; принимает только W4. Флаги по умолчанию выключены. -in варианты: только измерение.
Вход для W4: /workspace/koz3/w2/oof/merged/oof_cnx_<variant>_<seed>.geojson (5/5 фолдов, threshold_group=li_cnx, EPSG:3857).

## P-W2-1  c5-sc seed0 (5 каналов lrm_s,lrm_l,slope,svf_r,opp_r; scratch), 5 семейных фолдов
- Ядра: koz3-w2-cnx-c5sc-f01 / -f23 / -f4 (12000 it, crop 512, centerhead, D8 TTA; код = s1f23 + JOBS/seed).
- Измерение: (a) аддитивный слой li_cnx к стеку (как X у bot-10); (b) замена li-kurgan групп (как R).
- Сравнение с: c3-sc seed0 на тех же фолдах (тот же протокол). Гипотеза: c5 ≥ c3 на li-only; в стеке — гейт.
- Решение без подглядывания: если c3-sc и c5-sc оба отрицательны в (a) и (b) → ветка cnx закрывается.
## P-W2-2  c3-sc ансамбль сидов (seed0+seed1), усреднение карт maps_cnx_* до декодирования
- Только когда есть 5/5 фолдов обоих сидов. Измерение (a) и (b) как в P-W2-1. Порог/NMS — как в ядре, без подбора на OOF.
## P-W2-3  c3-sc seed1 одиночно — только как контроль дисперсии сидов (не кандидат на приём).

## P-W2-4  Idea 5: привязка центров курганов к пику рельефа (зарегистрировано 29.10 23:28 MSK, ДО измерений)
- Предыстория: bot-6 LOCALIZE (DEM 0,25 м из LAZ, разрежен, p50 заполнения 42%) — отрицательно. Отличие: полный DEM пайплайна (delaunay) → lrm_s 0,5 м из koz3-cnx-li-prep, 59 участков; snap только для li-групп.
- Преобразование (флаг --center-snap, по умолчанию выкл, tools/w2/center_snap.py): для kurgany_* с threshold_group, содержащим "li", центр → argmax gauss(σ=1 px) lrm_s в радиусе R=3 м, только если пик выше значения в исходной точке на ≥ δ=4 u8 (≈1,2 см на шкале 0,75 м); геометрия — та же фигура со сдвигом; confidence не меняется. Параметры фиксированы ЗДЕСЬ, без подбора на OOF.
- Ступень 1 (дёшево, критерий остановки): на стеке A (/workspace/koz3/w5data/stack_arch.geojson, sha .gz 42FF3087) для li-курганов считаем по точке representative_point: G = было вне любого GT того же класса и стало внутри; L = было внутри и стало вне. Стоп, если G−L ≤ 0 или G−L < 0,5% от числа TP-точек li-курганов.
- Ступень 2 (если прошла ступень 1): honest-10 по пайплайну W5 (prep → table → gate_v6c --vs base_counts) против A=0.314615. Гейт: Δ≥+0.003, P(A>B)≥0.9; полы по классам −0.002 (povr, tselye, gor), −0.005 прочие; классы с <5 GT в типе данных исключены. Принимают W4/W5.
- РЕЗУЛЬТАТ P-W2-4 ступень 1 (29.10 23:36 MSK, w2/t4/stage1_stackA.json): li-курганов в стеке A 1 209 619 (рельеф есть у всех 59 участков), сдвинуто 204 304 (p50 4,4 м в 3857 ≈ 2,4 м на местности); точек внутри GT до 3503; G=45, L=51, net −6 (−0,17% от TP); из 976 «кандидат в 5 м, центр вне GT» сдвинуто 239, попало 40. По conf≥0,5 net +4. → СТОП по критерию, ступень 2 (honest-10) не запускалась. Согласуется с bot-6 LOCALIZE: вершина холма ≠ центр разметки.

## P-W2-5  lidar.hillshade_style="li_match" (зарегистрировано 30.10 01:33 MSK, ДО измерений)
- Флаг: configs/default.toml [lidar] hillshade_style = "vendor" (по умолчанию) | "li_match": Horn 3×3 по DEM, гаусс σ=1 px, 315/45/z1, i = clip(253 − g). Разрешение DEM не меняется (0,1 м; родные 0,05 м у 036/041 без Li недоступны).
- Затрагивает только hillshade_set → li-модели (li_r18_hard2, li_mask, li_fort) на отмывке из DEM; cnx (каналы lrm/slope/svf) не затрагивает.
- Замер: honest-10 только по сценам, где li-модель видит отмывку из DEM (Li организаторов отсутствует) — срез lidar59 через scripts/li_real_paired.py-логику; остальные группы без изменений. Гейт: Δ≥+0.003, P(A>B)≥0.9, полы −0.002 povr/tselye/gor, −0.005 прочие, классы с <5 GT в типе исключены.
- Требуется (не на боксе): OOF-инференс li-моделей по фолдам на 59 LAS с флагом (GPU или ПК; фолдовые веса li_r18 нужны на руках). Пока не подтверждено, что такие веса есть на боксе → замер не запускаю.

## P-W2-6  вариант (b): замена li-слоя стека A на c3-sc s012avg (зарегистрировано 30.10 03:56 MSK, ДО измерений; запрос бота 1 30.10)
- Контекст: аддитивно c3-sc (s2 и s012avg) — FAIL (w2/notes/w2_c3sc_s2_gate.json). (b) — последний шанс c3-sc; s3+ сиды в hold до итога (b).
- Определение «li-групп»: все детекции стека A (/workspace/koz3/w5data/stack_arch.geojson, sha .gz 42ff3087) с markup_type == "li",
  т.е. threshold_group ∈ {li_dev, li_mosaic, li, li_coarse, li_fort, li_mask, gor_phys} = 1 214 793 строк (59 LAS-участков; классы
  kurgany_povr/tselye, gorodishcha, fortifikatsii). li_ae (markup_type=ae) и все прочие слои не трогаются.
- **b (основной, решающий)**: удалить ВСЕ li-строки выше (все классы, все 59 участков, включая 182_Красносельская_1.7км, где у cnx 0 детекций)
  и дописать /workspace/koz3/w2/oof/merged/oof_cnx_c3-sc_s012avg.geojson (82 996, sha a88c6505…, threshold_group=li_cnx, markup_type=li) без изменений conf.
- b-kur (вторичный, только диагностика, решения не меняет): удалить только li-строки классов kurgany_* и дописать только kurgany_* из s012avg; li gor/fort остаются.
- Протокол как в w2/gate_s2 (s2-гейт): w2/tools/stack_replace_li.py → prep.py → gate_v6c.py --vs w5data/work/base_counts.pkl (honest-10 по семьям,
  paired bootstrap 2000 по 103 семьям, metric materials/compute_metrics.py) → w2/tools/gate_detail.py. База A = 0.314615.
- PASS (b) ⇔ Δ ≥ +0.003 И P(B>A) ≥ 0.9 И нет нарушений полов по классам (−0.002 kurgany_povr/tselye/gorodishcha, −0.005 прочие; классы с n_GT<5 исключены)
  И Δ new34 ≥ −0.001 И Δ участки без лидара ≥ −0.001 (те же допуски, что в s2-гейте; будет показан и знак Δ). Принимают только W4/W5.
- Решение без подглядывания: (b) FAIL → ветка c3-sc закрыта целиком, s3f4 не используется, GPU → heatmap/r50c5 после Сб 03:00. (b) PASS → передать W4/W5, s3f4 — в s0123avg только после повторного гейта.

## P-W2-7  li_hm: lidar heatmap-детектор шага 1 с QFL (зарегистрировано 30.10 04:08 MSK, ДО запуска и измерений; запрос бота 1 30.10 03:59)
- Гипотеза (REVIEW_SOTA #8, CI2-4): c3-sc (centerhead, шаг 4 = 2 м, penalty-reduced focal по пикам) теряет мелкие/близкие курганы на
  огрублении до 2 м; теплокарта полного разрешения (0,5 м) с Quality-Focal (β=2) по гауссиане, нормировкой max(Σg·valid, 50), маской nodata
  и float32-лоссом даёт более точные пики курганов → полезный второй член ансамбля li-слоя.
- Вариант (единственный, без перебора): ядро elenakonkova/koz3-w2-lihm-c3sc-f01234 = проверенное ядро coord s3f4 (U-Net tu-convnext_tiny,
  scratch, каналы c3 lrm_s/lrm_l/slope, кроп 512, 12k ит., bs 8, copy-paste, height ×0,5–2,2, zoom ×0,7–1,45, d8-TTA, EMA, AMP) с изменениями
  ТОЛЬКО флагами: CNX_HEAD=w2hm_head:make_head (голова QflHeatHead, kaggle/2026-10-30/lidar_heatmap/w2hm_head.py: гауссианы σ=max(1, 0,15·D) px,
  декод maxfilter peak_k=11, min_score 0,05, класс = argmax, круг 0,5 м как у cnx), CNX_TGROUP=li_hm (threshold_group), seed 0,
  jobs c3-sc:0..4 (2 полосы T4×2), CNX_BUILD_RELIEF=1, deadline 8,5 ч. Маска-голова gor/fort/arch — без изменений. По умолчанию ядро
  (CNX_HEAD_DEFAULT=centerhead, группа li_cnx) не меняется. Фолды = region_fold s0/s1 (встроенный FOLDS, те же 5 семейных фолдов).
  c3 (а не c5) — чтобы отличие от c3-sc было только в голове; c5 закрыт ботом 1 и покрывается r50c5.
- Выход: oof_cnx_c3-sc_f{0..4}.geojson + слитый oof_cnx_c3-sc.geojson (имена ядра cnx, но threshold_group=li_hm, markup_type=li),
  gt_li_cnx_*, region_fold.json, report_cnx_full.json, maps/ (теплокарты шага 1), weights_cnx_*. Забор → w2/oof/lihm/koz3-w2-lihm-c3sc-f01234/.
- Ранний стоп (D5, без гейта): Li-only score (compute_metrics, лог ядра) li_hm на f0 и f1 против c3-sc s2 на тех же фолдах (0,2198 / 0,4786):
  если среднее(f0,f1) li_hm ≤ 0,3492 → линия heatmap закрыта, гейт не считается, квота → W1/r50c5. Иначе — гейт по полному OOF.
- Гейт (протокол w2/gate_s2, как P-W2-6: prep.py → gate_v6c.py --vs w5data/work/base_counts.pkl, honest-10 по семьям, paired bootstrap 2000
  по 103 семьям, materials/compute_metrics.py, база A = 0.314615):
  · H1 аддитивно (основной, решающий): стек A + все строки li_hm (своя группа порогов li_hm, conf без изменений).
  · H2 замена (вторичный): из стека A удалить li-строки классов kurgany_* (группы markup_type=li) и дописать kurgany_* li_hm; li gor/fort остаются.
    Решающий ТОЛЬКО если H1 FAIL, и тогда с ужесточением P(B>A) ≥ 0.95 (поправка на второе сравнение).
  · PASS ⇔ Δ ≥ +0.003 И P(B>A) ≥ 0.9 (H2: ≥ 0.95) И нет нарушений полов по классам (−0.002 kurgany_povr/tselye/gorodishcha, −0.005 прочие;
    классы с n_GT<5 исключены) И Δ new34 LAS ≥ −0.001 И Δ участки без лидара ≥ −0.001. Принимают только W4/W5.
- Стоп-правило: H1 и H2 FAIL → линия heatmap закрыта (без донастройки порогов/σ/β на тех же фолдах, без новых сидов); квота → W1/r50c5.
  PASS → передать W4/W5; доп. сиды — только новым PREREG. Ошибка ядра/таймаут до f0+f1 → одна повторная попытка с тем же кодом, затем стоп.
