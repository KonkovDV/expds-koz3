# REVIEW_SOTA: репо, логи, конкуренты, план SOTA-идей (КОЗ-3 «Комплексирование»)
_Составлено executor-ботом 29.10.2026, ~23:30 МСК. Только чтение, ничего не пушилось и не сабмитилось. Ядра Kaggle не трогались._

## 0. Ограничения обзора (важно)
- **ПК F:\expds3\work не прочитан.** Read вернул «Path is outside the allowed local-exec root», Shell вернул `spawn EPERM`, затем ПК ушёл offline. Вместо этого использованы копии этих логов, закоммиченные в репо: `notes/cross_family_*.json`, `notes/large_ceiling_2026-10-23.json`, `notes/gt_stats_summary.md`, `analysis/2026-10-28/rerank/*board*.json`, `docs/bots/2026-10-28/bot-3/results/pixgbm/pc/thr_pix_full.json`, а также опись файлов `local_ctx/ctx/treeF__expds3.csv`. Для полного разбора `scoreboard_base_v2.json` и `verify_*` нужен доступ к ПК или копия в /workspace.
- **Инцидент.** В начале работы я сделал `git checkout origin/main` в общем клоне `/workspace/koz3/repo`. Позже другой агент переключил клон на `bots/bot7-candidate-v4-nomodels`. После этого я работал только через `git show origin/<branch>:<path>`. Рекомендация: у каждого бота свой worktree, общий клон только для чтения.
- Kaggle-логи ядер здесь не разбирались. Опирался на сводки ботов в `research_archive_20261029/EXPERIMENTS.csv`.

## 1. Метрика: что реально решает (materials/compute_metrics.py, ТР прил. 4)
1. Матчинг идёт **по классу, глобально**: пул всех регионов и всех markup_type, жадно по confidence, one-to-one. bot7 подтвердил на 29.10, что LB ≡ global (EXPERIMENTS.csv, `delta_decomp_final_pc`). Режим «layer» на honest даёт −0.0206: 647 кросс-слойных TP.
2. Полигон засчитывается, если `area(P∩G)/min(area) ≥ 0.5` **и** representative_point(P) лежит внутри G. Значит, маленький круг с центром внутри GT всегда проходит: cov = 1. **Размер полигона почти не влияет. Решает только центр.** У TP курганов медиана pred/GT = 0.15 (GAP §5).
3. Точки (finds_points): расстояние от центроида до точки ≤ 2.0 м.
4. P = tp/(tp + 0.5·fp). **Правило безубыточности:** добавлять предсказание выгодно, если его шанс попадания p > F/(4−F). Это F1 класса до добавления (LB_LOG).
   - povr (F ≈ 0.32): p > 8.7 %
   - tselye (0.48): 13.7 %
   - gor (0.30): 8.1 %
   - arch (0.06): **1.5 %**
   - object_poly (0.13): **3.4 %**
   - fort (0.15): **4 %**
   - finds (0.55): 16 %
   - Вывод: для слабых классов пороги должны быть очень низкими, а лишние гипотезы дешёвые.
5. Класс, которого нет в GT, выпадает из нормировки весов. Пропущенный присутствующий класс теряет весь свой вес. Отсюда: **никогда не выдавать 0 предсказаний по присутствующему классу**, хотя бы немного лучших кандидатов.
6. Дубли в одном типе данных идут в FP. Между типами склеивать нельзя, но **копия на другой слой законна** и может взять GT другого слоя (п.1).
7. BIoU, HD95, PoLiS и P/R по типам данных справочные, в final_score не входят.

## 2. Ключевые находки в логах
### 2.1 Табло (scoring.log, обратный расчёт bot-10 REPORT13_lbv2)
| Тип | 27.10 (0.348) P/R | 29.10 v2 (0.412) P/R | tp/fp/gt v2 |
|---|---|---|---|
| Лидар | .508/.489 | .516/.504 | 486/910/964 |
| Спутник | .487/.292 | .456/.255 | 96/229/377 |
| Аэро | .091/.013 | .091/.013 | **1/20/75** |
| Орто | .368/.140 | .368/.140 | 7/24/50 |
| Геофизика | .791/.259 | .769/.656 | 946/567/1441 |

- Весь прирост 0.348 → 0.412 дала **геофизика**: R .26 → .66 (PIXGBM S + stack6).
- Публичный тест мал: ~2.9k GT, из них геофизика ~50 %, лидар 33 %, аэро 2.6 %, орто 1.7 %. **Финал закрытый, состав другой.** Оптимизировать под публичный ae/or бесполезно, но под закрытый, где ae/or могут быть большими, обязательно.
- Recall по размерам: 5–10 тыс. м² = **0**, >10 тыс. = .29. Крупные объекты почти не берутся.

### 2.2 Карта потерь на честной базе (bot-4 GAP.md, honest-10 A = 0.3146)
- Потери на FN 0.525 против **FP 0.083**. Мы теряем на полноте, а не на точности. При весе FP 0.5 это ожидаемо.
- Без лидара FN ≈ **0.38**, с лидаром 0.20. Городища без лидара: **0 из 73**.
- Потолки:
  - идеальные пороги 0.367 (+0.05);
  - **оракул-реранкер по нашим же кандидатам 0.658 (+0.29)**;
  - полнота кандидатов: povr 0.65, tselye 0.67, gor 0.48, arch 0.27, fort 0.40, finds 0.60.
- Вывод: главный резерв в **ранжировании и уверенности** (кандидатов у курганов в 100+ раз больше, чем GT), а для gor/arch/obj/fort ещё и в поиске.
- Геометрия:
  - Курганы: 527 povr и 185 tselye FN имеют кандидата в 5 м, но центр вне GT. Это **смещение центра**, а не размер.
  - finds: **809 FN с кандидатом в 2–5 м** (допуск 2 м). Уточнение координаты даёт до +0.017 к оракулу.
  - object_poly: GT крошечные (медиана 6 м²), 545 FN «рядом, но не проходит», потолок +0.046.
- Худшие клетки класс×тип по FN-потере:
  - object_poly·Geo 0.085
  - povr·Ae 0.072
  - povr·Or 0.069
  - gor·Or 0.068
  - arch·Or 0.058
  - povr·Li 0.055

### 2.3 Gate против LB: противоречия
| Сабмит | Gate | LB |
|---|---|---|
| v2 | honest-10 +0.011 / +0.0139 | **+0.0644** (сдвиг состава теста к геофизике) |
| arch 27.10 | +0.0036 | +0.0144 |
| 24.10 | self-fit порог 0.3538 | 0.3285; cross-family 0.3295 совпал с LB |

- Вывод: honest cross-family несмещён по уровню, но **плохо предсказывает Δ между вариантами**, если они бьют по разным модальностям. Нужна взвешенная по составу теста оценка, см. идею #12.
- **D-FINE SpOr.** В STATUS указано +0.0198 (P 0.985, «в v3»). Но mock-e2e в docker: A3 D-FINE ON **−0.0023**, а v3 compare **FAIL sp_or = 0.000**. Это противоречие нужно закрыть до включения: preflight, n_raw, пороги. Похоже на тот же класс багов, что и «тихий фолбэк» (MURDER_BOARD, LESSONS 3–4).
- gor_dense: OFF = ON, max conf 0.17–0.23 < τ 0.824, то есть NO-OP. Это не нейтральный результат, фича просто не включалась.
- PIXGBM clean-room: sha mismatch, расхождение предсказаний 0.566. Сборка из main ≠ архив LB (≈ −0.012). Критично для 11.11.

### 2.4 Отрицательные эксперименты (не повторять)
- MORPH и MORPH_RING
- ADAF zero-shot
- ORGATE_S, SATf, CAL1
- gormask: −0.0056
- neigh-prior в нашей реализации
- bigobj F1/F2/F12/F1s
- coarse_hillfort: −0.016; fort→gor: −0.006
- li_decode
- fusion-calib и WBF между детекторами
- kurgan ae c7g: −0.014
- kurgan subclass morph: AUC 0.59
- arch-порог Or
- ORSP с floors
- лидар: openness, SVF, CLAHE, TTA
- копии слоёв EtGr
- второе окно 300 м
- пороги по площади сцены
- match_scope=layer

## 3. Конкуренты и публичные работы
### 3.1 Конкурс (GitHub, Docker Hub, TG)
| Ресурс | Что полезно | Лицензия / статус |
|---|---|---|
| [yakirill99/ds-expedition](https://github.com/yakirill99/ds-expedition) | U-Net r34, 26 каналов рельефа, маска + **heatmap центра** | идеи |
| [vaitter/koz3-build](https://github.com/vaitter/koz3-build) | тайл 1024, overlap 30 %, conf 0.05, CLAHE | YOLO (AGPL), только идеи |
| [KateBuikina/expds-solution](https://github.com/KateBuikina/expds-solution) | U-Net R34; скелет вала → цепочка точек (КОЗ-2) | идеи |
| [Iskatel-potenzialov/gis](https://github.com/Iskatel-potenzialov/gis) | бесшовная вставка курганов (copy-paste) | **MIT**, можно брать |
| [MataNerdy/Geodata_Archaeology_CV](https://github.com/MataNerdy/Geodata_Archaeology_CV), [Egor-git-cloud/ds-expedition](https://github.com/Egor-git-cloud/ds-expedition) | КОЗ-2-эпоха | идеи |
| [vdurnov/xview2_1st_place_solution](https://github.com/vdurnov/xview2_1st_place_solution) | ансамбль U-Net, локализация + классификация | MIT |
| EhimenNathan/EXpedition, maratgaliulin/EXpedition | лидер-уровень (EN v22 0.505) | **по правилу владельца ничего не брать** |
| Docker: evgps (RF-DETR), niovol, restdockerse (TerraMind + LoRA), azalexandr87 fort-v2 и др. | стек лидеров: DETR-детекторы, RS-FM | только наблюдение |
| [t.me/s/expds](https://t.me/s/expds), [upgreat.one КОЗ-3](https://upgreat.one/contests/expds/koz-3), [ТР КОЗ-3](https://upgreat.one/upload/iblock/477/477560736561f633fc4e1bd983635da5/PROEKT_Tekhnicheskii_reglament_KOZ_3_Ekspeditsiya_DS.pdf) | правила, разъяснения | — |

Новых публичных GitHub-репо по КОЗ-3 поиск 29.10 не нашёл.

### 3.2 Статьи (переносимые идеи)
- [WODAN2.0: DL + Location-Based Ranking](https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/docId/124229): ранжирование по ландшафтному контексту и bagging дали +17–35 %. Для нас это контекстный реранкер.
- [Data-centric AI для курганов (IEEE Access 2023)](https://doi.org/10.1109/access.2023.3290305): copy-paste по маске допустимых зон и 3D-верификация формы кандидата. Precision выросла до 72 %.
- [Hillfort multimodal segmentation (Arch. Prosp. 2024)](https://doi.org/10.1002/arp.1958): fusion LRM + орто, hard positive/negative mining. FP 5678 → 40, F1 0.66.
- [Europe-wide hillfort search (JCAA 2025)](https://journal.caa-international.org/articles/10.5334/jcaa.178): ConvNeXt-классификатор виньеток + pseudo-label + регион-fine-tune, F1 0.34–0.38.
- [Ancient cemeteries, 5-channel LiDAR (npj Herit. Sci. 2025)](https://www.nature.com/articles/s40494-025-01798-5): DEM + slope + hillshade + roughness + curvature лучше 1–3 каналов.
- [Обзор DL по ALS (Remote Sens. 2025)](https://www.mdpi.com/2072-4292/17/15/2733).
- [Geomorphometric burial mound (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7070870/): поиск пиков и сегментов + RF, recall 93 % на внешних данных. Обосновывает центр по пику DEM.
- [D-FINE](https://arxiv.org/abs/2410.13842); [бенчмарк DETR на VisDrone](https://trepo.tuni.fi/handle/10024/232593): **D-FINE-X (Obj365 → COCO) лучший** из RT-DETR v1/v2, DEIM v1/v2, LW-DETR, RF-DETR. [HF-D-FINE](https://dl.acm.org/doi/10.1016/j.imavis.2025.105834): high-res инъекция признаков +3.2 AP на мелких объектах.
- [DINOv2 frozen linear probe, 4 последних слоя](https://arxiv.org/html/2407.18100v3); [DeepAndes (DINOv2 для археологии)](https://arxiv.org/html/2504.20303v2): замороженные патч-токены + линейная голова сильны в few-shot. Подходит для gor/arch и для реранкера.
- Геофизика:
  - [CNN-сегментация магнитки (C&G 2024)](https://doi.org/10.1016/j.cageo.2024.105847)
  - [AnomalyExtractor (dainst)](https://github.com/dainst/AnomalyExtractor): OBIA-векторизация аномалий
  - [Shallow ML segmentation 2D/3D geophysics (RS 2025)](https://www.mdpi.com/2072-4292/17/17/3092)

## 4. Приоритетный список идей
Оценки [оц] даны на LB-метрику, это экспертные оценки, а не замеры. Стоимость указана для GPU 4090 или Kaggle T4. «До 11.11» означает «реально успеть с gate».

| # | Идея | Ожид. Δ | Стоимость | Риск | Owner | До 11.11 |
|---|---|---|---|---|---|---|
| 1 | **Паритет сборки:** golden e2e main против архива v2 (PIXGBM sha, D-FINE preflight, n_raw>0 по каждой ступени, health-счётчики вместо `log.exception`) | защищает ~0.012–0.06 | 2–4 CPU-ч | низкий | W4 | да, **сначала** |
| 2 | **Ae D-FINE all-class** (f4pack 249b86fb) + ae fort τ 0.995 + исправление tile/fuse mismatch (bot3 Fix A/B/C). Сейчас в ae 1/75, курганы ae без лидара 0/510 в OOF | +0.005 публ. / **+0.02–0.05 закрытый** | 2–4 GPU-ч калибровки | средний (см. противоречие D-FINE SpOr) | W1 | да |
| 3 | **Пороги по правилу безубыточности** p > F/(4−F) per class×type, с floor «≥k лучших кандидатов» для каждого присутствующего класса (п.1.5). Особенно arch, obj, fort, gor | +0.005–0.02 | <1 CPU-ч | низкий-средний (шум семей) | W4 | да |
| 4 | **Кросс-слойная репликация:** копия уверенного объекта на каждый слой, который есть у участка (Li→Ae, Li→Sp, Sp↔Or); порог копии по безубыточности. li_ae сейчас фактически выключен (τ 0.993), global-матчинг стоит +0.02 | +0.005–0.02 | <1 CPU-ч | средний (FP, если разметки слоя нет) | W4 | да |
| 5 | **Уточнение центра:** snap к пику DEM/LRM для курганов (527+185 FN — центр за краем GT); snap к экстремуму или диполю аномалии для finds (809 FN в 2–5 м) | +0.01–0.02 (finds до +0.017 оракул) | 1–3 CPU-ч | низкий | W2 (курганы), W4/геофизика (finds) | да |
| 6 | **Мульти-гипотезы:** при расхождении центров у членов ансамбля на >2 м выдавать 2 микрокруга (второй с пониженной conf) для povr/gor/arch/obj; povr+tselye hedge при p2 ≥ 0.05 (S6/L7) | +0.005–0.015 | <1 CPU-ч | низкий-средний | W4 | да |
| 7 | **Кроп-реранкер на замороженных DINOv2-S/B (4 слоя) + LightGBM** по уже существующим кандидатам; цель — разрыв «идеальные пороги → оракул» +0.29 | +0.01–0.04 | 3–6 GPU-ч эмбеддингов + CPU | средний (время инференса: только кропы) | W4 (+W1) | да, если начать до 03.11 |
| 8 | **Lidar heatmap-детектор** (ConvNeXt-T/ResNet-50 U-Net, peak decoding, 5–8 каналов рельефа, zoom ×0.7–1.45, height ×0.5–2.2, d8-TTA) как второй член ансамбля | +0.02–0.04 [оц] | 15–30 GPU-ч | средний | W2 | впритык |
| 9 | **Городища и крупные объекты без лидара:** DINOv2 патч-признаки на 2–4 м/пикс + маска; hard-negative mining (FP 5678 → 40 в hillfort-статье); D-FINE Ae gor area_tau 0.20 (bot4 Fix C) | +0.005–0.02 | 5–10 GPU-ч | высокий (69 GT, LOFO-дисперсия) | W3 | частично (Fix C да) |
| 10 | **object_poly:** микрокруг в центре мелкого GT (медиана 6 м²); MagGr inference 0.05 м (G2 +0.0099), EtGr-изоляция слоёв (G4 +0.0098 LB), anchors MagGr 0.75 / EtGr 0.3 | +0.01–0.02 | 1–3 CPU-ч | низкий | W4/геофизика | да |
| 11 | **Satellite recall:** sat_thr_relax 0.85 (bot2), гетерогенный fusion Sp (S1: 0.27→0.37, кластер 8 м), Li→Sp twins α 0.1 | +0.003–0.01 | 1–2 GPU-ч | средний | W1 | да |
| 12 | **Gate с весами состава теста:** перевзвесить honest-10 по долям модальностей публичного теста (геофизика ~50 %) и отдельно по сценарию «закрытый»; критерий принятия = обе оценки ≥ 0 | точнее выбор (было +0.011 при LB +0.064) | <1 CPU-ч | низкий | W4 | да |
| 13 | Copy-paste курганов (MIT-код Iskatel) + hard-negative mining для Ae/Sp курганов | +0.005–0.015 | 5–10 GPU-ч | средний | W1/W2 | впритык |
| 14 | Снятие same-markup дублей (лидар 12, геофизика 5, спутник 2) | +0.001 | минуты | низкий | W4 | да |
| 15 | Ortho multiclass (O1: 0.5 м, только arch ≥ 0.45). Non-arch пока 0 TP, c′ отозван | +0–0.005 | 2–4 GPU-ч | высокий | W1 | низкий приоритет |

Не тратить время: повтор MORPH/ADAF zero-shot/WBF-калибровки/coarse_hillfort/fort→gor/kurgan-subclass morph; SAM3, DINOv3, YOLO (запрещены).

## 5. Решения, нужные владельцу
1. **Правило EN.** COMPETITORS v2 требует «ничего не брать», но COMPETITOR_IDEAS (clean-room) и линия PIXGBM / li_decode связаны с EN. Нужна явная позиция до 11.11, вместе с декларацией ИИ и THIRD_PARTY.
2. **Включать ли D-FINE SpOr в v3.** Сейчас gate (+0.0198) и mock-e2e (−0.0023, sp_or 0) противоречат друг другу.
3. **Дата сдачи кода:** 11.11 (Elena) или 14.11 (КЗ). Планируем под 11.11.
