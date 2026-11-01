# Глубокий разбор конкурентов КОЗ №3 «Комплексирование» — 01.11.2026

**Команда:** StrataTrace · **наш LB:** 0,41238 (#25/72, сборка v2 `eb268d6`, 29.10 08:47 МСК)  
**Лидер:** Nikolove 0,61636 · **топ-10:** ≥ 0,505 · **дедлайн кода:** 11.11.2026  
**Репо (read-only):** [KonkovDV/expds-koz3](https://github.com/KonkovDV/expds-koz3) · клон `/tmp/arch` (`git fetch origin`)  
**Правило:** чужой код не копируем; идеи — clean-room; YOLO/AGPL — только приёмы; данные/веса КОЗ №2 запрещены; правило «EN — ничего не брать» для кода EhimenNathan/EXpedition.

Метка **[ф]** — факт со ссылкой; **[оц]** — оценка без замера на нашей площадке. Факты не выдумывались: если в открытых источниках нет — помечено «неизвестно».

---

## 0. Стартовые документы и артефакты

| Источник | Что даёт |
|---|---|
| `origin/main:docs/COMPETITORS_2026-10-29.md` | таблица 14+ публичных репо на 29.10; правило EN; дыры Ae/Or/large |
| `origin/main:docs/MURDER_BOARD_2026-10-29.md` | провенанс PIXGBM / `li_decode`; ветки `bots/bot8-cpu-a-clean*` не clean-room |
| `origin/main:docs/PLAN_PIVOT_2026-10-29.md` | разворот на Ae/Or/крупные |
| `origin/main:docs/AUDIT_CLAUDE_2026-10-29.md` | таблица LB 29.10 (места 1–10, 20, 35); Nikolove — открытых материалов нет |
| `origin/main:docs/STATUS_2026-10-29.md` | хронология наших баллов; P/R по типам v2 |
| `origin/main:docs/research/2026-10-28/COMPETITORS_DEEP.md` | полный аудит EN |
| `origin/main:docs/MERGE_2026-10-29.md` | исключение bot8: спека с поведения EN @`d434e56` |
| ветка `bots/orch-protocol` (`orch/BEAT_EN_*.md`, `orch/plans/`) | оркестрация «обойти EN» |
| `archive/worked-20261031`, `archive/research-results-20261029` | архив исследований |
| `/workspace/jury-osint.md`, `/workspace/competitors-v3.md` | **НЕ КОЗ-3**: OSINT трека **ФСП «Талант»** (11.10.2026); к комплексированию не относятся |

### 0.1. Исключение `bots/bot8-cpu-a-clean` — какой репозиторий

**[ф]** `docs/MERGE_2026-10-29.md`: ветки `bots/bot8-cpu-a-clean` и `bots/bot8-cpu-a-clean-on-v2` **не вливались**, потому что `docs/CLEANROOM_SPEC_lidar_a.md` написана по поведению репозитория конкурента **EhimenNathan/EXpedition** на коммите **`d434e56`** (2026-10-26; сообщение: «Audit aerial verifier and rejected v21 variants»), а `tests/lidar_a_oracle.py` подгружает чужое репо. Это не clean-room.

Связанные упоминания: `docs/MURDER_BOARD_2026-10-29.md`, `docs/THIRD_PARTY.md` (PIXGBM «по мотивам» EN через `bot3-ehimen`; `li_decode` удалён).

---

## 1. Список найденных конкурентов / репозиториев

### 1.1. Привязка к местам LB (срез 29.10, `AUDIT_CLAUDE_2026-10-29.md`)

| Место | Команда | Балл | Публичное репо / материалы |
|---:|---|---:|---|
| 1 | Nikolove | 0,61636 | **не найдено** [ф]: GitHub API / code search по «Nikolove» + KOZ3-терминам — релевантного репо нет; `huntertomo`, `falexsun/*` — 404/удалены |
| 2 | CVшники | 0,57996 | открытых материалов нет [ф] |
| 3 | accuracy=0.5 | 0,56855 | открытых материалов нет [ф] |
| 4 | Family | 0,55848 | 1-е «Сканирование» (Омск); по Хабру — 8 моделей; разбор архитектуры в открытом доступе **не получен** (habr/t.me в сетевой политике недоступны на момент аудита 29.10) |
| 5 | GigaFlex | 0,54973 | 3-е «Сканирование» (Уфа); открытого КОЗ-3 репо нет |
| 6 | Фанаты Фишмана | 0,53492 | — |
| 7 | MTs | 0,52058 | 26.10 был ~0,50470 |
| 8 | котики | 0,51154 | — |
| 9 | **Сахалинский центр ИИ** | **0,50491** | **[ф]** https://github.com/EhimenNathan/EXpedition (= EN); форк https://github.com/maratgaliulin/EXpedition |
| 10 | Землекопы | 0,50232 | — |
| 20 | Кофиксики | 0,44631 | ориентир «реалистичной цели» в аудите |
| ~25 | **StrataTrace (мы)** | **0,41238** | https://github.com/KonkovDV/expds-koz3 |

**Вывод [ф]:** единственное публичное решение уровня топ-10 с полным журналом экспериментов и коммитами — **EN**. Топ-1…8 и Nikolove в открытом GitHub по именам с LB **не идентифицированы**.

### 1.2. Прочие публичные репо, связанные с КОЗ / археологией (не обязательно на LB)

Источники: `COMPETITORS_2026-10-29.md`, GitHub API 01.11.2026, `rg` по `/tmp/arch`.

| URL | LICENSE | Состояние на ~29.10–01.11 | Заявленный LB | Модальности / модели (кратко) | Риск |
|---|---|---|---|---|---|
| [EhimenNathan/EXpedition](https://github.com/EhimenNathan/EXpedition) | нет | активен; ветки `main`, `codex/v21-robust`, `codex/v24-dinov3-1024`, `claude/v23-dino-hillfort`, `feat/physics-gates`… | **v22 ≈ 0,50491** [ф] | эксперты по типам: LiDAR ConvNeXt heatmap, Sp fusion D-FINE+RT-DETR+hm, geo CNN+LightGBM/PIXGBM, Ae FCOS fort, Or=SpOr@0,5 м | **EN — код не брать**; только идеи; без LICENSE |
| [maratgaliulin/EXpedition](https://github.com/maratgaliulin/EXpedition) | нет | форк EN (+1 коммит `be6009a`) | то же | то же | то же + в форке были ключи S3 организаторов (не использовать) |
| [yakirill99/ds-expedition](https://github.com/yakirill99/ds-expedition) | нет | `develop`; коммиты до 01.11 (каркас) | нет | LiDAR U-Net r34, 26 каналов рельефа, LAS→слои в инференсе | только идеи |
| [vaitter/koz3-build](https://github.com/vaitter/koz3-build) | нет | HEAD ~27.10 `dfdafaa` | нет | YOLO E1 hillshade / E2 geo / E3 Ae·Sp·Or (1024, 30 % overlap, CLAHE) | **AGPL-идеи только** |
| [KateBuikina/expds-solution](https://github.com/KateBuikina/expds-solution) | нет | 27.05, без изменений | нет (своя метрика) | U-Net R34: SAT(SpOr+Ae), ORTO, LAS; скелет линейной маски | идеи |
| [MataNerdy/Geodata_Archaeology_CV](https://github.com/MataNerdy/Geodata_Archaeology_CV) | нет | 25.06 | нет | DeepLab / SegFormer | SegFormer — лицензионный риск |
| [Egor-git-cloud/ds-expedition](https://github.com/Egor-git-cloud/ds-expedition) | нет | 19.10 | нет | каркас + YOLO-бейзлайн организаторов | — |
| [Iskatel-potenzialov/gis](https://github.com/Iskatel-potenzialov/gis) | MIT | КОЗ-1 | — | вставки курганов | не КОЗ-3 |
| [iwannasome/expds-rubleincorp-KOZ2](https://github.com/iwannasome/expds-rubleincorp-KOZ2) | нет | КОЗ №2 | — | — | **ЗАПРЕЩЕНО** |
| [Strekalov/ultralytics](https://github.com/Strekalov/ultralytics) | AGPL | форк; drone/YOLO | — | не археология КОЗ-3 | AGPL; не использовать |
| falexsun/*, huntertomo/* | — | API 404 | — | — | удалены / не найдены |
| dulcisono/expds | — | push ~16.10 | неизвестно | не подтверждена привязка к LB КОЗ-3 | слабая релевантность |
| Docker Hub (osint_pass7, 26.10): `evgps/ds3v2-koz3`, `niovol`, `restdockerse`, `azalexandr87`… | разное | образы/окружения | неизвестно | RF-DETR / smp / TerraMind+LoRA / YOLO fort | веса часто закрыты; проверять лицензии |

Поиск GitHub API 01.11 по терминам `koz3`, `expds`, `gorodishcha`, `compute_metrics`, именам топ-25 **не дал** новых публичных репо уровня Nikolove/CVшники [ф].

---

## 2. Хронология роста баллов (что реконструируется)

### 2.1. StrataTrace (наши снимки LB) — `STATUS_2026-10-29.md`, `AUDIT_2026-10-24.md`

| Дата (МСК) | Балл | Примечание |
|---|---:|---|
| 23.10 23:36 | 0,06140 | первая посылка (в основном геофизика) |
| 24.10 | 0,32850 | полный стек; Ae R≈0,013; Or≈0; geo R≈0,26 |
| 27.10 | 0,33352 → 0,34796 (#33) | |
| 28.10 | 0,34757 | |
| **29.10 08:47** | **0,41238** (#25) | **v2 `eb268d6`**; PIXGBM на LB; geo P/R **0,77/0,66** |
| после 29.10 | — | в этом отчёте новых публичных снимков нашего LB нет |

**P/R v2 (29.10):** lidar 0,52/0,50 · geo 0,77/0,66 · sat 0,46/0,25 · ortho 0,37/0,14 · aerial 0,09/0,013; recall корзины **5000–10000 м² = 0**.

### 2.2. EN / «Сахалинский центр ИИ» — публичный журнал `docs/EXPERIMENTS.md` + README

Источник: https://github.com/EhimenNathan/EXpedition/blob/main/docs/EXPERIMENTS.md (и ветки `codex/*`).

| Версия | Дата (из журнала) | Публичный LB [ф] | Что дало скачок (по их тексту) |
|---|---|---:|---|
| v2 | 22–23.10 | **0,17023** (#34/59) | ансамбль фолдов; LiDAR падал без laspy на платформе |
| v4 | 23.10 | 0,18455 | калибровка ансамбля |
| **v5** | 23.10 | **0,33021** (#25) | LiDAR заработал (vendor laspy) + dual/mask rule |
| v9 | 24.10 | **0,41693** | полный стек; Ae ещё «включён» с P≈0,045 |
| v10–v11 | 24.10 | → **0,44032** | LiDAR ensemble; **Ae выкл.** |
| **v12** | 25.10 | **0,48874** | sat framing prior, class hedge, 6 членов fusion; geo P↑ |
| v13 | 25.10 | 0,48634 | geo anchors / neighbour — смесь; чуть ниже v12 |
| v14 | 25.10 | **0,48906** | LiDAR neighbourhood prior; geo MagGr 0,75 |
| v15–v17 | 25.10 | 0,486–0,489 | hillfort snap / gates — на LB нестабильно (−0,002 у v17) |
| **v19b** | 26.10 | **0,49890** | **EtGr layer preservation** (не схлопывать срезы) → geo R↑ |
| **v20** | 26–27.10 | **0,50190** | recovery **aerial fortification FCOS** (+~0,003 на LB) |
| **v22** | ~27–28.10 | **0,50491** | зафиксировано как строка команды #9; репетиция 26 рег. завышает LB ~+0,026 |
| v23–v24 (ветки) | 28–29.10 | **на LB не подтверждено в открытых источниках на 01.11** | DINOv3/DINOv2 hillfort localiser (OOF sat +0,03); ряд идей **отвергнуты** (DINOv3 sat detector, raw LAS, WiSE, half-scale LiDAR…) |

Цель в их `WORKSPACE_README.md` (22.10): обойти лидера **0,57645** — это снимок лидера на ту дату, не наш текущий Nikolove 0,616.

### 2.3. Прочие команды

Полной хронологии LB по Nikolove / CVшники / Family и др. **нет**: нет публичных репо, нет открытых снимков таблицы кроме наших аудитов 24.10 / 29.10. WebSearch 01.11 не вернул полный публичный LB площадки.

Шум публичного LB у EN и у нас оценивается **SD ≈ 0,034** [ф, EXPERIMENTS / STATUS] — мелкие дельты < ~0,03 на одной попытке ненадёжны.

---

## 3. По конкурентам: подход, скачки, чего нет у нас

### 3.1. EhimenNathan/EXpedition (Сахалинский ЦИИ, #9, 0,505)

**Подход [ф, COMPETITORS_DEEP + EXPERIMENTS]:**

| Модальность | Модели / препроцессинг | Постобработка / пороги |
|---|---|---|
| **LiDAR** | LAS→DTM (PMF осторожно на ground-only); 9–12 каналов рельефа; U-Net ConvNeXt-T heatmap + mask components для гор/форт; paste-aug | dual emission; neighbour prior (R~200 м); ансамбль сидов; TTA d8; thr~0,15; малые якоря (круги ~1 м) |
| **Спутник** | fusion D-FINE + RT-DETRv2 + heatmap (+ balq); scene gate; source_layer isolation | frame prior; kurgan hedge; gor_snap; thr fused ~0,25 (arch 0,15) |
| **Геофизика** | CNN heatmap + physics candidates + **LightGBM / pixel-GBM**; отдельная обработка **каждого** EtGr-слоя | подавление пиков уже (~0,75 м у PIXGBM-линии); thr по LORO |
| **Ae** | FCOS на тайлах 1024; до v20 почти off; v20 — только fort | fusion только внутри снимка; LB Ae всё ещё ~0,10/0,09 |
| **Or** | тот же SpOr-детектор на орто @0,5 м | arch thr ≥0,45; LB Or ~0,71/0,24 (мало GT) |
| **Крупные** | mask components `gor_min_px` / `fort_min_px`; hillfort snap / localiser (v23–v24 эксперименты) | coarse 2 м ветка у них **отвергнута** на sat OOF |

**Скачки:** v5 (+laspy) → ~0,33; v9→v12 (optics+geo) → ~0,49; v19b (EtGr) → ~0,499; v20 (Ae fort) → ~0,502; v22 → **0,505**.

**У нас относительно EN (пути сравнения):**

| Тема | EN | Мы (`expds/`, `configs/default.toml`) |
|---|---|---|
| Geo recall на LB | ~0,72 (v14) | v2: **0,66** (PIXGBM включён на сборке; модуль held-out / provenance) — `expds/geo_pixgbm.py` (сборка), `docs/PIXGBM_DECISION.md` |
| Sat recall | ~0,64 (v14) | **0,25** — `expds/sp_*.py`, `[sp_dfine] enabled=false` в default |
| Ae / Or детекторы всех классов | слабый Ae fort; Or=SpOr | `[aeor_dfine]` / `[dfine_aeor]` **OFF**; нет весов Ae+Or в сдаче — `expds/aeor_dfine.py`, `expds/dfine_aeor.py` |
| Neighbour prior / kurgan-in-hillfort | deploy | флаги часто OFF / не в v2 — см. post / `configs/default.toml` |
| Large 5–10k м² | частичный via mask components | `[large_objects] enabled=false` — `expds/large_objects.py`; **R=0** на LB |
| Hillfort localiser | DINOv2/v3 эксперименты v23–v24 | `[gor_dense]` / `[hillfort_loc]` заготовки, не в v2-сдаче |

### 3.2. yakirill99/ds-expedition

Каркас без заявленного LB. Сильная сторона идей: **26 каналов** DEM/LAS (SLRM multi-σ, SVF, TPI/TRI, intensity, returns), рендер LAS в инференсе, канонические классы. Коммиты 28.10–01.11 — инфраструктура обучения, не скачок скора. **У нас:** меньше каналов в li-стеке; внешний AHN (CC0) у них только для проверки — нам для сдачи не нужен.

### 3.3. vaitter/koz3-build

Три YOLO-эксперта (E1/E2/E3). Идеи без AGPL-кода: отдельный Ae/Sp/Or путь, тайл 1024 / overlap 30 %, CLAHE, DEM 0,2 м + hillshade. Балла нет. **У нас:** нет полноценного Ae/Or детектора всех классов (см. default.toml комментарии bot-1/bot-6).

### 3.4. KateBuikina/expds-solution

Отдельные U-Net на SAT (SpOr+Ae, native GSD, tile 384) и ORTO (640, overlap 7 %, temperature calibration); скелет → цепочка точек. Балла КОЗ-3 нет. Идея: **не смешивать Or с Sp в одном ресемпле без калибровки**.

### 3.5. Family / mlart / GigaFlex (этап «Сканирование»)

По аудиту: отдельные модели по **физическому смыслу** данных, не по расширению файла; Family — «8 моделей» (Хабр, первоисточник в этой сессии не открыт). Прямого КОЗ-3 репо нет → перенос только как принцип «эксперт на тип».

### 3.6. Nikolove и топ-5 без репо

**Неизвестно:** модели, препроцессинг, даты скачков. Гипотеза аудита 29.10 («опыт прошлого этапа») — **не факт**, помечена как вывод.

---

## 4. Таблица разрывов (gap), ранжированная по ожидаемому вкладу в **наш** LB

Фокус: aerial, ortho, satellite, large objects. Оценки прироста — **[оц]** из `AUDIT_CLAUDE`, `COMPETITORS`, `BOARD` / `ae_ortho_diagnosis`, арифметики классов; не замерены на новой посылке после v2. Трудоёмкость — человеко-дни до дедлайна 11.11. Легальность: только clean-room; внешние веса — проверять LICENSE (DINOv2 Apache-2.0 ок по нашим докам; ultralytics AGPL — нет; DINOv3/SAM3 — не предлагаем по правилам репо).

| # | Разрыв | Ожид. вклад в наш LB [оц] | Труд | Риск | Легальность | Что у нас уже есть (пути) | Идея (не код EN) |
|---:|---|---|---|---|---|---|---|
| 1 | **Детектор всех классов на Ae** (тайлы, native/близкий GSD, overlap, thr по OOF) | **+0,03…0,08** (сейчас R=0,013; Ae — крупный полигональный пул на публичке по реконструкции EN) | 3–6 д | высокий (FP как у EN v9 P≈0,05) | clean-room D-FINE/FCOS на COCO-весах; без YOLO | `[dfine_aeor]` / `[aeor_dfine]` OFF; `expds/dfine_aeor.py`, `expds/ae_recall_relax.py`, `reports/ae_ortho_diagnosis.md` | отдельный эксперт Ae; fuse только внутри снимка; не включать «всё подряд» |
| 2 | **Детектор / перенос на Or** (не только `surf_arch`) | **+0,015…0,04** (R=0,14) | 2–4 д | средний | то же | `or = surf_arch_r18`; `[dfine_aeor]` or OFF | отдельная голова Or или аккуратный transfer Sp→Or + temperature/OOF thr |
| 3 | **Крупные объекты 5–10k м²** (грубый масштаб / mask components / area τ) | **+0,01…0,03** (корзина R=0; вес гор+форт) | 1–3 д | средний (ложные огромные боксы) | свои правила площади | `expds/large_objects.py` всё OFF; `gor_dense` OFF | включить/замерить `large_objects` + coarse window; не копировать EN decode |
| 4 | **Sat recall 0,25→≥0,4** (пороги OOF, включение/починка Sp D-FINE, frame/hedge аналоги своими словами) | **+0,02…0,05** | 2–4 д | средний (v4 у нас SpOr D-FINE выключали) | свои веса bot-9 | `[sp_dfine] enabled=false`; `expds/sp_dfine.py` | честные пороги + TTA при бюджете времени ~8× |
| 5 | **Ансамбль LiDAR / TTA / neighbour-prior (уже почти паритет P/R)** | **+0,01…0,03** | 1–2 д | низкий–средний | свои ConvNeXt | li стек в `expds/`; prior часто OFF | добить то, что уже в репо, без EN-кода |
| 6 | **Hillfort / gor_dense локализатор на Sp** | **+0,005…0,02** | 1–2 д | средний (мало объектов) | DINOv2 probe Apache-2.0 | `expds/gor_dense.py`, `[gor_dense]` OFF | включить после honest-10 |
| 7 | **EtGr / geo layer isolation** (если ещё где-то схлопываем) | **+0…0,02** (у нас geo уже 0,66 после PIXGBM; у EN скачок v19b был с более низкой базы) | 0,5–1 д | низкий | своя реализация | PIXGBM + geo pipeline | аудит дедупа по `markup`/слою |
| 8 | **Калибровка по участку / реранк** | **+0…0,015** (rerank_all LOFO −0,012 — отвергнут) | 2+ д | высокий | — | `post.site_calibration` в планах аудита | только с честной приёмкой |

**Не включать в план как «взять у EN»:** копирование `koz3_infer`, PIXGBM binary paste, `li_decode`, oracle-тесты против EN, AGPL YOLO-веса, данные КОЗ-2.

---

## 5. Топ упущенного (5–8) с оценкой прироста

1. **Полноценный Ae-детектор (не только fort / li_ae-копии)** — ожид. **+0,03…0,08** LB [оц].  
2. **Or сверх архитектуры** — **+0,015…0,04**.  
3. **Крупные 5–10k м² (R=0)** — **+0,01…0,03**; код `large_objects` уже есть, флаги false.  
4. **Sat recall** (пороги + опционально Sp D-FINE) — **+0,02…0,05**.  
5. **Дожать LiDAR ensemble/TTA/prior из уже лежащего в репо** — **+0,01…0,03**.  
6. **gor_dense / hillfort point на Sp** — **+0,005…0,02**.  
7. **Проверка дедупа EtGr/слоёв** — **+0…0,02** (убывающая отдача после PIXGBM).  
8. **Бюджет времени площадки (~4 мин из 30)** — мультимасштаб/TTA без смены моделей — множитель к п.1–5.

Сумма верхних оценок **не складывается линейно**; реалистичный коридор дотягивания к ~0,50 при закрытии Ae+Or+large+sat — порядка **+0,08…0,15** к 0,41 **[оц, AUDIT]**, не гарантия 0,616.

---

## 6. Что неясно (не выдумывать)

1. Архитектура и хронология **Nikolove (0,616)** и мест 2–8 — публичных репо нет.  
2. Полная таблица LB после 29.10 в открытом доступе не найдена.  
3. Залит ли у EN **v23/v24** на площадку (в доках — репетиции/OOF; публичный зафиксированный — v22 0,50491).  
4. Точный вклад Family «8 моделей» и Docker-образов (`evgps`, `niovol`…) в текущий LB — нет привязки имён.  
5. Насколько наш PIXGBM на скрытой выборке повторит публичные 0,66 R (провенанс / hold-out модуля).  
6. Числа Хабра по «Сканированию» не перечитаны в этой сессии (сеть).  
7. `/workspace/jury-osint.md` и `competitors-v3.md` — **другой конкурс (ФСП «Талант»)**, не использовать как источник по КОЗ-3.

---

## 7. Источники (ссылки)

- Наше репо: https://github.com/KonkovDV/expds-koz3  
- EN: https://github.com/EhimenNathan/EXpedition  
  - EXPERIMENTS: https://github.com/EhimenNathan/EXpedition/blob/main/docs/EXPERIMENTS.md  
  - WORKSPACE_README: https://github.com/EhimenNathan/EXpedition/blob/main/docs/WORKSPACE_README.md  
  - коммит bot8-спеки: https://github.com/EhimenNathan/EXpedition/commit/d434e56  
- Форк: https://github.com/maratgaliulin/EXpedition  
- yakirill99: https://github.com/yakirill99/ds-expedition  
- vaitter: https://github.com/vaitter/koz3-build  
- KateBuikina: https://github.com/KateBuikina/expds-solution  
- MataNerdy: https://github.com/MataNerdy/Geodata_Archaeology_CV  
- Внутренние (origin/main): `docs/COMPETITORS_2026-10-29.md`, `docs/MURDER_BOARD_2026-10-29.md`, `docs/PLAN_PIVOT_2026-10-29.md`, `docs/AUDIT_CLAUDE_2026-10-29.md`, `docs/AUDIT_2026-10-24.md`, `docs/STATUS_2026-10-29.md`, `docs/research/2026-10-28/COMPETITORS_DEEP.md`, `docs/MERGE_2026-10-29.md`, `docs/PIXGBM_DECISION.md`, `configs/default.toml`, `expds/large_objects.py`  
- ФСП (не КОЗ-3): `/workspace/jury-osint.md`, `/workspace/competitors-v3.md`

---

*Отчёт собран 01.11.2026 (Europe/Moscow). Read-only: код репо не менялся, push не выполнялся. GITHUB_TOKEN использовался для API, в текст не выводился.*
