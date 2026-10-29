# bot-7 STATUS (геометрия под метрику)
- [x] COMMON.md прочитан; лимиты: OMP_NUM_THREADS=2, nice 10, RSS ≤ 1.5 ГБ (подвыборка top-6000 на (класс, группу) = тот же балл, work/base_sub.py).
- [x] Аналитика + МК форм: work/mc_shapes.{py,txt,json}
- [x] Статистика GT/предсказаний: work/geom_stats.txt; диагностика промахов: work/diag.txt
- [x] Дубли в эталоне: work/gt_dups.txt (объясняет, почему склейка дублей вредна)
- [x] Генератор: repo/expds/geom_opt.py (ветка bot7-geom-opt)
- [x] Точная честная оценка: work/eval_geom.py → eval_geom.pkl/log; сводка work/summarize.py → eval_geom_summary.txt
- [x] REPORT.md (готово)

## Задача 2 (28.10.2026, МСК): пакет мелких правок — ГОТОВО (20:45 МСК)
- [x] ae-маска фортификаций bot-6 — уже в базе (default.toml `[models] ae`, группа ae_fort; абляция bot-10 +0,00524) → не применялась
- [x] Ветка bot7-bundle 42ada8d (post.micro_circles bot-5 + радиус по классу, хук watchdog, oof_transform micro --radius-map, блок конфига) → bundle.diff
- [x] Ветка замера bot7-bundle-measure (инструменты bot-10 + scripts/micro_fam.py) → bundle_measure_tools.diff
- [x] База воспроизведена: honest-10 0,3146153226828423 (кривые побитово = bot-10/work/curves)
- [x] Замер по EXPERIMENTS.md: work2/hb_{kur,arch,bundle}.json, breakdown.json, fixed_tau.json, summary.txt
- [x] Пакет +0,00245 [+0,00076; +0,00447], P 0,995, LOFO +0,0016, 5/5 → принят; курганы +0,00137 (P 0,953), архитектура +0,00108 (P 0,867) — только пакетом
- [x] REPORT2.md
- Нужно от ПК: перекалибровка порогов на OOF после oof_transform micro (REPORT2 §5); без push, без LB
- [x] Обзор конкурентов (только чтение): NOTES_recara-cv-hackathon-nsk.md, NOTES_KateBuikina-expds-solution.md — оба решения КОЗ2, заявленных 0,5–0,6 в коде нет

## Задача 3 (28.10.2026, МСК): кандидат на посылку — ГОТОВО (~21:20 МСК)
- [x] Состав изменён по просьбе бота 1 (21:02): курганы 0,5 м без микрокругов архитектуры; crop-fix bot-8; реранк архитектуры ВКЛЮЧЁН
- [x] Ветка bot7-candidate: 42ada8d, 07bed29, dd7982e, 733d3ed, fe0e570 (дерево e75e311); прежний вариант — bot7-candidate-v1 (913e2be)
- [x] Доказательства: bot-5 analysis/joint/summary.json (RM−R в среднем −0,00035); bot-10 work4/rerank/*.log (k10 +0,0095, LOFO +0,0079, 5/5); EXPERIMENTS rerank +0,00974
- [x] pytest 315 passed; toml diff = 5 ключей; сборщик репо → zip; check_submission PASS 28; время реранка 6,6 с / 697 МБ на 206 тыс. объектов
- [x] candidate.diff, candidate.mbox, REPORT3.md
- Заблокировано (F:): честный перемер пакетом с crop verify, platform_sim на mock_val

## Задача 4 (28.10.2026, МСК): финальный кандидат + twins (выкл.) + reliability bot-8 — ГОТОВО (22:45 МСК)
- [x] bot7-candidate 4c3e29e: stack6 separate (c71af0b), курганы 0,5 м, rerank_arch ВКЛ, crop fix bot-8, reliability bot-8 (2dd45fa 2621e33 7a9f1d1 fd7dd50 = bot8-reliability-on-bot7-candidate, дублей нет), ae_fort в базе (тест), kurgan_twins_only bot-6 (1d8fc71) ВЫКЛ; без микрокругов архитектуры
- [x] Порядок safe_finish = finish (проверен, REPORT3 §2)
- [x] Guard writer.enforce_limits (f4d0159); прокси лимитов 103 семей OK; cap_all 54016→20000; RSS 2291 МБ на объединённом прокси — риск
- [x] pytest 327 passed / 9 skipped; check_submission PASS (синтетика 52,9 с / 979 МиБ; реальный 210_Гнёздово_Терраса_2 10,2 с / 313 МиБ)
- [x] Честно (коробка, без crop, keep): stack6+kur +0,016428 P 0,994 LOFO +0,015121 5/5; twins +0,000352 P 0,973 (work4/hb_pkg*.json)
- [x] ПК: work4/pc/pc_measure_pkg.ps1 (без twins / -Twins), только C:; candidate.diff/.mbox, measure_tools.diff, REPORT3.md
- Ждёт ПК: перемер пакета с crop (stack_verify_sp.geojson только на F:); решение по twins — с пользователем; LB — только с согласия Шаундры

## Задача 5 (28.10.2026, МСК): DEM грубый проход городищ (продолжение за bot-4) — ГОТОВО (22:43 МСК), вывод: НЕ БРАТЬ
- Конвейер bot-4: `bot-4/work/dem/run.sh`, `dem_eval.py`, `run.log`, `bot-4/STATUS.md` (пауза 20:58, готово 25 LAZ, 004 был прерван, лог пустой). Папка bot-4 — только чтение, поэтому запуск в своей копии: `bot-7/work/dem/run.sh` (код и venv bot-4 как есть; пропускает готовое в `bot-4/work/dem/out` и в своём `out`; шлюз памяти MemAvailable ≥ 2,5 ГБ; по одному файлу; nice 10, taskset 0,1, OMP/MKL/OPENBLAS/NUMEXPR/NUMBA=2, RAYON=1, ulimit -v 3 ГБ). Запуск из `bot-7/`: `nohup work/dem/run.sh >> work/dem/run.log &`.
- Обработано 34 LAZ, все rc=0: 004, 007, 028 + 31 файл, докачанный после 20:58 (034–045, 061, 066, 068, 069, 101–106, 179–182, 186–192, 200). Итого с bot-4 покрыто 59 участков с LAS. Ошибок нет, `.tmp` нет. Лог: `bot-7/work/dem/run.log`, `logs_*.txt`, `out/*.geojson`.
- Время: 22:21:53 → 22:36:30 МСК (14 мин 37 с); самые долгие 187_Судьбищи_1 142 с (огрублён до 1 м), 190_Муравлево 135 с, 037_Никола_Ленивец 110 с (+30 с ждал память). Ресурсы (`work/dem/mon.log`, проба каждые 5 с): пик RSS 741 МБ, load машины 11–25 (чужие процессы), минимум MemAvailable 1093 МБ.
- Кандидаты: 30 новых в 14 участках; всего у грубого прохода DEM 77 (bot-4 47 + bot-7 30). С эталоном городищ сопоставимы только 5 кандидатов (8 эталонов). **Ни одного нового эталона относительно кандидатов базы A**: все 8 уже покрыты (`work/dem/recall_chk.py` → `res_recall.json`; база покрывает 56/118).
- Честная оценка (`bot-7/work/dem/dem_eval.py all all` = копия dem_eval.py bot-4 + по классам, LOFO, 5 раскладок; лог `eval_all.log`, результат `res_dem_all.json`; пик RSS 813 МБ, ~5 мин). База A honest-10 0,314615 (= EXPERIMENTS.md), all-data 0,365386.
  - Δ honest-10 = **−0,000117**; парный бутстреп по 103 семьям (2000): среднее −0,000134, ДИ90 [−0,000399; 0,000000], **P(Δ>0) = 0,000**.
  - По классам: меняется только gorodishcha: F1 0,27933 → 0,27856 (ΔF1 −0,00078; взвеш. −0,000117); TP 25→25, FP 72→73. Остальные классы 0.
  - LOFO Δ −0,000114; 5 случайных раскладок: −0,000119 / −0,000226 / −0,000121 / −0,000117 / −0,000121 (0 из 5 > 0). All-data Δ 0 (sweep ставит порог группы gor_coarse_dem 0,998, оставляет 0). На покрытых 59 участках Δ −0,000385 (gor F1 0,5076 → 0,5050). Масштаб conf 1,0 и 0,4 — одинаково (отдельная группа порогов).
  - П1 ✗, П2 ✗, П3 ✗ → **отклонено**.
- Почему: потери FN по городищам в GAP.md — участки **без лидара** (0/73 найдено, 57/91 FN без кандидата), DEM на LAS туда не достаёт; на лидаре полнота кандидатов уже 28/30. Нужен поиск городищ по орто/аэро/спутнику, а не ещё один проход по ЦМР.
- Блокеров нет (F: не нужен; файлы все на месте; память хватило). Репозиторий bot-7/repo и его ветки не трогал.

## Задача 6. Городища без лидара: генератор кандидатов по оптике (28.10.2026, 23:20 МСК)
Папка `work/gor/`. Метки в генераторах не используются; выбор варианта — только на обучающих фолдах.

**Что уже делает bot-5** (`bot-5/work/gor_patch/extract_feats.py`, `gor_patch_eval.py`, NOLIDAR.md §2.2, §9): сплошная сетка виньеток SpOr 384 м (шаг 192 м) + контекст 768 м → DINOv2-S, лог-регрессия, точка = polylabel → круг 10 м. Покрывает **только SpOr** (175 файлов; на 23:00 извлечено 143/175). Ae/Or не покрыты. Свой генератор «где внутри виньетки» у bot-5 — пик p по виньеткам (точность ~192 м). Формат фрагмента — строка `meta.json` (`reg, file, ux, uy, corners, gsd`), id = индекс.

**Данные.** Стерео-/SRTM-DEM на боксе нет (`find` по train-expds-3: только laz/tif Li/SpOr/Ae/Or) — генератор «мыс/бровка» не делал, ничего не качал. 214 оптических сцен (`scenes.json`; 008_Герите Or пропущен: битый gsd, 5,2 ГБ). 99 из 118 эталонов городищ лежат в оптических сценах, все 73 безлидарные — да (60 sp_or, 11 ae, 2 or).

**Шаг 0, существующий стек** (`g0_stack.py` → `g0_stack.json/.log`): кандидаты класса gor (любой conf) официально совпадают с 15 из 73; центроид ≤50 м — 10, ≤100 м — 16.

**Генераторы** (`gen.py`, 2 м/пикс SpOr/Or, 3 м Ae; лог `gen.log`, 12 мин, RSS ≤ 504 МБ): радиальная симметрия FRST (светлый/тёмный центр/abs) — кольца и дуги валов/рвов; LoG-пятна; плотность сильных градиентов; радиусы 15–140 м, NMS 40 м. Плюс `centre` (центр сцены) и `stack` (реп. точки кандидатов gor из стека, любой conf). Кандидат = круг 10 м (официальный матч = центр внутри эталона, проверено `_match_quality`).

**Recall кандидатов** (`cand_eval.py` → `res_cand_D.json`, `res_cand_K.json`; офиц. `_match_quality`; вариант генератора выбран по recall на 9 обучающих фолдах):
| бюджет | honest-10 все (из 99) | honest-10 без лидара (из 73) | LOFO без лидара | 5 раскладок без лидара | кандидатов |
|---|---|---|---|---|---|
| только centre (1/сцену) | 28 | 23 | — | — | 214 |
| только stack | 32 | 17 | — | — | 1 474 |
| centre+stack | 48 | 33 | — | — | 1 688 |
| +K=10 на сцену | 53 | 38 | 38 | 38/38/38/38/40 | 3 598 |
| +K=50 на сцену | 59 | 44 | 46 | 46/43/43/54/43 | 9 387 |
| +D=10 /км² | 61 | 42 | 44 | 47/44/40/44/41 | 16 202 |
| +D=50 /км² | 74 | 52 | 52 | 51/52/52/51/60 | 59 029 |
| +D=200 /км² | 83 | 60 | 60 | 67/59/60/60/60 | 145 478 |
Справка (без выбора, `fixed`): регулярная сетка без centre/stack при D=50 — 40/73 (72 351 канд.), D=200 — 68/73 (288 812). Объём при D=50 (honest-10, `volume_honest10`): SpOr без лидара 55 канд./сцену (56/км², 162 сцены), Ae без лидара 2 610/сцену (39/км², 16 сцен, 1 077 км²).
Экспорт D=50: `cands_gor_oof_D50.json` (UTM) и `.geojson` (EPSG:3857), 59 029 канд.; независимая проверка `verify_export.py` → `verify_D50.log`: 74/118 всех, 52/73 без лидара (ae 10/11, or 2/2, sp_or 40/60). Попадания без лидара по генератору: frst 48, centre 23, stack 17. Пропуски — в основном Кисловодская котловина (15 шт.), 075, 165, 166.

**Формат**: `work/gor/FORMAT.md`; инструкция для bot 10: `work/gor/README_bot10.md`.
**Ограничения/риски.** (1) Это recall кандидатов, не балл: сами баллы генераторов фон от городища не отличают; прирост метрики зависит от классификатора bot-5 и меряется bot 10. (2) `centre` опирается на то, что сцены SpOr вырезаны вокруг объекта; на тесте может не держаться. (3) `frag_bot5` заполнен для 9 962 кандидатов, так как meta.json bot-5 неполный (идёт извлечение); после завершения нужно перезапустить `export.py` (~30 с). **[23:40: проверено — экспорт уже по финальному meta.json, изменений 0, см. дополнение 2]** (4) Один эталон 164_… — точка (площадь 0), кругом 10 м его не взять.

### Задача 6, дополнение: центр сцены на ТЕСТЕ (28.10.2026, 23:25 МСК)
**Где тест.** Сырья тестовых сцен на box нет (в `/workspace/data/` только трейн; S3-опись `s3_listing.txt` — только трейн). Есть опись организаторов публичной выборки: `koz3_shared/competitors/yakirill99-ds-expedition/vendor/organizers/data_structure.md` (48 участков `region_XXXX`, только имена файлов и размеры) и её сводка `expds-koz3/notes/public_val_2026-10-22.json`. `notes/site_overlap_public.json` НЕ доказательство: посчитан по mock_val из копий трейна (PLAN.md, срез 27.10). Поэтому сравнение — по размерам файлов, именам и числу сцен на участок; площадь/экстент/смещение GT на тесте посчитать нельзя (нет координат и пикселей).
Скрипт `work/gor/test_scene_check.py` → `work/gor/testchk/test_scene_check.{json,log}` (размер трейна — из S3-описи, МиБ).
| | тест (публичный) | трейн |
|---|---|---|
| сцен SpOr / участков | 31 / 30 | 175 / 132 |
| участков с ≥2 сценами | 1 (2 × 3 МБ) | 19 |
| файлов < 15 МБ | **14 (45 %)**, 1,4–3,7 МБ (+1 × 10,8) | 153 (87 %), p10–p90 2,8–4,5 МБ |
| 15–150 МБ | 0 | 9 |
| ≥ 150 МБ | **17 (55 %)**, 186–550 МБ | 13 |
| медиана | 191,5 МБ | 4,0 МБ |
Имена: тест `01_region_XXXX_SpOr_спутник.tif`, обезличены; шаблон тот же, что в трейне (`01_<Имя>_SpOr_спутник.tif`), имена объектов не видны.
Трейн по корзинам размера (`train_bins`): мелкие (<15 МБ) — 153 сцены, площадь p50 50 га, у 48 есть городище, смещение центроида GT от центра сцены p50 **77 м** (0,1 полудиагонали), 30 из 48 < 100 м. Крупные (≥150 МБ) — 13 сцен, площадь p50 258 га, GSD ≈ 0,1 м, городище в 3, смещение p50 **361 м** (0,3 полудиагонали), <100 м — 1. Средние — 4 городища, смещение 251–509 м, <100 м — 0.
**Вывод.** На тесте две популяции: 14 мелких сцен по размеру — копия трейновых кропов (там центр работает), 17 крупных — как 13 крупных сцен трейна, где городище НЕ в центре. Признаков, что крупные сцены теста вырезаны вокруг объекта, нет. Доля мелких на тесте 45 % против 87 % в трейне, так что трейновый recall центра завышен для теста.
**Сколько даёт центр сверх остальных генераторов (OOF, `verify_export.py`).** Полный D50 52/73 без лидара (`verify_D50.log`); **без центра 48/73** (`testchk/verify_D50_nocentre.log`, 58 815 канд.); **центр только в мелких сценах (<15 МиБ) 52/73** (`testchk/verify_D50_centre_small.log`, 58 976 канд., 161 центр). Все 4 эталона, которые берёт только центр, лежат в мелких сценах. Без центра выбор варианта по фолдам не переобучался (фильтр gen≠centre), это консервативно.
**Рекомендация.** В сдачу — `nocentre` (по правилу bot 1). Bot 10 мерит три варианта: полный, `nocentre`, `centre_small`. Если `centre_small` пройдёт честно, правило на тесте — размер файла сцены < 15 МБ (1 кандидат на сцену, на тесте ≤ 14 FP-рисков).
Файлы: `work/gor/cands_gor_oof_D50_nocentre.{json,geojson}`, `work/gor/cands_gor_oof_D50_centre_small.{json,geojson}` (тот же формат и cand_id, что у D50), скрипт `work/gor/export_nocentre.py`, лог `testchk/export_nocentre.log`.

### Задача 6, дополнение 2: финальный meta.json bot-5, его p и комбинированный балл (28.10.2026, 23:40 МСК)
Папка `work/gor/` (служебное — `work/gor/comb/`). Запуски: 1 поток, nice 10, сторож RSS (`comb/run.sh`), max RSS ≤ 942 МБ.
**Проверка заявлений bot-5.** `meta.json`: md5 `4ce45bd60826a57655f632ed3f5cf01c`, 3 982 строки, `feats.npz` F = 3982×1536 — совпадает (`comb/refrag.log`). `scores_cand7.json`: 11 870 кандидатов, без дублей, все cand_id есть в D50; SpOr 9 965 / Ae 1 872 / Or 33; не оценено 47 159 (`scores_cand7_report.json`). **p воспроизведён** копией скрипта bot-5 с проверками утечки (`comb/score_cand7_repro.py` → `comb/score_cand7_repro.{json,log}`, `comb/repro_compare.log`): max |Δp| = 6,9e-10, patch и fold совпадают у всех 11 870. В каждом из 10 фолдов (поле `fold`) assert: среди обучающих виньеток нет участков фолда и нет их семей (`site_family`) — OK (лог: «leak check … OK»). Наши фолды семьи не делят (`split_families: {}`). AUC по генераторам (метка — офиц. совпадение круга 10 м) повторяется: p 0,741 на оценённых; frst_dark p 0,786 против нашего score 0,596; stack_gor_phys p 0,117 против score 0,828 (`comb/comb_eval_D50.log`, `by gen`). Уточнение: 0,596 — AUC нашего сырого score внутри frst_dark, а не «комбинированного балла».
**frag_bot5** (`refrag.py`, правило как в `export.py`): заполнено 9 962 из 59 029 (все SpOr), **изменилось 0** (`comb/refrag_stats.json`). Экспорт 23:17 уже шёл по финальному meta.json (mtime 23:11); риск (3) выше снят. nocentre — 9 797, centre_small — 9 940. 727 «расхождений» у bot-5 — не ошибка meta: у bot-5 ближайшая виньетка ищется по всему участку (любой файл SpOr), у нас — в той же сцене (reg, file) по FORMAT. Все 727 — участки с ≥2 снимками SpOr (напр. 01/02_Шумгора), виньетка bot-5 из другого файла ближе. По той же причине 3 SpOr, 1 872 Ae, 33 Or имеют p без frag_bot5.
**Комбинация** (`comb_eval.py`; шкалы без меток: u_own — перцентиль score среди всех кандидатов файла, u_p — перцентиль p среди оценённых). Для оценённых при gen ≠ stack_gor_phys: w·u_p + (1−w)·u_own, иначе u_own. w ∈ {0; 0,25; 0,5; 0,75; 1} выбирается по AUC на обучающих фолдах: honest-10 — w = 1 во всех 10 фолдах, LOFO — во всех 77. То есть score_comb = «p, где есть, иначе свой балл». Справочно — заранее заданное ранговое среднее w = 0,5.
Recall — по эталонам без лидара из 73 (в скобках все из 118); «на сцену @k» = топ-k в каждой из 214 сцен; global @B = общий топ-B.
| файл | балл | AUC канд. все / без лидара | на сцену @1 | @3 | @10 | global @500 | @2000 | @10000 |
|---|---|---|---|---|---|---|---|---|
| D50 полный (покрытие 52/73) | свой score | 0,617 / 0,647 | 1 (1) | 5 (5) | 23 (23) | 0 | 3 | 24 (29) |
| | **score_comb (w=1, honest-10 = LOFO)** | **0,707 / 0,760** | 2 (4) | 10 (12) | 26 (31) | 4 | 9 | 28 (29) |
| | ранговое среднее w=0,5 | 0,676 / 0,720 | 1 (1) | 13 (15) | 29 (33) | 0 | 5 | 29 (32) |
| **nocentre** (покрытие 48/73) | свой score | 0,598 / 0,630 | 1 (1) | 6 (6) | 16 (16) | 0 | 3 | 24 (29) |
| | **score_comb** | **0,703 / 0,760** | 2 (4) | 11 (13) | 25 (30) | 4 | 9 | 28 (29) |
| | ранговое среднее w=0,5 | 0,663 / 0,710 | 1 (1) | 8 (8) | 23 (27) | 0 | 5 | 26 (29) |
Только оценённые bot-5 (D50: 11 870 канд., 172 положит., покрытие 46/73): свой score AUC 0,564, **p 0,741**, w=0,5 0,714; на сцену @3 — 5 / 10 / 14; global @1000 — 9 / 16 / 25 (`comb/comb_eval_D50.json`, раздел `scored`). nocentre: 0,538 / 0,743 / 0,697.
Бутстреп по семьям (1000, `boot`), разность score_comb − свой балл, AUC без лидара: D50 +0,113 [+0,026; +0,226], nocentre +0,129 [+0,029; +0,268]. Recall @3 на сцену: +5 [−1; +12], доля > 0 — 0,92.
**Файлы:** `cands_gor_oof_D50_comb.{json,geojson}`, `cands_gor_oof_D50_nocentre_comb.{json,geojson}`. Формат прежний, плюс `p_bot5` (null, если не оценён) и `score_comb`; в geojson confidence = score_comb, сырой балл — `score`. `export_comb.py`, `comb/export_comb.log`. Recall файлов не изменился: `comb/verify_*.log` — 52/73 и 48/73.
**Оговорки.** (1) Это ранжирование кандидатов, а не балл: официальная метрика (1-к-1, FP = 0,5 FN, пороги sweep_groups) — у bot 10. (2) Абсолютный recall в топе мал (@1 на сцену 2/73). Кандидаты одной сцены в радиусе 192 м делят одну виньетку и одинаковый p, их нужно гасить NMS. (3) 47 159 кандидатов (в основном Ae без снимка SpOr) идут со своим баллом, а у него AUC около 0,6. (4) score_comb — перцентили внутри файла; на тесте их надо считать по тестовым кандидатам (без меток). (5) w = 0,5 даёт больше @3/@10 на сцену на полном D50, но выбор шёл по AUC заранее, а на nocentre w = 1 лучше по всем метрикам. Поэтому в экспорте w = 1.

### Задача 6, итог bot-10 (29.10.2026, 00:55 МСК): городища без лидара — REJECTED
- bot-10 отклонил нелидарных кандидатов городищ: основной вариант `cands_gor_oof_D50_nocentre_comb` (conf = score_comb, NMS 192 м) — Δ **+0,000473** vs A, **P 0,698**, ДИ90 [−0,00106; +0,00211], LOFO +0,002042, 5/5 раскладок > 0, new34 0; П1 (<0,002) и П2 (P<0,85) не выполнены. **Нового TP нет**: gorodishcha TP 25 → 25, FN 93 → 93, FP 72 → 68 — весь плюс от снятия FP (`bot-10/work8/gor_diag.log`). `_comb` и `_centre_small` — тоже +0,000473. Источник: `bot-10/REPORT8_gor.md` (строки 5–11, 15, 36–39), логи `bot-10/work8/paired_gorc_noc.log/.json`, `paired_gorc_noc_site.log`, `paired_gorc_ref.log`; `bot-10/STATUS.md` 29.10 00:20.
- В кандидат v2 не входит. Дальше по городищам без лидара — только если появится классификатор, дающий новые TP (recall кандидатов 48–52/73 есть, ранжирование слабое: @1 на сцену 2/73).

## Задача 7. Курганы на аэрофото (Ae) без лидара: поиск кандидатов (28.10.2026, 23:25 МСК) — ВЫБОР И ПЛАН
**Почему.** GAP.md §2: потеря FN povr Ae 0,072 + tselye Ae 0,038 = 0,110 (самая крупная незанятая строка). Диагностика `work/kur/diag0.{py,log,json}` по table.pkl: на участках без лидара у курганов Ae кандидат есть у **58 из 510** эталонов (povr 53/359, tselye 5/151), т.е. это дыра поиска, а не порогов (у sp_or без лидара покрыто 732/1170, там ранжирование и crop на ПК). 12 семей (Гнёздово, Гочево, Изборск, Сельцо, Сож, Шниткино, Вышегжа, Шишкино…), 16 сцен Ae без лидара, GSD 0,15–0,98 м. Своей сети курганов на ae в сдаче нет (bot-6 STATUS); twins bot 6 покрывают только участки с лидаром.
**План (2–3 строки).** (1) Классический генератор на CPU по сценам Ae окнами (≤2 ГБ): LoG/FRST-пятна масштаба кургана (r 3–15 м) на 0,5–1 м + признаки окна; recall по семьям. (2) LightGBM-оценка кандидатов с обучением вне семьи (honest-10, LOFO), круг 0,5 м как в сдаче. (3) Добавить как новую группу в стек и мерить официальной метрикой по EXPERIMENTS.md: парный бутстреп, honest-10, LOFO, 5 раскладок, пороги sweep_groups, unseen одинаково; ни одной цифры без файла.

### Задача 7 — экспорт кандидатов для bot-5 (готов, 29.10 00:05 MSK)
- Файл: `/workspace/koz3_shared/bot-7/work/kur/cands_kur_ae.json` — 430 586 кандидатов (D ≤ 400 на км² сцены по рангу score), 16 нелидарных Ae-сцен / 14 участков.
  Поля: cand_id, reg, file, ux, uy (UTM), epsg, gsd, gen (масштаб LoG: log2…log13 = r м), score, rank (в сцене), D, fold (honest-10 = индекс в FOLDS из /workspace/rerank/work/base_honest.pkl).
  Полный набор (1 103 577, все D) с признаками и метками — `work/kur/cands_all.npz` (cand_id = индекс строки; cands_kur_ae.json — его подмножество с теми же id).
- Генератор `work/kur/gen_ae.py` (многомасштабный LoG ±, r=2…13 м, сетка 0,5 м или GSD, NMS 4 м, дедуп 3 м, ≤1500/км²). Метки нет в генерации.
- Recall (официальный _match_quality, круг r=2 м, `work/kur/recall_R2.0.json`): povr 132/359 при D≤400, 163/359 во всех; tselye 44/151 при D≤400, 118/151 во всех. Для сравнения у стека были кандидаты у 53/359 и 5/151 (diag0.json).
- Ждём от bot-5: `/workspace/koz3_shared/bot-5/analysis/kurg_ae/scores_cand7.json` [{cand_id, p, fold}].

## Задача 5. Кандидат v2 ветки сдачи (29.10.2026, ~00:40 МСК; обновлено 00:55 МСК) — ГОТОВ, twins ВКЛ, ждёт перемера на ПК и решения bot-10 по порогам *_pix
- Ветка `bot7-candidate-v2` (от 4c3e29e), голова **af97af0** (= b1a9d28 + `[post] kurgan_twins_only = true`), без push. Отчёт: `REPORT4.md` (§8 — обновление). Патчи перегенерированы: `work5/candidate_v2.{diff,mbox}` (10 патчей), `work5/candidate_v2_from_24bd916.mbox` (24); `git am` на 4c3e29e и 24bd916 даёт дерево af97af0.
- Вкл.: PIXGBM S + пороги *_pix bot-9 (bot-10 ещё выбирает bot-9 vs bot-3 — пока bot-9) + stack6 (pix→stack6) + **twins-only bot-6 с починкой** (приёмка bot-10: `bot-10/work8/twins78_stats.log` TO-A +0,001718, ДИ90 [+0,00039; +0,00377], P 0,986, LOFO +0,001603, 5/5; kurTO-A +0,003065, P 1,0). bot-8 cpu-a не включён (нет весов (а)).
- Точная оценка в порядке сдачи (`work5/exact/geo_stats.log`): Ss6−A keep +0,013938 / drop +0,011009; Ss6−S +0,001304 / +0,000061.
- af97af0: pytest 337/9/3 desel., 57 с, RSS 1198 МБ (`work5/pytest_af97af0.log`); сборка `work5/dist/submission_bot7cand_v2_af97af0.zip` 525 149 404 Б, sha256 a7b618dee86831616314d389517dea7c0a710b55d08a349d8fdb3b795dd95643 (`build_af97af0.log`; от b1a9d28 отличается только default.toml); check_submission PASS: синтетика 26 (5,3 с, 992 МиБ), 210 --lock 27 (23,5 с, 670 МиБ), 214 --lock 27 (13,2 с, 440 МиБ), 217 --lock 27 (79,5 с, 633 МиБ). Выходы = b1a9d28 (на этих регионах курганов нет).
- ПК: `work5/pc/pc_measure_v2.ps1` (пример -Zip → af97af0, распаковка в `zip_<имя>`, UTF-8 BOM для PS 5.1) + `tools_v2.tar` (не менялся) + `SHA256SUMS` (обновлён). Только C:.

### Задача 7 — предрегистрация замера ae_kur_b7 (29.10.2026 00:42 МСК, до запуска eval_kur.py)
- train_oof.py завершён 29.10 00:09 МСК (не перезапускался): `work/kur/train_oof.json`, `oof.npz`, `logs/train.log`.
- ОСНОВНОЙ вариант (зафиксирован до замера, параметры скрипта не меняются): `eval_kur.py`, D_MAX=400, круг r=2 м, conf = OOF p_povr / p_tsel (scale 1), группа `ae_kur_b7`, оба класса (`both`), scope global, unseen drop. Решение — только по нему (П1–П5).
- Вторичные (справочно, отбор по ним не делается): povr-only, tsel-only, stackfirst, unseen keep, scope site.
- Предрегистрация комбинации с bot-5 (29.10 00:58 МСК, до появления файла bot-5): `eval_kur_comb.py` (копия eval_kur.py, та же схема). MODE=comb: conf = 0,5·перцентиль(OOF p bot-7) + 0,5·перцентиль(p_povr/p_tsel bot-5), перцентили по всем 430 586 кандидатам без меток; нет оценки bot-5 (ok=false) → его перцентиль 0. Основной: both / global / drop. Справочно MODE=b5 (только bot-5). Сверка: все cand_id ⊂ наших, fold совпадает (assert).

- 29.10 01:45 MSK: исправлен pc_measure_v2.ps1 (замечание бота 1): в curves_S/s6/Ss6 копировались все pkl из curves_A → 3 изменённые семьи не пересчитывались (Δ=0 на ПК). Теперь pkl копируются только для единиц, побайтно совпадающих с fam_A; недостающие jsonl докладываются из fam_A; старые curves_* без _done_v2 удаляются; changed_families.txt. sha256 aaf89ad20c183bc90caef56b862b9a8b25934e5d94cc017308ae7da6958da9e1. Бэкап: work5/pc/pc_measure_v2.ps1.bak_before_fix.

- 29.10 02:00 MSK, задача 7 (курганы по Ae без лидара): ОТКЛОНЕНО. OOF: повр. AUC 0,744 / AP 0,0025; целые AUC 0,435. Метрика (eval_kur, ae_kur_b7, D400): Δ к A −0,00037 (keep и drop), CI95 [−0,00129; 0], P 0,000, LOFO −0,00036, раскладки 0/5 (+); site-scope −0,00036; stack-first −0,00036. Гочево/Изборск: сдвиг разметки не найден (лучший сдвиг даёт +1–2 совпадения) — дело в полноте генератора, а не в привязке (logs/eval_g_sec.log, diag_gi.json). scores_cand7.json бота 5 не используется: по сообщению бота 5 там промежуточная версия с утечкой; ждём честную.

- 29.10 02:50 MSK: PIXGBM S bot-3 (bot3-pixgbm 76f7317) интегрирован в bot7-candidate-v2, голова **67284ab** (af97af0 → bb0e0d8 [5deb94a] → ecb92ec [76f7317] → 4957729 → 67284ab). Взяты: абсолютный срок фазы + лестница full→finds_only→classic с классикой на упавших растрах, backend=lightgbm (без .so), потоки/пул по sched_getaffinity. budget_s = **600** (не 300; REPORT4 §9.2). Пороги *_pix = T9 bot-9 (bot-10 REPORT9_v2 §B; T3 bot-3 −0,000924). stack6 и twins вкл. check_submission: psutil/tl2cgen/lleaves — необязательные, pixgbm.maps — список (было 2 ложных FAIL). pytest 345/9/3 (`work5/pytest_67284ab.log`); архив `work5/dist/submission_bot7cand_v2_67284ab.zip` 525 155 201 Б, sha256 15d3122dc12d8fbae171e8753ed58b176e276d9b8ebd4a0ad3877a3c33cc4d22; check PASS: синтетика, 210/214/217 --lock, pixgbm уровень full везде; finds_points 210: v1 323 / af97af0 373 / 67284ab 373. diff/mbox/SHA256SUMS обновлены, pc_measure_v2.ps1 не тронут. REPORT4 §9.


- 29.10 03:25 MSK: bot7-candidate-v2, голова **eb268d6** (67284ab → 4623c76 [bot-8 26e12ae, мягкий потолок] → 31594d2 [bot-8 428257f] → 8594842 [procs=2, soft_ceiling_s=1200 явно, budget 600 временно до ПК] → eb268d6 [починка SIGTERM в fork-пуле PIXGBM]). Конфликтов нет. **8594842 на 217: FAIL** — дети пула наследовали обработчик SIGTERM main.py, после pool.terminate() пустой результат и зависание в join, 918 с (`work5/check_8594842_217.log`); ошибка есть и в 67284ab на многоядерной платформе. eb268d6: pytest 364 passed / 1 failed (float-флаки bot-8, при повторе прошёл; `work5/pytest_eb268d6*.log`); zip `work5/dist/submission_bot7cand_v2_eb268d6.zip` sha256 8f3f2671…d589; check --lock PASS: синтетика, 210, 214, 217 (129 с, пик дерева 1541 МиБ), pixgbm full, выходы побайтно = 67284ab, строка мягкого потолка есть. diff/mbox/SHA256SUMS обновлены (8594842 удалён), pc_measure_v2.ps1 не тронут. REPORT4 §10.

- 29.10 03:30 MSK, задача 7: предрегистрированная комбинация с честными скорами бота 5 (scores_cand7.json 02:02; MODE=comb и MODE=b5) не досчитана — оба прогона упёрлись в лимит 2 ГБ (watchdog kill, logs/eval_comb.log, eval_b5.log). Сверка совпала (430 286 из 430 586 со скором, фолды совпадают, база 0,314615 воспроизведена). Обе компоненты по отдельности отрицательны (моя −0,00037; у бота 5 gate −0,014), поэтому задача поставлена на паузу; при необходимости — пересчёт по классам по отдельности ради памяти.

## 29.10.2026 10:50 МСК — bot-7 v3: починка загрузки D-FINE (срочно), аудит, ORSP
- Ветка `bot7-candidate-v3` (без push), HEAD **85aee5e**. Цепочка: 9138bff → b5c51b9, c0790b4, 8f89294 (аудит: защита времени ВКЛ., выход побайтно тот же; gormask, neigh, calib, frame ВЫКЛ. по bot-10 REPORT11; KIG заменён gormask) → 4f03a3e (ORSP abae911, выкл.) → 1295c75, 1d6c822, 85aee5e (починка D-FINE).
- PC e2e бота 1 **недействителен**. Нативный Windows python, а вендор regex и tokenizers собран под linux, поэтому D-FINE молча не загрузился и выдача совпала с v2. Исправлено:
  - закреплённый импорт 5.21.0;
  - ERROR `SP_DFINE_UNAVAILABLE`, затем выдача v2;
  - check_submission FAIL на маркер и на пробу импорта;
  - лог `sp_dfine n_det`;
  - 14 тестов.
  Подробности: `work6/FOR_BOT1_e2e.md`.
- Архивы в `work6/dist` (sha256 — в `work6/dist/SHA256SUMS`):
  - `submission_bot7cand_v3on_1d6c822.zip`, 3e459278…d817 (на повтор PC e2e);
  - `submission_bot7cand_v3off_1d6c822.zip`, 1a12c286…6495;
  - `submission_bot7cand_v3off_8857343.zip`, хранится.
  v2 eb268d6 не тронут. Архивы 9138bff, 4f03a3e и TEST удалены.
- pytest 440 passed, 9 skipped. Флаги ВКЛ. на 103 (CPU, batch 1): n_det 154, 220,7 с, RSS 1557 МБ. 4f03a3e на synth, 210, 214, 217 побайтно = v2.
- Ждёт:
  - бот 1: повтор PC e2e (docker);
  - бот 10: приёмка, замер 4090;
  - тест деградации времени `work6/tt`;
  - проверки ORSP (a)–(e) — подготовлены, не запущены.

## 29.10.2026 ≈11:30 МСК — bot-7 v3: FR1h ВКЛ., D-FINE f0 (final — резерв), проверка весов, ORSP, деградация
- Ветка `bot7-candidate-v3` (без push), HEAD **36af6b7** (85aee5e → 1cefcee → 36af6b7).
- FR1h (frame_prior @d4edf2f) ВКЛ.: группы sp_or/sp_focal/sp_focal2, все классы кроме городищ, d0 0,5, edge 0,5; до порогов в finish и safe_finish; рамки = bot-10 (168/168). gormask, KIG, neigh, calib — ВЫКЛ. Оговорка про пороги — QUESTIONS.md.
- D-FINE: в сдаче **f0**; final — резерв, НЕ одной строкой (перенос порогов + приёмка; `work6/on/FINAL_VARIANT.md`). fp32 + autocast, без `.half()`.
- check_submission: FAIL при n_det=0 — проба весов на 12 окнах 103 SpOr (f0: 14 запросов ≥0,3) и проверка смоук-лога; тесты.
- Архивы (`work6/dist/SHA256SUMS`): **v3on_36af6b7** 9ec25962…74e9 (сдача: FR1h + D-FINE f0 R) — check --no-smoke PASS, проба 14/14; v3off_36af6b7 4d4107d1…c2f9; хранятся v3off_8857343, v2 eb268d6. Удалены 1d6c822, 1cefcee.
- pytest: 451 passed / 9 skipped / 1 failed (coco_roundtrip — нет cv2 в venv, окружение).
- ORSP на 103 (база v3off_36af6b7): povr 0,2665 и 0,3645 — Δ 0, выходы равны; курганы sp_or 1, sp_focal2 0. Под R ORSP — пустой шаг. Флаг выкл.
- ПК-набор: pc_e2e_v3.ps1 (-ZipV3 по умолчанию v3on_36af6b7), `work6/pc/SHA256SUMS` обновлён; FOR_BOT1_e2e.md обновлён.
- Деградация времени `work6/tt`: идёт (см. REPORT_v3 §4).

## 29.10.2026 — исправление D-FINE «tqdm found=None» + восстановление артефактов после отката бокса (bot-7)
- Причина: transformers 5.21.0 при импорте делает dependency_versions_check; в образе pantum/n7q:20261024 у dist-info нет METADATA → version()=None → ValueError. Исправление в expds/sp_dfine.py (коммит bc5aa15 в bot7-candidate-v3, cherry-pick 48f9b98 в v4): шим importlib.metadata.version (__version__ → версия из имени dist-info, WARNING SP_DFINE_DEPMETA), require_version мягче только при неопределимой версии и импортируемом модуле; tqdm 4.70.1 вендорится вместе с dist-info. check_submission получил вторую пробу dfine с пустыми dist-info (старый 36af6b7 её FAIL).
- ~13:12 MSK бокс восстановлен из бэкапа: потеряны work6/dist/*, work5/dist/*, новые файлы work6/pc (патч-кит, BOM-ps1, SHA256SUMS), запись STATUS, venv бота 8. Git цел. Новый venv: bot-7/venv (uv, py3.12, torch 2.11 CPU).
- Пересобрано (сборка не побайтно воспроизводима — меняются метки времени членов; число/CRC/размеры членов совпадают):
  - work6/dist/submission_bot7cand_v3on_bc5aa15.zip 669298047 B sha256 341a2394…d4bd — check --no-smoke PASS (FAIL 0, WARN 1, обе пробы dfine PASS, веса 14/14 max 0.7037). Было до отката: тот же размер, другой sha.
  - work6/dist/submission_bot7cand_v3off_bc5aa15.zip 525158241 B sha256 1cc21c42…7802.
  - work6/dist/submission_bot7cand_v3off_8857343.zip 525147148 B sha256 384e7099…bbb5 (оригинал 60ec6594…5651, размер тот же) — sha НЕ совпадает.
  - work5/dist/submission_bot7cand_v2_eb268d6.zip 525158901 B sha256 9fb763c9…53e2 (оригинал 8f3f2671…d589) — sha НЕ совпадает. На ПК остаются оригиналы.
  - 36af6b7 v3on пересобран во временном worktree: 3495 членов, 669213798 B (= оригиналу), sp_dfine CRC 1261576305 (=), sha 91275fad… ≠ 9ec25962 → для патча использованы CRC/размеры членов; дифф = ровно CHG expds/sp_dfine.py + 39 ADD (tqdm + dist-info), как до отката. Временный zip и worktree удалены.
- ПК-кит work6/pc: patch_v3on_341a2394.zip, manifest_v3on_341a2394.txt, apply_patch_v3on.ps1 (UTF-8 BOM, только C:, проверяет sha старого zip 9ec25962…, патча и манифеста; результат сверяется по манифесту, пишет sha в SHA256SUMS кита и .sha256), pc_e2e_v3.ps1 (BOM, -ZipV3 по умолчанию v3on_bc5aa15.zip), SHA256SUMS. Тест на linux-копии (pwsh 7.4.6): 3534/3534 членов совпали, патченный zip → check PASS.
- GTX 1650 4 ГБ: D-FINE f0 batch 8 ≈ 1.3 ГБ fp32 / 0.7–0.9 fp16 + веса + контекст — влезает с малым запасом; OOM ловится по региону (откат к v2); риск NaN fp16 на GTX 16xx — при проблемах `[sp_dfine] batch=2..4`, `fp16=false` (только ПК, не менялось).
- work5/dist/DELETED_ZIPS_SHA256.txt воссоздан (9ec25962 v3on_36af6b7).

## 29.10.2026 13:42 МСК — bot-7 v4: влиты ветки бота 6 (dfine_aeor v2, ORGATE G), все новые флаги ВЫКЛ.
- `bot7-candidate-v4` (без push): 48f9b98 → b34bcf5 (←3617be6) → 0acb840 (←f4e733b) → a780680 (←e0a6709) → 53a66df (←ca63a9d) → **885758e** (комментарий: место под gor_dense бота 5). Источник: `git fetch ../../bot-6/repo` → refs/remotes/bot6/bot6-aeor-detect @f4e733b, bot6/bot6-orgate @ca63a9d; bot-6/repo не менялся.
- Конфликт один: scripts/check_submission.py `dfine_lines` (3617be6) — сохранены LI_TO_AE_FAILED (FAIL li_to_ae) + SP_DFINE_UNAVAILABLE/GOR_GATE_UNAVAILABLE, добавлен DFINE_AEOR_UNAVAILABLE. Остальное автоматически. Проверка: diff(bot6-orgate..885758e^) == diff(360c870..48f9b98) построчно — патчи v4 (в т.ч. вторая проба depmeta, проверка li_to_ae) целы; follow-ups == bot6-followups-1312.diff.
- Фазы pipeline.py: _sp_dfine_phase → _dfine_aeor_phase → [gor_dense — место] → gor_gate → (watchdog.finish: orgate вместо rerank_arch → пороги/merge). Ключ merge_within_markup = (регион, markup, класс, threshold_group если dfine_ae/dfine_or).
- default.toml: [dfine_aeor] enabled=false, [orgate] enabled=false, [sp_dfine]/[gor_gate] false, union_name_re="" (dedupe_union выкл.), [thresholds_by_markup.dfine_or] gorodishcha=1.01. ⚠ `[thresholds_by_markup.sp_dfine]` в default.toml v4 НЕТ (sp_dfine выкл.; пороги вставляет work6/on/make_on.py, там gorodishcha=0.68847 из bot-9 sweep) — sp_dfine gorodishcha 1.01 в v4 не стоит, не менял, нужно решение.
- models/orgate_g.txt sha256 676f4c1b…db697a ✓.
- pytest (venv, 2 потока, CPU): v4 до слияния 471 passed/9 skipped/1 failed; v4 после 527 passed/9 skipped/1 failed (+56 = test_dfine_aeor 42 + test_bot6_orgate 14); tests/test_bot7_* 73 passed. Падение — test_coco_roundtrip (нет cv2 в venv, окружение).
- Identity: синтетический смоук (фикстура val_dir: region_0001 LAS + UTM.json, region_0002 MagGr tif; `work7m/smoke_in`), inference/main.py 48f9b98 vs 885758e → result.geojson побайтно равны (sha256 9222f5d1…83d9, 245 объектов); в логе нет aeor/orgate. Реальный смоук (103 и т.п.) не гонялся; zip не собирался.
- Рабочая копия возвращена на bot7-candidate-v3 (bc5aa15). work6/dist, work5/dist, work6/pc не трогались.
- Осталось: gor_dense бота 5 (фаза между dfine_aeor и gor_gate; городища dfine_or держатся 1.01 до совместного замера bot-10); веса bot-9 Ae/Or и пороги OOF для dfine_aeor; смоук ORGATE на опорном входе.

- 13:43 МСК (бот 7): решение по порогу городищ sp_dfine для v4 — 1.01 (как согласовано с ботом 1, до совместного замера с gor_dense у бота 10). Применить в work6/on/make_on.py (параметр/вариант для v4) при сборке v4 zip, не в default.toml. v3on_bc5aa15 не меняется (0.68847).

## 29.10.2026 — bot-7: A (кит v3fix), B (crsfix бота 6 в v4), C (gor_dense бота 5 в v4). Без push, все флаги v4 ВЫКЛ.
### A. ПК-кит v3fix — /workspace/koz3_shared/bot-7/pc_kit_v3fix/
- patch_v3fix_a9e62802.zip 22562 B sha256 685276b7…3dc; apply_patch_v3fix.ps1 11333 B 8e2965e0…b66d; manifest_v3fix_a9e62802.txt 517104 B 8cb8850a…ba5; manifest_v3on_341a2394.txt 516946 B 77a0dd82…8053; pc_e2e_v3.ps1 11269 B 9bc3a5d3…a031; README.txt 7515 B f1993554…6c04 (BOM+CRLF); SHA256SUMS 523 B.
- Новый zip work6/dist/submission_bot7cand_v3on_0876090.zip 669299212 B sha256 a9e62802a70fe2480e4860276e6fc075f2f0cc9abef01d962fb37e41f22c1fdc (worktree wt_v3fix @0876090). check_submission PASS (FAIL 0, WARN 1; work6/chk_on_0876090.log); pwsh-патч на linux → тот же манифест, check PASS (build/check_pwsh_patched.log); эмуляция PASS; подделанный вход → FAIL.
- ПК: `powershell -ExecutionPolicy Bypass -File C:\Sloi\pc_jobs\v3e2e\kit\apply_patch_v3fix.ps1`, затем pc_e2e_v3.ps1 -Mode docker -ZipV3 …\submission_bot7cand_v3on_0876090.zip -Out C:\Sloi\pc_jobs\v3e2e\out_v3fix. Герите: бокс /workspace/data/train-expds-3/008_Герите (5478984564 B), на ПК копировать в C:\Sloi\pc_jobs\v3e2e\regions\008_Герите (нужно ≥6 ГБ на C:).
### B. crsfix бота 6 → bot7-candidate-v4 (worktree wt_v4)
- cb826b9 (←f86a31a, конфликт в sp_dfine.utm_for сведён в одну страховку; gor_gate._epsg — слой bot-6; test_bot6_crsfix адаптирован: read_utm_json уже даёт EPSG:32638), 5cd5828 (←c67c385, без конфликта), 78269c9 (tests/test_bot7_crs_lidar_geophys.py, 3 теста).
- Лидар (build_dem, DemPrefetch, hillfort.build_dem, li_dev) и геофизика (geophys_feats, dedupe_points 2 м, geo_pixgbm.region_feats, geo_model_feats) уже берут CRS через pipeline.region_crs → read_utm_json: фикс кода не нужен. На v3fix тот же тест 3 passed (временно), v3fix не менялся (HEAD 0876090, кит не пересобран).
- pytest: 536 passed / 9 skipped (coco исключён). Identity 9222f5d1…83d9 (245 объектов) = v4base.
### C. gor_dense бота 5 → v4. **HEAD v4 = d22eac3**
- fetch ../../bot-5/repo → refs/remotes/bot5/bot5-gor-dense-v4 @86a60c5; cherry-pick без конфликтов: f452ce3 (←a6a3965), b9a6d13 (←e407ee8), d22eac3 (←86a60c5).
- Маркеры SP_DFINE/GOR_GATE/DFINE_AEOR/GOR_DENSE_UNAVAILABLE и проверки bot-6/bot-7 на месте. default.toml: [gor_dense]/[sp_dfine]/[dfine_aeor]/[gor_gate]/[orgate] enabled=false, [thresholds_by_markup.dfine_or] gorodishcha=1.01.
- pytest полный: 560 passed / 9 skipped / 1 failed (test_coco_roundtrip, нет cv2) — work9/pytest_v4C.log. test_bot5_gor_dense + test_bot6_crsfix + test_bot7_geographic_utm_json + test_bot7_crs_lidar_geophys: 31 passed. gor_dense берёт EPSG через gor_gate._epsg (4326 → 32638, тест бота 5).
- Identity work7m/smoke_in: work9/out_v4C/result.geojson sha 9222f5d1…83d9 = v4B (до C); в логе нет gor_dense/UNAVAILABLE.
- Веса НЕ в git и не копировались: models/gor_dense/gor_dense_head.npz, models/gor_gate/gor_v3_head.npz, dinov2-small — положить при сборке.
- Пороги городищ (ничего не менял): make_on.py вставляет [thresholds_by_markup.sp_dfine] с gorodishcha=0.68847. Включать gor_dense: `[gor_dense] enabled=true` (+ sp_dfine). НЕ раскомментировать второй заголовок [thresholds_by_markup.sp_dfine] (tomllib: «Cannot declare twice») и НЕ дописывать второй ключ gorodishcha в ту же таблицу («Cannot overwrite a value») — ЗАМЕНИТЬ 0.68847→1.01 в THR make_on.py (вариант для v4). [thresholds_by_markup.gor_dense] gorodishcha=0.824066 и [thresholds_by_markup.dfine_ae] gorodishcha=1.01 — новые одиночные таблицы, можно раскомментировать; dfine_or 1.01 уже активен — не дублировать. Семантика: noD (drop_dfine_classes) убирает городища sp_dfine только на участках, где gor_dense отработал; порог 1.01 убирает их везде, в т.ч. при отказе gor_dense.
