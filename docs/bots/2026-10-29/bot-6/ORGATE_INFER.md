# ORGATE G — интеграция в инференс (bot-6, 29.10 МСК)

Ветка `bot6-orgate` (от f4e733b), коммит **e0a67090b75e7e4c9ce09c6864c7652441e5d0e6**, не запушен.
Diff: `bot-6/bot6-orgate.diff` = `git diff f4e733b..e0a6709`, sha256 1d30c8fbfca5881054e03ac74bee9a7f7650f30900f24f900e905fada0bc816f.
Затронуто: expds/orgate.py (новый), expds/watchdog.py, expds/pipeline.py, configs/default.toml, models/orgate_g.{txt,json}, scripts/build_submission.py, scripts/check_submission.py, tests/test_bot6_orgate.py.

## (a) Флаг
`[orgate] enabled = false; model = "models/orgate_g.txt"; meta = "models/orgate_g.json"`.
При выключенном флаге выход побайтно совпадает с базой. Синтетика LAS+MagGr, 245 объектов: sha 9222f5d1…9a83d9 одинаковый у базы 360c870, флага off, флага on без модели и флага on с моделью (`work/orgate_identity/run1/identity.json`). Оговорка: в синтетике нет arch, поэтому on = off там ожидаемо. Работа гейта на arch проверена в п. (e) и (f).

## (b) Место в цепочке
`watchdog._finish_v3` и `_safe_v3`: only_known_classes → **[orgate, если включён, иначе rerank_arch R]** → site_calibration → kurgan_neighbour_prior → frame_prior (FR1h) → sp_score_fusion → apply_thresholds.
- Гейт **заменяет R** на его месте: идёт после sp_dfine/drop_groups (они раньше, в pipeline) и **до** FR1h frame_prior и порогов.
- Объекты, прошедшие гейт, как и у R, обходят базовые пороги. confidence у них = скор гейта, исходный скор лежит в conf_stack.
- Объекты dfine_ae/dfine_or (dfine_aeor) отделяются ещё раньше (`_aeor_split`) и в гейт не попадают.
- Гейт оценивает только arkhitektury групп из meta.thresholds {or, sp_arch}, с алиасом sp_or→sp_arch. Группы sp_dfine и dfine_* не оцениваются (тест `test_dfine_groups_not_scored`).

## (c) Признаки и правило LAS
- X = `rerank_arch.features` (35 X2) + 22 признака гейта 1:1 с `work/orgate/feats_gate.py`. Порядок сверяется с meta.names, иначе ORGATE_UNAVAILABLE.
- r_las и правило LAS берутся из **наличия LAS во входном каталоге участка**: `pipeline.run` после scan_root кладёт `cfg["_orgate_catalog"] = {участок: {las, utm_json}}`, только при enabled.
- На трейне каталог даёт 210 участков, из них 59 с LAS, и это ровно notes/las_boxes_59.json.
- Без каталога используется запасное правило «есть кандидаты li» — то же has_las, что у R; на трейне совпадает 209/209.
- Правило LAS: arch на LAS-участке получает скор −1 и удаляется.
- lat/lon берутся из UTM.json каталога. Если UTM.json нет, признаки = NaN (LightGBM это допускает), как и в трейне для участков без UTM.json.

## (d) Финальная модель (все данные трейна, гиперпараметры из PREREG)
- `models/orgate_g.txt`: sha256 676f4c1bbc8ff9d71b0bb86b39c9b870c10dc5a327bcf38d81c08e9d64db697a, 532 514 байт.
- `models/orgate_g.json`: sha256 d398032068f1abc3460083dcba4066b0bd2edcb757303144a465686686632c5d.
- Параметры: 300 раундов, lr 0,05, 31 лист, min_data_in_leaf 100, монотонность +1 по первым двум признакам, seed 0, 2 потока. Обучение на 7 793 строках пула, 79 позитивов (`work/orgate_final/fit.py`, `fit.log`).
- Порог — медиана 10 порогов фолдов k10, sweep_groups на inner-OOF с правилом LAS: **or 0,004086, sp_arch 0,008176** (`fit.log`, `fold_taus.json`).
- Рабочая точка вложенной процедуры воспроизведена: k10 G arch TP/FP = 54/327 против эталона 54/327, `REPRO_OK` в `fit.log`. Метрический эффект — ORGATE.md / REPORT12 bot-10 (G против R: k10 +0,001933).
- Финальная модель на трейне оставляет 302 из 47 518 кандидатов arch (`fit.log`).

## (e) Отказ, смоук, упаковка
- Любой сбой гейта (нет файлов, meta не той версии, число признаков не совпадает, LightGBM, исключение в decide) → в лог пишется `ORGATE_UNAVAILABLE: … — откат на rerank_arch/базу`, дальше работает R, если он включён, иначе база.
  - Модель грузится на каждом участке, даже без arch (кэш на процесс), поэтому маркер появляется в любом прогоне.
  - Тесты: `test_unavailable_marker_and_fallback`; в identity маркер есть в логе (`unready_marker_logged: true`).
- check_submission:
  - строка с маркером в логе смоука → FAIL (`orgate_lines`);
  - при enabled нет model/meta/expds/orgate.py в архиве → FAIL;
  - смоук `check_orgate` гоняет код и модель из архива в чистом процессе на опорном входе 008_Герите (43 кандидата arch, стек A). Требуется удаление ≥ 1 arch и ровно ожидаемых индексов (11 удалено, `[0,3,25,26,29,31,34,37,39,40,41]`).
  - **Опорный вход лежит вне git**, потому что это производное конкурсных данных (.gitignore `*.geojson`, п. 13.11 ТР): `/workspace/koz3_shared/bot-6/work/orgate_final/orgate_smoke_008.geojson`, sha256 e6ada8a7…b602b. Путь задаётся `--orgate-smoke`, если файла нет → FAIL.
  - При выключенном флаге ничего не требуется.
- build_submission: `orgate` добавлен в FLAG_MODULES, а модель и meta пакуются только при enabled (тесты `test_not_packaged_when_off` / `test_packaged_when_on`).

## (f) Эквивалентность онлайн/офлайн, время, RSS
- **Весь трейн** (`work/orgate_final/equiv_full.py` → `equiv/equiv_full.json`): decide_table на table.pkl, 47 518 целей. Решения по участкам совпали с офлайн-G: **0 расхождений из 134 участков**, оставлено 302. LAS берётся из каталога, LAS-участков с arch 11. Загрузка 2,5 с, decide 3,0 с, **maxrss 891 888 КБ**.
- **Путь пайплайна** (`equiv_geo.py` → `equiv/equiv_geo.json`): geojson 3857 → R.table → decide на 4 участках. Массивы таблицы совпадают (max abs 0), удалённые и оставленные id совпадают с офлайн-G:

  | участок | удалено / оставлено | id удалённых |
  |---|---|---|
  | 008_Герите | 12 / 31 | 716872, 716875, 716895, 716897, 716898, 716901, 716903, 716906, 716909, 716911, 716912, 716913 |
  | 014_Лешки | 100 / 28 | см. equiv_geo.json |
  | 031_Новотитаровская (LAS) | 20 / 0 | 670286…670305 (все arch, правило LAS) |
  | 116_Каменные_выкладки_6 | 5 / 5 | 669472…669476 |

  decide занимает 0,006–0,069 с на участок, maxrss 875 800 КБ (вместе с загрузкой стека трейна).
- **Оговорка, важна для bot-7:** gq (group_quantile) и пул (pool_mask) R считает по всему входу, и гейт это наследует. Поэтому решения зависят от состава тестового набора. Если на вход подан один участок (`equiv.py`), решения не совпадают с офлайн-G (контекст всего трейна): Герите 11 против 12, Лешки 102 против 100, у Каменные_6 пул шире. Новотитаровская совпадает точно. Одиночный прогон: maxrss 176 496 КБ, 0,8 с. Для R это поведение то же самое.

## Описание для bot-7
Включение: `[orgate] enabled = true`, файлы models/orgate_g.{txt,json} в архиве (build_submission добавит их сам). R при этом не применяется, гейт его заменяет, и `[rerank_arch]` оставить включённым как откат.
Ожидаемые строки лога: `ORGATE G: кандидатов arch N, LAS-участков K (catalog), оставлено …, удалено …`. Строка `ORGATE_UNAVAILABLE` означает, что работал откат на R, и check_submission тогда даёт FAIL.
Проверка: `check_submission.py dist/submission.zip --orgate-smoke <путь>` (по умолчанию путь на боксе выше).

Ограничения:
1. Модель обучена на стеке A **до** D-FINE drop_groups (без sp_dfine/dfine_aeor/FR1h-кандидатов), см. раздел (g). Если меняются кандидаты arch, гейт нужно переобучить: `work/orgate_final/fit.py` на новом table.pkl.
2. При включённом sp_dfine строки sp_dfine arch не оцениваются и в плотность r_arch_dens не входят.
3. Решения зависят от состава входа (gq/пул), как у R.
4. В meta.train.table записан абсолютный путь трейна — это только происхождение, код его не читает.

## (g) Для приёмки bot-10 (по запросу bot-7, 29.10 13:12 МСК)

### Стек обучения orgate_g.txt — ДО D-FINE drop_groups

- **Таблица.** `models/orgate_g.json` → meta.train.table = `/workspace/rerank/work/table.pkl`. Строит её `/workspace/rerank/prep.py` из `/workspace/rerank/stack_arch.geojson` (sha256 b412e299…f8e253, `/workspace/rerank/manifest.json`, лог `work/prep.log`): каждый объект стека переводится через `_collect` метрики, затем считаются пары `_match_quality` с GT.
  - D-FINE в prep.py не участвует: нет ни sp_dfine, ни drop_groups.
  - В таблице 18 групп, sp_dfine/dfine_* среди них нет (`work/orgate_stack/stack_check.json` → all_groups).
- **Состав arch в таблице** (`work/orgate_stack/stack_check.py` → `stack_check.json`, покрытие COV = 132 региона из `bot-10/work14/dfine_gor.json`, тот же набор, что в REPORT14 §2):

  | группа | mk | строк arch | из них в COV | выбросил бы R | OK-пар с GT | OK-пар выброшено R |
  |---|---|---|---|---|---|---|
  | sp_arch | sp_or | 46 735 | 46 735 | **46 735** | 205 | **205** |
  | or | or | 783 | 566 | 0 | 23 | 0 |
  | всего | | 47 518 | | 46 735 (98,4 %) | 228 | 205 (89,9 %) |

- **Вывод.** Гейт обучен (`work/orgate_final/fit.py`, 7 793 строки пула, 79 позитивов) и порог sp_arch 0,008176 подобран на стеке до R. Это 47 518 кандидатов, из которых 98 % — строки sp_arch.

### Совпадает ли это с тем, что bot-10 мерил как P = FR1h + G + R (REPORT14) — нет, не совпадает

- **Как в REPORT14.** Арх-вклад G и D-FINE R считаются по отдельности и разной машинерией:
  - REPORT14 §2: «G трогает только arch», а R восстановлен только для городищ — `pkg_gor.py`, `keep_rows` из `gor_rows`, выброшено 1006 строк городищ.
  - Остальные классы взяты из base_full A, `full()`.
  - Арх-эффект G (dR +0,001933 к rerank_arch, k10) измерен в REPORT12 на том же table.pkl, то есть без D-FINE R.
  - Арх-эффект R в REPORT10 §1 / `paired_5f.json` приведён только по povr/tselye/gor. arch там есть лишь для варианта C: +0,007111.
  - **Совместного замера G и D-FINE R по классу arch нет.**
- **Что реально будет в рантайме при обоих флагах.** Порядок такой: drop_groups в pipeline, потом гейт в watchdog (раздел (b)). На покрытых участках R уже выбросил все строки sp_arch, и вместе с ними 205 из 228 достижимых OK-пар стека.
  - Гейту остаются только 783 строки группы `or`, из них 23 OK. Порог sp_arch не применяется ни к одной строке.
  - Признаки участка и соседства у строк or (`rg_n`, `rg_rank`, `rg_pct`, `conf_rel`, `rg_mean`, `nb_same*`, `nb_strong*`, `r_arch_dens`, `r_arch_hi`, gq/пул) считаются по оставшимся arch участка. После R они сдвигаются относительно обучения: на обучении рядом были тысячи строк sp_arch.
  - Строки sp_dfine arch гейт не оценивает (тест `test_dfine_groups_not_scored`).
  - Значит, измеренный +0,001933 к пакету с R **не переносится**: его основа (отбор среди sp_arch) при R пуста.
- **Что нужно (не переобучал — по указанию bot-1).**
  1. **Перемер (минимум, без обучения).** Нужна таблица «после R»: table.pkl без строк sp_or/sp_focal/sp_arch/sp_focal2 на COV, плюс arch-круги sp_dfine из `bot-9/kaggle/dfine_spor/full_rest_out/oof_dfine_5f.geojson`, переведённые в UTM как в REPORT14 §2 (ошибка 3857/UTM).
     - На ней сравнить по классу arch три варианта: (i) R без гейта (база пакета), (ii) R + текущий orgate_g (`decide_table` из `work/orgate_final/equiv_full.py`, признаки пересчитаны на урезанной таблице), (iii) R + rerank_arch.
     - Протокол honest-10/LOFO/сиды, unseen=drop. Если (ii) − (i) ≤ 0 или в шуме, `[orgate]` при R держать выключенным.
  2. **Переобучение (если гейт при R нужен).** `work/orgate_final/fit.py` на таблице из п. 1: кандидаты arch = or + sp_dfine. Признаки sp_dfine-строк (grp/mk, conf) нужно добавить в словари meta. Затем новые фолд-пороги по группам or/sp_dfine и вложенный замер против п. 1(i). Код инференса меняется только в списке оцениваемых групп (сейчас {or, sp_arch}).
  3. Без R (D-FINE выключен) текущая модель соответствует обучению и замеру REPORT12.

### Время (риск, не замерен на целевой машине)

- Чтение Or без обзоров на боксе (2 потока CPU, заглушка детектора в RSS-замере `work/aeor_rss/`):
  - 033 Елизаветинское — 89,2 с на 4 окна (`or_033.log`);
  - 056 Фалхан — 88,0 с на 12 окон (`or_056.log`).
  - Потолок фазы dfine_aeor `budget_s` = 120 с. Сам гейт быстрый: decide 0,006–0,069 с на участок, раздел (f).
- Целевой ПК — **GTX 1650 4 ГБ**, а не 4090. Время инференса D-FINE на нём **не измерено**: ни tile_s, ни пиковая память при batch/fp16-autocast. Запас ×8 и `tile_s` в AEOR_DETECT.md взяты в расчёте на 4090.
- Что сделано в коде (коммит ca63a9d, AEOR_DETECT.md v3): срок фазы и окна сторожа проверяется перед каждым чтением окна. Фаза не ждёт, а пропускает остаток. Перерасход ограничен одним чтением окна — на боксе до ~22 с на Елизаветинском.
- Остаточный риск: на Or без обзоров выход по участку может быть пустым или неполным.

## Тесты
- `tests/test_bot6_orgate.py`: 13 passed. Если опорного входа нет на машине, модуль пропускается.
- Регресс из 17 файлов (15 прежних + test_dfine_aeor + test_bot6_orgate): **190 passed** (`work/orgate_regress.log`, скрипт `work/orgate_regress.sh`).
- 29.10 13:12 (коммит ca63a9d, diff `bot-6/bot6-followups-1312.diff` = `git diff e0a6709..ca63a9d`, sha256 2a75a4de…8c221d): нет файла `--orgate-smoke` → WARN «смоук не выполнен» (раньше FAIL). Тест `test_check_submission_missing_smoke_input_warns_not_pass`. test_bot6_orgate: 14 passed. Регресс 17 файлов: **195 passed** (`work/orgate_regress3.log`, bot-7 venv). Идентичность: `work/orgate_identity/run2/identity.json`, 4 из 4 sha b5f6b41c… (другое окружение, чем run1).
