# Доработки по решениям bot-1 (29.10, 11:09 МСК) — отчёт bot-3, пополняется по ходу

## 1) e3c4f41 → main — ОСТАНОВЛЕНО (main разошёлся нетривиально)
- Локальный main bot-3/repo = 24bd916 («Record completion of all 17 59-LAS models…»); это merge-base с bot3-pixgbm. В main нет всей линии PIXGBM (441bb71…76f7317, 7 коммитов) и нет audit01 (c315a9b): `git cat-file -e main:expds/geo_pixgbm.py` → «exists on disk, but not in 'main'».
- Пробный cherry-pick e3c4f41 на main во временном worktree (затем удалён, main не тронут): crs.py, geophys.py, pipeline.py и tests/test_ground_metres.py применяются чисто, а **expds/geo_pixgbm.py — конфликт DU** (файла нет в main). Тест test_ground_metres использует G.load_map из geo_pixgbm, поэтому на main он не пройдёт без переделки.
- Коммит LB eb268d6 (submission_bot7cand_v2_eb268d6.zip, /workspace/koz3_shared/LB_SCORING_v2_eb268d6_0.41238.log) в bot-3/repo отсутствует (`git log eb268d6` → unknown revision): сабмит собирается из репозитория bot-7, не из нашего main.
- Варианты для решения bot-1: (а) влить в main сначала bot3-pixgbm (+audit01), затем e3c4f41 — получится как задумано; (б) порт только частей без PIXGBM (crs.py + geophys.read_band + pipeline + тест на read_band); (в) переносить в ветку, из которой реально собирается сабмит (bot-7). Ничего не сливал.

## Пункт 4. Правило EtGr «каждый срез/профиль — отдельный эталон» (RULES_ORG_2026-10-29_misc.md п.4–5)

**Вердикт: в сдаче (bot7-candidate-v2 @ eb268d6) и в нашем коде по умолчанию слоёв EtGr не сливает ничто. Латентный дефект — правило U (`dedupe_union`), выключено в сдаче; исправление на ветке `bot3-etgr-union` (e0ec0ff), не смержено.**

Места (файл:строка в audit/wt = bot3-audit; в bot-7 — `bot7-candidate-v2:`):

| Место | Что | Сливает слои EtGr? |
|---|---|---|
| expds/post.py:99–111 `merge_within_markup` (bot-7 то же :99–111) — вызывается в candidates.py:337, blobs.py:184, watchdog.py:70 (bot-7 :76), scripts/geo_predict.py:78 | группировка (регион, метод, класс); `key[1]=="etgr"` → без слияния | **нет** (явно исключено) |
| expds/post.py:42 `merge_across_sources` | слияние пересекающихся полигонов разных source | для EtGr не вызывается (см. выше); для maggr/terfl — внутри одного метода (п.5 соблюдён) |
| expds/post.py:149 `finalize`, :180 `cap_features` | валидация/вершины; квота 20000 | нет (не объединяют) |
| expds/pipeline.py:185–190 `dedupe_points` (bot-7 :339) | только finds_points и только при `maps="all"` | нет: у EtGr нет находок; в сдаче maps не задан → "gray" |
| expds/pipeline.py:191–194 `layer_copies` (bot-7 :344) | копирует объекты в другие слои (добавляет, не сливает) | нет; `layer_copy` в сдаче не задан → False |
| expds/pipeline.py:207–214 → expds/geo_pixgbm.py:894 `dedupe_union` (bot-7 :368 → :830) | правило U: гасит pix-кандидата, если его центр внутри полигона базы **того же класса**, без учёта метода и слоя | **да, латентно**: и между слоями EtGr, и между методами (нарушение п.4 и п.5). Активно только при `union_name_re` ≠ "" (вариант H «Гнёздово»); в сдаче `union_name_re = ""` (bot7-candidate-v2:configs/default.toml:281; наш configs/default.toml:246) → **не вызывается** |
| expds/pipeline.py:215–227 замена S (`_drop` по (класс, метод) и по source done) | заменяет классику на pix пораслойно | нет |
| expds/geo_pixgbm.py:657 (bot-7 :595) `unary_union` | внутри одной компоненты одного растра | нет |
| expds/geophys.py:652–672 (bot-7 :489–509) | маска границ участка | не кандидаты |
| expds/gt.py:118/173 | эталон для обучения/оценки; строка 8: слои EtGr — отдельный эталон | нет |
| expds/geo_pixgbm.py:240 `pick_rasters`, :218 `raster_key`; geophys.py:565 `select_rasters` | выбор растров | трейн: 76 растров EtGr, коллизий ключей 0, классика и pix берут все 76, эталонных объектов на них 657, на отброшенных слоях 0 (`audit/etgr_layers.py` → out/etgr_layers.json). Замечание: `select_rasters` отбрасывает срезы глубже 2 м (ETGR_MAX_DEPTH_M, geophys.py:556) — на трейне таких нет; на тесте классика такой слой не увидит, pix (pick_rasters) — увидит |

**Сколько затронуто (трейн, если включить H):** `audit/union_affected.py` → out/union_affected.json, pix_lofo против стека geo_gap/sub_geo.pkl на 208–217: старое U гасит 252 из 675 pix-полигонов EtGr; из них **70 гасятся только полигоном другого метода** (точно), ещё 182 — полигоном EtGr (в стеке нет source → это верхняя граница межслойных гашений); то же для MagGr 44/405 (11 только чужим методом), TerFl 21/314 (7). Находки: 157/1169, все тем же методом (исправление их не меняет).

**Исправление** (ветка `bot3-etgr-union`, worktree audit/wt_etgr, коммит e0ec0ff, от bot3-audit): `_union_key` = (класс, метод, source у EtGr); тест tests/test_union_layers.py (5 тестов: чужой слой EtGr сохраняется, тот же слой гасится, чужой метод сохраняется, серая/цветная копия MagGr гасится, радиус находок); `pytest tests/test_union_layers.py tests/test_etgr_layers.py tests/test_ground_metres.py` → 10 passed. Не смержено. На сдачу (S) не влияет: dedupe_union там не вызывается.

Наблюдение (не проверено до конца): наша честная оценка сопоставляет по участку (EXPDS_MATCH_SCOPE=site), слой EtGr при сопоставлении не учитывается — если метрика организатора сопоставляет по слою, офлайн-оценка EtGr может быть завышена. Стоит сверить bot-10.

## Пункт 3. SATf как опция по классу — код готов, предрегистрация до перемера

- Предрегистрация: **audit/PREREG_SATf.md** (mtime 2026-10-29 11:15:12 МСК, sha256 в out/prereg_sha256.txt) — записана ДО любого перемера.
- Код: ветка **bot3-opt** (worktree audit/wt_opt, коммит 8c86700, от bot3-audit). `[geophys.pixgbm.sat_finds] = {core_max_m2, flag, model_dir}`; нет таблицы → выкл. Включено: только MagGr и только finds_points — второй load_map с ядрами и флагом и своя модель finds; object_poly — load_map без насыщения (явно 0/False, env PIXGBM_SAT_* на poly не действует) и модель S. Цена: второй load_map + второй проход окон только для finds на растрах MagGr.
- Проверки (scripts/geo_predict.py, трейн 201+214, 2 потока, procs=1, audit/opt_runs.sh → out/opt/):
  - выкл. (bot3-opt) = S (bot3-audit): sha256 geojson **совпадает** (074336a0c5ab…, 542 объекта);
  - SATf вкл.: finds = путь SAT (env SAT + модели var/SAT/models/all) — 354 объекта, хеш без fid 68bea54cc6d8 = 68bea54cc6d8; poly = S — 164, dc565b5fbcef = dc565b5fbcef (out/opt/satf_identity.json).
  - юнит-тесты tests/test_pix_optin.py (+ test_ground_metres, pixgbm-тесты): 13 passed.
- Перемер honest по предрегистрации: см. ниже (после записи PREREG).

## Пункт 2. C2 — код готов, предрегистрация

- Предрегистрация: **audit/PREREG_C2.md** (11:15:12 МСК).
- Код: тот же коммит 8c86700, `[geophys.pixgbm] ensemble_dirs = []` (выкл.). Вкл.: P = среднее моделей каталогов, признаки окна считаются один раз (prob_map с бустерами «poly#i»/«finds#i»). Каталог без нужной модели → C2 выключается (лог), работает model_dir. Проверка: C2 с одной моделью [all] = S байт в байт (тот же sha256 074336a0c5ab).
- Замечание: априорная скорость лестницы бюджета (PRIOR_S_PER_MPX) не умножена на число моделей; после первого растра скорость берётся из замера (PHASE["rate"]), а срок проверяется перед каждым окном (_Budget → классика). Для LB достаточно, но при budget_s 600 и ×2,5 на prob_map это надо учитывать.
- mock_val на box не было (был на F: у ПК); собран `scripts/make_mock_val.py --structure bot-1/pc_files/B/materials/public_val_structure/data_structure.yaml` в audit/mock_val (жёсткие ссылки, 48 участков; etgr 0,00 из 0,03 ГБ, maggr 0,01 из 0,02 ГБ — геофизики в макете мало, это ограничение замера). Время — в работе.

### Пункт 2 — время ветки геофизики на mock_val (out/opt/mock_timing.json)

`scripts/geo_predict.py --root audit/mock_val`, вариант S (enabled, replace, модель all) против C2 (ensemble_dirs = 3 fold-модели), 2 потока (OMP/EXPDS_PIXGBM_THREADS=2), procs=1, nice 10, budget_s без лестницы, ulimit -v 2,3 ГБ:

| | wall, с | user, с | объектов |
|---|---|---|---|
| S | **150,7** | 195,9 | 5543 |
| C2 | **318,4** | 466,6 | 4890 |

Источник: out/opt/mock_S.log, out/opt/mock_C2.log. Прирост +167,7 с (×2,11 по стене) на 2 потоках. Это согласуется с ens_timing (prob_map ×2,44–2,67 при 3 моделях: признаки общие, растут только предсказания). Ограничения: (1) в обоих прогонах классика упала по памяти под ulimit 2,3 ГБ на двух больших картах MagGr (01_Гнёздово_Усадьба_MagGr_цвет_2, 01_Судьбищи_2_MagGr_серый), то есть это нижняя оценка; на платформе (96 ГБ) этих падений не будет; (2) в макете геофизики мало (etgr 0,00 из 0,03 ГБ, maggr 0,01 из 0,02 ГБ по make_mock_val). Вывод по времени: с LB v2 на 3:48 и TimeGuard 1500 с запас есть; при `budget_s = 600` (сдача bot-7) C2 при 2 потоках по макету укладывается (318 с), но лестница при нехватке срока сама перейдёт на классику.

### Пункт 3 — перемер honest по предрегистрации (out/honest_audit_SATf_prereg.json, лог out/eval_SATf_prereg.log)

Файл варианта var/SATf_prereg/pix_lofo.geojson (создан 11:32:13 МСК, после PREREG_SATf.md 11:15:12) = finds из var/SAT + poly из S; по sha256 он совпадает с прежним var/SATf/pix_lofo.geojson.

| режим | d_honest10 = d_lofo | P(>0) | П1 П2 П3 П4 | П5 (<MDE) |
|---|---|---|---|---|
| global/drop | +0,004016 | 0,649 | да нет да да | да |
| global/keep | +0,004016 | 0,649 | да нет да да | да |
| site/drop | +0,004131 | 0,742 | да нет да да | да |
| site/keep | +0,004131 | 0,742 | да нет да да | да |

Finds (global/drop): F1 0,644 → 0,684 (TP 2191 → 2253, FP 2267 → 1707); object_poly без изменений (TP 110, FP 552 в global/drop). **По правилу успеха из PREREG_SATf.md: П2 не выполнен → не принимать**, опция остаётся выключенной. Результат: «слабо положительный, ниже MDE». Решение за bot-10/bot-1.

## Итог (11:36 МСК)
1. e3c4f41 → main: **остановлено** (main расходится нетривиально, см. §1); main не тронут.
2. C2: opt-in, bot3-opt 8c86700; PREREG_C2.md; mock_val S 150,7 с против C2 318,4 с (2 потока). Приёмку делает bot-10.
3. SATf: opt-in, bot3-opt 8c86700; PREREG_SATf.md; выкл. = S байт в байт; перемер: +0,0040…+0,0041, П2 нет → не принимать.
4. EtGr: в сдаче слои не сливаются; латентный дефект dedupe_union (только вариант H) → фикс на ветке bot3-etgr-union e0ec0ff, не смержен.

## Решение бота 1 (11:34): патчи на линию bot-7 (v4)
- Ветка `bot3-v4-patches` (от bot7-candidate-v3 36af6b7) содержит 4e63897 (= e3c4f41, метры на земле) и deafc30 (= e0ec0ff, правило U по класс+метод+слой EtGr).
- Конфликт в `load_map` разрешён так: поведение v3 сохранено, добавлен только множитель k. Код аудита `satc`/`sat_flag` не перенесён.
- Патчи и README: `patches_v4/` (0001, 0002, README_v4.md).
- Тесты: 58 passed в наборе geo/pixgbm/etgr. Расширенный набор дал 70 passed / 4 failed / 2 errors, и это те же провалы окружения, что на чистой v3 (`audit/out/v4_fail_*.txt`).
- Идентичность на трейне 200/204/212/214: v3, v4 по умолчанию, v4 + ground_metres (конфиг) и v4 + ground_metres (env) дают одинаковые 8210 объектов, sha256 bbdfadc68a23…21ae (`audit/out/v4/`, `audit/v4_identity.sh`).
- Новее v3 у bot-7 есть только docs-коммиты 360c870 в той же ветке. `git apply --check` и `git am` на них проходят чисто.
