# FORMAT — обмен кандидатами городищ (bot-7 → bot-5 → bot-10), 28.10.2026

Совместим с `bot-5/analysis/gor_patch/meta.json`: те же ключи `reg, file, ux, uy, corners, gsd`; координаты — UTM региона (`{reg}/UTM.json`).
Фрагмент bot-5 = индекс строки в его `meta.json` (виньетка 384 м, шаг 192 м, контекст 768 м с тем же центром).

## 1. `cands_gor_oof_D{D}.json` — список словарей (UTM), одна строка = один кандидат-точка
| ключ | тип | смысл |
|---|---|---|
| cand_id | int | сквозной id кандидата в файле |
| reg, file, kind | str | регион, растр сцены, `SpOr`/`Ae`/`Or` |
| lidar | bool | регион из `notes/las_boxes_59.json` |
| epsg | int | UTM-зона региона |
| ux, uy | float | центр кандидата, UTM, м |
| corners | 4×[x,y] | бокс 384 м с центром в кандидате (UTM; порядок ↖ ↗ ↘ ↙) — как `corners` у bot-5 |
| ctx_corners | 4×[x,y] | бокс-контекст 768 м, тот же центр |
| gsd | float | м/пикс исходного растра |
| gen | str | генератор: `frst_abs/bright/dark`, `log_bright/dark`, `grad`, `centre` (центр сцены), `stack_sp_or`/`stack_ae`/`stack_or`/`stack_gor_phys` (репрезентативная точка кандидата из стека, любой conf) |
| R | float/null | радиус структуры генератора, м (15–140) |
| score | float | сырой балл генератора (НЕ калиброван; для stack_* — conf стека; centre = 1) |
| rank | int | ранг внутри сцены после NMS 40 м (−1 для centre/stack) |
| variant | str | вариант генератора, выбранный честно для фолда региона |
| fold | int | номер фолда honest-10 (`family_folds(regions, site_family, 10)`), на остальных 9 выбран variant |
| frag_bot5 | int | индекс ближайшей виньетки bot-5 (`meta.json`) той же сцены, центр которой ≤192 м по обеим осям; −1 — нет (Ae/Or или сцены ещё не в meta) |

## 2. `cands_gor_oof_D{D}.geojson` — то же для стека/табло (EPSG:3857)
Геометрия: круг r = 10 м вокруг (ux, uy) (официально засчитывается, если центр внутри эталона: `_match_quality` проверен).
properties: `class_name='gorodishcha'`, `region`, `markup_type` (`sp_or`/`ae`/`or`), `confidence` (= score), `cand_id`, `gen`, `rank`, `fold`, `frag_bot5`, `scene_file`.

## 3. Ожидаемый возврат от bot-5 (предложение)
`bot-5/analysis/gor_patch/scores_cand7.json`: `[{"cand_id": int, "p": float, "fold": int}]` — OOF-вероятность городища для кандидата
(по его `frag_bot5`, либо по виньетке, вырезанной по `corners`/`ctx_corners`), обученная без фолда кандидата.
Тогда bot-10 ставит conf = p (или p·f(score)) и меряет как отдельную группу.

## 4. Бюджет
`D` = кандидатов на км² сцены (мин. 3 на сцену). Внутри сцены — первые `rank` генератора; срез по меньшему D = фильтр `rank < max(3, D'·km²)`.

## 5. `cands_gor_oof_D50{,_nocentre}_comb.{json,geojson}` — с оценкой bot-5 (28.10.2026, 23:40 МСК)
Те же строки, cand_id и поля, что в исходном файле (frag_bot5 — по финальному meta.json bot-5, md5 4ce45bd6…). Добавлено:
| ключ | тип | смысл |
|---|---|---|
| p_bot5 | float/null | OOF-вероятность городища от bot-5 (`bot-5/analysis/gor_patch/scores_cand7.json`). Виньетка SpOr — ближайшая на всём участке, \|dx\|,\|dy\| ≤ 192 м, любой файл SpOr. Поэтому p есть и у части Ae/Or. null — не оценён |
| score_comb | float [0,1] | ранжирующий балл: если p_bot5 есть и gen ≠ stack_gor_phys — перцентиль p среди оценённых, иначе перцентиль своего score среди всех кандидатов файла. Вес w = 1 выбран по AUC на обучающих фолдах (honest-10 и LOFO). Не калиброван |
geojson: `confidence` = score_comb, сырой балл генератора — `properties.score`. Перцентили считаются внутри файла; на тесте — по тестовым кандидатам, без меток.
