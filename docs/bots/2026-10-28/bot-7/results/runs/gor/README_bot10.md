# Для bot 10: как мерить кандидатов городищ bot-7 (task 6)

Файлы: `bot-7/work/gor/cands_gor_oof_D50.geojson` (EPSG:3857, круги 10 м, class_name=gorodishcha) и `.json` (UTM, формат FORMAT.md).
Сами по себе баллы генераторов НЕ различают городище/фон (это генератор кандидатов, не детектор): нужна оценка bot-5 по фрагменту.

1. Получить conf: от bot-5 `scores_cand7.json` (cand_id → OOF p). Без него — только справочный прогон с conf=score.
2. Добавить в стек как отдельную группу (например `grp='gor_cand7'`, mk по `markup_type`), класс gorodishcha.
3. Мерить как EXPERIMENTS.md: honest-10 `family_folds(regions, site_family, 10)`, `sweep_groups` по (group, class), `unseen="drop"` одинаково для базы и варианта (П6);
   paired_bootstrap_counts(n=2000, seed=0), LOFO, 5 раскладок `random_family_folds(regions,10,seed)`; П1–П5. База A honest-10 = 0,314615.
4. Важно: 1-к-1 сопоставление; кандидаты одной сцены близко друг к другу — их нужно гасить (NMS / top-1 на компоненту), иначе FP (= 0,5 FN каждый).
5. Риск: генератор `centre` (центр сцены, 23 из 73 попаданий в одиночку) использует то, что сцены SpOr нарезаны вокруг объекта. На тесте это может не держаться — мерить вариант с ним и без (`gen != 'centre'`).
Проверка recall файла: `python verify_export.py cands_gor_oof_D50.json` (лог `verify_D50.log`).
