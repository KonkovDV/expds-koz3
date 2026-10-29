# Для bot 10: как мерить кандидатов городищ bot-7 (task 6) — обновлено 28.10.2026, 23:40 МСК

**Какой вариант мерить**
- **В сдачу — `bot-7/work/gor/cands_gor_oof_D50_nocentre_comb.geojson`.** Это nocentre (без центра сцены), conf = `score_comb`.
- Для сравнения: `cands_gor_oof_D50_comb.geojson` (полный, с центром сцены) и `cands_gor_oof_D50_centre_small.geojson` (центр только в сценах < 15 МиБ). Для centre_small отдельного comb-файла нет: у него те же cand_id, p_bot5 можно взять из полного `_comb.json`.
- Справочно: старые файлы без `_comb` (conf = сырой score генератора).
Форматы — `FORMAT.md` §1–2, §5; JSON в UTM — те же файлы с расширением `.json`.

**Что такое score_comb.** Если у кандидата есть OOF-p от bot-5 (`p_bot5`, 11 870 из 59 029; nocentre — 11 696) и он не `stack_gor_phys`, берётся перцентиль p. Иначе — перцентиль своего балла генератора. Для stack_gor_phys свой балл намного лучше p (AUC 0,828 против 0,117). Вес выбран на обучающих фолдах: w = 1 во всех фолдах honest-10 и LOFO. p bot-5 обучен без фолда кандидата (наше поле `fold`) и без его семьи. Это проверено воспроизведением с assert, см. `STATUS.md`, задача 6, дополнение 2.
AUC кандидатов без лидара (honest-10): nocentre — свой балл 0,630, score_comb 0,760; полный — 0,647 и 0,760. Это ранжирование, а не официальный балл: его нужно мерить вам.

1. Добавить в стек отдельной группой (например, `grp='gor_cand7'`, mk по `markup_type`), класс gorodishcha, conf = `confidence` (= score_comb).
2. Мерить как в EXPERIMENTS.md: honest-10 `family_folds(regions, site_family, 10)`, `sweep_groups` по (group, class), `unseen="drop"` одинаково для базы и варианта (П6). Дальше paired_bootstrap_counts(n=2000, seed=0), LOFO, 5 раскладок `random_family_folds(regions,10,seed)`; П1–П5. База A honest-10 = 0,314615.
3. Важно: сопоставление 1-к-1. Кандидаты одной сцены стоят близко, и в радиусе 192 м у них одна виньетка bot-5 и одинаковый p. Их нужно гасить (NMS / top-1 на компоненту), иначе пойдут FP (каждый = 0,5 FN).
4. Риск центра сцены: на тесте 17 из 31 сцены SpOr крупные, и там центр не работает (`STATUS.md`, задача 6, дополнение). Поэтому в сдачу nocentre.
Проверка recall файла: `python verify_export.py <файл>.json` (логи `comb/verify_*.log`: полный 52/73, nocentre 48/73 без лидара).
