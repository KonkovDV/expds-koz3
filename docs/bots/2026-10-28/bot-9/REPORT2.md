# REPORT2 bot-9: 01 подкласс кургана, ядра s4/s5 (28.10.2026, МСК)

Правила COMMON.md соблюдены: ≤2 потока (`OMP_NUM_THREADS=2`), `nice -n 10`, без GPU. Ничего не пушилось на Kaggle и в GitHub, запусков и загрузок не было, сообщений никому не отправлял. Токен Kaggle использован только в read-only `kernels status` / `kernels output`.

## 1. Ноутбук 01_kurgan_subclass — готов к запуску (ждёт датасет)

**Файлы**
- Папка для push: `/workspace/koz3_shared/bot-9/kaggle/01_kurgan_subclass/` = `koz3-kurgan-subclass.ipynb` + `kernel-metadata.json`.
- Копии ноутбука: `notebooks/01_kurgan_subclass.ipynb` и `notebooks/01_kurgan_subclass.py` (тот же код скриптом, py_compile OK).
- Исходники ячеек: `notebooks/k01/c0_config.py … c6_final.py`, сборка `notebooks/k01/build.py`.
- Конфиг: `configs/01_kurgan_subclass.json` (v2; старый сохранён как `01_kurgan_subclass.v1.json`).

**kernel-metadata.json**
- `id` dmitrijkv/koz3-kurgan-subclass, `is_private` true.
- `enable_gpu` true, `machine_shape` "NvidiaTeslaT4" (T4×2), `enable_internet` false.
- `dataset_sources`: dmitrijkv/koz3-kurgan-subclass-crops и dmitrijkv/koz3-li-tin-59-t128-bundle (из бандла берутся smp и кэш весов ImageNet resnet18/34, по тем же путям, что у s4).
- `docker_image` тот же, что у s4.

**Вход**
- Путь `/kaggle/input/koz3-kurgan-subclass-crops/kurgan_subclass_crops.npz`. Запасной поиск — каталог с этим именем через rglob (Kaggle иногда монтирует в `/kaggle/input/datasets/<owner>/<slug>`), затем единственный `*.npz`.
- Схема npz проверяется до обучения, ошибка понятная и перечисляет найденное:
  - обязательные поля `X` (uint8, N×3×H×W, квадрат ≥32; на NHWC есть подсказка), `y` (int из {−1, 0, 1}; 1 — повреждённый, 0 — целый, −1 — без метки) и `region` (str, длина N);
  - необязательные `family`, `kind`, `stack_idx`, `cand_class`, тоже длины N;
  - нужно ≥50 размеченных и оба класса.
- Формат совпадает с выходом `scripts/bot9_kurgan_subclass.py crops` (ветка bot9-kaggle-queue). Если на ПК npz собирают другим скриптом, ключи должны быть ровно такими.

**--require-gpu**
- `gpu_guard(2)` требует 2 CUDA-устройства, тестовый тензор на каждом и `nvidia-smi -L`.
- Обход через `KOZ3_ALLOW_CPU_FALLBACK` запрещён, без GPU ядро падает с явным текстом.

**Валидация**
- Leave-family-out: 10 блоков `family_folds`. Это дословная копия `expds/metric.py:538`, семья — `site_family` из `expds/naming.py:108`.
- Есть assert, что ни одна семья не попадает в два блока.

**Выход в /kaggle/working**
- `kurgan_subclass_report.json` содержит:
  - AUC всего, по кандидатам, по эталону и по блокам;
  - по семьям: n, n_povr, n_tsel, auc; macro по семьям;
  - бутстреп по семьям p5/p50/p95 и P(AUC ≥ 0,9);
  - `target_auc_0.9_met`;
  - решение при пороге 0,4: acc, fixed/broken.
- `kurgan_subclass_auc_by_family.csv`.
- `kurgan_subclass_oof.csv` с колонками stack_idx, region, family, fold, kind, cand_class, y, p_povr, p_<модель>.
- `weights/<модель>_all.pt` — финальные модели на всех метках.

**Рецепт v2.** Цифр AUC до запуска нет, ниже только обоснование.
- ImageNet-предобучение: resnet18 на GPU0 и resnet34 на GPU1, ансамбль из двух.
- GPU-аугментации: поворот на любой угол (рельеф изотропен), отражение, масштаб ±10 %, сдвиг ±4 px, усиление и смещение каналов ±0,1, шум 0,03.
- AdamW + OneCycle, 20 эпох, AMP, TTA из 8 видов (D4).
- BCE без класс-весов, чтобы вероятность осталась калиброванной. Порог решения выводится из весов метрики, а не подбирается:
  - вес повреждённых 30, целых 20 (`materials/compute_metrics.py:84–85`);
  - отсюда p > 20/(30+20) = **0,4**.
- Достигнута ли цель AUC ≥ 0,9, покажет только `target_auc_0.9_met` и бутстреп в отчёте ядра.

**Проверка на CPU** (`tests/k01/run_local.py`, лог `tests/k01/run_local.log`)
- Синтетический npz той же схемы, 1 эпоха × 1 батч, 3 фолда, без предобучения.
- Результат: exit 0, maxrss 1568 МБ, 2 потока.
- Созданы все 5 выходных файлов: 12 строк семей, 240 строк OOF. AUC 0,549 на шуме ничего не значит.
- Негативные случаи дают понятные ошибки: нет датасета, нет `y`, NHWC, y=2, float X. `gpu_guard` на CPU падает.
- `tests/k01/test_hf.py` (лог test_hf.log): symlink-кэш HF собирается, отсутствие весов resnet34 даёт явную ошибку. Настоящие веса на коробке не проверялись, их нет.

**Команда запуска.** Только после того, как датасет dmitrijkv/koz3-kurgan-subclass-crops появится на Kaggle, и только когда на аккаунте не идёт другое GPU-ядро: второе одновременное молча получает CPU, и `--require-gpu` его уронит.
```
kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/01_kurgan_subclass
```
На коробке — `KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/01_kurgan_subclass`. На ПК — скопировать папку и `kaggle kernels push -p <папка>`.

**Применение на ПК после ядра**
- Порог теперь **0,4**, а не 0,5 из REPORT.md §4:
```
python scripts/bot9_kurgan_subclass.py apply --stack F:/expds3/work/stack_base.geojson --oof kurgan_subclass_oof.csv --thr 0.4 --dedup-m 2.0 --out F:/expds3/work/stack_subclass.geojson
python scripts/scoreboard.py <флаги базы> --rerank F:/expds3/work/stack_subclass.geojson --honest 10
```
- Затем приёмка П1–П5 бота 10 (раздел 4).

## 2. Ядро dmitrijkv/koz3-gpu-s5 — COMPLETE, выход скачан

- Статус `KernelWorkerStatus.COMPLETE` (`kaggle kernels status`, 20:16 МСК).
- Выход лежит в `/workspace/kaggle_out/koz3-gpu-s5/`, лог скачивания — `/workspace/kaggle_out/koz3-gpu-s5.dl.log`.
- **GPU был**: строка 1 `koz3-gpu-s5.log` — `GPU OK: ['Tesla T4', 'Tesla T4']` (t=9,5 с).
- `logs/summary.json`: 12 шагов, у всех rc=0.
  - GPU-секунд: {0: 1457,5, 1: 1821,4} (хвост koz3-gpu-s5.log).
  - Ядро шло ≈1950 с. Время последней записи лога 1949,9 с.
- Веса из `02_centers_fixed_out.tar.gz` (888 МБ) и `out/`:
  - `li_r18_unet_centers_fix_fold0–4` + `_final` по 57,4 МБ;
  - `li_r34_fpn_centers_fix_fold0–4` + `_final` по 92,8 МБ.
- Хвост лога:
```
GPU-секунд по очередям: {0: 1457.5, 1: 1821.4}
12 логов; архив /kaggle/working/02_centers_fixed_out.tar.gz
[NbConvertApp] Converting notebook __script__.ipynb to html
[NbConvertApp] Writing 318066 bytes to __results__.html
```

**best_pixel_f1 на валидации обучения** (`out/train_summary.json`). Это НЕ OOF и НЕ табло: валидация — одна семья, Муравлево или Боровичи (`configs/folds_li_tin_59_t128.json`).

| Модель | f0 | f1 | f2 | f3 | f4 | final (64 тайла вал.) |
|---|---|---|---|---|---|---|
| r18 Unet центры | 0,3996 | 0,1740 | 0,1849 | 0,3666 | 0,1890 | 0,6462 |
| r34 FPN центры | 0,3288 | 0,1146 | 0,1442 | 0,2563 | 0,1369 | 0,3161 |

- По центрам r18 fold0 на эпохе 30 (`logs/gpu0_step0.log`) center_f1 povr 0,347, tselye 0,458. У старых edge-моделей было ≤0,045 (REPORT.md 1.1e).
- Это признак, что починка центров сработала, но не доказательство: приёмка только по честному табло.
- Тестовые семьи фолдов (`configs/folds_li_tin_59_t128.json`): f0 Гнёздово, f1 Тимерево, f2 — 10 семей (Вышенское…Эсбю), f3 — 15 (Бервенец…Шетнево), f4 — 17 (Армиево…Юшково).

**OOF-проверка s5 на ПК** (точные команды, CPU/GPU ПК; `nn_oof.py --resume` с готовыми `fold{k}/li.pt` и `done.json` не обучает, а только предсказывает тестовые семьи фолда):
```
:: 1) распаковать
mkdir F:\expds3\work\s5 && tar -xzf 02_centers_fixed_out.tar.gz -C F:\expds3\work\s5
:: 2) разложить веса (скрипт /workspace/koz3_shared/bot-9/stage_s5_for_nn_oof.py, только stdlib, проверен на коробке)
python stage_s5_for_nn_oof.py --out-dir F:\expds3\work\s5\out --run li_r18_unet_centers_fix --work F:\expds3\work\oof_cfix_r18
python stage_s5_for_nn_oof.py --out-dir F:\expds3\work\s5\out --run li_r34_fpn_centers_fix --work F:\expds3\work\oof_cfix_r34
:: 3) OOF-инференс (в клоне с патчем bot9.diff: train.py/centers.py)
python -X utf8 scripts\nn_oof.py --markup li --root F:\2\train-expds-3 --cache <кэш li_tin_59_t128> --work F:\expds3\work\oof_cfix_r18 --out F:\expds3\work\li_cfix_r18_oof.geojson --folds 5 --target centers --center-sigma 0 --center-radius 4.0 --center-min-dist 4.0 --resume
python -X utf8 scripts\nn_oof.py --markup li --root F:\2\train-expds-3 --cache <кэш li_tin_59_t128> --work F:\expds3\work\oof_cfix_r34 --out F:\expds3\work\li_cfix_r34_oof.geojson --folds 5 --target centers --center-sigma 0 --center-radius 4.0 --center-min-dist 4.0 --arch FPN --ema 0.99 --resume
:: 4) табло, как li_focal (docs/PLAN.md:111)
python -X utf8 scripts\scoreboard.py <флаги базы> --add F:\expds3\work\li_cfix_r18_oof.geojson:li_cfix --honest 10
python -X utf8 scripts\scoreboard.py <флаги базы> --add F:\expds3\work\li_cfix_r34_oof.geojson:li_cfix34 --honest 10
```
- При `--resume` чекпойнт читает `Segmenter` (энкодер и архитектура берутся из меты .pt), `done.json` с `train_tiles`/`best_val_loss` пишет stage-скрипт из `train_summary.json`.
- **Обязательно** сверить, что тестовые семьи каждого фолда (`<work>\fold{k}\fold_report.json` → `test`) совпадают с `configs/folds_li_tin_59_t128.json`. Тот же `family_folds` по позитивам, но если разбиение не совпадёт, OOF будет с утечкой, и прогон надо выбросить.
- Затем приёмка П1–П5 (раздел 4) с `--base-add F:\expds3\work\li_cfix_r18_oof.geojson:li_cfix`.

## 3. Ядро dmitrijkv/koz3-gpu-s4 — COMPLETE, выход скачан, OOF оценён

**Итог (обновлено 21:05 МСК).** Статус `KernelWorkerStatus.COMPLETE` (`kaggle kernels status`, 20:48 и 21:05). Выход в `/workspace/kaggle_out/koz3-gpu-s4` (cropverify/{r18_scratch,r18_imnet}/{oof_fam10.csv,oof_lofo.csv,summary.json,train.log,weights}). Оценка: `python3 eval_s4_oof.py /workspace/kaggle_out/koz3-gpu-s4 > s4_eval.json` (rc=0, 20:49). `grep "pretrained load failed"` по cropverify/*/ — совпадений нет, т.е. r18_imnet действительно с весами ImageNet.

| модель / схема | n | pos | AUC всего | macro AUC по семьям (30 семей с AUC) |
|---|---|---|---|---|
| r18_scratch fam10 | 38362 | 14875 | 0,9316 | 0,9025 |
| r18_scratch lofo | 38362 | 14875 | 0,9281 | 0,8787 |
| r18_imnet fam10 | 38362 | 14875 | 0,9386 | 0,9024 |
| r18_imnet lofo | 38362 | 14875 | 0,9379 | 0,8987 |

- r18_imnet fam10 по блокам (s4_eval.json → auc_by_fold): 0,9287 / 0,9663 / 0,9145 / 0,861 / 0,8934 / 0,9366 / 0,9173 / 0,7583 / 0,992 / 0,9737 (блоки 0–9); худший — блок 7 (0,7583, n=1790, pos=231).
- Слабейшие семьи r18_imnet fam10 (by_family): Осечки 0,6047 (n=96, pos=18), Тешилово 0,6764 (n=309, pos=7), Примаки 0,6874 (n=136, pos=42), Новенькое 0,7267 (n=848, pos=245), Эрентуна 0,764 (n=1227, pos=130).
- Справочно, AUC povr vs tselye среди позитивов: 0,4062 / 0,4547 / 0,4602 / 0,4242 (scratch fam10 / scratch lofo / imnet fam10 / imnet lofo) — около случайного, классификатор подкласс не различает (он ему и не учился).
- Переранжирование `stack_arch.geojson` через `scoreboard --honest 10` и приёмка П1–П5 на коробке не выполнены (причина ниже) — делаются на ПК командами ниже, рекомендуется начать с `r18_imnet` (лучше AUC всего при равном macro).

**Исходные заметки (до завершения):**

- Статус `KernelWorkerStatus.RUNNING` при проверках в 20:16, 20:21, 20:23, 20:24 и 20:25 МСК (`kaggle kernels status dmitrijkv/koz3-gpu-s4`).
- Выход ещё не скачивался, оценок s4 нет.
- Что ядро делает (`/workspace/kpull/s4/koz3-gpu-s4.py`):
  - две модели параллельно: r18_scratch на cuda:0 и r18_imnet на cuda:1;
  - классификатор «есть курган в центре 6×6 px кропа 64 px» (kurgan vs фон, НЕ повреждённый vs целый);
  - две схемы: fam10 (10 блоков family_folds) и lofo (по семье), плюс final_all;
  - на каждую схему пишет `cropverify/<variant>/oof_<scheme>.csv`, `weights/<scheme>_foldKK.pt` с `heldout_regions` и `summary.json`.

**Готово к обработке** (запустить, когда статус станет COMPLETE):
```
mkdir -p /workspace/kaggle_out/koz3-gpu-s4 && KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) nice -n 10 ~/.local/bin/kaggle kernels output dmitrijkv/koz3-gpu-s4 -p /workspace/kaggle_out/koz3-gpu-s4
python3 /workspace/koz3_shared/bot-9/eval_s4_oof.py /workspace/kaggle_out/koz3-gpu-s4 > /workspace/koz3_shared/bot-9/s4_eval.json
grep -h "pretrained load failed" /workspace/kaggle_out/koz3-gpu-s4/cropverify/*/train.log   # при совпадении r18_imnet = scratch
```
- `eval_s4_oof.py` (проверен на синтетическом CSV) считает:
  - AUC всего, по fold и по семьям, macro по семьям;
  - справочно — AUC povr vs tselye среди позитивов по `y_sub`. Модель этому не училась, это не ответ на вопрос 01.

**Переранжирование кандидатов курганов честно по семьям**
- На коробке невозможно: кропы для кандидатов `stack_arch.geojson` режутся из LAS (TIN 0,1 м), а в `/workspace/data/train-expds-3` перенесено 7 файлов .laz (`find … -name "*.laz"`, xfer идёт). К тому же это тяжёлый инференс.
- Написан `repo/scripts/s4_rerank.py` (коммит 194a691 в bot9-kaggle-queue, bot9.diff обновлён, без push):
  - кандидат участка R оценивается только тем чекпойнтом fam10, где R в `heldout_regions`;
  - чекпойнтов должно быть 10, участок не может быть в двух блоках;
  - кропы режутся тем же `crop_verify.extract_crops`;
  - смешение `geo`, медиана группы для кандидатов без кропа;
  - загрузка 10 чекпойнтов и predict проверены на коробке на фиктивных весах.

Команды на ПК:
```
python -X utf8 scripts\scoreboard.py <флаги базы> --honest 10 --dump-stack F:\expds3\work\stack_base.geojson
python -X utf8 scripts\s4_rerank.py --root F:\2\train-expds-3 --stack F:\expds3\work\stack_base.geojson --gt F:\expds3\work\gt_train_v2.geojson --weights <s4>\cropverify\r18_imnet\weights --groups li li_dev li_mosaic --out F:\expds3\work\stack_s4_imnet.geojson --report notes\s4_imnet_rerank.json
python -X utf8 scripts\scoreboard.py <флаги базы> --rerank F:\expds3\work\stack_s4_imnet.geojson --honest 10
```
- То же с `r18_scratch`.
- Затем П1–П5 (раздел 4), где кандидат = `stack_s4_imnet.geojson`.

## 4. Приёмка бота 10 (П1–П5) — как применять

Источник — `/workspace/koz3_shared/bot-10/REPORT.md` §4–5 и EXPERIMENTS.md. Нужен клон с `honest_bootstrap.diff` (ветка bot10-honest-bootstrap, коммит a839da2).

**Честная база A** = 0,3146153 (`/workspace/rerank/surf_arch_board_2026-10-27.json` → `honest.after[0]`).

**`<флаги базы>`**: из того же JSON (`add`, `replace`, `gt`, `geo`):
```
--add F:\expds3\work\surf_arch_or.geojson:or:arkhitektury:or --replace sp_arch=F:\expds3\work\surf_arch_sp.geojson --gt F:\expds3\work\gt_train_v2.geojson --geo F:\expds3\work\geo_cur.geojson --expect 0.367363
```
Плюс `--work`/`--train` вашего ПК. Полный балл этой базы — `all` 0,367363 в том же JSON.

**Команды**
```
for %s in (base cand) do python -X utf8 scripts\split_stack_by_family.py --stack F:\expds3\work\stack_%s.geojson --out F:\expds3\work\fam_%s
python -X utf8 scripts\family_clusters.py --fam F:\expds3\work\fam_base --gt F:\expds3\work\gt_train_v2.geojson --out F:\expds3\work\family_clusters.json
for %s in (base cand) do python -X utf8 scripts\family_curves.py --fam F:\expds3\work\fam_%s --gt F:\expds3\work\gt_train_v2.geojson --clusters F:\expds3\work\family_clusters.json --out F:\expds3\work\curves_%s
python -X utf8 scripts\honest_bootstrap.py --a F:\expds3\work\curves_cand --b base=F:\expds3\work\curves_base --boot 2000 --lb 2000 --seeds 5 --out notes\accept_<имя>_<дата>.json
```
- `stack_cand` — это rerank-стек (`stack_s4_imnet.geojson`, `stack_subclass.geojson`) или дамп `--base-add <ветка> --dump-stack`.

**Правило принятия**
- П1: Δhonest-10 ≥ +0,002.
- П2: P(Δ>0) ≥ 0,85.
- П3: Δ LOFO > 0 и Δ > 0 в ≥4 из 5 раскладок.
- П4: povr, tselye и gor каждый Δ ≥ −0,0005; new34/optical ≥ base − 0,001.
- П5: Δ < 0,006 принимается только пакетом.

## 5. Итог
- 01 готов, ждёт датасет.
- s5 COMPLETE, с GPU, без ошибок; веса скачаны, OOF на ПК — командами раздела 2.
- s4 COMPLETE: OOF AUC r18_imnet fam10 0,9386 (macro по семьям 0,9024), r18_scratch fam10 0,9316 (0,9025); честный rerank и П1–П5 — на ПК (раздел 3–4).
- Оценок по табло нет: честное табло считается только на ПК (LAS и полный кэш).

## 6. S3 бакета организаторов (train-expds-3) и вариант 01 из S3 — кратко (20:58 МСК)

Ключи — только `set -a; . ~/.koz3s3/env; set +a` внутри команд; значения нигде не печатались и не записаны.

**Бакет** (`s3/listing.txt`, полный `s5cmd ls`, 2132 строки): 1382 файла, 223,3 ГБ: tif 558 шт. / 183,4 ГБ, las 59 / 39,78 ГБ (несжатые .las), geojson 555 / 0,09 ГБ, json 210. По слоям: Li_карты (c/ch/g/i) 244 / 103,66 ГБ; Or 6 / 65,46; La (LAS) 59 / 39,78; Ae 36 / 7,42; SpOr 175 / 5,40; MagGr 19 / 1,23. Каналы li-модели (lrm_s, lrm_l, slope) строятся из LAS (TIN 0,1 м), готовые Li_*.tif — не те каналы → для 01 нужен LAS. 210 участков, из них 59 с LAS; курганы gt_train_v2 есть в 106, из них 42 с LAS = 25,64 ГБ LAS (`s3/regions_table.json`).

**Скорость** (`s3/speedtest.log`, box, range-GET по LAS Муравлево): 1 поток 15,4 МБ/с; 4 потока 40,9; 8 потоков 36,6 МБ/с. На Kaggle не мерилось (организаторы заявляют ~72 МБ/с). ⇒ LAS курганных участков 25,64 ГБ ≈ 10–12 мин, все LAS 39,8 ГБ ≈ 16–18 мин, весь бакет ≈ 1,5–1,7 ч и не влезает в /kaggle/working (~20 ГБ, допущение).

**DEM+кропы** (`s3/bench.log`, `s3/bench_crops.py`, 1 поток, участки 030_Эсбю 8,5 МБ/282 665 точек и 179_Гнёздово_Лесная_1 41,3 МБ/1 214 070 точек, все 298+460 и 186+308 кропов собраны):
| бэкенд | Эсбю, проход GT / кандидаты, с | 179, с | пик RSS |
|---|---|---|---|
| NumPy | 23,1 (с прогревом) / 8,1 | 14,8 / 15,3 | 0,78–1,21 ГБ |
| Rust `expds_core.abi3.so` (repo/inference/lib, RAYON_NUM_THREADS=1) | 1,7 / 1,2 | 1,3 / 1,2 | 0,35–0,45 ГБ |

Линейная экстраполяция Rust (~1 с на 1,2 млн точек; ~34 байт/точку) на 25,64 ГБ LAS (~750 млн точек) ≈ 10–20 мин CPU на проход; крупные облака (Муравлево 6 ГБ) упираются в max_pixels=60 млн → шаг огрубляется, время не линейно — не мерилось.

**ВАЖНО для идентичности кропов** (`s3/identity.log`): NumPy и Rust дают РАЗНЫЕ кропы (доля отличных пикселей 2,5–3,7 %, макс. разница 13–37 уровней uint8; sha256 не совпадают). Причина: другая заливка TIN (`expds/lidar.py:81–110`, и при >4 млн занятых ячеек NumPy уходит в пирамиду, Rust — TIN до 20 млн). Значит ядро должно грузить тот же бэкенд, что и ПК: класть `inference/lib/expds_core.abi3.so` в бандл и `sys.path.insert(0, …/inference/lib)`, проверять `accel.backend()` в логе. Какой бэкенд был на ПК при сборке `koz3-kurgan-subclass-crops` — не проверено (README: `scripts/build_rust.py dev`, т. е. скорее Rust). Проверка идентичности: собрать в ядре кропы одного малого участка (Эсбю) и сравнить sha256 X/y/stack_idx с npz ПК.

**Вывод по 01_s3:** технически выгодно (скачивание ~11 мин + DEM ~15–30 мин вместо ожидания датасета ПК), но делать в CPU-ядре (CPU-подготовка в GPU-сессии оплачивается как GPU-часы). По указанию бота «1» 01 уже запущен (koz3-kurgan-subclass), вариант 01_s3 — низкий приоритет, ядро не готовилось. Стек: в `stack_arch.geojson` 1 386 709 признаков (`s3/stack_counts.json`), `fid` не уникален (6575 повторов в паре region+markup) → id кандидата = позиционный индекс в файле. li-курганов с conf ≥ 0,05: povr 28 719, tsel 3223.
