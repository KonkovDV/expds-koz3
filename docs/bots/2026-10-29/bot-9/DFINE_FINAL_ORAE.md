# D-FINE: spor-final + Or/Ae (obj2coco) — ядра для пуша (bot 9 → bot 1), 29.10.2026

Корень: `/workspace/koz3_shared/bot-9/kaggle/dfine_spor/` (генератор `build.py`, исходники `src/`).
Все ядра собираются `python3 build.py`; легаси-сборки (smoke/full/full_rest) байт-в-байт как до правок
(сверка с `backup/20261029_0724/SHA256_before.txt`: все `same`); final и o2c-a/b после правок Or/Ae не изменились.
Ключи S3 в файлах отсутствуют: проверка `rg -F -f <значения из ~/.koz3s3/env>` по bot-9 → 0 файлов.

## 1. Ядра, sha256, команды пуша

| # | ядро (папка) | sha256 .py | sha256 kernel-metadata.json | статус |
|---|---|---|---|---|
| A | `final/` koz3-dfine-spor-final (COCO, все SpOr, сохраняет `dfine_final`) | d190370a52629ffd76bf36a0c0dbb63a0016aba2bbce50354b880e092dab286c | 6a2202931a53515b6f507c16f3274770fd66b9eb5f2e224d22919d7b30edcc49 | **READY** |
| B0 | `or_prep/` koz3-dfine-or-prep (CPU, без GPU) | f598b2928941e33249622c509ae7c7566ab7e5af9db8591b9d6c1c5dbd8a45c5 | 26ec2b5d74dccd19a696a4845cfd3291bc00e8bde76dc06366003a66e929c274 | ready |
| B1 | `or_o2c/` koz3-dfine-or (obj2coco, **3 фолда leave-one-family-out**, §8; было 6 LORO, sha 02dba66d…) | 4576c82abbab5601d6f374133072935f14123acde8fb01c454ebf52bb50880c5 | e6b677ad63cf3e3da9b0ee907a96f2c3cbb74d00342694a198921b3853a27294 (не изменился) | ready (после B0) |
| B2 | `ae_o2c/` koz3-dfine-ae (obj2coco, k5-семейные фолды) | fd7d3317ebcfc06ea60d42fdd42686cb2e9ce2348d59199cf210aed30409d0ab | ec6288d492855824ebe6acc6631e794d5d69dd2a70c396d9b43d2db0687fa719 | ready |
| опц. | `full_o2c/` koz3-dfine-spor-o2c-a | 5cb9cf4a36145e9b0bfa83716b1b5c325a2e6a95119ecc206ec249fc67ba015f | eb04653dda91f51828f70f857e6b24efeb464cdbc1836091ca99603fe30c6276 | ready |
| опц. | `full_rest_o2c/` koz3-dfine-spor-o2c-b | 5d33eb5fc9afe9b16de73535f07aa866ba7aa39e2d275c33f10047cbfdf58bf2 | 661cdd75eb1c1e1e31fa157f961959baa5f11bbfc38723d75fc7d03e495655e5 | ready |

(команда: `cd .../dfine_spor && sha256sum final/* or_prep/* or_o2c/* ae_o2c/* full_o2c/* full_rest_o2c/*`)

Пуш (делает bot 1; D = /workspace/koz3_shared/bot-9/kaggle/dfine_spor):
```
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p $D/final        # A, сразу
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p $D/or_prep      # B0, CPU
#   ждать статуса COMPLETE у dmitrijkv/koz3-dfine-or-prep, только потом:
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p $D/or_o2c       # B1
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p $D/ae_o2c       # B2, независимо
# опционально (SpOr на obj2coco, ~4.9 GPU-ч):
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p $D/full_o2c
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p $D/full_rest_o2c
```
Все ядра: T4 (кроме or_prep — CPU), internet on, dataset `dmitrijkv/koz3-s3-cred`, тот же docker_image, что у фолдов.
B1 имеет `kernel_sources: ["dmitrijkv/koz3-dfine-or-prep"]` (поиск `**/or_prep/<имя>` рекурсивно в /kaggle/input).
Если B0 не готов/упал — B1 всё равно работает: качает и даунсэмплит Or-тифы в самом ядре (+~65 ГБ трафика в GPU-время, см. §3).

## 2. Что внутри

**Общее (рецепт фолдов SpOr, DFINE.md):** D-FINE-L, transformers 5.21.0 (пин), 0.5 м UTM через WarpedVRT, тайл 640, bs 8,
AdamW 1e-4, OneCycle, fp16, overlap 160, flip-TTA, NMS 0.5; бюджет-стоп трейнера bot5.
Базы (пин ревизий): COCO `ustc-community/dfine-large-coco@aa8f8211545e4cff5adf52ea512f9ac74a9824ec`;
obj2coco `ustc-community/dfine-large-obj2coco-e25@c89de47153e338c07cddc81fc21a776b97cbf75e` (Apache-2.0).
Переключатель `DFINE_BASE=coco|obj2coco` (env), дефолт задаётся сборкой.
Проверка obj2coco на CPU: `kaggle/dfine_spor/fix/test_model_block_o2c_t211.log` → ALL_OK.

**A (final):** DEADLINE 3.0 ч, ITERS 6000, база coco, без OOF; обучение на всех SpOr → сохраняет `dfine_final/`
(model.safetensors + конфиг) и блок тайминга fp32/fp16 мс/окно + пиковая VRAM.
Dryrun CPU (`dryrun_final/`): rc=0, peak RSS 1474860 kB, model.safetensors 125001980 B, reload diff 0.0.

**B0 (or_prep):** для каждого Or-тифа ≥1.5 ГБ (`DFINE_PREP_MIN_BYTES`) из 6 Or-участков: проверка диска → скачивание
→ даунсэмплинг average до ≈0.25 м/пикс (целая кратность, ≤3 канала, маска/альфа → 0, nodata=0, deflate, тайлы, исходная CRS)
→ sha256 → удаление исходника. Выход: `/kaggle/working/or_prep/*.tif` + `manifest.json`. Ошибки → SystemExit.
Имя: `<stem>__<размер_байт>__r0.25.tif` (размер в имени — защита от смены исходника).
Почему: Or-тифы огромные (Герите 12.36, Лешки 18.97, Елизаветинское 3.62, Фалхан 3.68, Старый_Барах 26.74, Изборск 0.108 ГБ;
`s3/listing.txt`), на 0.5 м даунсэмплинг ×2 от 0.25 м не теряет информации для сетки 0.5 м.
Альфа 4-го канала uint8 распознаётся и по colorinterp, и по значениям {0,255} (синтетический тест: маска обнулена, bounds совпали;
int16 4-канальный — маска не применяется, корректно).

**B1 (Or):** MOD=Or, markup_type `or`, threshold_group `or_dfine`, база obj2coco, ITERS 1200, DEADLINE 5.0,
~~фолды 0..5 = leave-one-region-out~~ (заменено 29.10 07:48 → 3 семейных фолда, см. §8) по 6 Or-участкам (008_Герите, 014_Лешки, 033_Елизаветинское, 056_Фалхан, 093_Старый_Барах, 185_Изборск),
каждый участок — своя семья. OOF в EPSG:3857, полигоны через buffer(0); файл `oof_dfine_Or.geojson`; в REPORT — gt_coverage.

**B2 (Ae):** MOD=Ae, markup_type `ae`, threshold_group `ae_dfine`, obj2coco, ITERS 4000 (env `DFINE_ITERS`), DEADLINE 5.5,
фолды семейные k5 (порядок 3,4,0,2,1 — тяжёлые первыми). Ae: 32 участка с тифами (7.42 ГБ), 29 размечены, все в k5;
участков по фолдам {3:11, 0:7, 4:7, 2:4, 1:3}, ГБ {3:4.14, 4:1.56, 2:1.0, 1:0.55, 0:0.18} (`s3/listing.txt`).

**Опц. o2c-a/b:** SpOr тем же рецептом, база obj2coco, свои 5-фолдовые OOF, threshold_group `sp_dfine_o2c`.
Dryrun `dryrun_o2c_a/dryrun.log`: rc=0, 3 фолда × 3 итерации, 1592 детекции, peak 1428508 kB.

## 3. Dryrun-факты (CPU, ящик)

| что | лог | итог |
|---|---|---|
| Ae: 001_Гнёздово_Нивленское (f0), 015_Демидовка (f2), 189_Макеево (f3), DFINE_FOLDS=0,2 | `dryrun_ae/dryrun.log` | rc=0, peak 1421004 kB, 1981 детекция (ae_dfine/ae), покрытие GT чипами 7/7, 1/1, 14/14 |
| or_prep: 185_Изборск, MIN_BYTES=5e7 | `dryrun_orprep/prep.log` | rc=0; 0.11 ГБ за 4 с (28 МБ/с); 0.0678 м/пикс → ×3 → 2493×1610×3, 2.6 МБ |
| Or (B1) с готовым prep | `dryrun_or/dryrun.log` | prep-файл найден (исходник не скачивался); «фолды OK: 6 участков, 6 семей»; покрытие gt_total 4 / в растре 3 / покрыто 3, чипов 5 |
| Or (B1) без prep (fallback в ядре) | `dryrun_or2/dryrun.log` | скачал 0.1 ГБ (31 МБ/с) + prep в ядре; то же покрытие 3/3 |

Ожидаемо в обоих Or-dryrun: фолд 5 → `ValueError('high <= 0')`, rc=1 — при одном участке трейн пуст (на Kaggle 6 участков, трейн = 5).
Недостающий 1 GT Изборска — факт данных: один полигон фортификации площади 0 (вырожденный), один объект городища на 564 м вне границ растра.
Полный Or-dryrun на ящике невозможен (свободно ~4.7 ГБ, `df -h /`; Or ≈65.5 ГБ) — логика обучения/инференса та же, что в Ae-dryrun.

## 4. GPU-часы (wall-часы сессии T4, как в DFINE.md)

s/it: 0.72 (одна T4), 0.98 (2 GPU параллельно) — DFINE.md.

| ядро | оценка | расчёт |
|---|---|---|
| A final | ≈1.5–2.0 | setup ~0.2 + 6000×0.72–0.98 с = 1.2–1.63 ч + бенч ~0.1–0.2 |
| B0 or_prep | 0 GPU (≈1–1.5 ч CPU) | ~65.5 ГБ скачивания; на ящике 27–31 МБ/с ⇒ ~0.6–0.7 ч + даунсэмплинг; скорость S3 с Kaggle неизвестна |
| B1 Or | ≈1.0 (было 1.35 при 6 фолдах; §8) | 6 фолдов / 2 GPU = 3 раунда × (1200×0.98 с = 0.33 ч + инференс ~0.05) + setup 0.2; без B0 + время скачивания 65 ГБ |
| B2 Ae | ≈3.9 | 5 фолдов / 2 GPU = 3 раунда × (4000×0.98 с = 1.09 ч + ~0.1) + 0.3; бюджет трейнера (19800−1100−1800)/3−1200 = 4433 с ≥ 3920 с |
| опц. o2c-a + o2c-b | ≈4.9 | a ≈3.07 + b ≈1.86 (DFINE.md) |

Итого обязательное: A ≈1.5–2 + Or+Ae ≈5.3 GPU-ч (оценка bot 1 для Or/Ae: 5–6).

## 5. Ожидаемый выигрыш (источники)
- SpOr D-FINE: +0.0198 (DFINE.md, 5-фолдовое парное сравнение).
- Or/Ae D-FINE: +0.005…0.015 (bot 1, PRETRAINED_WEIGHTS_2026-10-29.md) — оценка, не измерена.
- obj2coco vs COCO на SpOr: не измерено; опциональные o2c-a/b дают это сравнение на тех же фолдах.

## 6. Риски
1. Or LORO оптимистичен: 4 кавказских участка в радиусе ~35 км друг от друга (Герите–Старый_Барах ~3 км) и не входят
   ни в folds_family_k5, ни в family_clusters bot-10 → это leave-site-out, не leave-family-out.
2. Or-GT мало (~188, из них архитектура 179) → пороги `or_dfine` шумные; GAP «Or» частично закрывается SpOr.
3. Порядок пуша: B1 после COMPLETE у B0; иначе B1 уйдёт в fallback (скачивание в GPU-время, диск /kaggle/working ~20 ГБ — prep идёт по одному тифу под PREP_LOCK с проверкой диска, исходник удаляется).
4. Ae ITERS снижены 6000→4000 ради бюджета ≤5.5 ч; при желании `DFINE_ITERS` (но проверять бюджет).
5. Фолды Or имеют id 0–2 (не k5; до 29.10 07:48 было 0–5) — при сшивке OOF учитывать (§8).
6. Сообщение лога «участков со SpOr» в Or/Ae косметическое.
7. Скорость S3 с Kaggle для B0 неизвестна (на ящике 27–31 МБ/с).

## 7. Открытые вопросы
- Нужна ли для стека отдельная семья для кавказского кластера Or (честнее: 2 фолда «Кавказ / прочие»)?
- Запускать ли опциональные o2c-a/b (~4.9 GPU-ч) — решение bot 1.
- После B1/B2: подбор порогов `or_dfine`/`ae_dfine` на OOF и проверка, что markup_type `or`/`ae` совпадают с ожиданием верификатора (по rg в bot-10/work9, bot-5, bot-1 — совпадают).

## 8. Or: leave-one-family-out, 3 фолда (решение bot 1, 29.10 07:43; сделано 07:46–07:50 MSK)

Причина: 4 кавказских участка «архитектуры» в ≤25 км друг от друга (расстояния центроидов GT: Герите–Старый_Барах 1.9 км,
Лешки–Старый_Барах 6.1, Герите–Лешки 8.0, Фалхан–… 17.7–24.9 км; Елизаветинское от них ~535–557 км, Изборск ~2030–2046 км;
`kaggle/dfine_spor/or_gt_counts.log`) → leave-one-site-out оптимистичен. Они объединены в одну семью **«Кавказ»**.

Изменение только в `build.py` (`folds_or`, `OR_FAMILIES`, `FOLDS_DEFAULT "0,1,2"` для or_o2c; + assert «каждый участок ровно в одном
тестовом фолде, фолд = одна семья»). Бэкап до правки: `backup/20261029_0746_or3fold/` (build.py, or_o2c/*, SHA256_before.txt);
сверка после сборки: `backup/20261029_0746_or3fold/compare.txt` — изменился только `or_o2c/koz3-dfine-or.py`
(02dba66d… → **4576c82abbab5601d6f374133072935f14123acde8fb01c454ebf52bb50880c5**); `or_o2c/kernel-metadata.json` без изменений
(e6b677ad…; от фолдов не зависит). final, or_prep, ae_o2c, full_o2c, full_rest_o2c, smoke, full, full_rest, src/* — `same`;
легаси против `backup/20261029_0724/SHA256_before.txt` — `diff` пуст.

**Нумерация фолдов OOF (поле `fold` в `oof_dfine_or.geojson`) и состав** (GT = features в `*_Or_*.geojson`, `/workspace/data/train-expds-3`;
скрипт-проверка `kaggle/dfine_spor/or3_folds_check.log`):

| fold | семья (тест) | тестовые участки (GT) | трейн-участки (GT) |
|---|---|---|---|
| 0 | Кавказ | 008_Герите 46, 014_Лешки 11, 056_Фалхан 53, 093_Старый_Барах 69 = **179** (все «архитектуры») | 033_Елизаветинское 4 (2 с геометрией), 185_Изборск 5 = 9 (7) |
| 1 | Елизаветинское | 033_Елизаветинское **4** (2 с геометрией, «городища») | Кавказ 179 + Изборск 5 = 184 |
| 2 | Изборск | 185_Изборск **5** (городища 2, фортификации 3) | Кавказ 179 + Елизаветинское 4 = 183 (181) |

Итого 188 features (186 с геометрией). Каждый из 6 участков в тесте ровно один раз; семья теста ни в одном фолде не попадает в трейн
(assert в build.py + рантайм-assert «утечка семьи» в ядре). Порядок очереди 0,1,2: GPU0 берёт f0 (тяжёлый инференс 4 больших растров),
GPU1 — f1, затем f2.

**GPU-часы (T4×2, s/it 0.98 параллельно — DFINE.md):** 2 раунда вместо 3: setup ~0.2 ч + max(GPU0: f0 1200×0.98 с = 0.33 ч + инференс
Кавказа ≈0.28 ч [объём инференса не зависит от разбиения: прежние 6×0.05 ч × доля байтов Кавказа 61.7/65.5 ГБ, `s3/listing.txt`];
GPU1: f1 0.33 + f2 0.33 + ~0.02 ч) ≈ **0.9–1.0 ч** (было 1.35). Число итераций за фолд не зависит от размера трейна; ITERS **не менял** (1200).
Бюджет трейнера на фолд на Kaggle ≈ (5.0 − 0.5 − 0.2)/2 ч − 1200 с ≈ 1.8 ч ≫ 0.33 ч → урезания нет; без B0 (+~0.6–0.7 ч скачивания 65 ГБ)
тоже укладывается в DEADLINE 5.0 ч.

**or_prep совместим:** `or_prep/koz3-dfine-or-prep.py` не изменён (f598b292…); prep не использует фолды (`rg -i fold` по
`src/or_prep_main.py`, `src/or_prep_lib.py`, ядру or_prep — 0 совпадений); имя файла `<stem>__<байты>__r0.25.tif` от фолдов не зависит.
Уже запущенный bot 1 в 07:44 or_prep остаётся валидным.

**CPU-dryrun** (`kaggle/dfine_spor/dryrun_or3/`, `fix/run_dryrun.sh`, venv_t211, 2 потока, nice 10; SCRIPT=or_o2c, DFINE_REGIONS=185_Изборск,
DFINE_PREP_DIR=dryrun_orprep, DFINE_PREP_MIN_BYTES=5e7, фолды по умолчанию 0,1,2): «фолды OK: 6 участков, 3 семей»; prep-файл найден
(исходник не скачивался); покрытие GT 4/3/3, чипов 5; f0 и f1: трейн 5 чипов (Изборск), 3 ит., тестовых участков на ящике 0 → 0 детекций;
f2 (тест = Изборск): трейн 0 чипов → `ValueError('high <= 0')`, **rc=1 — ожидаемо** (как раньше f5); peak RSS 1383560 kB.
Побочно: первый прогон без DFINE_PREP_MIN_BYTES скачал исходный тиф Изборска (0.11 ГБ < порога 1.5 ГБ — на Kaggle так же), лог
`dryrun_or3/dryrun_noprep_minbytes_default.log`, тот же rc=1; скачанный тиф и модели dryrun удалены.

**Риск (важно для стека):** в фолде 0 трейн = Елизаветинское+Изборск, 7 GT с геометрией и **ни одного «архитектуры»** — модель f0
не видела этот класс, а все 179 «архитектуры» Or лежат в f0. OOF Or по «архитектуре» будет близок к нулю: это честная оценка
переноса на новую семью, но подбирать порог `or_dfine` для «архитектуры» по такому OOF фактически не на чем. Решение — за bot 1
(варианты: принять как есть; для «архитектуры» опираться на SpOr/другие источники; финальную Or-модель всё равно учить на всех 6 участках).

## 9. 29.10 ≈12:05 МСК — Or: тайл 1024 (запрос бота 1)
- Правка только в `build.py` (патч `P_OR1024` для or_o2c): tile 640→**1024**, overlap 160→**288** (28 %), bs 8→**4**, ITERS 1200→**2400** (столько же чипов за обучение: 2400×4 = 1200×8). Классы все 5, база obj2coco, 3 семейных фолда, DEADLINE 5.0 — без изменений.
- `or_o2c/koz3-dfine-or.py` 4576c82a… → **b984f6dc3b3562b1b385cd94a2e25fec6f412895706042d00220266d676f11a4**; `kernel-metadata.json` без изменений (e6b677ad63cf3e3da9b0ee907a96f2c3cbb74d00342694a198921b3853a27294); остальные ядра (ae_o2c, final, …) байт-в-байт прежние (`backup/20261029_1150_or1024/SHA256_{before,after}.txt`).
- or_prep (0,25 м, kernel_sources) от тайла не зависит — готовый выход годен.
- CPU-dryrun `dryrun_or1024/` (Изборск, 1015×655 px при 0,5 м < тайла): фолды OK, покрытие GT 3/3, чипов 5 — нарезка 1024 на малом растре работает [изм]. Обучение 1024 на CPU не проверено: сторож 2 ГБ убил на шаге train (RSS 2 255 956 кБ) — ограничение ящика.
- VRAM [оц]: 6,77 ГБ при 640/bs8 (`koz3-dfine-spor-full`) × 2,56 (площадь) / 2 (bs) ≈ 8,7 ГБ из 15 ГБ T4. с/ит [оц] ≈1,2–1,3 → фолд ≈0,8–0,9 ч, 2 раунда на 2×T4 ≈ 2 ч с setup.
- Push (после c3sc-f4): `KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/dfine_spor/or_o2c`
