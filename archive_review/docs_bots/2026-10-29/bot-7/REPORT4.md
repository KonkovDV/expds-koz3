# REPORT4 — кандидат v2 ветки сдачи (bot-7, 29.10.2026 ~00:40 МСК; обновлено 00:55 МСК — twins ВКЛ, af97af0; обновлено 02:50 МСК — PIXGBM bot-3 76f7317, голова **67284ab**, §9)

Ветка `bot7-candidate-v2` в `bot-7/repo`, база `4c3e29e` (кандидат v1, REPORT3), голова **`af97af0`** (было b1a9d28; af97af0 = b1a9d28 + `kurgan_twins_only = true`, см. §8). Без push, без LB, без Kaggle. Все числа ниже взяты из названных файлов в `bot-7/work5/`, если не указано иное.

## 1. Вывод (5 строк)
- В v2 **включён PIXGBM S** (bot-3, bot-10 REPORT7_geo: ACCEPTED → v2) с порогами `*_pix` от bot-9 (`thr_pix_full.json`). stack6 (multipole + filter) остаётся включён: это независимый флаг, порядок pix → stack6.
- **Twins-only bot-6 с починкой `_ground_scale` ВКЛЮЧЁН** (`[post] kurgan_twins_only = true`, коммит af97af0): bot-10 принял 78add0e в пакет v2 — +0,001718 vs A, ДИ90 [+0,00039; +0,00377], P 0,986, LOFO +0,001603, 5/5; с курганами 0,5 м +0,003065, P 1,000 (`bot-10/work8/twins78_stats.log`, строки TO/TOfix-A, kurTO-A; `bot-10/STATUS.md` 29.10 00:20). Код `kurgan_cross_modal_twins` и `_ground_scale` в ветке побайтно равен замеренному bot-10 (`bot-10/work8/b6_78/expds/post.py`).
- **Конфликт −0,001587 — артефакт порядка.** bot-10 мерил stack6 на сырых стеках до порогов, а в сдаче пороги идут раньше. Точная оценка в порядке сдачи (§4): оба (Ss6) − S = keep +0,001304 / drop +0,000061, оба − A = keep +0,013938 / drop +0,011009. Решение по умолчанию = документ bot-10 (S вкл., stack6 вкл.). Перемер на ПК подготовлен (§6).
- Проверки на af97af0: pytest **337 passed / 9 skipped**; build 500,8 МБ (`submission_bot7cand_v2_af97af0.zip`, sha256 a7b618de…5643); check_submission **PASS** (синтетика 26/26; реальные 210, 214, 217 с `--lock` — 27/27 каждый).
- Время: на коробке pix добавил к регионам 210 / 214 / 217 ≈ 27 / 10 / 188 с. Оценка bot-3 для площадки — 258–431 с [оц] при лимите 1200 с. Полный регион на площадке не мерен.

## 2. Коммиты (4c3e29e..af97af0)
| коммит | что |
|---|---|
| 92ab33e | cherry-pick bot-6 78add0e: `post.kurgan_cross_modal_twins` — один cos φ на участок; регресс-тест |
| 32519b8, 9e2ba2c, 3516e3c, 594602e | PIXGBM bot-3 (= 441bb71, 82d23a0, b1ea251, 7cdace5 ветки bot3-pixgbm) |
| 79248f3 | bot-7: `model_dir` от корня репо; `budget_s` (участок целиком без pix при превышении); нет моделей → warning; сборщик кладёт models/pixgbm/all |
| cc174a9 | модели `models/pixgbm/all/{poly,finds}.txt` (sha256 d1fe3efa…ec800 / 5fa8c09d…eb04a) |
| fd6a3bf | default.toml: `[geophys.pixgbm] enabled=true, replace=true, budget_s=600`; пороги `maggr_pix` finds 0,902915 / op 0,684725, `etgr_pix` op 0,545825, `terfl_pix` op 0,746421; twins выкл. (до af97af0); тесты |
| b1a9d28 | check_submission: `geophys.pixgbm.model_dir` — папка (poly.txt, finds.txt при enabled), без ложного FAIL; тест |
| **af97af0** | default.toml: `[post] kurgan_twins_only = true` (приёмка bot-10, `bot-10/work8/twins78_stats.log`); тесты `test_bot7_candidate_v2::test_flags_independent_and_as_documented`, `test_bot7_limits::test_candidate_twins_only_on_and_in_fallback` ждут true |

Файлы (перегенерированы на af97af0): `work5/candidate_v2.diff`, `work5/candidate_v2.mbox` (4c3e29e..af97af0, 10 патчей) и `work5/candidate_v2_from_24bd916.mbox` (24bd916..af97af0, 24 патча, если на ПК нет v1). Проверено `git am` на чистых клонах от 4c3e29e и от 24bd916: дерево = af97af0^{tree} fe3cec17… Хеши — `work5/pc/SHA256SUMS`.

## 3. Флаги и умолчания
| флаг | v2 | основание |
|---|---|---|
| `[geophys.pixgbm] enabled`, `replace` (вариант S) | true, true | bot-10 PREREG_pixgbm / REPORT7_geo: ACCEPTED → v2 |
| `[post] multipole_object_poly`, `filter_finds_points` (stack6) | true, true | bot-10 «ACCEPTED условно» (keep +0,015057 / drop +0,006703) |
| порядок при обоих | pix → stack6 | pix — источник кандидатов в `geophys_feats`, stack6 — постобработка в `watchdog.finish` |
| `[post] kurgan_twins_only` | **true** (с af97af0) | bot-10 `work8/twins78_stats.log`: TO-A +0,001718, P 0,986, LOFO +0,001603, 5/5; kurTO-A +0,003065, P 1,000; STATUS 29.10 00:20 «→ только в пакет». Внутри: `dedup_ae_vs_li_m = 0`, ae_twin и sp_twin вкл. = вариант TO bot-10 |
| `budget_s` | 600 | потолок PIXGBM; атомарен по участку, поэтому может быть превышен на один участок |

Флаги независимы: тесты `tests/test_bot7_candidate_v2.py` — `test_flags_independent_and_as_documented`, `test_flags_off_independently`, `test_replace_S_then_stack6_sees_pix` (порядок pix→stack6), `test_pix_time_cap_keeps_classic`.

## 4. Точная оценка в порядке сдачи (коробка)
Как считалось: `work5/exact/geo_exact.py` + `geo_stats.py`, лог `exact/geo_stats.log`.
1. Пороги по фолду из кривых: A — bot-10 work/curves; S — work7/curves_g_S.
2. Затем multipole + filter из кода ветки v2.
3. Затем официальный match по семьям.

Считались только finds_points и object_poly. Весь их GT лежит в 3 семьях: Судьбищи 221, Семибратнее 2495, Гнёздово 1845. Поэтому Δ = Δ итогового балла.

| сравнение | keep Δ k10 (= LOFO) | P27 | drop Δ | P27 |
|---|---|---|---|---|
| S − A | +0,012634 | 1,0 | +0,010949 | 1,0 |
| s6 − A | +0,012285 | 0,852 | +0,003768 | 0,852 |
| Ss6 − A | **+0,013938** | 1,0 | **+0,011009** | 1,0 |
| Ss6 − S | +0,001304 | 0,963 | +0,000061 | 0,704 |
| Ss6 − s6 | +0,001653 | 0,704 | +0,007241 | 0,704 |

- Ss6 − S по 5 раскладкам: keep +0,001233 / +0,001304 ×3 / +0,000595; drop −0,000010 / +0,000061 ×3 / +0,000074.
- Без одной семьи (keep): без Гнёздово +0,00008, без Семибратнее +0,00179, без Судьбищ +0,00130.
- Для сравнения, в порядке bot-10 (воспроизведено точно, `work5/pc/selftest_paired.log`): S +0,011935 / +0,010980, s6 +0,015057 / +0,006703, Ss6 +0,01347 / +0,01104.

**Почему результаты расходятся.** В порядке сдачи счётчик «≥ 50 поселений» и terfl > 150 считаются после порогов, а вторичные полюса multipole заново не порогуются. Поэтому stack6 сам по себе в сдаче слабее (keep +0,0123 против +0,0151 у bot-10), а поверх S он уже не вредит.

**Оговорки:**
- всего 3 семьи, P27 = точный бутстрэп по 27 выборкам;
- `merge_within_markup` не моделировался;
- S взят из pix_lofo (LOFO-модели), а в сдаче модель all.

Это не замена приёмки bot-10: формально решение остаётся за перемером на ПК (§6).

## 5. Проверки, лимиты, время и ОЗУ
**pytest и сборка:**
- pytest `-k "not coco_roundtrip"` на **af97af0**: **337 passed, 9 skipped, 3 deselected**, 57,4 с, RSS 1198 МБ (`work5/pytest_af97af0.log`). Ранее на b1a9d28 — то же число, 55,4 с, 1215 МБ (`pytest_b1a9d28.log`).
- build af97af0: **`work5/dist/submission_bot7cand_v2_af97af0.zip`**, 50 файлов, 525 149 404 Б (500,8 МиБ), sha256 **a7b618dee86831616314d389517dea7c0a710b55d08a349d8fdb3b795dd95643**, 22 с, RSS 334 МБ (`build_af97af0.log`). От архива b1a9d28 (1741b23e…a302, `build.log`) отличается только `configs/default.toml` (сверка CRC по 50 файлам).

**check_submission:**

| прогон | итог | main.py, время | пик RSS | geo | выход |
|---|---|---|---|---|---|
| **af97af0**: синтетика (`check_af97af0.log`) | PASS 26, FAIL 0, WARN 0 | 5,3 с | 992 МиБ | 1 с | 230 Polygon (= b1a9d28) |
| **af97af0**: 210_Гнёздово_Терраса_2, `--lock` (`check_real_af97af0_210.log`) | PASS 27, lock PASS | 23,5 с | 670 МиБ | 22 с | 383 (= b1a9d28) |
| **af97af0**: 214_Гнёздово_Лесная_2, `--lock` (`check_real_af97af0_214_Гнёздово_Лесная_2.log`) | PASS 27 | 13,2 с | 440 МиБ | 12 с | 56 (= b1a9d28) |
| **af97af0**: 217_Гнёздово_Селище_3, `--lock` (`check_real_af97af0_217_Гнёздово_Селище_3.log`) | PASS 27 | 79,5 с | 633 МиБ | 78 с | 298 (= b1a9d28) |
| b1a9d28: синтетика (`check.log`) | PASS 26 проверок | 17,4 с | 994 МиБ | — | 230 Polygon |
| b1a9d28: 210, `--lock` (`check_real.log`) | PASS 27, lock PASS | 30,8 с (v1 10,2 с) | 669 МиБ | 29 с | 383 объекта |
| b1a9d28: 214 | PASS | 13,0 с | 436 МиБ | — | 56 |
| b1a9d28: 217 | PASS | **194,2 с** (v1 12,0 с) | 631 МиБ | 191 с | 298 |

Время шумное: при b1a9d28 load ≈ 6,6, при af97af0 load 3,2–4,0 (217: 194 → 80 с — это нагрузка коробки, код geo тот же). Регионы 210/214/217 — только MagGr, курганов нет, поэтому twins там не срабатывает и выход совпадает с b1a9d28; на синтетике курганов по 3, двойников не появилось. **Путь twins на реальном регионе с курганами li/ae/sp в main.py на коробке не прогонялся** (покрыт юнит-тестами `test_bot6_twins_only.py`, `test_bot7_limits.py` и замером bot-10 на стеке).

**Лимиты:**
- 20000 объектов / 3000 вершин гарантирует код (`finalize` / `cap_features` / `writer.enforce_limits`, REPORT3 §3).
- Все выходы check ≤ 383 объектов, validate OK.
- Архив 500,8 МиБ, распакованный 546,1 МиБ ≤ 2048.
- Прокси лимитов с pix-семьями заново не гонялся.

**Время PIXGBM при лимите 1200 с:**
- bot-3 BUDGET_PIXGBM §2 [изм]: 7 участков / 43 растра — 420 с (1 поток), 275 с (2 потока), пик 706 МБ.
- bot-3 §4 [оц]: площадка 258–431 с.
- Наш `budget_s=600` при норме не срабатывает.

**Не мерено:**
- main.py на площадке;
- крупный мультимодальный регион целиком (RAM ≤ 2 ГБ);
- тестовый объём.

**Риски:**
- bot-9 `geo_predict` по всему трейну одним процессом — пик 3052 МБ. В main.py это один регион, но у самых крупных регионов это не проверено.
- bot-3 §8.6: классика geophys на 204/212 превысила сторож 1,95 ГБ. Это не PIXGBM и относится к v1 тоже.

## 6. Перемер на ПК (только C:, F: не трогать)
Скрипт `work5/pc/pc_measure_v2.ps1`. Он трёхсторонний: {S, s6, Ss6 (pix→stack6)} против A, плюс Ss6 против S и против s6.
- Порядок bot-10: `split_stack_by_family` → `mkfam_pix_v2` → `stack6_fam --mode separate` → `family_curves` → `paired.py`.
- Порядок сдачи: `geo_exact.py` + `geo_stats.py`.
- По желанию: main.py архива на одном участке с временем и пиком памяти.

**Положить на C:**
- `C:\expds3c\in\`:
  - `stack_base.geojson` — с коробки `/workspace/rerank/stack_arch.geojson`, sha256 b412e299…e253;
  - `gt_train_v2.geojson` — `/workspace/rerank/gt_train_v2.geojson`, d4bc1a3b…ff86;
  - `pix_lofo.geojson` — `bot-3/pixgbm/pix_lofo.geojson`, efec0552…0ac650.
- `C:\expds3c\bot7v2\`: `tools_v2.tar`, `family_clusters.json`, `pc_measure_v2.ps1`, `SHA256SUMS`.
- По желанию: `submission_bot7cand_v2_af97af0.zip` (sha256 a7b618de…5643, есть в SHA256SUMS; строка b1a9d28 оставлена для старого архива).

Нужен Python с numpy, shapely и geopandas (как для pc_measure_pkg.ps1).

**Команды:**
```
powershell -ExecutionPolicy Bypass -File C:\expds3c\bot7v2\pc_measure_v2.ps1
powershell -ExecutionPolicy Bypass -File C:\expds3c\bot7v2\pc_measure_v2.ps1 -MainRegion C:\expds3c\regions\<участок> -Zip C:\expds3c\bot7v2\submission_bot7cand_v2_af97af0.zip
```
Изменения скрипта 00:50: (1) пример с новым архивом; (2) папка распаковки `$Out\zip_<имя архива>` вместо общей `zip_v2` — иначе старый распакованный b1a9d28 молча использовался бы повторно; (3) добавлен UTF-8 BOM — без него Windows PowerShell 5.1 читает файл как cp1251, и «ф» в строке `"нет файла…"` (байты D1 84 = «С„») закрывает строку → ошибка разбора. `tools_v2.tar` не менялся: expds/ в b1a9d28 и af97af0 одинаков (`git diff --stat b1a9d28 af97af0 -- expds scripts inference` пуст); перемер geo (finds/object_poly) twins не затрагивает.

**Проверить:**
- `paired_vsA.log`, строка k10: A_keep = 0,333237, A_drop = 0,314615;
- `geo_stats.log` сверить с §4.

**Самопроверка tools на коробке:**
- `geo_exact` из tools_v2.tar — строки geo_part ИДЕНТИЧНЫ `exact/geo_exact.log` (`pc/selftest_geo_exact.log`, RSS 772 МБ, около 40 с);
- `paired.py` из tools — числа bot-10 точно (`pc/selftest_paired.log`, boot 200, RSS 1130 МБ, около 610 с на нагруженной коробке).

## 7. Не включено, ожидает
- **bot-3 5deb94a** (lleaves/treelite, пул признаков, срок фазы и лестница; BUDGET_PIXGBM §8) не взят: нет приёмки, нужны `.so` рядом с моделью и сборка. Ускорение wall ×1,83–2,97 [изм у bot-3]; кандидаты побайтно равны lightgbm. Кандидат на v3.
- **Пороги.** bot-3 `thresholds_v2.toml` (maggr_pix finds 0,93 / poly 0,686, etgr_pix 0, terfl_pix 1,01) отличаются от bot-9 `thr_pix_full.json`, которые стоят в v2. Какие брать, должен решить bot-10.
- **bot-8 `bot8-cpu-a-clean`.** Есть ветка `bot8-cpu-a-clean-on-v2` @18b2f66 на 4c3e29e: pytest 410 / 12 skipped / 1 failed (coco), check PASS. Но lidar_a не подключён и весов сценария (а) в models/ нет (bot-8 STATUS 23:50: вариант (а) не начат). Условие включения не выполнено, **не включён**.

## 8. Обновление 29.10 00:55 МСК: twins-only включён (af97af0)
- Основание проверено по файлу: `bot-10/work8/twins78_stats.log` (00:14 МСК) — `TO-A`/`TOfix-A: d10=+0.001718 P=0.986 CI90=[+0.00039;+0.00377] lofo=+0.001603 seeds=5/5`, `kurTO-A: d10=+0.003065 P=1.000`; `bot-10/STATUS.md` 29.10 00:20: «twins-only 78add0e воспроизведён: +0,001718 … → только в пакет». Скрипт bot-10 (`work8/twins78.py`) мерит код `work8/b6_78` (= bot-6 78add0e); в нашей ветке `kurgan_cross_modal_twins` и `_ground_scale` текстуально равны ему, а `kurgan_twins_only = true` даёт ровно spec варианта TO (`dedup_ae_vs_li_m = 0`, ae+sp twins).
- Коммит af97af0: только `configs/default.toml` + 2 теста. pytest 337/9, сборка, check ×4 PASS (§5). diff/mbox и `pc/SHA256SUMS` перегенерированы.
- Пороги `*_pix` — по-прежнему bot-9 (`thr_pix_full.json`); bot-10 ещё выбирает между bot-9 и bot-3.
- Честный перемер пакета Ss6 + twins + kur одним прогоном на коробке не делался: twins (курганы) и pix/stack6 (finds/object_poly) трогают разные классы, поэтому оценки складываются по классам (kurTO-A +0,003065 у bot-10 + §4), но это оценка, а не замер.

## 9. Обновление 29.10 02:50 МСК: PIXGBM S от bot-3 (bot3-pixgbm 76f7317), голова 67284ab

Источник: `bot-3/repo` ветка `bot3-pixgbm` @76f7317, `bot-3/pixgbm/PACKAGING_S.md`, `config_v2_S.toml`, `thresholds_v2.toml`, `packer_pixgbm.patch`, `BUDGET_PIXGBM.md`.

### 9.1. Что отличалось (v2 af97af0 ↔ 76f7317) и что взято
| часть | v2 af97af0 | bot-3 76f7317 | решение |
|---|---|---|---|
| срок pix | свой: суммарное время pix > budget_s → весь регион на классике | абсолютный срок фазы `min(guard.t0, старт процесса) + budget_s` (env `PIXGBM_BUDGET_S` важнее), липкая лестница full → finds_only → classic, `status{done,fell}` → классика остаётся на упавших растрах | **взят** (5deb94a → bb0e0d8, cherry-pick -x; мои тесты stop/pix_spent переписаны) |
| бэкенд | только lightgbm | lightgbm (умолч.) / lleaves / treelite opt-in с фолбэком на lightgbm | **взят код**, `backend = "lightgbm"`; .so lleaves/treelite в архив НЕ кладутся |
| потоки/пул | потоки = OMP | `avail_cores` по sched_getaffinity; `pix_threads`: EXPDS_PIXGBM_THREADS → EXPDS_THREADS → min(8, avail); fork-пул `procs=0` → min(threads, 4, avail); `_rate` по procs; `process_t0` без psutil; `resolve_model_dir` от корня | **взят** (76f7317 → ecb92ec, чисто) |
| модели | `models/pixgbm/all/{poly,finds}.txt` | те же | побайтно равны: poly.txt 2 122 321 Б sha256 d1fe3efa…ec800, finds.txt 2 111 596 Б 5fa8c09d…eb04a |
| упаковщик | `scripts/build_submission.py` кладёт оба .txt при enabled | patch: finds условно + .so бэкендов | **не взят**: для lightgbm эквивалентно; в архиве `models/pixgbm/all/poly.txt`, `finds.txt`, .so бэкендов нет |
| ключи `[geophys.pixgbm]` | maps, t_pix, budget_s, enabled | + backend, guard_reserve_s, procs, poly_mode="pole" | взяты backend="lightgbm", guard_reserve_s=0, procs=0; `poly_mode` наша ветка не читает (сверка tomllib: из pixgbm-ключей расходится только budget_s) |
| budget_s | (свой смысл) | 300 | **600** (§9.2) |
| пороги `*_pix` | T9 bot-9: maggr_pix finds 0,902915 / op 0,684725; etgr_pix op 0,545825; terfl_pix op 0,746421 | T3: 0,93 / 0,686; 0,0; 1,01 | **оставлены T9** (§9.3) |
| audit01 `dedupe_maps`, split/pole_circle (c315a9b, c573ea9, b8cfdf4) | — | по умолчанию выключены/no-op | **не взяты** (вне PIXGBM S) |
| stack6, twins-only | вкл. | — | **оставлены вкл.**, порядок pix → stack6 (bot-10 REPORT9_v2 §A: Ss6−A keep +0,013938) |

Свои коммиты поверх: 4957729 (тесты под новый API, комментарии конфига), 67284ab (check_submission: psutil/tl2cgen/lleaves — необязательные пакеты с фолбэком, не FAIL imports/lock; `[geophys.pixgbm] maps` — список типов, не режим gray/all; без этого check давал 2 ложных FAIL — `work5/check_4957729_falsefail.log`). Ветка: af97af0 → bb0e0d8 → ecb92ec → 4957729 → **67284ab**.

### 9.2. Бюджет 600 с, а не 300
- Лимит площадки 1800 с, сторож 1620 с, `[time] budget_s` 1500. 1200 с («≤20 мин») — цель приёмки (docs/IMPROVEMENT_PLAN.md:198, docs/ARCHITECTURE_TARGET_2026-10-25.md:54; bot-8/BUDGET.md:28, 72), а не лимит площадки. **Расхождение зафиксировано:** в задаче названо «1200 с лимит площадки»; по документам это цель, жёсткий предел — 1800/1620.
- Срок pix абсолютный, от старта процесса: при 600 с остальному конвейеру остаётся ≥600 с до 1200 с и ≈1020 с до сторожа. Остальное ≈191–202 с (bot-3 BUDGET_PIXGBM.md:6), ×2 запаса ≈400 с.
- Оценка pix на площадке 258–431 с при 2 потоках (bot-3 BUDGET_PIXGBM.md:6, [оц]) **больше 300**. При 300 есть риск, что на публичном наборе сработает finds_only/classic и S частично потеряется. При 600 запас есть, а лестница всё равно страхует.
- Риск ОЗУ [оц]: на площадке с ≥4 ядрами `procs=0` даёт fork-пул до 4 детей, ≈+0,35 ГБ на ребёнка (PACKAGING_S). На коробке не мерено (по COMMON ≤2 процесса прогон шёл с `--cpus 0` → avail 1 → без пула). Если ОЗУ площадки под вопросом, ставим `procs = 1`.

### 9.3. Пороги: T9 (bot-9), а не T3 (bot-3)
bot-10 REPORT9_v2.md §B: T9 0,347983 против T3 0,347059, T3−T9 = −0,000924, P(T3>T9) 0,259, ДИ90 [−0,00184; +0,00033]. bot-10 выбрал T9, поэтому `thresholds_v2.toml` bot-3 не применяется. Расхождение записано в комментарии `configs/default.toml`.

### 9.4. Проверки на 67284ab
- pytest `-k "not coco_roundtrip"` (venv bot-8, OMP 1, EXPDS_PIXGBM_THREADS=1): **345 passed, 9 skipped, 3 deselected**, 63 с, RSS 1219 МБ (`work5/pytest_67284ab.log`). На 4957729 было 343/9/3 (`pytest_4957729.log`), на af97af0 — 337.
- Сборка: **`work5/dist/submission_bot7cand_v2_67284ab.zip`**, 50 файлов, 525 155 201 Б (500,8 МиБ), sha256 **15d3122dc12d8fbae171e8753ed58b176e276d9b8ebd4a0ad3877a3c33cc4d22** (`build_67284ab.log`). От af97af0 по CRC отличаются только `configs/default.toml`, `expds/geo_pixgbm.py`, `expds/pipeline.py`. Есть `models/pixgbm/all/poly.txt` и `finds.txt`. Единственный .so — ускоритель `inference/lib/expds_core.abi3.so`, как раньше. Сборка на 4957729 дала тот же sha256 (scripts/ в архив не входят).
- check_submission (`--cpus 0`, чтобы не было пула: по COMMON ≤2 процесса; поэтому время **не сравнимо** с af97af0, где было `--cpus 0,1`):

| прогон | итог | main.py | пик RSS | строка pixgbm (run.log) | выход | finds_points v1 / af97af0 / 67284ab |
|---|---|---|---|---|---|---|
| синтетика `--lock` (`check_67284ab.log`, `_run.log`) | PASS, FAIL 0, WARN 0, 29 проверок | 8,7 с | 993 МиБ | region_0002: ступени {'full': 1}, уровень full | 230 (= af97af0) | — / 222 / 222 |
| 210 `--lock` (`check_real_67284ab_210.log`, `_run.log`) | PASS 29, lock PASS | 38,9 с (pix 36,9 с) | 603 МиБ | {'full': 1}, уровень full | 383 = af97af0, файл 280 748 Б | **323** (`work3/check_real.log`) / 373 / **373** |
| 214 `--lock` (`check_real_67284ab_214.log`) | PASS 29 | 18,0 с (pix 13,4 с) | 453 МиБ | {'full': 21}, full | 56 (= af97af0) | 0 / 0 / 0 |
| 217 `--lock` (`check_real_67284ab_217.log`) | PASS 29 | 111,4 с (pix 109,0 с) | 631 МиБ | {'full': 10}, full | 298 (= af97af0; v1 170) | 0 / 0 / 0 |

На всех трёх регионах выход совпадает с af97af0 по числу объектов и классам. Для 210 размер файла тоже совпал (280 748 Б). Это ожидаемо: при уровне full новый код даёт те же кандидаты.
- Ограничения: срок 600 с и лестница на коробке сработали только в юнит-тестах (`test_region_feats_deadline_and_no_models`, `test_pix_fell_raster_keeps_classic`). Пул на площадке не мерен.

### 9.5. Артефакты для ПК
- `work5/candidate_v2.diff` (4c3e29e..67284ab), `candidate_v2.mbox` (14 патчей), `candidate_v2_from_24bd916.mbox` (28 патчей). Каждый проверен `git am` / `git apply` на чистую базу: дерево = 67284ab.
- `work5/pc/SHA256SUMS` обновлён: diff/mbox и новый архив. Строка `pc_measure_v2.ps1` aaf89ad2… не менялась, файл не трогали.
- `tools_v2.tar` не пересобирался. В нём лежит expds af97af0, а замер на ПК берёт только `post.py`/metric (geo_exact.py импортирует `expds7.post`), а post.py с af97af0 не менялся (`git diff --stat af97af0 67284ab`).


## 10. Обновление 29.10 03:25 МСК: мягкий потолок bot-8, procs = 2, починка пула PIXGBM — голова eb268d6

Ветка `bot7-candidate-v2` (без push): 67284ab → **4623c76** (cherry-pick -x bot-8 26e12ae: мягкий потолок) → **31594d2** (cherry-pick -x bot-8 428257f: тест, что срок pixgbm отсчитывается от guard.t0) → **8594842** (флаги) → **eb268d6** (починка пула). У bot-8 не брались f7cde8e (слияние) и 5deb94a (дублирует наш bb0e0d8). Текстовых конфликтов не было. Сверка `git diff 428257f HEAD -- expds/` показала только ожидаемые различия: нет dedupe_maps bot-3 audit01, своё разрешение model_dir, ecb92ec. Код мягкого потолка совпадает с bot-8 побайтно.

### 10.1. Флаги (решения бота 1)
- `[time] soft_ceiling_s = 1200` записан явно; это умолчание bot-8. Приоритет: EXPDS_SOFT_CEILING_S > конфиг > 1200, значение ≤0 выключает. Пороги пропуска: wide_window 120 с, sigma 60 с (+60 с на verify), verify 60 с. TimeGuard 1500 с и сторож 1620 с не менялись.
- `[geophys.pixgbm] procs = 2` (было 0, то есть до 4 детей). Причина: пик ОЗУ прогона у bot-9 уже 3 ГБ.
- `budget_s = 600` — **ВРЕМЕННО**. Решение принимается после полного прогона на ПК: если mock_val укладывается в ≤1200 с, оставляем 600, иначе возвращаем 300. Комментарий об этом есть в default.toml.
- Тест флагов: `tests/test_bot7_candidate_v2.py` (procs == 2, soft_ceiling_from == 1200).

### 10.2. Найдена и исправлена ошибка: SIGTERM в fork-пуле PIXGBM (eb268d6)
- 8594842 на 217 дал **FAIL**: 918,2 с, записано 0, файл 45 Б, в run.log «сигнал 15» на 16-й секунде (`work5/check_8594842_217.log`, `_run.log`).
- Причина. `watchdog.arm_signal` ставит в main.py обработчик SIGTERM, который пишет результат и вызывает os._exit. Дети `mp.get_context("fork").Pool` наследуют этот обработчик. `pool.terminate()` шлёт детям SIGTERM, ребёнок пишет **пустой** result.geojson поверх и может зависнуть, а родитель висит в `pool.join()` до таймаута проверки (900 с).
- Ошибка есть и в 67284ab: на платформе procs=0 даёт пул из ≥2 процессов при ≥2 ядрах. На коробке с `--cpus 0` в 67284ab пула нет (1 ядро), поэтому ошибка не проявлялась. Она гоночная: 8594842 на 210 с пулом прошёл.
- Починка `expds/geo_pixgbm.py`:
  - `make_pool(n)` блокирует SIGTERM в родителе на время fork (pthread_sigmask) и восстанавливает маску в finally.
  - initializer `_pool_init` ставит детям SIG_DFL и снимает блокировку.
  - Обработчик родителя не меняется.
- Тест `tests/test_bot7_pix_pool_sigterm.py`. В нём есть отрицательный контроль: обычный Pool оставляет маркеры обработчика, `make_pool` не оставляет. Тест прошёл 3/3 раза.

### 10.3. Проверки на eb268d6
- pytest (`-k "not coco_roundtrip"`, venv bot-8): 364 passed / **1 failed** / 9 skipped / 3 deselected, 71 с, RSS 1026 МБ (`work5/pytest_eb268d6.log`).
  - Упал `test_bot8_window_deadline.py::test_watchdog_deadline_bounds_window_loop`: `19.999999999996362 == 20`. Это точное сравнение float с `PROCESS_START + 1600` в тесте bot-8, наш код ни при чём; на 8594842 тот же тест прошёл.
  - Повтор этого файла вместе с тестами пула, флагов и soft_ceiling: 34 passed (`work5/pytest_eb268d6_rerun.log`).
- Сборка: `work5/dist/submission_bot7cand_v2_eb268d6.zip`, 525 158 901 Б, sha256 `8f3f267107ce1bf5ac2c0b2d15346749521b0ce9c809ac4f339649085422d589` (`work5/build_eb268d6.log`).
- check_submission (`--cpus 0 --lock requirements.lock --keep`, последовательно; `work5/check_eb268d6_*.log`, `work5/chk9.log`):

| вход | итог | main, с | пик RSS дерева, МиБ | записано / байт | pixgbm | = 67284ab |
|---|---|---|---|---|---|---|
| синтетика | PASS 29 | 7,1 | 1009 (67284ab 989) | 230 / 162 622 | full | побайтно |
| 210 | PASS 29 | 43,0 | 1449 (67284ab 615) | 383 / 280 748 | full | побайтно |
| 214 | PASS 29 | 18,5 | 436 (67284ab 438) | 56 / 63 321 | full ×21 | побайтно |
| 217 | PASS 29 | 129,3 | 1541 (67284ab 633) | 298 / 337 079 | full ×10, пул работал (фаза 126,5 с, cpu родителя 76,2 с) | побайтно |

- Строка «мягкий потолок 1200 с от старта процесса…» есть во всех четырёх run.log. Строк пропуска нет: прогоны короткие.
- 67284ab на 217 для сравнения: PASS, 123,2 с (`work5/check_67284ab_217.log`).
- Рост пика RSS на 210 и 217 объясняется двумя fork-детьми. Это сумма RSS дерева, разделяемые COW-страницы в ней учтены дважды. Лимит проверки 2000 МиБ, запас около 450 МиБ.

### 10.4. Артефакты
- `work5/candidate_v2.diff` (от 4c3e29e), `candidate_v2.mbox` (18 патчей), `candidate_v2_from_24bd916.mbox` (32). `git am` на 4c3e29e и на 24bd916 и `git apply` дают дерево eb268d6 `2ce37540ff8a6aee7da178d9e755b276c224f678`.
- `work5/pc/SHA256SUMS` обновлён: новые diff/mbox и zip eb268d6; строка zip 8594842 **удалена**, этот архив с ошибкой пула брать нельзя. pc_measure_v2.ps1 не тронут (aaf89ad2…). Для прогона на ПК нужен `-Zip submission_bot7cand_v2_eb268d6.zip`.

### 10.5. Риски
- Ошибка пула есть и в уже собранных 67284ab/af97af0 на многоядерной платформе. Сдавать их нельзя, только eb268d6 и новее.
- procs=2 добавляет до ~0,9 ГБ к RSS дерева (210: 615 → 1449 МиБ). На ПК нужно смотреть общий пик.
- budget 600 временный, ждём ПК.
- Принудительный пропуск шагов по мягкому потолку проверен только синтетическим тестом и юнит-тестами bot-8, на реальных данных он не срабатывал.
- Флаки-тест bot-8: точное сравнение float, нужна правка у bot-8 (pytest.approx).
