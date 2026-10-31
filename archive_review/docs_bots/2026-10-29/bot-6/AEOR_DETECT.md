# AEOR_DETECT — все-классовый D-FINE на Ae/Or в линии bot-7 (bot-6, 29.10)

**Ветка:** `bot6-aeor-detect` от `360c870` (bot7-candidate-v3, код = 36af6b7), коммит **`3617be6dceb38cb4b7045d20ca81d91e38c8f209`** (только локально).
**Diff:** `bot-6/bot6-aeor-detect.diff` (1154 строки; 6 файлов, +1016/−11 по `git diff --stat 360c870..bot6-aeor-detect`).

## 1. Ревизия скелета (aeor_dfine.py → expds/dfine_aeor.py)
- В скелете был `S.DFineUnavailable`, которого в v3 нет. Заменён на собственный `AeOrUnavailable`, маркер `DFINE_AEOR_UNAVAILABLE`.
- Разрешение Or 0,25 м исправлено на 0,5 м, как в рецепте bot-9: DFINE_FINAL_ORAE.md, dfine_main.py:53,66.
- Tile и overlap были глобальными. Теперь они заданы отдельно для каждой модальности: Ae 640/160, Or 1024/288.
- Прямой импорт transformers убран. Модели грузятся только через `sp_dfine.load_models` → `import_dfine` (проверка num_labels==5).
- Имена: флаг `[dfine_aeor]`, markup_type нативный `ae`/`or`, threshold_group `dfine_ae`/`dfine_or`, пороги `[thresholds_by_markup.dfine_ae/dfine_or]`.

## 2. Точка вставки (expds/pipeline.py)
- Фаза `_dfine_aeor_phase` (стр. 1129) вызывается после `_sp_dfine_phase` (стр. 1337) и до gor_gate (стр. 1344). Импорт ленивый, `phases.append("dfine_aeor")`, `state.add`.
- На каждый регион вызывается `_soft_ok(guard,"dfine_aeor",...)`. Логи: `dfine_aeor n_det=N region=…` и `dfine_aeor n_det=N total`.
- Preflight в run() работает только при `enabled`:
  - проверяет импорт через sp_dfine.import_dfine, наличие файлов весов, соответствие id2label классам compute_metrics и готовность порогов;
  - при проблеме пишет ERROR с маркером (check_submission → FAIL), и сабмит собирается без фазы.
- Кэш моделей общий с sp_dfine (`dfine_cache`, стр. 1332). Если пути совпадают, модели не грузятся повторно.

## 3. Дизайн
- **Тайлинг.** Тайлы нарезаются в разрешении обучения через Scene/WarpedVRT в UTM. Последний ряд и столбец прижимаются к краю. Пустые тайлы (доля nodata ≥ 0,95) пропускаются.
- **Пороги.**
  - Детектор работает с низким полом `min_score` 0,05.
  - Пороги по группам и классам берутся из `thresholds_file` (JSON `{"dfine_ae":{класс:порог},"dfine_or":{...}}`) и сливаются в `thresholds_by_markup`; при конфликте выигрывают ключи из TOML.
  - При `require_thresholds=true` фаза без порогов не запускается: так нельзя выпустить детекции с полом 0,05.
  - Сейчас порогов нет. Это честная заглушка для bot-10, цифры не придуманы.
- **TTA и масштабы.** TTA — горизонтальный flip. `scales` по умолчанию `[1.0]`, опционально `[1.0, 1.5]`.
- **Слияние детекций.** WBF (IoU 0,55) или NMS (0,5). Сливаются только виды одного растра одной модальности, между снимками и типами слияния нет (это проверено тестом).
- **Геопривязка.** Боксы переводятся из UTM в 3857 и становятся эллипсами (32-угольник), затем `cluster_circles` внутри растра. Классы сверены с `materials/compute_metrics.py`.
- **Time guard.**
  - `plan_level` проходит уровни full → no_flip → one_scale → skip с запасом ×8 на измеренное `t_view` и выбранный уровень пишет в лог.
  - Бюджет фазы — 120 с на датасет (`budget_s`), регионы идут от дешёвых к дорогим.
  - `max_objects` 20000 на группу.
- **Упаковка.** build_submission кладёт модуль, sp_dfine, vendor, веса и thresholds_file только при `enabled`. check_submission ищет маркер, а при включённом флаге требует наличие модуля и vendor и проверяет импорт.

## 4. Проверки
- `tests/test_dfine_aeor.py`: **29 passed** (`bot-6/work/aeor_pytest_final.log`), детектор заглушечный. Что покрыто:
  - тайлинг (покрытие и перекрытие);
  - WBF/NMS не сливает разные снимки и типы;
  - round-trip геопривязки;
  - деградация guard и её логи;
  - дедлайн фазы;
  - merge_within_markup;
  - вариант R не удаляет dfine_or;
  - frame_prior не трогает dfine_*;
  - verify не сбрасывает conf;
  - пороги групп;
  - гейтинг, ленивый импорт и порядок фаз;
  - без флага ничего не упаковывается;
  - маркер в check_submission.
- **Побайтовая идентичность** (`bot-6/work/aeor_identity/check_identity.py`, результат в `run2/identity.json`). На синтетике conftest (LAS + MagGr) сравнивались sha256 выхода `pipeline.run` в трёх режимах:
  - база 360c870;
  - ветка с выключенным флагом;
  - ветка с флагом без весов.

  Все три sha = `9222f5d1…9a83d9`, n_features=245, маркер залогирован. Первый прогон run1 упал на базе из-за лимита 2 ГБ AS (`memory allocation of 60000000 bytes failed`). Повтор прошёл с MALLOC_ARENA_MAX=2 и RAYON_NUM_THREADS=2.
- **Смежные тесты при prlimit 2 ГБ.** Все падения — из-за окружения: `can't allocate memory`, `failed to map segment` (scipy, lazrs).
  - v3_flags, dfine_import, check_submission, crop_verify, candidate_v2: на ветке 102 passed / 9 failed (`work/aeor_pytest_flags.log`). Все 9 упавших — в crop_verify и candidate_v2, где torch и scipy упираются в лимит памяти. На базе pytest упал с INTERNALERROR (`work/aeor_pytest_flags_base.log`).
  - test_pipeline и др.: 5 failed / 43 passed на ветке и те же 5 на базе (`work/aeor_pytest_pipeline.log`, `work/aeor_pytest_pipeline_base.log`).

## 5. Как включить
```toml
[dfine_aeor]
enabled = true
weights = { ae = ["models/dfine_ae"], or = ["models/dfine_or"] }   # HF-папки: config.json + model.safetensors
thresholds_file = "models/dfine_aeor/thresholds_oof.json"          # {"dfine_ae": {класс: порог}, "dfine_or": {...}}
# scales = [1.0, 1.5]
# budget_s = 120
```

## 6. Что ждём от bot-9 / bot-10
- **bot-9:** чекпойнты Ae и Or (HF-папки, 5 классов sp_dfine.CLASSES) и замер времени на тайл на 4090.
- **bot-10:** вложенные OOF-пороги в формате выше. Пороги arch для Or слабые, потому что fold 0 не видит arkhitektury.

## 7. Вопросы к bot-7
1. `merge_within_markup` объединяет круги dfine_ae с другими ae-источниками того же класса (union, max conf). Так и задумано, или нужен отдельный ключ markup/source?
2. Будет ли ребейз на v4? Сейчас ветка идёт от 360c870.
3. Нужен ли потоковый detect_tif для больших Or-растров? Сейчас растр читается тайлами через Scene, память ограничена батчем.
4. Реальный tile_s на 4090 нужен, чтобы откалибровать ×8. Стоит ли включать scales 1,5?
5. Для dfine_aeor в check_submission нет smoke-эталона (аналога DFINE_KNOWN). Добавлять ли его?

## v2 — правки по замечаниям bot-7 (29.10 12:31 и 12:49 МСК), коммит f4e733b (родитель 3617be6, база 360c870, не запушен)
Diff целиком: `bot-6/bot6-aeor-detect.diff` (`git diff 360c870..f4e733b`, 97 785 байт, sha256 a45f5e4d…0ea8c2).

**1. Ключ слияния.** `post.merge_within_markup` теперь сливает по ключу (участок, markup_type, класс, threshold_group, если группа из `post.ISOLATED_GROUPS` = dfine_ae/dfine_or, иначе None). То есть dfine_* сливаются только сами с собой. Кроме того, объекты dfine_* идут по отдельной цепочке: `watchdog._aeor_split` → `_finish_v3(старые)` + `_aeor_finish(dfine, остаток MAX_FEATURES)`, и так же в `safe_finish`. Объединение с другими источниками — `[dfine_aeor] merge_union = true`, по умолчанию выключено. Тест `test_flag_on_old_groups_byte_identical_to_off`: на реальном `configs/default.toml` при включённом флаге старые объекты побайтно совпадают с выдачей при выключенном (и finish, и safe_finish). `test_aeor_tail_respects_feature_limit` проверяет потолок.

**2. Память (≤ 2 ГБ).** Окна читаются через `dfine_aeor.Scene`, подкласс `sp_dfine.Scene`. Если до 0,5 м нужно сильно уменьшать, Scene берёт уровень обзора (ovr_max_frac 0,5) и не warp'ит полный растр. Батч идёт в один заранее выделенный float32-буфер (bs,3,T,T), /255 на месте, `torch.from_numpy` без копий, `GDAL_CACHEMAX` = `gdal_cache_mb` 256. Замер — заглушка детектора, flip включён, `prlimit --as=2000000000`, 2 потока, ru_maxrss (`work/aeor_rss/maxrss.py`; /usr/bin/time на боксе нет):
| растр Or | байт | уровень обзора | окон | макс. RSS | время | лог |
|---|---|---|---|---|---|---|
| 008 Герите (крупнейший в трейне) | 5 478 863 788 | 3 (×16; src 0,0095 м) | 9 | **605 308 КБ** | 4,9 с | work/aeor_rss/gerite_or.log, scene_info.log |
| 033 Елизаветинское | 959 937 737 | нет пригодных (×272) | 4 | 749 392 КБ | 90,7 с | work/aeor_rss/or_033.log |
| 056 Фалхан | 461 291 996 | нет обзоров | 12 | 791 660 КБ | 89,5 с | work/aeor_rss/or_056.log |
Для сравнения, старый `sp_dfine.Scene` на Герите: GDAL «cannot allocate 718185316 bytes»; одно окно — 225,6 с и 1 604 312 КБ (`work/aeor_rss/plain_scene_gerite.log`).
**Риск по времени, не исправлен.** На растрах без обзоров (Елизаветинское, Фалхан) одно только чтение и warp на 2 потоках CPU занимают около 89 с при потолке фазы 120 с. На платформе потолок проверяет guard, и участок уходит на уровень one_scale или пропускается (без падения). Но если на тесте такие Or, выход по ним может быть неполным.

**3. check_submission (только при enabled).**
- Маркер AEOR_UNAVAILABLE даёт FAIL.
- Есть проба импорта в чистом процессе (та же, что у sp_dfine, она покрывает expds/dfine_aeor.py).
- Опорный прогон требует n_det > 0 на двух участках. Ae: `181_Гнёздово_Лесная_3`, растр `01_Гнёздово_Лесная_3_Ae_немецкая.tif` (398×1678, самый маленький Ae с разметкой Ae). Or: `185_Изборск`, `01_Изборск_Or_ортофото.tif` (7479×4830, самый маленький Or). Корень задаётся `--dfine-aeor-root`, по умолчанию /workspace/data/train-expds-3.
- Если весов нет, выдаётся FAIL «нужны веса …», а не PASS. Если веса есть, а участка нет — FAIL «… опорного участка» (**в v3 заменено на WARN «смоук не выполнен», см. ниже**). С `--no-weights` будет WARN.
- При выключенном флаге ничего нового не требуется (`test_check_submission_flag_off_requires_nothing_new`).

**4. Добавки 12:49.**
- Веса bot-9: `/workspace/koz3_shared/bot-9/models/dfine_ae` и `/workspace/koz3_shared/bot-9/models/dfine_or` (HF-каталоги config.json + model.safetensors fp32, 5 классов в порядке sp_dfine.CLASSES; на 12:52 МСК каталога bot-9/models ещё нет).
  - В примере конфига указаны `models/dfine_ae` и `models/dfine_or`, это симлинки на них: `ln -s /workspace/koz3_shared/bot-9/models/dfine_ae models/dfine_ae` и так же для dfine_or.
  - Абсолютный путь в default.toml не прошёл бы `test_no_personal_paths_in_packaged_code` и не работал бы в архиве, поэтому `build_submission` на абсолютных путях `[dfine_aeor].weights` останавливается с сообщением.
- Рецепт проверен тестом `test_recipe_autocast_only_no_half`:
  - на вход модели идёт float32 RGB/255 без нормализации (255→1,0, 102→0,4, 51→0,2);
  - второй вид — горизонтальное отражение того же батча;
  - nms_iou 0,5, flip_tta true;
  - `.half()` в модуле нет, fp16 только через `torch.autocast`, и только на CUDA.
- Загрузчик порогов принимает имена ae_dfine/or_dfine (`GROUP_ALIASES`, тест `test_thresholds_file_aliases_and_gor_off`).
- `[thresholds_by_markup.dfine_or] gorodishcha = 1.01`, городища D-FINE Or выключены, пока bot-10 не замерит их вместе с gor_dense bot-5. Ключ TOML главнее thresholds_file; тот же тест проверяет, что conf 0,999 отсекается.

**5. Проверки.**
- `tests/test_dfine_aeor.py`: 38 passed (`work/aeor_pytest_final2.log`).
- Вместе с test_bot7_v3_flags и test_bot8_check_submission: 82 passed (`work/aeor_pytest_final3.log`).
- Регресс из 15 файлов: ветка 139 passed до правок 12:49 (`work/aeor_regress_branch.log`), база 139 passed (`work/aeor_regress_base.log`). После правок — 138 passed и 1 failed (`work/aeor_regress2_branch.log`): это test_no_personal_paths на абсолютных путях в комментарии, исправлено, файл теперь проходит (final3.log).
- Побайтная идентичность на 245 объектах, sha 9222f5d1…9a83d9 одинаковый у базы 360c870, флага off и флага on без весов, маркер в логе (`work/aeor_identity/run4/identity.json`, run3 — до правок 12:49).

## v3 — правки по запросу bot-7 (29.10 13:12 МСК), коммит ca63a9d (родитель e0a6709, не запушен)
Diff: `bot-6/bot6-followups-1312.diff` (`git diff e0a6709..ca63a9d`, 21 073 байта, sha256 2a75a4de…8c221d).

**1. Срок фазы: участок пропускается, сдача не ждёт.**
- Что было и осталось (проверено тестом `test_phase_skips_region_via_soft_ok_and_never_waits`). Перед каждым участком `pipeline._dfine_aeor_phase` проверяет `guard.ok(60)`, `phase_left() > 0` и `soft_ok` (мягкий потолок). Если хоть одна проверка не проходит, фаза сразу выходит (`break`): ни ожидания, ни вызова region_feats. Внутри `region_feats` перед каждым растром идут `soft_ok` и `plan_level`, это лестница full → no_flip → one_scale → skip.
- **Найденная дыра.** `detect_scene` проверял срок только между батчами окон. Батч Or — 4 окна по 1024 px, и на Or без обзоров чтение+warp окна занимает около 22 с: 033 Елизаветинское — 89,2 с на 4 окна (`work/aeor_rss/or_033.log`, детектор-заглушка). Батч, начатый за 1 с до потолка, переехал бы его примерно на 90 с.
  - Кроме того, при первом растре (`tile_s` = None) план — «дешёвый проб» one_scale без оценки стоимости.
- **Исправлено** (`expds/dfine_aeor.py`):
  - перед каждым чтением окна проверяются `keep_going()` и `left() > max(чтение окна) + max(прямой проход батча)`, по замерам этой сцены;
  - `left` = min(срок фазы, `guard.window_left()` − `window_reserve_s`);
  - прерванный батч прогоняется, только если на прямой проход ещё есть время;
  - перед каждым следующим масштабом в `detect_image` стоит `keep_going()`.
  - Перерасход ограничен одним чтением окна, потому что GDAL-чтение не прерывается. Это худший случай до первого замера на сцене, на боксе около 22 с.
- **Тесты** (поддельные часы модуля, чтение окна двигает их):
  - `test_deadline_checked_before_each_window_not_only_batch`: окно 23 с, бюджет 30 с, батч 4 → прочитано 1 окно, часы ≤ 30 с; до правки было 4 чтения и 92 с.
  - `test_deadline_guard_window_left_also_bounds_reads`: срок сторожа ограничивает чтения (3 окна, 30 с).
  - Оба теста падают на старом коде (`git stash` dfine_aeor.py → 2 failed) и проходят на новом.

**2. GTX 1650 4 ГБ, а не 4090 — перекалибровать.**
- Запас `safety` ×8, `tile_s` (None, замер на первом растре) и пример в §5/§7 рассчитаны на 4090 (замер bot-9). На целевом ПК — GTX 1650 4 ГБ VRAM, и время инференса на ней **не измерено**.
- Перекалибровать нужно:
  - (а) `tile_s` на окно-вид для Ae (640 px, batch 8) и Or (1024 px, batch 4) на 1650;
  - (б) пиковую VRAM при `fp16` = true (только `torch.autocast`, веса fp32 — `test_recipe_autocast_only_no_half`). Веса в fp32 плюс активации D-FINE на 4×1024² могут не уложиться в 4 ГБ. Если OOM — снизить `batch.or` до 1–2 и `batch.ae` до 2–4: CUDA OOM уходит в `except` участка (пропуск, не падение), но участок теряется;
  - (в) `safety` по фактическому разбросу времени на 1650, а не ×8 «на глаз»;
  - (г) `budget_s` 120 с: чтение Or без обзоров (~89 с на боксе) от GPU не зависит и съедает почти весь потолок.
- До замера на 1650 включать `[dfine_aeor]` только как PACKAGE-ONLY с этим риском.

**3. check_submission: «смоук не выполнен» — явный WARN, не тихий PASS.**
- `[dfine_aeor]` enabled:
  - `--no-weights` → WARN «смоук не выполнен: --no-weights …»;
  - пустой список опорных участков → WARN «смоук не выполнен: … нет опорных участков»;
  - нет опорного растра в `--dfine-aeor-root` → WARN «смоук не выполнен: … веса на эталоне НЕ проверены». Раньше здесь был FAIL: это состояние машины проверки, а не дефект архива.
  - Нет весов в архиве — по-прежнему FAIL.
- `[orgate]` enabled: нет файла `--orgate-smoke` → WARN «смоук не выполнен: нет опорного входа смоука гейта …» (раньше FAIL). Нет модели, маркер ORGATE_UNAVAILABLE или неверные решения — по-прежнему FAIL.
- Тесты:
  - `test_check_submission_missing_train_data_warns_not_pass` (dfine_aeor: данных нет → WARN×2, пустой список участков → WARN);
  - `test_check_submission_missing_smoke_input_warns_not_pass` (orgate);
  - `test_check_submission_smoke_needs_weights` (WARN при --no-weights).
  - Модуль test_bot6_orgate теперь не пропускается целиком без опорного входа: пропускаются только тесты, которым вход нужен.

**4. Проверки.**
- `tests/test_dfine_aeor.py` + `tests/test_bot6_orgate.py`: 56 passed (42 + 14).
- Регресс из 17 файлов: **195 passed** (было 190, +5 новых; `work/orgate_regress3.log`, `work/orgate_regress.sh`).
- Python — `bot-7/venv`: bot-8 venv не восстановлен после рестора бокса около 13:11 МСК.
- Побайтная идентичность на 245 объектах (`work/orgate_identity/run2/identity.json`): база 360c870, флаг off, on без модели, on — sha b5f6b41c…5f8f1b у всех четырёх. От run1 (9222f5d1…, bot-8 venv) хэш отличается для всех, включая базу, то есть дело в окружении, а не в коде.

## v4 — веса dfine_or bot-9, чтение растров Ae/Or, исправление CRS (29.10 13:35–14:06 МСК)

**Веса** /workspace/koz3_shared/bot-9/models/dfine_or (sha256sum): config.json ff8bb6305c227c5e52ba7f7cc16bf0edaa7a9ac085104452b0aec3b6b9e90789,
model.safetensors f80cf9a30a52253901535b8191bd82bbbc6c1f356fb221bbe0d52126b47cfd39 (125 001 980 B). Препроцессор не нужен
(sp_dfine.load_models: config.json + safetensors; RGB/255 в коде). Вендор — колёса bot-9 (sha256sum -c 10/10 OK) → work/aeor_v4/vendor,
transformers 5.21.0, source=vendor. LOAD 3,3–3,6 с, RSS 618 МБ (work/aeor_v4/smoke_or*.log).

**Смоук 185_Изборск Or** (CPU, 2 потока, prlimit --as 2 ГБ; work/aeor_v4/smoke_or.log, smoke_or_b1.log): сцена 1015×655, 1 окно.
no_flip: n_det 94 (gorodishcha 63, fortifikatsii 24, arkhitektury 5, kurgany_tselye 2), max fort 0,4608 / gor 0,2346 / arch 0,0916;
2,8–3,2 с/окно (CPU 4,4–5,3 с), RSS ≈1,07–1,17 ГБ. full (flip, batch 1): n_det 169, 4,7 с/окно; MAXRSS процесса 1 196 576 КБ.
**batch or = 4 (по умолчанию) под --as 2 ГБ → OOM** (на 185 full и на 008 no_flip, work/crsfix/smoke008_after_b4_oom.log); batch 1 проходит.
Пороги: заглушка models/thresholds_stub_v4.json (work/aeor_v4/root) — все классы dfine_or = 1,01 (согласовано bot-9/bot-1 до нового
чекпойнта Or) → kept 0. check_submission с весами (без --no-weights, локальная копия конфига): ИТОГ PASS, FAIL 0, WARN 1
(«лишние .py в inference/» — файлы вендора, давний ложный WARN), 23 проверки, 162,6 с, MAXRSS 1 092 020 КБ (work/aeor_v4/check_v4b.log).
Дефект: check_submission падал TypeError (вердикт смоука — кортеж из 2, report ждёт 3) → исправлено с тестом: 770abda (bot6-orgate),
перенесено на bot6-crsfix как c67c385, bot6-v4-checksub.diff.

**Чтение растров Ae/Or** (work/aeor_v4/raster_probe.log, raster_headers.log): подсетов нет ни у одного из 40 tif Ae/Or; полосы 1..min(3,count)
(альфа отбрасывается); обзор GDAL берётся, если шаг×фактор ≤ 0,5×res_m; сетка окон закрывает растр (последнее окно прижато к краю).
008 Or: 108386×98570, RGBA, EPSG:4326, обзоры 2..511 → уровень 3 (6775×6161) → сцена 2070×1883 @0,5 м, 9 окон. Сам ридер исправен; 1×1
был от UTM.json (ниже). Замечания: 185 Ae 0,98 м → 9232×8773, 342 окна (цена); 033 Or — единственный обзор [272], непригоден → чтение
полного разрешения.

**CRS-фикс** (находка bot-9): 008_Герите/UTM.json = urn:ogc:def:crs:EPSG::4326 (единственный географический из 210). Ветка bot6-crsfix
от 885758e (bot7-candidate-v4): f86a31a — sp_dfine.utm_for: географическую fallback не брать, зона UTM по центру растра; gor_gate._epsg —
так же через utm_for по первому SpOr; тест tests/test_bot6_crsfix.py (2 теста, без фикса оба падают). bot6-crsfix.diff
(sha256 d064d306…050f5e). До/после на 008 (pipeline-путь read_utm_json → utm_for → Scene, work/crsfix/before.log, after.log):
EPSG:4326, 1×1, шаг 0,5°, 1 окно, n_det 0 → EPSG:32638, 2070×1883, 0,5 м, 9 окон, n_det 397 (arch 365, gor 17, fort 15; max arch 0,9129),
5,9 с/окно, RSS 1159 МБ, batch 1 (smoke008_before.log / smoke008_after.log). Сверка по всем 210 участкам (work/crsfix/sweep.log):
215 растров ae/or/sp_or + gor_gate._epsg — отличие ровно одно: 008 Or 4326 → 32638. Тесты: 142 passed (10 файлов, work/crsfix/pytest.log),
после cherry-pick 69 passed (test_dfine_aeor, test_bot6_crsfix, test_bot8_check_submission). Флаги sp_dfine/dfine_aeor/gor_gate в
configs/default.toml выкл. → сдача по умолчанию эти функции не вызывает.
Остальные потребители region_crs/read_utm_json: orgate._lat_lon — NaN для 4326 намеренно (как feats_gate.py обучения, не трогать);
fusion.copy_layers, _raster_with_model, map_footprint (рамка кадра, экспорт в 3857) — fallback только если CRS растра непригодна
(у 008 растр сам 4326 → берётся своя) — безопасно; лидар (build_dem/DemPrefetch/hillfort, crs_region 458/886/1286), геофизика
(305, dedupe_points 2,0 «м» 342/795, geo_pixgbm) — у 008 нет LAS и геофизики → не достигается, но при географическом UTM.json с LAS было
бы неверно (латентно, не правил); health, candidates/tiles/gt — отчёт и обучение.
