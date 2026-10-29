# Ревью 54cfab6 (bot8-cnx): check_submission.py под [li_cnx]
Бот 4, 29.10.2026, ~07:50 МСК. Ревьюировался коммит 54cfab6 (`scripts/check_submission.py` +150/−4, `tests/test_bot8_check_cnx.py`, 8 тестов) против родителя 4fb79c6. Дополнительно — слияние с 7878087 и корневой `bot-8/check_submission.py`.

## Вердикт (54cfab6): **OK с замечаниями** → после повторного ревью 5bc6ad1: **OK, замечания 1–6 закрыты** (см. последний раздел)
1. **Существующие проверки не ослаблены.** Формат, CRS, геометрия, классы, лимит 20000, размер, время, forbidden, metadata, PIXGBM и основной сетевой сканер не тронуты. `validate_output`, `check_forbidden`, `check_size`, `check_metadata`, `check_so`, `scan_imports` и `sandbox_cmd` в диффе отсутствуют. Правка только добавляет проверки и снимает один ложный FAIL: ссылки `li_cnx.*` при выключенном cnx.
2. **Новые проверки cnx мягче, чем могли бы быть** (это не регресс, а неполнота): (а) сетевой анализ подготовки cnx видит только статические `import`, а `NET_CALLS`, `importlib`, `__import__`, `subprocess` и недостижимые модули пропускает; (б) MANIFEST.json, пустой или без обязательного файла, даёт PASS+WARN вместо FAIL; (в) вендорный `expds_core.abi3.so` не проверяется на ELF, GLIBC и AVX-512; (г) WARN в смоуке не ловит «cnx не сработал ни на одном LAS».
3. Сейчас `[li_cnx] enabled=false`, блокера нет. **Перед включением cnx** надо внести п. 1–4 рекомендаций. Кроме того, ключи смоука уже рассинхронизированы с незакоммиченной работой бота 8 в wt-cnx: см. «Внимание» ниже.

## Проверка заявлений бота 8
| Заявление | Итог | Доказательство |
|---|---|---|
| inference/cnx/*.py не считаются «лишними» | ✔, но фильтр **широкий**: любая глубина, любое имя, независимо от enabled | `check_layout`: `not n.startswith("inference/cnx/")`. Сигнал не теряется: при выкл. любой файл там → WARN cnx (L1), при вкл. файл вне MANIFEST → WARN. Единственная слепая зона — файл, **внесённый в MANIFEST** и недостижимый из cnx_prep: тишина (L2). Раньше там тоже было лишь WARN «лишние .py» (15 штук шума на каждый архив с cnx) |
| li_cnx.* не «missing» при выключенном | ✔, подделать «выключен» нельзя | Пайплайн читает только `configs/default.toml` (inference/main.py:87), `lidar_cnx.enabled()` = `bool(enabled) and bool(weights)`, чекер — `lc.get("enabled")`, истинность та же. `enabled="false"` (строка): пайплайн считает cnx **включённым**, и чекер тоже, FAIL (C2). `enabled=0`: оба выключены (C3). `enabled=true, weights=[]`: пайплайн выключен, чекер даёт FAIL, то есть он строже. Других источников конфига (env, overlay) у пайплайна нет |
| check_cnx: выкл.+файлы → WARN; вкл. → FAIL на пустых weights, отсутствующем файле или whl, sha≠MANIFEST, сетевом импорте | ✔ по всем пунктам (тесты бота 8 + M3, N1, N2) | Слабые места: см. N3–N7, M1, M2 |
| Проба: импорт vendor-prep в подпроцессе; с весами — lidar_cnx.load(cpu) | ✔ | Подпроцесс наследует песочницу пробы (`unshare -rn`, если доступен) и `child_env` (PYTHONDONTWRITEBYTECODE=1, без EXPDS_*/CUDA). cwd — временный каталог, который удаляется. timeout=300 с при внешнем `--timeout` 900. Проверяется, что expds, relief_ext и expds_core взяты из vendor, а rvt.vis — из `_deps`. Побочных эффектов и сети нет. `lidar_cnx.load` выполняет `check_prep_code`, то есть sha вендора сверяется с контрактом весов, **но только при `--weights`** |
| Смоук: WARN на «cnx не сработал» и «cnx пропущен» | ✔ (S1) | Раньше эти строки не ловились вообще, то есть это усиление. «веса [li_cnx] не загрузились» ключом не ловится: «не загрузились» ≠ «не загрузилась». Ловится через `Traceback` от `log.exception` |
| Корневой bot-8/check_submission.py = sha d97042f9…, PIXGBM нетронут | ✔ sha совпадает. Это **не** байтовая копия ветки: ветка fb94431c…, слияние a492237c… | Построчно: (корень − 7878087) = (54cfab6 − 4fb79c6) + отсутствие блоков `rerank_arch` бота 7. Их в корне не было и раньше (bot-8/STATUS.md 07:30), к этому коммиту это не относится. Порядок секций PIXGBM/cnx в корне другой. Тесты на корневом файле: 31 passed |
| Слияние с 7878087: 1 тривиальный конфликт, 31 passed | ✔ | Конфликт один: обе секции вставлены перед «# ---- импорты (статически)», оставлены обе. 8 (cnx) + 23 (check_submission) = **31 passed**. Фичевые test_bot8_cnx.py — 19 passed |

## Подробно по вопросам
- **Ослабление существующих проверок — нет.** Дифф затрагивает только `check_layout` (фильтр WARN), `check_config_static` (одно условие missing), новый `check_cnx`, PROBE/`run_probe` (только добавления), ключи `smoke` (добавления) и `main` (распаковка inference/cnx, `third | cnx_third` — только расширение). Сторонние пакеты подготовки (laspy, lightgbm, numpy, pyproj, rasterio, scipy, shapely) теперь проверяются пробой и `--lock`. Это усиление.
- **WARN при выключенном cnx с файлами — приемлемо.** Пайплайн не вызывает `run_prep` при `enabled()==False`, код не исполняется. Цена — объём архива, а `check_size` работает как раньше.
- **Сетевая детекция — только статическая и уже основного сканера.** Ловит `import requests` и `from urllib import request` (N1, N2). Пропускает PASS-ом:
  - `importlib.import_module('soc'+'ket')` (N3);
  - `__import__('socket')` (N4);
  - `subprocess.run(['curl',…])` (N5);
  - `torch.hub.load(...)` (N6) — основной `scan_imports` ловит `NET_CALLS`/`torch.hub`, `check_cnx` — нет;
  - модуль с `import socket`, загружаемый динамически и недостижимый в графе (N7).
  
  Недостижимые модули с сетью вообще не дают WARN, а основной сканер в таком случае выдаёт WARN. Динамические импорты и subprocess не ловит и основной сканер, так что это не регресс. На платформе реальная защита — сетевая изоляция, а в пробе — `unshare -rn`.
- **sha256 vs MANIFEST.** Покрыты все файлы inference/cnx, кроме самого MANIFEST: 18 из 19 (`tests/test_bot8_cnx.py:306`). **Веса .pt не покрыты:** их наличие ловит `check_config_static` (M4, FAIL), загрузку и связь «веса↔вендор» — только проба с `--weights`, а бот 8 гоняет архивы с `--no-weights` (STATUS 07:30). MANIFEST самосогласован: его пишет та же сборка, поэтому он защищает от порчи и рассинхрона, а не от подмены. Дыры: пустой `{"sha256": {}}` даёт PASS «0 файлов совпадают» + WARN (M1). Изменённый relief_ext.py, вычеркнутый из MANIFEST, даёт только WARN (M2). Поле `prep_code_sha_expected` не проверяется.
- **Смоук: WARN при «cnx failed/skipped» опасен?** Откат идёт на модель li со своими порогами, то есть балл не ниже базового. Это потеря ожидаемого прироста, а не поломка. Но если cnx включён и не сработал **ни на одном** LAS (сломан вендор или .so, нет rvt, SIGILL), смоук покажет лишь WARN, и архив уйдёт с тихо выключенной модальностью. Это ровно тот риск R3, ради которого существуют проверки моделей.

## Внимание: незакоммиченная работа в wt-cnx
`git status` в /workspace/koz3_shared/bot-8/wt-cnx показывает изменённые `expds/lidar_cnx.py` (+215/−…, 07:46 МСК) и `expds/pipeline.py` (07:44 МСК): cnx переделывается в «добавочный» `cnx_phase`. В этой версии строк «cnx не сработал» и «cnx пропущен» в pipeline.py **больше нет**, так что ключи смоука из 54cfab6 ничего не поймают. Ревью относится к закоммиченному 54cfab6. После новой версии пайплайна ключи смоука надо пересверить.

## Тесты (воспроизводимо)
- Каталог `work/review_cnx/t/`. Там лежат: `cs_old.py` (4fb79c6), `cs_new.py` (54cfab6), `cs_merged.py` (54cfab6 + 7878087, обе секции), `cs_root.py` (корневой d97042f9…), `merged_vs_7878087.diff`, `merge_log.txt`.
- Логи pytest (python из bot-8/venv, на клоне с `--shared`, без models/; клон удалён):
  - `t_cnx.log` — 8 passed;
  - `t_merge.log` — 31 passed после слияния;
  - `t_root.log` — 31 passed на корневом файле;
  - `t_cnxfeat.log` — test_bot8_cnx 19 passed.
- Крафтовые случаи: `python test_cnx_review.py` (bot-8/venv), лог `test_cnx_review.log`. Исходники берутся из bot-8/wt-cnx только на чтение.

| Случай | Старая | Новая | Нужно |
|---|---|---|---|
| L1 выкл., evil.py (import socket) в inference/cnx | WARN layout | WARN cnx | WARN ✓ |
| L2 вкл., evil.py в MANIFEST, недостижим | — | **PASS** | WARN |
| N1 `import requests` / N2 `from urllib import request` в cnx_prep | — | FAIL | FAIL ✓ |
| N3 importlib / N4 `__import__` / N5 subprocess curl | — | **PASS** | хотя бы WARN |
| N6 `torch.hub.load` | — | **PASS** | FAIL (как в основном сканере) |
| N7 динамическая загрузка модуля с socket | — | **PASS** | WARN |
| M1 пустой MANIFEST | — | **PASS+WARN** | FAIL |
| M2 relief_ext.py изменён и вычеркнут из MANIFEST | — | **WARN** | FAIL |
| M3 байт изменён, MANIFEST старый | — | FAIL | FAIL ✓ |
| M4 вкл., весов .pt нет в zip | FAIL | FAIL | FAIL ✓ |
| C1 выкл., weights указаны, файлов нет | FAIL (ложный) | PASS | PASS ✓ (во всех трёх новых версиях) |
| C2 `enabled="false"` (строка) | — | FAIL (оба считают включённым) | FAIL ✓ |
| C3 `enabled=0` | — | PASS (пайплайн тоже выкл.) | PASS ✓ |
| S1 ключи смоука | — | «не сработал» и «пропущен» ловятся, «не загрузились» нет (только через Traceback) | — |

## Рекомендации (до enabled=true)
1. **Сеть в check_cnx — как в основном сканере.** Вынести тело `scan_imports` в общую функцию: `NET_CALLS`, `torch.hub`, `pretrained=True`. Для недостижимых модулей inference/cnx — WARN, как делает основной сканер. Плюс WARN на `importlib.import_module`, `__import__`, `subprocess`, `os.system` в достижимом коде: в cnx_prep их сейчас нет, ложных срабатываний не будет.
2. **MANIFEST строже.** FAIL, если `man` пуст или любой файл из `CNX_REQUIRED` либо любой `.py`/`.so` под inference/cnx в нём отсутствует. WARN «вне MANIFEST» оставить только для README/LICENSE. Сверять `prep_code_sha_expected` с пересчитанным хэшем, по той же формуле, что в сборке.
3. **check_so на вендорный .so.** Прогнать ту же проверку (ELF x86-64, GLIBC, `%zmm`) по `inference/cnx/vendor/expds_core*.so`. Текущий файл чистый: GLIBC_2.14, zmm 0 (`objdump -d`). Но при SIGILL cnx молча откатится на li.
4. **Смоук при enabled.** FAIL, если в логе нет ни одной строки успеха `cnx_prep <las>: ch=` при ≥1 LAS, или есть «веса [li_cnx] не загрузились». WARN оставить на частичные отказы. Привязать ключи к реальным строкам тестом (grep по `expds/*.py`), чтобы рефакторинг (уже идёт в wt-cnx) их не рассинхронизировал.
5. **Узкий фильтр layout:** исключать из «лишних .py» только файлы, перечисленные в MANIFEST, или только при enabled.
6. **Веса cnx:** при enabled запускать загрузку `lidar_cnx.load(cpu)` пробой даже при `--no-weights`, она дешёвая для 1–2 файлов. Либо записывать sha256 весов в MANIFEST при сборке и сверять.
7. Мелочь, не про этот коммит: корневой bot-8/check_submission.py по-прежнему без проверок `rerank_arch` (в repo `[rerank_arch] enabled=true`, `configs/default.toml:351`). Явные `model`/`meta` ловятся общим поиском путей, но пробы загрузки `rerank_arch` в корне нет. Лучше перейти на файл из ветки.

---

# Повторное ревью 5bc6ad1 (bot-4, 29.10.2026, МСК)

**Итоговый вердикт: OK — замечания 1–6 закрыты, ни одна проверка не ослаблена. Блокеров для enabled=true со стороны чекера нет.** Остались мелкие обходы статического сканера (ниже, п. «Остаточные замечания»), это не блокеры: реальную изоляцию даёт `unshare -rn` в смоуке и офлайн-платформа. Замечание 7 (rerank_arch в корне) осталось как было, оно не из этого коммита.

## Что проверено (независимо, в скретч-клоне @5bc6ad1, без models/)
- Ветка bot8-cnx = 5bc6ad1, wt-cnx чистый (`git status` пуст). 54cfab6..5bc6ad1: 2f08af2 (слияние 17c2e2c), 9ff14e3, ee1d788, **692b503 (правки чекера по ревью)**, 5bc6ad1 (docs).
- `scripts/check_submission.py` @2f08af2 = мой `cs_merged.py` (54cfab6+7878087) с точностью до 2 пустых строк. Значит, слияние с PIXGBM корректно, и правки по ревью = чистый дифф 2f08af2→5bc6ad1 (`t/diff_2f08af2_5bc6ad1.diff`, 468 строк).
- AST-сравнение 2f08af2 → 5bc6ad1: изменены только `check_cnx`, `check_layout`, `check_so`, `scan_imports`, `run_probe`, `PROBE`, `smoke`, `main`; добавлены cnx-функции и константы, ничего не удалено.
  - `scan_imports`: тело сетевых вызовов вынесено в `net_calls()` дословно, поведение то же.
  - `validate_output`, `check_forbidden`, `check_size`, `check_metadata`, `check_config_static`, `check_pixgbm_static`, `OPTIONAL_LAZY`/`guarded_import_sites`: **не тронуты** (регрессий нет).
- pytest (venv bot-8, nice 10, 2 потока): `test_bot8_check_cnx.py` (46 собрано) + `test_bot8_check_submission.py` + `test_bot8_cnx.py` = **100 passed**.
- Единственный падающий тест полного прогона bot-8 (`456 passed, 1 failed, 9 skipped`) — `test_coco_roundtrip::test_grid_cache_roundtrip_gives_one_circle_inside_each_object`: `ModuleNotFoundError: cv2` в `scripts/coco_from_cache.py:96`. Файл теста не менялся в 54cfab6..5bc6ad1 (последний коммит e4c24a4), cv2 в venv нет → **не связан с cnx, падал и раньше**. Подтвердил прогоном этого файла: 1 failed, 2 passed.

## Мой харнесс против 5bc6ad1 (`t/test_cnx_rereview.py`, лог `t/test_cnx_rereview.log`)
Версии: `cs_fix` = scripts/check_submission.py@5bc6ad1, `cs_rootfix` = корневой bot-8/check_submission.py. Обе дают **одинаковые** результаты во всех случаях. Конфиг ON — с реальными весами из WEIGHTS_MANIFEST и группой порогов li_cnx. Базовый случай B0 (чистый inference/cnx, вкл.) = PASS.

| Случай | 54cfab6 | 5bc6ad1 | Оценка |
|---|---|---|---|
| L1 zip только из inference/cnx + evil.py, выкл. | FAIL «нет обязательных файлов» | тот же FAIL | **артефакт харнесса**: в zip нет metadata.json/main.py/…. С этими файлами (L1b) FAIL исчезает, остаются WARN layout «лишние .py: inference/cnx/evil.py» (узкий whitelist, п.5 ✓) и WARN cnx «выключен, лишний объём» |
| L2 недостижимый evil.py с socket, в MANIFEST | PASS | WARN «недостижим из cnx_prep» | ✓ как основной сканер |
| N1 `import requests` / N2 `from urllib import request` | PASS | FAIL | ✓ |
| N3 `import_module('soc'+'ket')` | PASS | WARN (цель неизвестна) | ✓ разумно, см. ниже |
| N4 `__import__('socket')` / N4b `import_module('urllib.request')` | PASS | FAIL (константа идёт в граф) | ✓ строже, чем я просил |
| N5 `subprocess.run(['curl',…])` | PASS | WARN | ✓ разумно |
| N6 `torch.hub.load` | PASS | FAIL | ✓ |
| N7 динамическая загрузка недостижимого модуля с сетью | PASS | WARN (+FAIL prep_code_sha, потому что я положил helper в vendor/) | ✓ |
| M1 пустой MANIFEST | PASS+WARN | FAIL | ✓ |
| M2 relief_ext изменён и вычеркнут из MANIFEST | WARN | FAIL (обязательный вне MANIFEST + prep_code_sha) | ✓ |
| M3 байт изменён, MANIFEST старый | FAIL | FAIL | ✓ |
| M3b вендор изменён, MANIFEST пересобран самосогласованно | — | FAIL prep_code_sha ≠ expected | ✓ новое |
| M3c нет `prep_code_sha_expected` | — | FAIL | ✓ |
| M5 лишний .so вне MANIFEST | WARN | FAIL | ✓ |
| M6 README2.md вне MANIFEST | WARN | WARN (текст) | ✓ |
| M4 веса из default.toml не в zip | FAIL | FAIL | ✓ |
| W1 веса с чужим sha/size | — | FAIL | ✓ (п.6) |
| W2 путь весов не в WEIGHTS_MANIFEST / W3 нет WEIGHTS_MANIFEST | — | FAIL / FAIL | ✓ |
| T1 нет `[thresholds_by_markup.li_cnx]` / T2 group=`li` / T3 tta=4 | — | FAIL / FAIL / FAIL | ✓ новое |
| C1 выкл., веса указаны, файлов нет | PASS | PASS | ✓ |
| C2 `enabled="false"` | FAIL | FAIL (оба считают включённым, как пайплайн: `bool("false")=True`) | ✓ согласовано |
| C3 `enabled=0` | PASS | PASS (пайплайн: выкл.) | ✓ |
| S1 `smoke_cnx`: успех / нет строки `cnx_prep …: ch=` / «веса … не загрузились» / «объекты cnx НЕ выдаются» / выкл. / 0 LAS | — | PASS / FAIL / FAIL / FAIL / PASS / PASS | ✓ (п.4) |

Заявления bot-8 по харнессу подтверждаются полностью (S1 у него FAIL — это случаи «нет строки/весов/порогов», как и у меня).

## По замечаниям 1–6
1. ✓ Общий `net_calls()` (NET_CALLS, torch.hub, pretrained, weights="imagenet"); WARN на динамический импорт и запуск процессов; WARN для недостижимого с сетью. Динамический импорт с константой проходит в граф и в FAIL.
2. ✓ Пустой MANIFEST, обязательные файлы вне него, код/данные вне него → FAIL. Вне MANIFEST допустимы только README/LICENSE/NOTICE/COPYING (WARN). `prep_code_sha_expected` сверяется по формуле контракта весов.
3. ✓ `check_so` идёт и по `inference/cnx/vendor/expds_core*.so`. В логах смоука bot-8: GLIBC_2.14, AVX-512 нет.
4. ✓ `smoke_cnx`: FAIL без строки успеха при ≥1 LAS и FAIL на фатальные строки; частичные отказы → WARN через общий список. Тест `test_smoke_keys_match_pipeline_logs` сверяет ключи с `expds/*.py`, и каждая warning/exception-строка про cnx в pipeline.py ловится каким-то ключом. Строки в коде на месте: `lidar_cnx.py:384` `cnx_prep %s: ch=`, `pipeline.py:791/798/806/810/826/845/860/1287/1294`.
5. ✓ Layout исключает только .py, перечисленные в MANIFEST.
6. ✓ И то и другое: проба грузит веса cnx и при `--no-weights` (только если enabled; извлекаются только cnx-веса). Плюс `WEIGHTS_MANIFEST.json` (sha256+size, never_pack для smoke-весов).

## WARN, а не FAIL, для N3/N5/N7 — разумно
Цель `import_module(<выражение>)` и действие подпроцесса статически не определить. FAIL на любой subprocess/importlib дал бы ложные срабатывания на легитимный код (у самого пайплайна cnx_prep — подпроцесс). В исходном ревью я просил «хотя бы WARN», это выполнено. Константные цели (N4/N4b) — FAIL, это строже запрошенного. Реальная защита — `unshare -rn` в смоуке и офлайн-платформа.

## Остаточные замечания (не блокеры, на будущее)
- a. **Обходы статического сканера**, все PASS в обеих версиях:
  - `from subprocess import run; run([...])` и `import subprocess as sp; sp.run(...)`: `dynamic_calls` смотрит только на `owner` по буквальному имени `subprocess`/`os`;
  - `exec('import soc'+'ket')`;
  - `getattr(importlib, 'import_'+'module')('socket')`;
  - `ctypes`;
  - `torch.utils.model_zoo.load_url(...)`: `load_url` нет в NET_CALLS, у основного сканера та же дыра.

  Дёшево закрыть первые два случая (учесть `ImportFrom` из subprocess/os и алиасы `import … as`), а также добавить WARN на `exec`/`eval`/`compile`, `ctypes` и `load_url` в NET_CALLS. Это общий предел статики, не регрессия.
- b. `CNX_SOFT_NAMES` проверяет только префикс имени: `README_evil.py` вне MANIFEST получает лишь WARN «тексты вне MANIFEST». Сетевой сканер этот файл всё равно видит (WARN недостижимого или FAIL, если он достижим). Лучше считать мягкими только не-.py/.so (например, `.md`/`.txt`/без расширения).
- c. `enabled="false"` (строка) и чекер, и пайплайн считают включённым — согласовано, но опасно для человека. Стоит отдельный WARN/FAIL на тип `li_cnx.enabled`, отличный от bool.
- d. Заявление «смоук off байт-в-байт 56be33ff» я не перепроверял (диск 2,4 ГБ, смоук не запускал). Оно есть в STATUS bot-8; check_off_5bc6ad1.log = PASS, 0 FAIL. Логи c3 tta8/tta1 @5bc6ad1 = PASS, «cnx: подготовка прошла на 1 LAS», вендорный .so проверен.
- e. (п.7, старое) Корневой `bot-8/check_submission.py` (sha 8c04bc99…, обновлён 07:57 МСК) синхронизирован со всеми cnx-правками 692b503 (AST: отличается от ветки только `check_config_static` и `PROBE`, и только отсутствием `rerank_arch`). Это прежняя проблема, не из этого коммита.
