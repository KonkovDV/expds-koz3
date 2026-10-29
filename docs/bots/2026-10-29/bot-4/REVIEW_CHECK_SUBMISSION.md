# Ревью правки check_submission.py (бот 8, под PIXGBM / архив koz3_softceil_pixgbm_428257f_20b4ccea.zip)
бот 4, 29.10.2026 ~07:35 МСК

## Вывод (3 строки)
1. **Для архива 428257f — OK.** Проверки формата, CRS, геометрии, имён классов, лимита 20000, раскладки файлов, размера и времени не менялись. Диффа в `validate_output`, `check_layout`, `check_forbidden`, `check_size`, `check_metadata`, `smoke` нет.
2. Правка снимает три ложных FAIL (PIXGBM выключен, `maps` задан списком, ленивые psutil/tl2cgen). Но в двух местах **проверка ослаблена**, и это сработает, **если PIXGBM включить**: (а) папка моделей проходит проверку при любом файле внутри, наличие poly.txt и finds.txt не проверяется; (б) любой из трёх пакетов, импортированный внутри функции, даёт WARN вместо FAIL, даже если импорт без фолбэка.
3. Рекомендую: до включения PIXGBM (enabled=true) поправить (а) и (б) по пунктам ниже. Сейчас блокера нет.

## Что именно сравнивалось
- Эталон: `/workspace/expds-koz3/analysis/2026-10-28/bots/bot-8/check_submission.py`, коммит f932945, 42831 Б, md5 82f56b6b…. Те же байты лежат в koz3_stage, gh_sync_wt и /tmp/lid.
- Новая версия: `/workspace/koz3_shared/bot-8/check_submission.py` (02:01 МСК, 45383 Б, md5 edcd6146…).
- **Внутри zip check_submission.py нет.** Архив содержит только configs/, expds/, inference/, models/ и metadata.json (`unzip -l`). Это инструмент бота 8, а не часть решения. По bot-8/STATUS.md (02:10) архивом занимается именно эта правленая версия.
- Дифф: `diff -u <эталон> <новая>`. Изменились три участка, всё остальное совпадает побайтно.

## Изменённые проверки
| # | Где | Было | Стало | Оценка |
|---|---|---|---|---|
| 1а | `check_config_static`: пути весов из default.toml | ключ `geophys.pixgbm.model_dir="models/pixgbm/all"` всегда отсутствует в списке файлов zip → FAIL | `geophys.pixgbm.*` при enabled≠true пропускаются | **Корректно.** pipeline.py:352 использует `pix.get("enabled", False)`, умолчания совпадают |
| 1б | то же | — | при enabled=true model_dir засчитан, если в zip есть **любой** файл `models/pixgbm/all/…` | **Ослабление.** `geo_pixgbm.load_models` молча пропускает недостающие poly.txt/finds.txt, а при обоих отсутствующих только пишет в лог «классика». Это ровно тот риск R3 (модальность молча выключена), ради которого проверка и существует |
| 2 | PROBE: `geophys.*.maps` из MAP_MODES | у pixgbm `maps` — список → `list not in MAP_MODES` → ложный FAIL | для pixgbm элементы списка проверяются по {maggr, etgr, terfl}, для остальных секций — как раньше | **Корректно.** Совпадает с geo_pixgbm.py:683,694–696. Строка вместо списка тоже ловится (посимвольно). Мелочь: если `maps` — число, PROBE упадёт с TypeError, но это попадёт в общий except пробы |
| 3 | `run_probe`: пакеты, не установленные в образе | FAIL | psutil, tl2cgen, lleaves → WARN, если на уровне модуля их никто не импортирует | **Частичное ослабление.** Для текущего кода верно: psutil (geo_pixgbm.py:642) в try/except с фолбэком, tl2cgen (:172) вызывается под try в load_models:158, `import lleaves` в коде нет вовсе. Но правило **не проверяет наличие фолбэка**: будущий `import psutil` без try внутри функции тоже станет WARN, а в рантайме упадёт |

## Доказательства (тесты)
Скрипт `work/review/cs_test/test_cs.py`, лог `work/review/cs_test/test_cs.log`. Обе версии грузятся через importlib, без правок. Вход — синтетические zip в памяти с default.toml из архива 428257f; все остальные веса считаются присутствующими.

| Случай | Старая | Новая | Ожидается |
|---|---|---|---|
| T1 PIXGBM выкл., папки нет (как в 428257f) | FAIL | PASS | PASS → починен ложный FAIL |
| T2 вкл., папки нет | FAIL | FAIL | FAIL ✓ |
| T3 вкл., в папке только README.md | FAIL | **PASS** | FAIL → **ослабление** |
| T4 вкл., есть poly.txt, нет finds.txt | FAIL | **PASS** | FAIL → **ослабление** (finds_points от pix молча не будет) |
| T5 вкл., poly.txt + finds.txt | FAIL | PASS | PASS → починен ложный FAIL |
| T6 вкл., похожая папка all_old/ | FAIL | FAIL | FAIL ✓ (граница префикса «/» соблюдена) |
| T7 контроль: порог несуществующего класса | FAIL | FAIL | FAIL ✓ (проверка имён классов не изменилась) |
| T8 `_module_level_imports` на модуле с `def f(): import psutil` без try | — | psutil не считается модульным → уйдёт в WARN | FAIL → **ослабление (потенциальное)** |

Проверка: `cd /workspace/koz3_shared/bot-4/work/review/cs_test && python3 test_cs.py`

## Рекомендации
1. Пункт 1б заменить явной проверкой при enabled=true: для каждого класса в `geophys.pixgbm.classes` нужен свой файл (`finds_points` → `<model_dir>/finds.txt`, `object_poly` → `<model_dir>/poly.txt`). Отсутствие → FAIL. Для backend=lleaves/treelite отсутствие `.so` → WARN, фолбэк на lightgbm там есть.
2. Пункт 3: понижать до WARN, только если каждый такой импорт по AST лежит внутри `try`, у которого есть обработчик `ImportError`, `ModuleNotFoundError` или `Exception`. Иначе FAIL.
3. Для текущего архива 428257f (PIXGBM выкл.) повторно гонять проверку не нужно: на нём правка влияет только на ложные FAIL. Если включать PIXGBM, сначала внести п. 1 и 2.
