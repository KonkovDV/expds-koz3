# PIXGBM S — follow-up к merge-тесту bot-8: потоки, зависимости, упаковка (bot-3, 29.10.2026 02:30 МСК)

**Вывод (3 строки).**
1. Коммит **76f7317** в bot3-pixgbm (поверх 5deb94a, без push). Потоки и пул признаков теперь считаются по `os.sched_getaffinity(0)`: если его нет — `os.cpu_count()`, потолок 8, env важнее. Относительный `model_dir` ищется и от корня решения. bot3-geosnap перебазирован: **7a5ee57**.
2. В образе (`requirements.lock` / `docker/Dockerfile`) нет **psutil, lleaves, llvmlite, tl2cgen, treelite**. Для дефолта (lightgbm 4.7.0 есть в lock) и для lleaves ничего ставить не нужно. tl2cgen и treelite нужны только для `backend="treelite"`. psutil опционален: `process_t0()` без него падает на момент импорта.
3. Для S в архив идут 2 файла (`models/pixgbm/all/poly.txt`, `finds.txt`): 4 233 917 Б, в zip +1 919 839 Б (≈1,83 МиБ). Нужен патч упаковщика, см. `packer_pixgbm.patch`. Opt-in конфиг: `pixgbm/config_v2_S.toml`. `default.toml` не трогал.

## (1) Потоки — `expds/geo_pixgbm.py` (коммит 76f7317)
- `avail_cores()` = `len(os.sched_getaffinity(0))`. При `AttributeError`/`OSError` берётся `os.cpu_count()`, результат не меньше 1.
- `pix_threads()`: `EXPDS_PIXGBM_THREADS` → `EXPDS_THREADS` → `min(8, avail_cores())`.
- `pix_procs(pix, nt)`: `[geophys.pixgbm] procs > 0` — как задано, иначе `min(nt, 4, avail_cores())`. Раньше было `min(nt, 4)`, и при `EXPDS_PIXGBM_THREADS=8` на 2 ядрах пул шёл в 4 процесса; теперь в 2.
- `_rate()` (прогноз с/Мпикс для лестницы бюджета) делит априорную скорость на фактическое число процессов, а не на `min(nt, 4)`. При полном affinity результат тот же.
- Если `model_dir` относительный и его нет от cwd, он ищется от `<корень решения>` = `Path(expds/__file__).parents[1]`, так же как веса НС (`pipeline._segmenters: root / f`). Раньше `load_models` брал `Path(model_dir)` от cwd. `inference/main.py` не делает chdir, поэтому при запуске не из корня архива pix молча уходил бы в классику (`log.warning … нет poly.txt/finds.txt`).
- Тест `tests/test_geo_pixgbm_threads.py` (6 тестов):
  - `sched_getaffinity` → {0,1} при `cpu_count`=64 даёт threads = 2, procs = 2;
  - 32 ядра → 8 / 4;
  - без `sched_getaffinity` → `cpu_count`;
  - `OSError` и `cpu_count=None` → 1;
  - env-приоритет;
  - `_rate` по procs;
  - `process_t0` без psutil == `_T_IMPORT`;
  - `resolve_model_dir` при chdir.
- Прогоны:
  - На bot3-pixgbm, полный `tests/` (venv bot-8 только на чтение, в нём есть torch/lightgbm): `1 failed, 321 passed, 9 skipped` (`pixgbm/pytest_threads.log`). Упал только `test_coco_roundtrip::test_grid_cache_roundtrip_gives_one_circle_inside_each_object` — тот же coco, что у bot-8 в базе.
  - На bot3-geosnap: peaks + threads — 8 passed.
- Фактически на box: `avail 8, cpu_count 8 → threads 8, procs 4`.
- `expds/health.py:47` по-прежнему пишет в отчёт `os.cpu_count()`; это только диагностика, не трогал.

## (2) Зависимости
Проверены `requirements.txt`, `requirements.lock`, `requirements-dev.txt`, `docker/Dockerfile` (ставит `requirements.lock`) и `pyproject.toml` (зависимостей там нет).

| пакет | в образе (lock/txt/Dockerfile) | где есть | нужен в рантайме? |
|---|---|---|---|
| lightgbm | **есть**: `lightgbm==4.7.0` (lock:21), `>=4.3` (txt) | — | да, `backend="lightgbm"` (по умолчанию) |
| psutil | **нет** | только `requirements-dev.txt` (`psutil>=6` и дубль `psutil>=5.9`) | нет: `process_t0()` → `except Exception: return _T_IMPORT` |
| lleaves | **нет** | нигде | нет: `<k>.lleaves.so` грузится через ctypes (`SoForest`), `objdump -p`: NEEDED пуст |
| llvmlite | **нет** | нигде | нет (нужен только для компиляции: `scripts/pixgbm_compile.py`) |
| tl2cgen | **нет** | нигде | только `backend="treelite"` (`TreeliteModel`: `import tl2cgen`); `.tl2cgen.so` требует GLIBC_2.29 и libm |
| treelite | **нет** | нигде | зависимость tl2cgen (`treelite>=4.1.2`) |

- Без psutil (это подтверждено и задокументировано в docstring `process_t0`) срок фазы `budget_s` отсчитывается от импорта `expds.geo_pixgbm`, а не от старта процесса, то есть чуть позже. `pix_deadline` всё равно берёт min с `guard.t0`, если guard передан, а `region_feats` ещё режет срок по `guard.left() − reserve`.
- Патч (не применён): `pixgbm/requirements_pixgbm.patch`.
  - `requirements.lock`: `psutil==7.2.2` (BSD-3-Clause); для treelite opt-in — `tl2cgen==1.0.0` и `treelite==4.8.1` (Apache-2.0).
  - `requirements-dev.txt`: `lleaves==1.3.0` (MIT), `llvmlite==0.44.0` (BSD-2-Clause), tl2cgen и treelite — для компиляции.
  - Колёса под образ python 3.12 x86_64 проверены через `pip download --python-version 3.12 --only-binary=:all:`:
    - psutil cp36-abi3 manylinux_2_12, 155 560 Б;
    - tl2cgen py3 manylinux2014, 774 824 Б;
    - treelite py3 manylinux_2_28, 924 776 Б;
    - lleaves py3-none-any, 22 209 Б;
    - llvmlite cp312 manylinux2014, 42 361 901 Б.
  - Версии взяты из venv bot-3, где собирались .so (lleaves 1.3.0 / llvmlite 0.44.0 / tl2cgen 1.0.0 / treelite 4.8.1); psutil — последняя на PyPI (`pip index versions`).
  - Лицензии взяты из METADATA колёс; llvmlite — по тексту LICENSE (2 пункта, без пункта об endorsement).
  - **Минимум для S**: ничего (backend lightgbm). Строка psutil желательна, но не обязательна. tl2cgen/treelite ставить только если выбран treelite.
  - Образ публичный, тег не переписывать (шапка Dockerfile). Любая правка lock — новый тег.
- Бэкенд по умолчанию не менял: `backend = "lightgbm"`, lleaves — opt-in.

## (3) Упаковка S
- В `configs/default.toml`: `[geophys.pixgbm] enabled = false`, флаг не переключал.
- Opt-in конфиг: **`bot-3/pixgbm/config_v2_S.toml`**:
  - `enabled=true`, `replace=true`, все классы и методы, `t_pix 0.5`, `backend lightgbm`, `model_dir models/pixgbm/all`;
  - три секции `[thresholds_by_markup.*_pix]` из `thresholds_v2.toml` — проверено, что равны: `tomllib`, `==`;
  - от default отличаются только `enabled`, `replace` и `poly_mode="pole"`. Ключ `poly_mode` из bot3-geosnap в bot3-pixgbm игнорируется.
  - Платформа читает только `configs/default.toml` (`inference/main.py:87`), поэтому при сборке секцию нужно заменить, а пороги дописать.
  - Готовая склейка: `pixgbm/packtest/default_S.toml`. Остальной конфиг сравнён с default через `tomllib` (`rest equal: True`).
- Файлы в архив (backend lightgbm, sha256 совпали в zip и в источнике):

| путь в архиве | источник | байт | в zip (deflate) | sha256 (начало) |
|---|---|---|---|---|
| `models/pixgbm/all/poly.txt` | `bot-3/pixgbm/models/all/poly.txt` | 2 122 321 | 963 640 | d1fe3efafcd4377d |
| `models/pixgbm/all/finds.txt` | `bot-3/pixgbm/models/all/finds.txt` | 2 111 596 | 956 197 | 5fa8c09d77d0914c |
| **итого** | | **4 233 917** | **1 919 837** | |

- Если `backend="lleaves"`, добавляются `poly.lleaves.so` (1 133 208), `finds.lleaves.so` (1 125 016) и два `.lleaves.json` (по 219): ещё 2 258 662 Б. Если `"treelite"` — `poly.tl2cgen.so` (1 105 752) и `finds.tl2cgen.so` (1 130 328).
- Моделей нет в git (`repo/models/pixgbm` отсутствует), значит их нужно подавать через `--weight`.
- **Где упаковщик должен их включить** (чужие файлы не правил, патч `pixgbm/packer_pixgbm.patch`):
  - `bot-8/build_submission.py`, `model_refs()`, после строки 64 (`if rr.get("enabled") else []`), перед `seen, uniq = …` (стр. 65). Сейчас там только `[models]`, `candidates.lgbm_model`, `verify`, `rerank_arch`. Без правки `--weight models/pixgbm/...` печатает «не упомянуты в конфиге — в архив не попадут».
  - `repo/scripts/build_submission.py`, `collect()`, стр. 40–45 (после `verify = …`, перед `seen = set()`). Та же вставка; файлы берутся из `ROOT/models/pixgbm/all/`.
  - Вставка: если `geophys.pixgbm.enabled`, добавить `<model_dir>/poly.txt` и `finds.txt` (finds — только если есть `finds_points` и `maggr`), а для lleaves/treelite — их `.so`/`.json`.
- Сухая сборка копией упаковщика bot-8 с патчем (`pixgbm/packtest/build_submission_b8_patched.py`, ref bot3-pixgbm 76f7317; zip удалены, манифесты оставлены в `packtest/dist/`):
  - S: `koz3_s-pix-dry_76f7317_2071b5d3_wefee57da.zip`, 566 935 388 Б, 47 файлов;
  - база: `koz3_base-dry_76f7317_73019339.zip`, 565 015 549 Б, 45 файлов;
  - **Δ = +1 919 839 Б**, до лимита 2 ГБ далеко.
  - `check_submission` на этой сборке не запускал.

## Что нужно от PC / владельца упаковки (bot-8)
1. Применить `packer_pixgbm.patch`. Первая часть (`a/bot-8/build_submission.py`): `cd /workspace/koz3_shared && patch -p1 --dry-run -i bot-3/pixgbm/packer_pixgbm.patch`; если patch спросит про `scripts/build_submission.py` — пропустить. Вторая часть (`a/scripts/build_submission.py`) применяется из корня репозитория.
2. Собрать:
```
B3=/workspace/koz3_shared/bot-3; M=$B3/pixgbm/models/all
python bot-8/build_submission.py --repo <слитое дерево> --label v2-s --config $B3/pixgbm/packtest/default_S.toml \
  --weight models/pixgbm/all/poly.txt=$M/poly.txt --weight models/pixgbm/all/finds.txt=$M/finds.txt --check
```
   `default_S.toml` собран из default bot3-pixgbm. Для слитого дерева 428257f нужно повторить склейку на его default.toml: заменить `[geophys.pixgbm]` и дописать три `*_pix` секции из `config_v2_S.toml`.
3. Зависимости для S ставить не нужно. psutil — по желанию, в новый тег образа.
