# Патчи bot-3 для линии bot-7 (v4)

Подготовлено 29.10.2026 (около 12:10 МСК), автор коммитов: bot-3. Ничего не пушилось; ветка bot-7 не трогалась (remote `bot7` — только fetch).

## Патчи

| Файл | Коммит в bot-3/repo (ветка `bot3-v4-patches`) | Источник | sha256 файла |
|---|---|---|---|
| `0001-geophys-opt-in-EPSG-3857-cos-geophys-ground_metres-E.patch` | 4e63897 | cherry-pick -x e3c4f41 | d12fd9c3605861dc8228e9e913c04a49639195bb6533b13c777f482cad70c5b4 |
| `0002-geo_pixgbm-U-dedupe_union-EtGr-29.10-.4-5.patch` | deafc30 | cherry-pick -x e0ec0ff | 233b6af38cdd7a36b7a991e871508ad3fc92c80ec40d9c69da5dc5fe47267989 |

**0001 — метры на земле для EPSG:3857 (opt-in).**
- Включается через `[geophys] ground_metres = true` или env `EXPDS_GROUND_M=1`. По умолчанию выключено: ключа нет в `configs/default.toml`.
- Когда включено, шаг пикселя растра в EPSG:3857 умножается на cos φ центра растра (`crs.ground_scale`). Это затрагивает:
  - выбор шага f и up в `load_map`;
  - площадь ядер насыщения;
  - буферы 0,3 / 1,0 м в `label_masks`;
  - `area_m2` кандидатов;
  - классический путь (`geophys.py`, `pipeline.py`).
- Для UTM и прочих метрических CRS множитель равен 1,0.
- Тест: `tests/test_ground_metres.py`.
- При переносе был конфликт в `expds/geo_pixgbm.py::load_map`. Причина: в e3c4f41 есть код аудита (`satc`, `sat_flag`), которого нет в v3. Разрешение такое:
  - поведение v3 сохранено: `restore_sat_cores` + `del sat`, возврат без `sat`/`satc`/`sat_feat`;
  - добавлен только множитель `k`: `* k * k` в площади ядра, `up = abs(T.a) * k / gsd`, `gsd=... * k, k=k`;
  - `crs` теперь вычисляется до `f`. Это то же выражение, просто раньше по коду.

**0002 — правило U (`dedupe_union`).**
- Сравнение идёт только внутри ключа (класс, markup_type, слой). Слой учитывается только для EtGr. Это правила 29.10, п. 4–5.
- Прежний код мог гасить кандидата PIXGBM об объект другого класса или метода, либо об объект другого слоя EtGr.
- Используется только при `replace = true` и непустом `union_name_re`. В v3 `union_name_re = ""`, поэтому на сдаче v3 патч ничего не меняет.
- Тест: `tests/test_union_layers.py` (5 тестов).
- Применился без конфликтов.

## База
- Патчи сделаны от `bot7/bot7-candidate-v3` = **36af6b7** («config: формулировка порогов sp_dfine…»).
- К 11:42 МСК bot-7 добавил в эту же ветку 2 коммита: c1a6b7e и **360c870**. Оба меняют только `docs/THIRD_PARTY.md`.
- Других веток новее v3 у bot-7 нет. Проверено командой `git for-each-ref --sort=-committerdate refs/heads` в bot-7/repo: свежие — v3 (11:42), v2 (03:12), measure (28.10).
- На 360c870 патчи применяются чисто. Проверка шла во временном worktree `bot-3/tmp_v3tip`, после неё он удалён:
  - `git apply --check 0001… 0002…` → OK;
  - `git am` → OK, 6 файлов, +174/−16.

## Тесты (2 потока, nice 10)
Команда:
`OMP_NUM_THREADS=2 EXPDS_PIXGBM_THREADS=2 PYTHONPATH=…/bot-3/ehimen/pylib nice -n 10 …/venv/bin/python -m pytest -q -p no:cacheprovider <тесты>`

1. Набор `test_ground_metres`, `test_union_layers`, `test_etgr_layers`, `test_geo_*` (density, maps, nn, pixgbm_threads, plateau, regime, stages), `test_geophys_signal`, `test_post_writer_geo` дал **58 passed**.
2. Набор `test_bot8_window_deadline`, `test_bot7_pix_pool_sigterm`, `test_claude_pix_sat_cores`, `test_raster_crs`, `test_pipeline`, `test_bot7_v3_flags`, `test_bot7_candidate_v2`, `test_platform_robustness`, `test_raster_infer` дал **70 passed, 9 skipped, 4 failed, 2 errors**. Ровно то же на чистой v3 (36af6b7).
   - Списки FAILED/ERROR совпадают побайтно: `audit/out/v4_fail_patch_v4.txt` = `audit/out/v4_fail_tmp_v3base.txt`.
   - Причина — окружение бокса, например `ModuleNotFoundError` (нет torch). Патчи тут ни при чём.
   - `test_audit_fixes.py` не собирается в обоих деревьях: нет torch.

## Побайтная идентичность на трейне
- Скрипт: `audit/v4_identity.sh`. Выход и логи: `audit/out/v4/`, сводка: `audit/out/v4/runs.txt`.
- Запуск: `scripts/geo_predict.py --root /workspace/data/train-expds-3 --regions 200_Судьбищи_2 204_Семибратнее_Городище_4 212_Гнёздово_Усадьба 214_Гнёздово_Лесная_2`.
- Настройки: конфиг v3 по умолчанию (PIXGBM S: enabled+replace, `union_name_re=""`, модели `models/pixgbm/all` из дерева). Единственная правка `--set geophys.pixgbm.procs=1`, одинаково во всех прогонах.
- Ограничения: 2 потока, nice 10, `ulimit -v 2 ГиБ`.

| Прогон | Дерево | Объектов | sha256 geojson | пик RSS, кБ |
|---|---|---|---|---|
| base_v3 | 36af6b7 без изменений | 8210 | bbdfadc68a239d7d9ac6a0fe27d7f6695301c109c81b38a035be0df9ed4421ae | 1625632 |
| v4_default | патчи, по умолчанию | 8210 | bbdfadc68a239d7d9ac6a0fe27d7f6695301c109c81b38a035be0df9ed4421ae | 1587236 |
| v4_gm_cfg | патчи + `--set geophys.ground_metres=true` | 8210 | bbdfadc68a239d7d9ac6a0fe27d7f6695301c109c81b38a035be0df9ed4421ae | 1704000 |
| v4_gm_env | патчи + `EXPDS_GROUND_M=1` | 8210 | bbdfadc68a239d7d9ac6a0fe27d7f6695301c109c81b38a035be0df9ed4421ae | 1636716 |

- По участкам в каждом прогоне (`base_v3.log`): 200 — 3171, 204 — 1436, 212 — 3539, 214 (EtGr) — 64.
- Включённый ground_metres на UTM-трейне выход не меняет. Весь трейн в UTM, см. `audit/out/crs_inventory.json`.
- **Ограничение.** Под лимитом 2 ГиБ классика падает по памяти на двух больших картах, одинаково во всех 4 прогонах. Это `01_Судьбищи_2_MagGr_серый.tif` и `01_Гнёздово_Усадьба_MagGr_цвет_2.tif` (`ошибка геофизики (Unable to allocate …)`). Классика на этих двух картах в сравнении не участвует. На платформе (96 ГБ) этих падений нет.
- Полный трейн не гонялся: около 6–7 мин на 4 участка на прогон, 2 потока.
- Правило U с непустым `union_name_re` на трейне не мерилось. Оно выключено в v3, его покрывает `test_union_layers`.

## Как применить и проверить
```bash
cd <repo bot-7>                      # чистое дерево на bot7-candidate-v3 (36af6b7 или 360c870)
git am /workspace/koz3_shared/bot-3/patches_v4/0001-*.patch /workspace/koz3_shared/bot-3/patches_v4/0002-*.patch
export OMP_NUM_THREADS=2 EXPDS_PIXGBM_THREADS=2
nice -n 10 python -m pytest -q tests/test_ground_metres.py tests/test_union_layers.py tests/test_etgr_layers.py tests/test_geo_*.py
# идентичность (по умолчанию и с ground_metres): ожидаемый sha256 bbdfadc6…21ae, 8210 объектов
nice -n 10 python scripts/geo_predict.py --root /workspace/data/train-expds-3 --out /tmp/v4.geojson \
  --regions 200_Судьбищи_2 204_Семибратнее_Городище_4 212_Гнёздово_Усадьба 214_Гнёздово_Лесная_2 --set geophys.pixgbm.procs=1
sha256sum /tmp/v4.geojson
```
- Хэш воспроизводится под `ulimit -v 2097152`, когда две большие карты классики падают.
- Без лимита памяти выход будет другим, но тоже одинаковым для v3 и v4: сравнивайте с прогоном чистой v3 в тех же условиях.
- Откат: `git reset --hard HEAD~2`. Отдельной ветки у bot-7 патчи не создают.
