# RELIEF.md — задание боту 9: многоканальный рельеф из LAS в Kaggle-ядре

Автор: bot-2, 28.10.2026 21:18 МСК. Статус: рецепт и заготовка кода; на Kaggle ничего не запускалось, на ПК — тоже.
Модуль-заготовка: `/workspace/koz3_shared/bot-2/relief/relief_ext.py` (проверен на синтетике: `relief/selftest.py`, вывод ниже, §8).

## 0. Зачем и что сравниваем
Сейчас лидарная модель курганов `models/li_r18_hard2.pt` видит 3 канала. Метаданные чекпойнта (прочитаны без torch, через zip/pickle): `in_channels 3, li_tags [lrm_s, lrm_l, slope], li_source las, res 0.5, tile 128, dem_res 0.1, dem_fill delaunay, encoder resnet18, epoch 30`. Гипотеза: 9 дополнительных каналов, не зависящих от азимута (горизонт, локальный рельеф, облако), улучшат курганы на честном A. Сравнение — **парное**: те же тайлы, фолды, рецепт и зёрна, отличается только набор каналов.
Важно: прошлая честная проверка 59-LAS моделей дала отказ, −0,09 (`/workspace/koz3_shared/COMMON.md:10`). Поэтому сначала проверяем, что ядро воспроизводит базовые 3 канала бит в бит (§6, этап 0).

## 1. Входные данные (S3) и доступ
- Бакет `train-expds-3`, endpoint `https://s3.ru-7.storage.selcloud.ru`, регион `ru-7`. Клиент boto3 — готовый `s3_client()` и `fetch()` из `bot-9/kaggle/matanerdy_repro/src/mn_lib.py:9-60`.
- Ключи берутся только из Kaggle Secrets `KOZ3_S3_ACCESS_KEY_ID` / `KOZ3_S3_SECRET_ACCESS_KEY` (`mn_lib.py:11,26-27`). Ключи не печатаются и не пишутся.
- Состав бакета (`bot-9/REPORT2.md` §6, по `s3/listing.txt`):
  - LAS (слой `01_…_La_точки`, несжатые `.las`): 59 файлов, 39,78 ГБ, по одному на участок. Курганы gt_train_v2 есть на 42 из них (25,64 ГБ).
  - Готовые карты `Li_*.tif` (c/ch/g/i) — **не те каналы**: li-модель строится из LAS (TIN 0,1 м). Поэтому нужен LAS, а tif не нужны.
- CRS участка — `UTM.json` участка (как в `expds.tiles.iter_sources`, `repo/expds/tiles.py:179-200`, `read_utm_json`).
- Эталон: `gt_train_v2.geojson` (как у кэша `li_tin_59_t128`). Положить его в датасет ядра.
- Скорость скачивания: на box 40,9 МБ/с при 4 потоках (`bot-9/s3/speedtest.log`). На Kaggle не мерилась.

## 2. Каналы и точные параметры
Шаг сетки модели — **0,5 м**. DEM строится так же, как для hard2: TIN по ячейкам 0,1 м, затем усреднение до 0,5 м (`MODEL_DEM`, `repo/expds/lidar.py:39`; `to_work_grid` в `relief_channels`, `lidar.py:570`). Радиусы заданы в метрах, пиксели = round(м / 0,5). У yakirill пиксель 1 м (`configs/dataset_006.yaml:30-31`), поэтому его «пиксели» = метры.

Первые 3 канала — **без изменений**: `lrm_s` (z − среднее окна r 8 м, клип ±0,75 м), `lrm_l` (r 30 м, ±2 м), `slope` (0…45°). Это `repo/expds/lidar.py:48-54,577-587`. Они нужны, чтобы базовая ветка совпала с hard2. Дальше 9 новых каналов с суффиксом `_r` (site_robust). Существующие теги `opns`/`svf`/`curv`/`int` с фиксированной шкалой не трогаем.

| тег | что | параметры | основание |
|---|---|---|---|
| `svf_r` | фактор вида на небо, SVF = 1 − ср. max(0, sin γ) (Zakšek 2011) | луч 10 м = 20 пикс., 16 направлений | yakirill `dataset_006.yaml:67-68` (openness_radius 10, n_directions 16). Наш `RELIEF opns_r_m 10, opns_dirs 16` (`lidar.py:50`), `accel.openness_svf` (`repo/expds/accel.py:378-412`) |
| `opp_r` | положительная открытость, ° (Yokoyama 2002) | те же 20 пикс. / 16 направлений | yakirill `relief.py:183-271` (pos = 90 − max угла) |
| `opn_r` | отрицательная открытость = открытость(−z) | те же | yakirill `relief.py:183-231` (neg = 90 + min угла, то же самое) |
| `tpi5_r` | TPI = z − среднее окна (2r+1)² | r = 5 м = 10 пикс. | yakirill `relief.py:339-365`, `dataset_real.yaml:30` (`tpi_radius_5`), `dataset_006.yaml:70` |
| `tri3_r` | TRI = √(локальная дисперсия z) | r = 3 м = 6 пикс. | yakirill `relief.py:368-392`, `dataset_real.yaml:31` (`tri_radius_3`) |
| `curv_r` | лапласиан z после гаусса, 1/м | σ = 1,5 м (3 пикс.) | наш `_curvature`, `lidar.py:517-524` и `RELIEF curv_sigma_m` (`lidar.py:54`). У yakirill σ 2 (`relief.py:308-331`, `dataset_real.yaml:29`) — это только абляция |
| `slrm8_r` | SLRM = z − гаусс(z) | σ = 8 м = 16 пикс., нормированная свёртка по съёмке | yakirill `relief.py:128-149`, `dataset_real.yaml:25` (`slrm_sigma_8.0`) |
| `dens_r` | log1p(плотность ВСЕХ точек, 1/м²) на ячейке 0,5 м | — | yakirill `cloud_features.py:91-127`, «по всем точкам, не только земле» (`src/expds/data/build_site.py:375`). log1p — моё решение (тяжёлый хвост на перекрытиях полос), фиксировано заранее |
| `int_r` | средняя интенсивность точек земли (класс 2; если его нет — все точки), пустые ячейки — среднее окна 2 м | — | наш `intensity_grid` / `intensity_unit` (`lidar.py:447-496`, `INTENSITY_FILL_M 2,0`, `lidar.py:60`), yakirill `cloud_features.py:136-173` |

Чего у yakirill **не повторять**:
- Его SVF — это упрощение `pos_openness/180` (`relief.py:274-305`), а не настоящий SVF.
- Его открытость считается через `np.roll` (`relief.py:183-231`), а он заворачивает противоположный край растра в горизонт.
- У нас лучи обрываются на краю (`accel.py:393-411`). Разность z перед квадратом сдвигается на среднее участка (как `accel.local_dev`, `accel.py:423-438`), чтобы float32 не терял дециметры.

## 3. Нормировка site_robust
Основа — yakirill `src/expds/tiles/dataset.py:38-39,77-100` и `configs/dataset_real.yaml:38-39`: `(x − median) / (p98 − p2)` по валидным пикселям участка, при `p98 − p2 < 1e-6` масштаб равен 1.
- **Участок = весь LAS-регион.** Статистика считается по всему растру региона на шаге 0,5 м до нарезки и без эталона, поэтому на тесте она считается так же.
- Перевод в uint8 (наш контракт: 0 — вне съёмки, 1…255 — данные, `lidar.py:615`): `y = clip((x−med)/(p98−p2), −1, 1)`, `u8 = rint(1 + (y+1)/2·254)`. Медиана даёт 128, p2/p98 дают ≈ 64/192.
- Статистику {регион: {тег: [p2, med, p98]}} сохранять в `site_stats.json` рядом с кэшем.
- **Ловушка — мозаика.** На инференсе большие облака при `lidar.mosaic` режутся на окна (`repo/expds/pipeline.py:288-330`, `build_dem_mosaic`). В этом случае статистику нужно брать с целого грубого DEM участка и передавать в `extra_channels(..., stats=…)`, а не считать по окну.

## 4. Библиотеки на Kaggle и запасные варианты
- **`laspy[lazrs]`** — чтение LAS кусками (`chunk_iterator`), как в `lidar.py:461`. Уже используется нашим кодом. Запасного варианта нет, он и не нужен.
- **Наш `expds`** — бандл репо, ветка hard2: `build_dem`, `to_work_grid`, `relief_channels`, `expds.tiles.build_cache`, `cut_window`.
  - Бэкенд обязан совпадать с ПК. Положить `inference/lib/expds_core.abi3.so`, сделать `sys.path.insert(0, …/inference/lib)` и писать `accel.backend()` в лог.
  - NumPy и Rust дают разные тайлы: 2,5–3,7 % пикселей, до 37 уровней (`bot-9/REPORT2.md` §6, `s3/identity.log`).
- **SVF/открытость** — основной путь `expds.accel.openness_svf` (Rust или NumPy). Запасной — **`rvt-py`**: `pip install rvt-py --no-deps`, функция `rvt.vis.sky_view_factor(dem, resolution, compute_svf, compute_opns, svf_n_dir, svf_r_max[пикс.], svf_noise, no_data)` возвращает `{"svf","asvf","opns"}`.
  - API проверен по колесу rvt_py-2.2.1, `rvt/vis.py:673-688`. В зависимостях колеса есть `gdal`, но `rvt.vis` импортирует только numpy/scipy, поэтому `--no-deps`. `rvt.vis` берёт `scipy.ndimage.morphology` — это устаревший путь в новых SciPy.
  - **Не смешивать бэкенды**: rvt и accel на одной синтетике расходятся до 38–41 уровня uint8 (§8). Кэш и инференс — строго одним путём.
  - `rvt.vis.slrm` не годится: там среднее окна радиусом 10–50 пикс. (`vis.py:433-470`), а не гаусс.
- **TPI/TRI/SLRM/кривизна** — только numpy/scipy (`box_mean` на интегральном изображении, `gaussian_filter` с нормировкой по маске). Внешние библиотеки не нужны.
- **pdal** — не ставится через pip (только conda), не использовать.
- **whitebox** (pip) скачивает бинарь WhiteboxTools при первом вызове, нужен интернет ядра. Не нужен: всё покрыто numpy/scipy.

## 5. Выход: формат, dtype, тайлы (совместимо с `expds.train` и li_r18_hard2)
- **Тайлы — те же центры, что у `li_tin_59_t128`**: `/workspace/kaggle_out/bundle_index/li_tin_59_t128/index.csv`, 4645 тайлов, 3518 положительных, 59 участков, 44 семьи, столбцы `shard,idx,region,family,file,cx,cy,positive`.
  - Окна вырезаются тем же `cut_window(arr, tr, cx, cy, 128)` (`tiles.py:143-158`) из массива каналов участка. Маски копируются из старого кэша как есть, они от каналов не зависят.
  - Так обе ветки отличаются только каналами, и сравнение строго парное.
- Файлы (формат `expds.tiles.build_cache`, `tiles.py:475-540`, чтение — `train.py:234,444,472`):
  - `images_XXXX.npy` — (N, 12, 128, 128) **uint8**, порядок каналов = `li_tags`;
  - `masks_XXXX.npy` — (N, 128, 128) uint8, бит k = класс k из `classes`, плюс IGNORE_BIT, как в исходнике;
  - `index.csv` — как исходный;
  - `meta.json` — копия исходного meta, поменять `"li_tags": ["lrm_s","lrm_l","slope","svf_r","opp_r","opn_r","tpi5_r","tri3_r","curv_r","slrm8_r","dens_r","int_r"]` и `"channels": 12`, добавить `"relief_ext": P` (параметры §2), `"site_stats": "site_stats.json"`, `"backend": accel.backend()`. `flips` оставить `true`: все каналы инвариантны к повороту.
- Размер: 4645 × 12 × 128² B ≈ 0,91 ГБ (базовые 3 канала ≈ 0,23 ГБ). Это арифметика, не замер.
- Дополнительно, по желанию: растры участков `relief_<region>.tif` (GeoTIFF uint8, 9 полос, nodata 0, deflate, блоки 512, CRS — UTM участка, описания полос = теги) — для инференса и визуальной проверки.
- Обучение — тот же рецепт, что у hard2_59 (`repo/notes/kaggle_gpu_s1_benchmark_2026-10-28.json:31-68`): `python -m expds.train --cache <cache>/foldK --encoder resnet18 --arch Unet --target mask --epochs 30 --batch 64 --lr 1.4e-3 --val-families <auto_val_family> --device cuda:0`.
  - `in_channels` берётся из meta (`train.py:472`); первый слой smp для 12 каналов собирается сам.
- **Инференс.** Чтобы веса работали в `_li_with_model` (`pipeline.py:658-662`), в `expds/lidar.py:relief_channels` нужно добавить ветку тегов `*_r`, которая вызывает `relief_ext.extra_channels`.
  - Это отдельный патч в репо. Он должен быть тем же кодом, что строил кэш: один модуль для обоих путей.

## 6. План валидации (честный, по семьям)
0. **Паритет.** В ядре пересобрать 3 базовых канала для 030_Эсбю и 179_Гнёздово_Лесная_1, вырезать тайлы по `index.csv` и сравнить с тайлами `li_tin_59_t128` (sha256 и доля отличных пикселей).
   - Критерий — 0 отличий. Иначе стоп и разбор бэкенда (§4).
1. **Ветки:** A0 = 3 канала (пересобраны в ядре) и A1 = 12 каналов. Те же 5 фолдов семей `bot-9/configs/folds_li_tin_59_t128.json`, тот же рецепт (§5), 5 зёрен на ветку (`--seed 0..4`).
2. **OOF на целых участках.** Каждая fold-модель прогоняется по отложенным регионам целиком через `_li_with_model`, с теми же кандидатами и `rescore`, что в проде. Получаются признаки в формате стека (`markup li`, `threshold_group li`, conf, region).
   - Тайловая val_loss — только для контроля обучения, не для решения.
3. **Счёт.** На box/ПК в стеке заменить li-курганы ветки на OOF A0 и OOF A1. Затем honest-10 A (`family_folds(…,10)`, пороги по train-семьям, сопоставление global, как в `base_honest.pkl`).
   - Решение: **A1 против A0**.
   - Санити-проверка: A0 против текущей базы 0,314615 — должно быть ≈ 0.
4. **Приём — правило bot-10 П1–П5** (`/workspace/koz3_shared/EXPERIMENTS.md`):
   - П1 Δ ≥ +0,002;
   - П2 парный бутстреп по семьям n ≥ 1000, P(Δ>0) ≥ 0,85;
   - П3 LOFO Δ > 0 и ≥ 4 из 5 раскладок семей (`bot-10/repo/scripts/honest_bootstrap.py --seeds 5`);
   - П4 povr/tselye/gor взвешенная Δ ≥ −0,0005 каждая, new34/optical ≥ база − 0,001;
   - П5 при Δ < 0,006 — только «в пакет».
5. **Pre-declaration.** До запуска записать строку в EXPERIMENTS бота 9: время, sha256 этого файла и `relief_ext.py`, список тегов, параметры §2, зёрна. После запуска параметры не менять.
6. **Абляция** — только если A1 принят. Убирать группы каналов: {svf, opp, opn}, {tpi, tri, curv, slrm}, {dens, int}. Отдельно: кривизна σ 2 м против 1,5 м.
   - Риск: `dens`/`int` зависят от сканера и полётов, а не от рельефа. site_robust снижает этот риск, но не снимает.

## 7. Оценки времени и памяти (ОЦЕНКИ, не замеры на Kaggle)
- **Скачивание.** 39,8 ГБ LAS при 36–72 МБ/с ≈ 10–19 мин. Только 42 курганных участка (25,6 ГБ) ≈ 6–12 мин.
  - Качать по одному LAS, удалять после обработки. Максимум на диске — один LAS (самый крупный около 6 ГБ, Муравлево) плюс выход. Это меньше ~20 ГБ `/kaggle/working`.
- **DEM TIN 0,1 м.** Rust ≈ 1 с на 1,2 млн точек (`bot-9/s3/bench.log`) ⇒ все 59 LAS ≈ 15–30 мин CPU. NumPy в 8–13 раз медленнее (там же).
  - Крупные облака упираются в `max_pixels` 60 млн (`lidar.py:31`), там время нелинейное и не мерилось.
- **Новые каналы.** На box (NumPy-бэкенд, 1 поток) 7 рельефных каналов на 1000×1000 пикс. заняли 5,6 с (§8). Регион на 0,5 м обычно ≤ 10 млн пикс. (при 60 млн пикс. DEM 0,1 м ≈ 2,4 млн пикс. на 0,5 м) ⇒ ≤ ~1 мин на регион, ≤ ~30–60 мин на 59 регионов в NumPy.
  - Плотность и интенсивность — один дополнительный проход по LAS на регион. Время сопоставимо с чтением при DEM.
- **Память.** DEM 0,1 м ≤ 60 млн × 4 B = 0,24 ГБ; TIN Rust до 20 млн вершин × ~72 B ≈ 1,4 ГБ (`lidar.py:34`); каналы 12 × 10 млн × 4 B ≈ 0,5 ГБ. Пик ≈ 2–3 ГБ на регион, регионы обрабатываются по одному.
- **Где запускать.** Подготовку — в CPU-ядре: CPU-работа в GPU-сессии тратит GPU-часы (`bot-9/REPORT2.md` §6). Обучение 2 ветки × 5 фолдов × 5 зёрен — в GPU-ядре, время по замеру hard2_59 в `repo/notes/kaggle_gpu_s1_benchmark_2026-10-28.json`. При 12 каналах вместо 3 время шага почти не растёт: первый слой ≈ 1 % FLOPs resnet18 (оценка).

## 8. Проверка заготовки (box, синтетика, 1 поток)
`cd /workspace/koz3_shared/bot-2/relief && PYTHONPATH=_rvt OMP_NUM_THREADS=1 /workspace/rerank/venv/bin/python selftest.py`
Синтетика: 1000×1000, шаг 0,5 м, уклон, шум 3 см, 3 гауссовых «кургана» 1,2 м, дыра NaN.
- accel-бэкенд numpy: 7 каналов `(7,1000,1000) uint8` за 5,6 с.
- В центре кургана: tpi5 255, tri3 255, slrm8 255, curv 1 (купол отрицательный), opp 175. На фоне все каналы 92–147.
- rvt против accel: до 41 / 38 / 39 уровней разницы по svf/opp/opn ⇒ бэкенды не смешивать.

## 9. Заготовка ядра (CPU, подготовка кэша)
```python
# koz3-relief-prep: CPU-ядро. Вход: датасет с бандлом репо (+ expds_core.abi3.so), relief_ext.py,
# li_tin_59_t128 (index.csv, masks_*.npy, meta.json), gt не нужен (маски берём из старого кэша).
import json, os, sys, shutil, hashlib, numpy as np, pandas as pd
from pathlib import Path
B = Path("/kaggle/input/koz3-bundle"); W = Path("/kaggle/working"); OUT = W / "cache_li12"; OUT.mkdir(exist_ok=True)
sys.path[:0] = [str(B / "repo"), str(B / "repo/inference/lib"), str(B / "relief")]
from expds import accel
from expds.lidar import MODEL_DEM, build_dem, relief_channels
from expds.candidates import to_work_grid
from expds.tiles import cut_window
from expds.catalog import read_utm_json
import relief_ext as RX
from mn_lib import s3_client, fetch                      # bot-9/kaggle/matanerdy_repro/src/mn_lib.py
print("backend:", accel.backend())                       # должен совпасть с ПК (§4)
SRC = B / "li_tin_59_t128"; idx = pd.read_csv(SRC / "index.csv"); meta = json.loads((SRC / "meta.json").read_text())
TAGS = list(RX.BASE_TAGS) + list(RX.EXTRA_TAGS); RES, TILE = meta["res"], meta["tile"]
keys = json.loads((B / "las_keys.json").read_text())    # {region: (s3_key, utm_json_key)} — из listing.txt
cli = s3_client(); stats_all = {}; tiles = {}            # (shard, idx) -> 12×128×128
ONLY = os.environ.get("RELIEF_ONLY")                      # этап 0: "030_Эсбю,179_Гнёздово_Лесная_1"
for region, g in idx.groupby("region"):
    if ONLY and region not in ONLY.split(","): continue
    las_key, utm_key = keys[region]
    las = fetch(cli, las_key); utm = fetch(cli, utm_key)
    crs = read_utm_json(Path(utm))                        # «EPSG:326xx», как в expds.tiles.iter_sources
    d = build_dem(las, res=MODEL_DEM["res"], fill=MODEL_DEM["fill"], crs=crs)
    base, tr = relief_channels(d, RES, RX.BASE_TAGS)      # 3 канала hard2 — бит в бит (этап 0)
    z, step = to_work_grid(d.dem, d.res, RES)
    assert abs(step - tr.a) < 1e-9 and z.shape == base.shape[1:]
    extra, st = RX.extra_channels(z, step, d.left, d.top, las_path=las, accel=accel)
    stats_all[region] = st
    arr = np.concatenate([base, extra])                   # 12×H×W uint8
    for r in g.itertuples():
        win, _ = cut_window(arr, tr, r.cx, r.cy, TILE); tiles[(r.shard, r.idx)] = win
    os.remove(las); del d, base, extra, arr, z
for shard in sorted(idx["shard"].unique()):
    rows = idx[idx["shard"] == shard].sort_values("idx")
    if not all((shard, i) in tiles for i in rows["idx"]): continue
    np.save(OUT / f"images_{shard:04d}.npy", np.stack([tiles[(shard, i)] for i in rows["idx"]]))
    shutil.copy(SRC / f"masks_{shard:04d}.npy", OUT)
shutil.copy(SRC / "index.csv", OUT)
meta.update(li_tags=TAGS, channels=len(TAGS), relief_ext=RX.P, site_stats="site_stats.json", backend=accel.backend())
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
(OUT / "site_stats.json").write_text(json.dumps(stats_all, ensure_ascii=False, indent=1))
# этап 0: сравнить первые 3 канала с SRC/images_*.npy по тем же (shard, idx): sha256 и доля отличных пикселей
```
Примечания к заготовке:
- `las_keys.json` собирается из `bot-9/s3/listing.txt` на box и кладётся в датасет. Ключи S3 — только через Secrets.
- Хранение всех тайлов в словаре ≈ 0,9 ГБ — допустимо. Иначе писать шарды по мере готовности.
- Фолдовые кэши `cache/foldK` строить тем же способом, что для hard2_59, из `index_fold*.csv`.
- Заготовка на Kaggle не запускалась. Синтаксис и функции `relief_ext.py` проверены только на box (§8).
