# speedup: ранние находки (для bot-8, чтобы не дублировать) — 28.10, 22:20 МСК

База: копия HEAD 24bd916 в `speedup/base` (git не трогал), выборка `speedup/sample` (7 регионов, симлинки файлов без разметки),
харнесс `tools/prof_run.py` (cProfile, prefetch=0, сеть заменена дешёвой детерминированной заглушкой,
путь накопления — как на CUDA, `_predict_on_device`, на CPU-torch). Профиль: `prof/base1.txt`.

Сквозной прогон выборки (7 рег., 2 маленьких LAS, SpOr/MagGr/EtGr/TerFl): 27,3 с; фазы геофизика 1,5 / лидар 8,2 / оптика 10,4 / σ 4,8.

| # | функция | файл:строка | время в выборке | почему |
|---|---|---|---|---|
| 1 | `_predict_on_device` (питоновский цикл по окнам 128², 3–4 мелких ядра на окно) | expds/infer_nn.py:176 | 4,8 с tottime (CPU-torch) | на 4090 — ~40 мкс запусков на окно ≈ как сам forward r18 на окно; Ae 029 ≈ 1e5 окон |
| 2 | `expds_core.fill_tin` (Delaunay-заполнение DEM) | lidar.py:81 → accel.py:202 | 3,55 с на 6 вызовах (крошечные LAS!) | масштабируется с LAS; проверить rayon-потоки в контейнере |
| 3 | `expds_core.las_grid` | accel.py:126 | 2,0 с | чтение LAZ + сетка |
| 4 | checkpoint/write_geojson (`mapping()`+round по точке) | writer.py:23 | 2,25 с на 8350 объектов | чистый Python на каждой вершине |
| 5 | создание smp.Unet со случайной инициализацией перед load_state_dict | infer_nn.py:75–81 | ≈1,3–2 с на 8 моделей | kaiming/normal_ впустую → meta-device + assign=True |
| 6 | `circle_3857` по одной точке (pyproj CRS + buffer + to_crs) | raster.py:151, centers.py:252 | 1,15 с на 7933 круга | можно одним векторным вызовом shapely/pyproj |
| 7 | импорт torch._dynamo/custom_ops при загрузке | — | ≈3 с разово | выяснить, кто тянет |
| 8 | GDAL без GDAL_NUM_THREADS / GDAL_CACHEMAX | Dockerfile:15 | (меряю на Ae 029, 15k×16k zstd) | декодирование блоков однопоточное |

DTM/рельеф/чтение растров — зона bot-8: мои цифры выше (п.2,3,8), подробнее не копаю, кроме GDAL-переменных окружения.
Я беру: п.1 (GPU-накопление), 4 (writer), 5 (загрузка моделей), 6 (круги), постобработку probs_to_features/mask_to_polygons, рецепт GPU.
