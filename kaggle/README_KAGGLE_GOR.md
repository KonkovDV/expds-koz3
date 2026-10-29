# Городища без лидара: виньетки SpOr + DINOv2 (bot-5)

**ТОЛЬКО ЧЕСТНЫЙ ЗАМЕР. Ничего не идёт в сдачу до письма организаторов.** Запуски на Kaggle — только bot 9.

## Что это
1. `scripts/gor_patch/extract_feats.py` — виньетки 384 м (шаг 192 м) + контекст 768 м с тем же центром, по 224 px,
   замороженный DINOv2 (Apache-2.0), признак = [CLS, mean token] обеих шкал. Метки не используются.
   Каждый tif читается целиком один раз на шкалу (resampling=average), растяжка 2–98 % по файлу.
   Env: `DATA OUT MODEL DEVICE TAG THREADS LIMIT_FILES` (LIMIT_FILES>0 пишет `*_test.*`).
2. `scripts/gor_patch/gor_patch_eval.py` — только на боксе bot-5 (нужен `work/hillfort/gor_scene.py`, база A, кривые):
   метки виньеток из SpOr_городища, чистка Landauer 2025 (p<0,85 pos / p>0,05 neg на внутреннем OOF по семьям),
   logreg на PCA-64, декод: компонента виньеток → пересечение → polylabel → круг 10 м → официальная метрика honest-10,
   П1–П5 (EXPERIMENTS.md). Гиперпараметры заданы заранее, не подбирались.

## Версии
`kaggle/requirements_gor.txt`: transformers==5.21.0 (пин; после загрузки `assert_no_meta`, затем `.to(device)`),
rasterio==1.5.2, pyproj==3.8.0, shapely==2.2.0. torch на Kaggle не трогать.

## Kaggle (bot 9)
`kaggle/gor_patch_kaggle.ipynb`: smoke на 3 файлах → полный прогон `MODEL=facebook/dinov2-base DEVICE=cuda TAG=_b`
→ `gor_feats_b.tgz` (feats_b.npz + meta_b.json). Вернуть bot-5.

## Бокс
`tar xzf gor_feats_b.tgz -C analysis/gor_patch && FEATS_TAG=_b python scripts/gor_patch/gor_patch_eval.py`
→ `analysis/gor_patch/gor_patch_eval_b.json`. Принимать только по П1–П5, primary = вариант с чисткой.
