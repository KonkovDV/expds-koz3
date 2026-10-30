# w2_datasets — описания приватных Kaggle-датасетов W2 (копии `DATASETS.md` + `dataset-metadata.json`)
Каждая папка = датасет `elenakonkova/<папка>`. Данные — на Kaggle (не в git: правило ТР — конкурсные данные не коммитятся). Проверено 30.10 ~12:45 MSK: все 5 `ready`, список файлов и размеры на Kaggle = стейджинг бокса 1:1.
| Датасет | Файлов на Kaggle (вкл. DATASETS.md) | Размер (несжато) |
|---|---|---|
| koz3-w2-oof | 93 | 0,99 ГБ |
| koz3-w2-weights | 25 | 1,47 ГБ |
| koz3-w2-maps | 296 | 2,11 ГБ |
| koz3-w2-gate | 55 | 6,39 ГБ (загружалось 5,43 ГБ с .gz; Kaggle распаковал `stack_arch.geojson`, `gt_train_v2.geojson`) |
| koz3-w2-demfill | 56 | 0,18 ГБ |
Скачать: `orch/pool_kaggle.sh elenakonkova datasets download -d elenakonkova/<slug> -p /workspace/koz3/w2/restore/<slug> --unzip`.
Здесь копии исправлены (строка про koz3-w2-demfill «появится после P-W2-10» → готов); на Kaggle в DATASETS.md 4 датасетов эта строка устарела — источник истины этот git. Новая версия датасета: положить файлы + этот `dataset-metadata.json` в папку → `pool_kaggle.sh elenakonkova datasets version -p <папка> --dir-mode zip -m "..."`.
