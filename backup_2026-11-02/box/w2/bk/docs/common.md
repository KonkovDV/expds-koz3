## Общее (для любого датасета W2)
- Проект KOZ-3, поток W2 (лидар, ConvNeXt c3-sc, тег `[li_cnx]`). Код и записка: GitHub `KonkovDV/expds-koz3`, ветка `bots/w2-lidar`, файл `HANDOFF_W2.md` (раздел «Данные и как ими пользоваться»).
- Скачать на бокс: `orch/pool_kaggle.sh elenakonkova datasets download -d elenakonkova/<slug> -p /workspace/koz3/w2/restore/<slug> --unzip` (обычный `kaggle` CLI на боксе не авторизован; токены не печатать).
- Подключить к ядру: в `kernel-metadata.json` → `"dataset_sources": ["elenakonkova/<slug>"]`; в ядре файлы в `/kaggle/input/<slug>/...` (на новых образах Kaggle бывает `/kaggle/input/datasets/elenakonkova/<slug>/...` — искать `glob('/kaggle/input/**/<имя файла>', recursive=True)`). Датасеты приватные: ядро должно быть на elenakonkova или датасет надо расшарить.
- Kaggle при загрузке распаковывает .gz → файлы лежат несжатыми (`stack_arch.geojson`, а не `.gz`).
- Все геометрии: GeoJSON, CRS EPSG:3857 (`original_crs` — исходная UTM). Свойства детекций: `class_name, confidence, region_name, markup_type="li", threshold_group, fold`.
- Фолды: 5 региональных фолдов (`region_fold.json` в каждом каталоге ядра). Сиды s0/s1/s2 — одинаковый конфиг c3-sc (ConvNeXt-tiny, 3 канала рельефа, centerhead), разный seed.
- Метрика: F1 KOZ-3 (`materials/compute_metrics.py`, попадание: покрытие ≥ 0,5 и внутренняя точка в эталоне). Гейт honest-10: база A=0,314615; PASS ⇔ Δ≥+0,003, P(B>A)≥0,9, полы по классам (−0,002 kurgany_povr/tselye/gorodishcha, −0,005 прочие), new LAS и «без лидара» ≥ −0,001.
- Итог по всем W2-линиям: ни одна не прошла (c3-sc s2 Δ−0,0162; s012avg Δ−0,0071; вариант b Δ−0,0195; li_hm ранний стоп FAIL). Эти данные — для воспроизведения и новых экспериментов, не готовое улучшение.
