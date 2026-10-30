# koz3-w2-maps — сырые карты предсказаний c3-sc, ~2,1 ГБ, 295 файлов

**Что это:** выход сети до декодирования в полигоны, по одному `.npz` на участок: `maps_cnx_c3-sc_f{K}_{участок}.npz`.
Ключи: `hm` (2×H/4×W/4 float32, теплокарта центров), `wh` (размер), `off` (смещение), `valid`/`valid_h` (маска покрытия рельефа), `mask` (3×H×W uint8, маски классов), `meta` (JSON-строка: участок, геопривязка, шаг растра).

| Путь | Что |
|---|---|
| `w2maps/s0/<ядро>/maps/…`, `w2maps/s1/<ядро>/maps/…` (+логи) | сиды 0 и 1 |
| `w2maps/s2/<ядро>/…`, `koz3-w2-cnx-c3sc-s2f23/`, `koz3-w2-cnx-c3sc-s2f4/` | сид 2 (s2f23/s2f4 — копии прямо из output ядер) |
| `koz3-w2-lihm-c3sc-f01234/` | li_hm (P-W2-7, FAIL) |

## Зачем
- Пересобрать детекции с другими порогами/NMS/усреднением сидов без GPU: усреднить `hm` нескольких сидов → декодировать → `merge_cnx.py`.
- Отладка конкретного участка (например 182_Красносельская_1.7км — 0 детекций).
- Сравнение с картами DEM fill (P-W2-10).
## Общее (для любого датасета W2)
- Проект KOZ-3, поток W2 (лидар, ConvNeXt c3-sc, тег `[li_cnx]`). Код и записка: GitHub `KonkovDV/expds-koz3`, ветка `bots/w2-lidar`, файл `HANDOFF_W2.md` (раздел «Данные и как ими пользоваться»).
- Скачать на бокс: `orch/pool_kaggle.sh elenakonkova datasets download -d elenakonkova/<slug> -p /workspace/koz3/w2/restore/<slug> --unzip` (обычный `kaggle` CLI на боксе не авторизован; токены не печатать).
- Подключить к ядру: в `kernel-metadata.json` → `"dataset_sources": ["elenakonkova/<slug>"]`; в ядре файлы в `/kaggle/input/<slug>/...` (на новых образах Kaggle бывает `/kaggle/input/datasets/elenakonkova/<slug>/...` — искать `glob('/kaggle/input/**/<имя файла>', recursive=True)`). Датасеты приватные: ядро должно быть на elenakonkova или датасет надо расшарить.
- Kaggle при загрузке распаковывает .gz → файлы лежат несжатыми (`stack_arch.geojson`, а не `.gz`).
- Все геометрии: GeoJSON, CRS EPSG:3857 (`original_crs` — исходная UTM). Свойства детекций: `class_name, confidence, region_name, markup_type="li", threshold_group, fold`.
- Фолды: 5 региональных фолдов (`region_fold.json` в каждом каталоге ядра). Сиды s0/s1/s2 — одинаковый конфиг c3-sc (ConvNeXt-tiny, 3 канала рельефа, centerhead), разный seed.
- Метрика: F1 KOZ-3 (`materials/compute_metrics.py`, попадание: покрытие ≥ 0,5 и внутренняя точка в эталоне). Гейт honest-10: база A=0,314615; PASS ⇔ Δ≥+0,003, P(B>A)≥0,9, полы по классам (−0,002 kurgany_povr/tselye/gorodishcha, −0,005 прочие), new LAS и «без лидара» ≥ −0,001.
- Итог по всем W2-линиям: ни одна не прошла (c3-sc s2 Δ−0,0162; s012avg Δ−0,0071; вариант b Δ−0,0195; li_hm ранний стоп FAIL). Эти данные — для воспроизведения и новых экспериментов, не готовое улучшение.
## Связанные источники (не эти датасеты)
| Источник | Что | Зачем |
|---|---|---|
| `elenakonkova/koz3-cnx-li-prep` | рельеф `cnx_relief/<участок>.npy` (каналы lrm_s, lrm_l, slope, svf_r, opp_r, uint8) + `.json` (геопривязка), 119 файлов, 1,6 ГБ; ядро dmitrijkv/koz3-cnx-li-prep | вход всех c3-sc/r50c5 ядер и `infer_cpu.py` (c3 берёт первые 3 канала, r50c5 — все 5) |
| `aaaa0000004/koz3-w2-gaudit-las` | LAS 015/021/186/190 (1–6 ГБ каждый), `*_Li_{gorod,povr,tselye}.geojson`, `*_UTM.json`; 16 файлов | аудит классов земли (CI2-3/P-W2-8, закрыт) и DEM fill; доступен только аккаунту aaaa0000004 |
| S3 `train-expds-3` (`https://s3.ru-7.storage.selcloud.ru`) | все LAS, карты/разметка Li | первоисточник; ключи — приватный датасет elenakonkova/koz3-s3-cred (не печатать), загрузка `w2_tools/s3get.py` |
| `elenakonkova/koz3-w2-demfill` | (готов 30.10, ≈177 МБ, 55 файлов; P-W2-10 FAIL) рельеф/карты/итог DEM fill | см. HANDOFF, п. 8 |
