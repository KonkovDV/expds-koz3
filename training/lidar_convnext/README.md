# Обучение ConvNeXt-лидара (курганы лидара финальной версии)

Курганы лидара финальной версии (`configs/default.toml`, `[li_cnx]`) — среднее логитов десяти сетей: U-Net с
энкодером ConvNeXt-T (инициализация ImageNet, `timm convnext_tiny.fb_in1k`) и головой CenterNet, по пять фолдов
семей участков на двух входах:

| Вариант | Каналы рельефа (сетка 0,5 м) | Веса |
|---|---|---|
| `c5-in` | `lrm_s`, `lrm_l`, `slope`, `svf_r`, `opp_r` | `models/cnx/weights_cnx_c5-in_f{0..4}.pt` |
| `c12-in` | c5 + `opn_r`, `tpi5_r`, `tri3_r`, `curv_r`, `slrm8_r`, `dens_r`, `int_r` | `models/cnx/weights_cnx_c12-in_f{0..4}.pt` |

Сети обучены в ноябре 2026 ядрами Kaggle (GPU T4, 12 000 итераций, окно 512 px, батч 8, AdamW 3·10⁻⁴,
EMA 0,999, смешанная точность). Ядра собраны из одного шаблона и десяти правок. Здесь лежит всё, чтобы собрать их
заново.

## Состав

| Файл | Что это |
|---|---|
| `convnext_main.py` | шаблон ядра обучения: данные, модель, потери, обучение, OOF-инференс, веса в формате контракта инференса |
| `make_kernel.py` | сборщик ядра: встраивает константы, фолды и вендор на место `# @@EMBED@@` и применяет правки 1–9 (описаны в docstring) |
| `folds_li_59.json` | фолды семей: 59 участков с лидаром, семья и кластер соседних участков целиком в одном проверочном фолде |
| `vendor/vendor_manifest.json` | sha256 каждого файла вендора ядра обучения |
| `vendor/centerhead/`, `vendor/cnx_head.py` | голова CenterNet, цели, потери и декодирование в версии обучения |
| `prep12.py`, `make_dataset12.py` | двенадцать каналов c12 из LAS и папка входов для ядра |
| `cnx_contract.py`, `validate_oof.py` | граф сети в формате контракта и проверка OOF-выхода |

Остальные файлы вендора берутся из репозитория: код каналов — `inference/cnx/vendor` (тот же, что у инференса),
метрика — `materials/compute_metrics.py`. Сборщик сверяет каждый файл с манифестом, а тест
`tests/test_cnx_training_kit.py` проверяет манифест и сборку обоих вариантов.

Ядро, собранное из репозитория, функционально совпадает с ядрами 05.11, которые обучили веса: тот же код
вендора, те же фолды и правки 1–9. Код вендора сверен с вендором тех ядер по синтаксическим деревьям: отличаются
только комментарии, строки документации и тексты двух сообщений об ошибках. Два вспомогательных скрипта анализа
радиусов, которые в обучении не участвовали, в комплект не входят. Расходятся ещё строки журналов и метаданных и
выбор числа воркеров загрузчика; правка 1 по умолчанию задаёт `CNX_WORKERS=0`, как в ядрах 05.11. Обучение на GPU со смешанной точностью побитно не воспроизводится. Поэтому повторное обучение даст
другие веса того же рецепта, а не побайтно те же.

## Порядок

Нужны: GPU с CUDA, интернет для pip-пинов ядра (`timm 1.0.29`, `segmentation-models-pytorch 0.6.0`,
`rvt-py 2.2.1` и др.) и весов ImageNet, обучающая выборка КОЗ №3.

1. **Эталон Li.** Файл эталона трейна в EPSG:3857 (`python -m expds.gt --root DATA --out work/gt_train.geojson`;
   на Kaggle ядро ищет файл с именем `gt_train_v2.geojson` среди подключённых датасетов).
   Без файла ядро читает LAS и разметку участков прямо из S3 организаторов: ключи — в `KOZ3_S3_ACCESS_KEY_ID` и
   `KOZ3_S3_SECRET_ACCESS_KEY`.
2. **Каналы c5.** Режим prep ядра строит `cnx_relief/{участок}.npy|.json` и `manifest.json` из LAS. Для этого
   нужен S3, и файл эталона не подключается:

   ```bash
   python training/lidar_convnext/make_kernel.py --jobs c5-in:0 --mode prep --slug koz3-cnx-li-prep --owner OWNER --out BUILD
   CNX_OUT=work/prep5 python BUILD/koz3-cnx-li-prep/koz3-cnx-li-prep.py
   ```

3. **Каналы c12.** Из локальной копии выборки, затем папка для ядра со сверкой первых пяти каналов с c5:

   ```bash
   python training/lidar_convnext/prep12.py --root DATA --out work/relief12
   python training/lidar_convnext/make_dataset12.py work/relief12 work/prep12 --ref work/prep5/cnx_relief
   ```

4. **Обучение.** По одному ядру на фолд (на Kaggle T4 около 1,7 ч на фолд):

   ```bash
   python training/lidar_convnext/make_kernel.py --jobs c5-in:0 --owner OWNER --out BUILD
   CNX_GT_FILE=work/gt_train.geojson CNX_PREP_DIR=work/prep5/cnx_relief CNX_OUT=work/c5_f0 \
     python BUILD/koz3-li-cnx-c5-in-0/koz3-li-cnx-c5-in-0.py
   python training/lidar_convnext/make_kernel.py --jobs c12-in:0 --owner OWNER --prep-dataset OWNER/koz3-cnx-li-prep12 --out BUILD
   CNX_GT_FILE=work/gt_train.geojson CNX_PREP_DIR=work/prep12/cnx_relief CNX_OUT=work/c12_f0 \
     python BUILD/koz3-li-cnx-c12-in-0/koz3-li-cnx-c12-in-0.py
   ```

   Так же — фолды 1–4. На Kaggle: `kaggle kernels push -p BUILD/<slug>`. Каналы и эталон подключаются
   датасетами, их имена — в `kernel-metadata.json`.
5. **Веса и пороги.** Каждое ядро пишет веса в формате контракта инференса (`weights_cnx_<вариант>_f<k>.pt`)
   и OOF-объекты своего проверочного фолда. Схему OOF и балл официальной метрики проверяет
   `validate_oof.py`; «идеальные» круги в эталонах должны дать около 1,0:

   ```bash
   python training/lidar_convnext/validate_oof.py work/c5_f0/oof_cnx_c5-in_f0.geojson work/c5_f0/gt_li_cnx_c5-in_f0.geojson
   ```

   Пороги группы `li_cnxin` — медиана порогов фолдов по официальной метрике (`python -m expds.metric sweep`).
   OOF ядро считает со смешанной точностью (`infer.autocast` в контракте весов), а решение — в FP32.
