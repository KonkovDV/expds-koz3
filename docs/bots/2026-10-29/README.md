# Выгрузка 29.10.2026 — отчёты, код и Kaggle-ядра ботов 2–10

Дельта-снимок `/workspace/koz3_shared` на 29.10.2026 ~07:30 МСК: только файлы, которых нет в репозитории (сверка по хешу содержимого со всем `main`). Снимок 28.10 — [`../2026-10-28/`](../2026-10-28/). Статус и прогноз — [`../../STATUS_2026-10-29.md`](../../STATUS_2026-10-29.md).

Раскладка (каталоги ботов сохранены; `work`→`runs`, `data`→`data_`, `cache`→`cache_`):

- `docs/bots/2026-10-29/bot-N/` — отчёты `.md`, логи `.log/.txt`;
- `analysis/2026-10-29/bots/bot-N/`, `analysis/2026-10-29/speedup/` — код (`.py .sh .ps1 .toml .yaml .rs .cmd .ipynb` без выходов);
- `kaggle/2026-10-29/` — исходники Kaggle-ядер бота 9 и `kernel-metadata.json` (ключей нет, креды читаются из приватного датасета/env);
- `patches/2026-10-29/bot-N/` — `.patch/.diff`;
- `docs/research/2026-10-29/` — общий EXPERIMENTS.md (`EXPERIMENTS_shared.md`), материалы speedup.

Не входят по правилам: конкурсные данные и производные (tif, geojson, npy/npz, pkl, parquet, las/laz, png/jpg), веса (.pt .pth .safetensors .ckpt .onnx .bin, модели LightGBM), zip-сабмиты, все `.json` кроме метаданных ядер, файлы > 1 МБ, `docs/DATA_PROTOCOL.md`, код EhimenNathan (нет LICENSE), чужие репозитории (`competitors*`, `osint/`), вендорные библиотеки (`pylib`, `hub`, `.cargo`, `.rustup`), резервные копии `backup/`.

## Отчёты (.md)

- [`bot-10/PREREG_gor.md`](bot-10/PREREG_gor.md) — Предрегистрация: городища без лидара (bot-7 + p бота 5) — bot-10, 28.10.2026 23:48 МСК
- [`bot-10/PREREG_pixgbm.md`](bot-10/PREREG_pixgbm.md) — Предрегистрация PIXGBM (bot-10, 28.10.2026 23:35 МСК) — зафиксировано ДО замера bot-10
- [`bot-10/REPORT7_geo.md`](bot-10/REPORT7_geo.md) — REPORT7 — геофизика: A2+R3 (bot-3), PIXGBM (bot-3), GEO.md и LOCALIZE (bot-6). bot-10, 28.10.2026 ~23:35 МСК
- [`bot-10/REPORT8_gor.md`](bot-10/REPORT8_gor.md) — REPORT8 — городища без лидара (bot-7 + p бота 5), bot-10, 29.10.2026 00:15 МСК
- [`bot-10/REPORT9_v2.md`](bot-10/REPORT9_v2.md) — REPORT9_v2 — bot-10, 29.10.2026 (00:40–01:30 МСК)
- [`bot-10/STATUS.md`](bot-10/STATUS.md) — STATUS bot-10
- [`bot-10/work9/exp_block.md`](bot-10/work9/exp_block.md) — 29.10.2026 01:32 МСК — bot-10: пакет S+stack6, пороги pix, ORSP bot-6 (`bot-10/REPORT9_v2.md`)
- [`bot-2/EXPERIMENTS.md`](bot-2/EXPERIMENTS.md) — bot-2 — заранее объявленные эксперименты
- [`bot-2/REPORT2.md`](bot-2/REPORT2.md) — REPORT2 (bot-2): морфометрический классификатор «повреждённый / целый» курган (топ-5, п. 2)
- [`bot-2/STATUS.md`](bot-2/STATUS.md) — STATUS bot-2
- [`bot-3/STATUS.md`](bot-3/STATUS.md) — bot-3 STATUS (geophys)
- [`bot-3/backbone/VARIANTS_DECLARED.md`](bot-3/backbone/VARIANTS_DECLARED.md) — backbone — варианты, объявленные ДО замера (29.10.2026 07:18 МСК, бот 3)
- [`bot-3/geo_snap/GEO_SNAP_REPORT.md`](bot-3/geo_snap/GEO_SNAP_REPORT.md) — GEO_SNAP — GAP.md §5 на базе S (bot-3, 29.10.2026 02:16 МСК)
- [`bot-3/geo_snap/VARIANTS_DECLARED.md`](bot-3/geo_snap/VARIANTS_DECLARED.md) — GEO_SNAP — варианты, объявленные ДО замера (29.10.2026 02:01 МСК, bot-3)
- [`bot-3/pixgbm/BUDGET_PIXGBM.md`](bot-3/pixgbm/BUDGET_PIXGBM.md) — BUDGET_PIXGBM — время и память PIXGBM (бот 3, 28.10.2026 ~23:35 МСК)
- [`bot-3/pixgbm/PACKAGING_S.md`](bot-3/pixgbm/PACKAGING_S.md) — PIXGBM S — follow-up к merge-тесту bot-8: потоки, зависимости, упаковка (bot-3, 29.10.2026 02:30 МСК)
- [`bot-3/pixgbm/pc/PC_STATUS.md`](bot-3/pixgbm/pc/PC_STATUS.md) — PIXGBM на ПК — статус от bot-9 (28.10, 23:10 МСК)
- [`bot-4/RED_TEAM_v2_eb268d6.md`](bot-4/RED_TEAM_v2_eb268d6.md) — RED TEAM — кандидат v2 eb268d6 (бот 4, 29.10 ≈07:20 МСК)
- [`bot-4/REVIEW_CHECK_SUBMISSION.md`](bot-4/REVIEW_CHECK_SUBMISSION.md) — Ревью правки check_submission.py (бот 8, под PIXGBM / архив koz3_softceil_pixgbm_428257f_20b4ccea.zip)
- [`bot-4/STATUS.md`](bot-4/STATUS.md) — STATUS bot-4 (28.10.2026, MSK)
- [`bot-4/THR.md`](bot-4/THR.md) — THR — честные пороги (класс × группа) с вложенным выбором, поверх реранкера архитектуры (bot-4)
- [`bot-5/FOR_BOT9_GOR.md`](bot-5/FOR_BOT9_GOR.md) — FOR BOT 9 — городища: признаки DINOv2-B по виньеткам SpOr (bot-5, 28.10.2026, 23:45 МСК)
- [`bot-5/NOLIDAR.md`](bot-5/NOLIDAR.md) — NOLIDAR — дыра «без лидара» (bot-5, 28.10.2026, МСК)
- [`bot-5/STATUS.md`](bot-5/STATUS.md) — bot-5 STATUS (28.10.2026, 23:58 МСК)
- [`bot-5/analysis/kurg_ae/FORMAT_BOT5.md`](bot-5/analysis/kurg_ae/FORMAT_BOT5.md) — Формат обмена bot-7 ⇄ bot-5: курганы на Ae (без лидара)
- [`bot-6/GEO.md`](bot-6/GEO.md) — GEO: честная проверка object_poly etgr/terfl (Гнёздово 212–217)
- [`bot-6/ORSP.md`](bot-6/ORSP.md) — ORSP — ортофото/спутник (or/sp): крупнейшая невзятая потеря и одна поправка
- [`bot-6/PREREG_ORSP.md`](bot-6/PREREG_ORSP.md) — PREREG_ORSP — предрегистрация правки для or/sp (bot-6)
- [`bot-6/REPORT.md`](bot-6/REPORT.md) — bot-6 — канал оптики / аэрофото 1940-х (ae): разбор «0,09 / 0,013»
- [`bot-6/STATUS.md`](bot-6/STATUS.md) — bot-6 STATUS — канал оптики / аэрофото 1940-х
- [`bot-7/REPORT4.md`](bot-7/REPORT4.md) — REPORT4 — кандидат v2 ветки сдачи (bot-7, 29.10.2026 ~00:40 МСК; обновлено 00:55 МСК — twins ВКЛ, af97af0; обновлено 02:50 МСК — PIXGBM bot-
- [`bot-7/STATUS.md`](bot-7/STATUS.md) — bot-7 STATUS (геометрия под метрику)
- [`bot-7/runs/gor/FORMAT.md`](bot-7/runs/gor/FORMAT.md) — FORMAT — обмен кандидатами городищ (bot-7 → bot-5 → bot-10), 28.10.2026
- [`bot-7/runs/gor/README_bot10.md`](bot-7/runs/gor/README_bot10.md) — Для bot 10: как мерить кандидатов городищ bot-7 (task 6) — обновлено 28.10.2026, 23:40 МСК
- [`bot-7/runs/gor/comb/status_add.md`](bot-7/runs/gor/comb/status_add.md) — Задача 6, дополнение 2: финальный meta.json bot-5, его p и комбинированный балл (28.10.2026, 23:40 МСК)
- [`bot-8/BUDGET.md`](bot-8/BUDGET.md) — BUDGET.md — бюджет контейнера на RTX 4090 (24 ГБ), лимит 30 мин
- [`bot-8/DISK_CLEANUP_PROPOSAL.md`](bot-8/DISK_CLEANUP_PROPOSAL.md) — Disk cleanup proposal (bot-8, 2026-10-28 23:40 MSK) — NOTHING DELETED
- [`bot-8/FROM_BOT9_platform_sim_v2.md`](bot-8/FROM_BOT9_platform_sim_v2.md) — bot-9 → bot-8: платформенная симуляция 4c3e29e + twins — ГОТОВО (29.10 00:21 МСК)
- [`bot-8/STATUS.md`](bot-8/STATUS.md) — bot-8 — STATUS (надёжность и скорость сдачи)
- [`bot-8/bench/acc_gpu_T4.md`](bot-8/bench/acc_gpu_T4.md) — §7.12 (5) — замер на T4 (для бота 9; бот 8 его не запускал, GPU на box нет)
- [`bot-8/bench/las_grid_fast_result.md`](bot-8/bench/las_grid_fast_result.md) — lidar_a: ground.rasterise → Rust expds_fast (las_grid_fast), бот 8, 28.10.2026 (МСК)
- [`bot-9/CNX_CONTRACT_FOR_BOT8.md`](bot-9/CNX_CONTRACT_FOR_BOT8.md) — Контракт ConvNeXt-лидар (bot-9) ↔ стек bot-8 — ответ боту 8
- [`bot-9/CONVNEXT.md`](bot-9/CONVNEXT.md) — CONVNEXT.md — ядра Kaggle: ConvNeXt-T U-Net по лидару (KOZ3 №3)
- [`bot-9/CONVNEXT_TPU.md`](bot-9/CONVNEXT_TPU.md) — ConvNeXt-T U-Net (лидар, scratch c3sc/c5sc) — вариант для Kaggle TPU
- [`bot-9/DFINE.md`](bot-9/DFINE.md) — D-FINE-L на SpOr (детектор рамок): ядро Kaggle (бот 9)
- [`bot-9/PC_RESULTS.md`](bot-9/PC_RESULTS.md) — PC_RESULTS (ПК DESKTOP, GTX 1650) — бот-9
- [`bot-9/STATUS.md`](bot-9/STATUS.md) — STATUS bot-9 (Kaggle-очередь)
- [`bot-9/for_bot8/.CNX_CONTRACT_FOR_BOT8.before_s6.md`](bot-9/for_bot8/.CNX_CONTRACT_FOR_BOT8.before_s6.md) — Контракт ConvNeXt-лидар (bot-9) ↔ стек bot-8 — ответ боту 8
- [`../../research/2026-10-29/EXPERIMENTS_shared.md`](../../research/2026-10-29/EXPERIMENTS_shared.md) — KOZ3 — журнал экспериментов (общий; ведёт bot-10, дописывать строками)
- [`../../research/2026-10-29/speedup/FOR_BOT8.md`](../../research/2026-10-29/speedup/FOR_BOT8.md) — Для бота 8: ветка speedup-2026-10-28 (коммит 6602bb2, база 14d419a)

## Файлы по каталогам

- `analysis/2026-10-29/…` — 172
- `docs/bots/…` — 269
- `docs/research/…` — 19
- `kaggle/2026-10-29/…` — 175
- `patches/2026-10-29/…` — 17

## Ветки (push 29.10, без force)

Fast-forward: `bots/bot3-geophys` b8cfdf4→72ceb28, `bots/bot3-pixgbm` 7cdace5→76f7317, `bots/bot8-cpu-a-clean` cb4742e→95bc0a6.

Новые: `bots/bot2-li-real` 2871697, `bots/bot3-backbone` 76f7317, `bots/bot3-geosnap` 7a5ee57, `bots/bot5-nopretrain` 43d0163, `bots/bot5-rfdetr-coco` 97a9f21, `bots/bot6-orsp` c050028, `bots/bot7-candidate-v1` 913e2be, `bots/bot7-candidate-v2-old-fe0e570` fe0e570, `bots/bot7-candidate-v3pre` 8a6ce89, `bots/bot7-candidate-v2-nomodels` 1b62124 (дерево eb268d6 без весов PIXGBM), `bots/bot8-cnx` 428257f, `bots/bot8-softceil-pixgbm` 428257f, `bots/bot8-soft-ceiling` 26e12ae, `bots/bot8-cpu-a-clean-on-v2` 18b2f66, `bots/bot8-reliability-ro` 8343483, `bots/bot10-tools` a839da2.

Не отправлены: `bot7-candidate-v2` / `bot7-candidate-v3` @ eb268d6 — в истории модели LightGBM PIXGBM (`models/pixgbm/all/*.txt`, по 2,1 МБ); `bot3-ehimen` и `bot8-cpu-a-speedup` — код EhimenNathan без LICENSE (DO_NOT_DISTRIBUTE). Ветки, уже совпадающие с origin, не трогались.

## Исключено при выгрузке файлов

- `bot-8/bench/bench_cpu_a.py` — код EhimenNathan без LICENSE (DO_NOT_DISTRIBUTE / построчный порт)
- `bot-8/patch/bot8-cpu-a-speedup.NOT_FOR_DISTRIBUTION.diff` — код EhimenNathan без LICENSE (DO_NOT_DISTRIBUTE / построчный порт)
- `bot-8/patch/bot8-cpu-a-speedup/0003-li_a-6-DtmPool-DTM-LAS-LPT-RAM-LAS-.npy-tmpfs-predic.patch` — код EhimenNathan без LICENSE (DO_NOT_DISTRIBUTE / построчный порт)
- `bot-8/patch/bot8-cpu-a-speedup/DO_NOT_DISTRIBUTE.md` — код EhimenNathan без LICENSE (DO_NOT_DISTRIBUTE / построчный порт)
- `bot-8/patch/bot8-cpu-a-speedup/0005-li_a-5-hm_acc-LI_A_ACC-device_exact-upstream-device-.patch` — код EhimenNathan без LICENSE (DO_NOT_DISTRIBUTE / построчный порт)
- `bot-8/patch/bot8-cpu-a-speedup/0004-li_a-3-DTM-exact_fast-EDT-d2-int-gather-upstream-ban.patch` — код EhimenNathan без LICENSE (DO_NOT_DISTRIBUTE / построчный порт)
- `bot-8/patch/bot8-cpu-a-speedup/0002-li_a-2-9-x12-0.7-11-relief-1.5-1.8-bench-cpu_a_speed.patch` — код EhimenNathan без LICENSE (DO_NOT_DISTRIBUTE / построчный порт)

## Дополнение ~09:00 МСК (вторая выгрузка 29.10)

Дельта `/workspace/koz3_shared` по файлам, изменённым после 07:20 МСК: 537 файлов (~4,5 МБ) — новые/обновлённые отчёты (bot-10 REPORT10_accept, PREREG_audit, PREREG_scan, REVIEW; bot-4 GAP, PREREG_MORPH, PREREG_CROPVERIF, CROPVERIF, REVIEW_CHECK_SUBMISSION_CNX; bot-9 PC_RESULTS, DFINE_FINAL_ORAE, DATA_PROTOCOL_SCAN_NOTE; bot-2 REPORT2, EXPERIMENTS; bot-3 BACKBONE_REPORT, audit/VARIANTS_DECLARED; bot-6 ORSP, PREREG_ORGATE; bot-5 FOR_BOT7_GOR_V3, FOR_BOT9_RFDETR; STATUS.md ботов), логи, скрипты, исходники Kaggle-ядер бота 9 (dfine_spor и др.) и общий `EXPERIMENTS_shared.md`.

Изменение правила: теперь входят **малые `.json` с агрегированными метриками** (белый список имён: paired/stats/metrics/report/summary/counts и т. п., ≤ 120 КБ, без координат и `POLYGON`; максимум — значения по участкам/фолдам/бутстрепу). Сайдкар `rerank_arch_lgbm.json` (1,8 КБ: имена признаков, словари, пороги) — метаданные, не веса. По-прежнему не входят: данные и производные, веса и модели LightGBM (`*.txt`-дампы), zip, `kernel-metadata.json`, `kaggle.json`, `DATA_PROTOCOL.md` ботов, код EhimenNathan. Собранные ядра `kaggle/2026-10-29/bot5/gorloc/kernels/*/*.py` **не выложены**: в них встроен (gzip+base64) `kaggle/gor_dense_inputs.json` с разметкой городищ; выложены сборщик `build.py`, шаблоны `tpl/` и исходники `src_6984c67/` без этого файла.

Удалено из головы `main`: два текстовых дампа LightGBM `rerank_arch_lgbm.txt` (по 559 КБ), попавшие в 76b3540 (в истории остаются).

### Ветки (push ~08:55, только fast-forward или новые, без force)

Fast-forward: `bots/bot2-li-real` → 361a9b5, `bots/bot3-backbone` → 49d0200, `bots/bot5-gor-patch-kaggle` → 82058b8, `bots/bot5-rfdetr-coco` → c43cb08, `bots/bot8-cnx` → 1aecea9, `bots/bot8-softceil-pixgbm` → 17c2e2c.

Новые: `bots/bot2-adaf` 7e3639d, `bots/bot2-morph` e918a3b, `bots/bot3-audit` e3c4f41 (порт исправлений аудита, opt-in), `bots/bot5-gor-dense-loc-nodata` 8dfeb0b и `bots/bot5-dinov2reg-nodata` 3fd4743 (пересобраны поверх 82058b8 без `kaggle/gor_dense_inputs.json` — это разметка городищ), `bots/bot7-candidate-v3-nomodels` d420ceb (коммиты eb268d6..9138bff, перенесённые без `models/pixgbm/all/{poly,finds}.txt`; дерево отличается от 9138bff только этими файлами и `MODELS_NOT_IN_GIT.md`).

Не отправлены: оригиналы `bot7-candidate-v2` eb268d6 / `-v3` 9138bff (веса PIXGBM), `bot5-gor-dense-loc` 6984c67 / `bot5-dinov2reg` a0ac2ef (разметка), `bot3-ehimen`, `bot8-cpu-a-speedup` (код EhimenNathan без LICENSE). Ветка аудита `claude/audit-2026-10-29` уже в origin (d4edf2f, содержит 6e26062).
