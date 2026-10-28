# Kaggle-ядра 28.10.2026 (бот 9, 2×T4)

Ядра собираются из `src/` скриптом `build.py`: он встраивает `FOLDS` (у верификатора ещё и `TABLE_B64`). Поэтому линтер видит в `src/*.py` «неопределённые» `FOLDS`/`TABLE_B64` — так задумано.

| Каталог | Что | Статус |
|---|---|---|
| `nolidar_verifier/` | верификатор «объект/фон» для участков без лидара (см. `docs/bots/2026-10-28/bot-9/NOLIDAR_VERIFIER.md`) | собранное ядро не коммитится: в нём таблица кандидатов, производная от эталона. Пересобрать: `build_candidates.py` → `add_gt.py` → `build.py` |
| `dfine_spor/` | D-FINE на спутнике и ортофото, `smoke/` и `full/` (см. `bot-9/DFINE.md`) | только обучение и замер; в сдачу — после письма организаторов о предобученных весах |
| `matanerdy_repro/` | своя реализация по описанному рецепту MataNerdy (DeepLabV3+ R34, без предобучения) | отменено (`bot-9/REPRO.md`); чужой код не копировался |
| `01_kurgan_subclass/`, `notebooks/` | подкласс кургана; ячейки ноутбуков k01 (`notebooks/k01/c*.py` склеивает `build.py`, конфиги — `configs/2026-10-28/bot-9/`) | отклонено (AUC 0,6) |
| `bench_t4/` | замер тайлов бота 8 на T4 (`src/bench_main.py`, `build.py`, HF-конфиги D-FINE-L и RT-DETRv2, `logs/` с проверками на CPU); см. `docs/bots/2026-10-28/bot-9/BENCH_T4.md` | подготовлено, не запущено. Собранное ядро `koz3-bench-t4.py` и pip-отчёты `dry_report*.json` не коммитятся: `build.py` пересобирает ядро из `src/`, `kernel/bench_tile.py` и конфигов |
| `convnext_lidar/` | ConvNeXt по лидару (centerhead) на T4, smoke `cnx-li-smoke` (см. `bot-9/CONVNEXT.md`) | в очереди после D-FINE full_rest |
| `convnext_lidar_tpu/` | 4 ядра для TPU v5e-8 (`smoke`, `sc`, `c3sc`, `c5sc`), собираются из GPU-исходников (см. `bot-9/CONVNEXT_TPU.md`) | собрано и проверено на CPU/XLA-CPU, не запущено; закрыть, если TPU >1,5× медленнее T4 |
| `dfine_spor/full`, `dfine_spor/full_rest`, `dfine_spor/fix` | D-FINE-L: фолды 4,2,1 (запущено 28.10 22:39 МСК), фолды 0,3 (после COMPLETE full), правка чипов | идёт обучение |
| `runs_out/` | логи и JSON-метрики прогонов с Kaggle (без геометрии и весов) | — |

**Ключи S3.** Порядок: Kaggle Secrets `KOZ3_S3_ACCESS_KEY_ID`/`KOZ3_S3_SECRET_ACCESS_KEY` → env `AWS_*` → `/kaggle/input/**/cred.json` (приватный датасет; поля `id`/`key`). Значения в лог не пишутся. В репозитории ключей нет.

Ядра работают на Kaggle T4 и не входят в 30-минутный прогон контейнера на 4090.

Это снимок рабочего каталога бота 9. Скрипты сборки рассчитаны на его раскладку (`configs/` рядом, `/workspace/expds-koz3/notes/...`). Из этого каталога без правки путей они не запустятся.
