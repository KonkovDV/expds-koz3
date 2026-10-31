# ConvNeXt-T U-Net (лидар, scratch c3sc/c5sc) — вариант для Kaggle TPU

Статус на 28.10.2026 22:55 MSK: **ядра собраны и проверены на боксе (CPU и XLA-CPU), ничего не запушено и не запущено, сообщений никому не отправлено.**
Папка: `bot-9/kaggle/convnext_lidar_tpu/`. GPU-папки (`kaggle/convnext_lidar/`, dfine_spor, nolidar_verifier) не изменялись: `build.py` только читает их.

## 1. Ответы по исследованию (с источниками)

| Вопрос | Ответ | Источник |
|---|---|---|
| Можно ли одновременно запускать сессии TPU и GPU | **Официального подтверждения нет.** Квоты раздельные: TPU 20 ч/нед, сессия до 9 ч, «TPUs are a separate resource». Для batch-сессий TPU лимит — **1 активная или стоящая в очереди сессия на пользователя**. То, что TPU-сессия идёт параллельно с 2×T4, надо проверить на практике: запустить smoke во время GPU-ядра и посмотреть статус обоих | kaggle.com/docs/tpu; форум Kaggle 173129; kaggle-cli issue #1197 (stevemessick, 18.09.2026 02:34 MSK) |
| Что сейчас стоит за «TPU v3-8» | v3-8 выведен из эксплуатации; CLI заменяет его на `TpuV5E8`, то есть v5e-8: 8 чипов × 1 TensorCore | kaggle CLI 2.2.5 `kaggle_api_extended.py:486-489` (`_RETIRED_ACCELERATORS`), `docs/kernels_metadata.md` |
| Значение accelerator в метаданных | `"enable_tpu": true`, `"enable_gpu": false`, `"machine_shape": "TpuV5E8"`, `"docker_image_pinning_type": "latest"`, поле `docker_image` **не указывать** | issue #1197; файлы `koz3-cnx-li-tpu-*/kernel-metadata.json` |
| Поддерживает ли `kaggle kernels push` TPU | Да, через `machine_shape` (см. строку выше). С закреплённым `docker_image` TPU-образ не поднимается (#1197), поэтому стоит pinning `latest` | issue #1197; `docs/kernels_metadata.md` |
| Версии TPU-образа | Python 3.12; torch 2.8.0 (CPU); torch_xla 2.8.0; torchvision 0.23.0; libtpu 0.0.17; TF 2.20; timm без пина; `ISTPUVM=1`. Хост: 96 CPU и 330 ГБ RAM. Фактические версии ядро пишет в `REPORT.versions` (torch_xla/libtpu/jax/jaxlib) | Kaggle/docker-python, ветка main: `tpu/config.txt`, `tpu/requirements.in`, `tpu.Dockerfile`; kaggle.com/docs/notebooks |
| Совместимость timm/smp | Чистый PyTorch, кастомных CUDA-ядер нет. Encoder `tu-convnext_tiny` прошёл forward+backward на XLA (PJRT CPU) как `convnext_atto`: тот же код timm, меньше по памяти. `aten::upsample_nearest2d_backward` (декодер smp) уходит на CPU **только на не-TPU**, на TPU он снижается в HLO. smp/timm доставляются вендором из prep, как в GPU-ядре; `_deps` ограничивает pip, чтобы он не трогал torch/torch_xla/libtpu/jax | pytorch/xla v2.8.0 `torch_xla/csrc/aten_xla_type.cpp:3789-3801` (`CheckTpuDevice`); `logs/dry_xla.log`; `logs/xla_sync_probe.log` |
| bf16 | `XLA_USE_BF16` устарел; используем `torch.autocast("xla", dtype=torch.bfloat16)`, а EMA и оптимизатор держим в fp32 | release notes PyTorch/XLA 2.6; документ r2.6 «amp» |
| Будущее PyTorch/XLA | Проект объявлен устаревшим в пользу TorchTPU, архивация репозитория 15.11.2026. Сейчас образ на 2.8 работает, но это риск | README pytorch/xla |

## 2. Устройство кода

`build.py` читает GPU-исходник `../convnext_lidar/src/convnext_main.py` (sha256 `00db7df6eb92b3b4…`). Каждый патч обязан совпасть ровно один раз, иначе сборка падает. Из `src/` подставляются:

- `tpu_step1.py`:
  - `PJRT_DEVICE`, удаление `XLA_USE_BF16`;
  - проба TPU в отдельном подпроцессе;
  - **жёсткая ошибка, если TPU нет** (кроме `--allow-cpu`).
- `tpu_step1b.py`:
  - выбор устройства, `NPROC`;
  - устройство XLA создаётся *до* self-test головы. Иначе CPU-backward до инициализации XLA ломает autograd engine: `logs/dry_xla_fail1_engine.log`, `engine.cpp:1539`.
- `tpu_train.py`, цикл обучения:
  - статические формы: crop 512, фиксированный bs 8, `drop_last=True`, на инференсе последний батч дополняется (`_pad_bs`);
  - `XAdamW`: lr и шаг — тензоры на устройстве, поэтому смена lr от warmup/cosine не вызывает перекомпиляцию;
  - `head_loss_nosync`: та же формула, что `HEAD.loss` бота 2, но без `.item()`. Тест: `logs/test_tpu_parts_cpu.log` — расхождение 0.0, XAdamW против AdamW 5.96e-08; на `xla:0` — 1.67e-06 и 2.98e-08 (`logs/test_tpu_parts_xla.log`);
  - лог только через `xm.add_step_closure` каждые `log_every` шагов. Проверка на NaN тоже в closure;
  - замеры: компиляция = первая итерация, s/it считается после `WARM_IT=3`; есть счётчик перекомпиляций после прогрева и доля ожидания данных.
- Таргеты centerhead и copy-paste строятся на CPU в DataLoader; декодирование пиков — только на инференсе, на CPU. Выход: OOF-npz, gt, ckpt, `region_fold.json`. Ключи и dtype совпадают с GPU-ядром (сравнение `dry_cpu` с GPU dry, см. §4).
- `tpu_lanes.py`, многочиповый режим: **один независимый процесс на чип, а не `xmp.spawn`/DDP.** Почему:
  1. Заданий 10, чипов 8 — каждое задание обучается отдельно, коллективы не нужны.
  2. Рецепт bs 8 остаётся идентичным GPU, поэтому OOF сравнимы.
  3. Скрипт Kaggle — верхнеуровневый код, и `spawn` переимпортировал бы его целиком.

  Как это устроено:
  - главный процесс **не импортирует torch_xla**: `_setup_default_env` выставил бы переменные топологии на все чипы;
  - он запускает воркеры с `TPU_VISIBLE_CHIPS=i`, `TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1`, `TPU_PROCESS_BOUNDS=1,1,1` и своими портами `8476+i`/`8576+i` — по образцу `torch_xla/_internal/tpu.py:230-237, 272-290` (v2.8);
  - логи пишутся в `OUT/logs_tpu/chip{i}.log`, отчёт каждого чипа — в `report_cnx_<mode>_chip{i}.json`;
  - запасной путь: `CNX_TPU_PROCS=1` — всё в одном процессе на одном чипе.
- Креды: Kaggle Secrets → env → `cred.json` датасета `dmitrijkv/koz3-s3-cred`, без изменений относительно GPU. Prep (рельеф и вендор) берётся из `kernel_sources: dmitrijkv/koz3-cnx-li-prep` (полные ядра). Smoke считает рельеф двух регионов сам.

## 3. Ядра (слаги и пути)

| Ядро (`dmitrijkv/…`) | Путь | Задания | Ключевые env | Размер |
|---|---|---|---|---|
| `koz3-cnx-li-tpu-smoke` | `kaggle/convnext_lidar_tpu/koz3-cnx-li-tpu-smoke/` | c3-sc:3, c5-sc:4 (2 чипа параллельно) | ITERS 20, LOG_EVERY 5, TPU_PROCS 2, DEADLINE_H 2.0, регионы `191_Осечки_1.3км,016_Хога`, без kernel_sources | 914294 Б |
| `koz3-cnx-li-tpu-sc` | `…/koz3-cnx-li-tpu-sc/` | c3-sc:0..4 + c5-sc:0..4 (10 заданий, 8 чипов, 2 волны) | 12000 ит., DEADLINE_H 8.3, INFER_RESERVE_H 0.6, WORKERS 6 | 914149 Б |
| `koz3-cnx-li-tpu-c3sc` | `…/koz3-cnx-li-tpu-c3sc/` | c3-sc:0..4 (5 чипов) | те же | 914109 Б |
| `koz3-cnx-li-tpu-c5sc` | `…/koz3-cnx-li-tpu-c5sc/` | c5-sc:0..4 (5 чипов) | те же | 914109 Б |

Источник размеров и хешей: `kaggle/convnext_lidar_tpu/logs/build.log` (relief_sha `6ed689552b4a9780`, centerhead `788202a1…`). Окружение задаётся в начале каждого `.py` и метаданных; все 4 файла проходят `py_compile`. В `pyflakes` нет undefined names, только неиспользуемые импорты (`logs/pyflakes.log`, запуск в venv).

**Влезает ли в 9 ч.** GPU-оценка — 0.37–0.45 s/it при bs 8 на T4 (`CONVNEXT.md:86`, по `bench_t4/validity.json`). Даже если TPU в 1.5 раза медленнее, выходит 12000 × 0.45 × 1.5 ≈ 2.25 ч на задание. Для `-sc` это 2 волны ≈ 4.5 ч + инференс, меньше дедлайна 8.3 ч. Кроме того, бюджетный контроль в `train_job` после `min(100, iters)` итераций сам урезает число итераций (не ниже `min_iter_frac`), если не успевает. Поскольку разрешена 1 batch-сессия TPU (#1197), выгоднее одно ядро `-sc` (8 чипов заняты), чем пара c3sc/c5sc подряд (по 5 чипов из 8).

## 4. Проверка на боксе (2 потока, ≤2 ГБ, `nice -n 10`, `_scratch/run_guard.sh`)

| Проверка | Результат | Лог |
|---|---|---|
| Сборка | 4 ядра, все патчи применены ровно один раз, sha вендора сверен | `logs/build.log` |
| Импорт torch_xla 2.8 (PJRT CPU) | OK | `logs/venv_install.log`, `logs/xla_cpu_probe.log` |
| Части цикла (лосс, XAdamW, pad) CPU / XLA | OK, расхождения в §2 | `logs/test_tpu_parts_cpu.log`, `logs/test_tpu_parts_xla.log` |
| Сухой прогон: собранный smoke, CPU-путь, главный процесс + 1 воркер-чип, атто-encoder, crop 128 | rc 0; «ИТОГ воркера: OK»; npz/OOF/gt/ckpt/region_fold/отчёты чипа есть; пик RSS 1,372,676 КБ | `logs/dry_cpu.log`, `logs/dry_cpu.rss` |
| Сухой прогон: XLA (PJRT CPU), 1 процесс, 10 ит. | rc 0; компиляция + 1-я ит. 35.0 с; steady 1.4016 s/it; **0 перекомпиляций после прогрева**; data_wait 0.007; инференс OK | `logs/dry_xla.log:75-77`, `logs/dry_xla.rss` |
| Синхронизации хоста в шаге обучения | `aten::_local_scalar_dense` = **0** на forward, loss, backward, opt и sync за 2 шага. Счётчик 48 в `dry_xla.log` дают `float()` в step-closure (после исполнения графа, вне трассировки) и инференс/метрики, а не цикл | `tests/xla_sync_probe.py`, `logs/xla_sync_probe.log` |
| grep S3-ключей по папке TPU | 2 значения проверены, 61 файл, совпадений 0 (значения не печатались) | `logs/key_grep.log` |

Ограничения проверки:
- Два воркера одновременно на боксе не проверены: убито на 2,006,164 КБ из-за лимита 2 ГБ.
- Цифры XLA-CPU ничего не говорят о скорости TPU; вердикт «ЗАКРЫТЬ» в `dry_xla.log` получен на CPU и не информативен.
- `dry_cpu.log` записан до исправления нормировки T4 на bs 8, поэтому его блок «TPU vs T4» устарел. В `dry_xla.log` нормировка уже правильная.

## 5. Команды push (НЕ выполнялись; сначала smoke)

```bash
cd /workspace/koz3_shared
# 1) smoke: 2 чипа × 20 ит., отдельно время компиляции и steady s/it, блок «TPU vs T4»
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p bot-9/kaggle/convnext_lidar_tpu/koz3-cnx-li-tpu-smoke
~/.local/bin/kaggle kernels status dmitrijkv/koz3-cnx-li-tpu-smoke
# 2) только если вердикт smoke не «ЗАКРЫТЬ» и prep-ядро koz3-cnx-li-prep COMPLETE:
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) ~/.local/bin/kaggle kernels push -p bot-9/kaggle/convnext_lidar_tpu/koz3-cnx-li-tpu-sc
#    либо по очереди (1 batch-сессия TPU): koz3-cnx-li-tpu-c3sc, затем koz3-cnx-li-tpu-c5sc
```

Для точного сравнения надо подставить реальный s/it GPU-smoke: env `CNX_T4_S_PER_IT=<a>,<b>` (по умолчанию `0.37,0.45` из `CONVNEXT.md:86`) или пересобрать с правкой `T4_S_PER_IT_DEFAULT` в `build.py`.

## 6. Критерий

**Если TPU медленнее T4 больше чем в 1.5 раза на сэмпл, закрываем.** Smoke печатает `TPU vs T4`:
- медиана s/sample на чип;
- `ratio_per_device` против каждой оценки T4;
- пропускная способность сессии (8 чипов) против 2×T4;
- время компиляции.

Вердикт:
- «ЗАКРЫТЬ», если min ratio > 1.5;
- «ПОГРАНИЧНО», если превышает только верхняя оценка;
- «ПРОДОЛЖАТЬ» в остальных случаях.

## 7. Риски

1. **Образ.** Pinning `latest` (#1197) означает, что версия torch_xla/libtpu может смениться без предупреждения. Фактические версии пишутся в отчёт; несовпадение libtpu и torch_xla проявится падением пробы TPU, то есть жёсткой ошибкой в начале.
2. **v5e, а не v3.** Другая архитектура и объём памяти на чип (по докам Kaggle не проверял). Скорость на ConvNeXt (depthwise 7×7, LayerNorm) заранее неизвестна, и это главный вопрос smoke.
3. **Перекомпиляции.** На XLA-CPU их 0, но на TPU нужно смотреть `recompiles_after_warm` в отчёте. Инференс компилирует отдельный граф (один раз на форму).
4. **Только 1 batch-сессия TPU** (#1197). Ядра TPU идут строго по очереди; одновременность с GPU не подтверждена (§1).
5. **Зависимость от prep.** Полные ядра требуют готового `koz3-cnx-li-prep`; smoke от него не зависит.
6. **Многопроцессный env** (`TPU_VISIBLE_CHIPS` и порты) на Kaggle не проверен. Если воркеры не стартуют, fallback — `CNX_TPU_PROCS=1` (1 чип, медленнее в 8 раз по сессии).
7. **Нагрузка на хост.** 8 воркеров × 6 DataLoader-процессов = 48 процессов на 96 CPU / 330 ГБ, должно хватить. Copy-paste и таргеты на CPU могут стать узким местом; смотреть `data_wait_frac`.
8. **Численность bf16.** Autocast против AMP fp16 на GPU — OOF может немного отличаться; EMA и оптимизатор в fp32.
9. **Эталон T4 — оценка**, а не замер этого же ядра (§5, `CNX_T4_S_PER_IT`).
10. **Fallback upsample-backward на CPU** есть только на не-TPU (`aten_xla_type.cpp:3799`). На TPU его быть не должно; если счётчик `aten::upsample_nearest2d_backward` в отчёте TPU ненулевой, это сигнал проблемы.
11. **Два параллельных воркера на боксе не проверены** (лимит 2 ГБ); проверен главный процесс + 1 воркер.
12. **Устаревание PyTorch/XLA** (архив 15.11.2026). На горизонте недель это не проблема, но долгосрочно путь тупиковый.
13. Мелочь: в шапке собранного файла комментарий «smoke (10 ит.)» унаследован от GPU; фактически smoke-ядро берёт `CNX_ITERS=20` из env в начале файла.

## Наблюдение рутины 28.10 23:05 МСК
`koz3-cnx-li-tpu-smoke` (пуш 22:54 МСК) стоит в **QUEUED** ~11 мин, пока GPU-ядро `koz3-dfine-spor-full` в RUNNING. Параллельный старт TPU с GPU пока не подтверждён; перепроверка на следующем прогоне.
- 28.10 23:20 МСК: smoke всё ещё QUEUED (~26 мин), GPU dfine-spor-full RUNNING → параллельный старт TPU с GPU не наблюдается.
- 28.10 23:36 МСК: smoke всё ещё QUEUED (~42 мин), GPU dfine-spor-full RUNNING; koz3-cnx-li-prep COMPLETE.
- 28.10 23:38 МСК: smoke всё ещё QUEUED (~44 мин), GPU dfine-spor-full RUNNING → параллельного старта TPU с GPU по-прежнему нет.
- 29.10 01:01 МСК: smoke всё ещё QUEUED (~2 ч 07 мин), GPU dfine-spor-full RUNNING → TPU параллельно с GPU не стартует.
- 29.10 01:26 МСК: smoke всё ещё QUEUED (~2 ч 32 мин), GPU dfine-spor-full RUNNING → TPU параллельно с GPU не стартует.
- 29.10 01:41 МСК: smoke всё ещё QUEUED (~2 ч 47 мин), GPU dfine-spor-full RUNNING (~3 ч 01 мин) → TPU параллельно с GPU не стартует.

## Наблюдение рутины бота 1 (29.10 01:58 МСК)
- koz3-cnx-li-tpu-smoke (запушен 28.10 22:54 МСК) всё ещё QUEUED, ≈3 ч. Параллельно с GPU он не пошёл: всё время работы GPU-ядра koz3-dfine-spor-full ждал машину TPU в очереди.
- koz3-cnx-li-prep COMPLETE. koz3-dfine-spor-full COMPLETE, сейчас на GPU идёт koz3-dfine-spor-full-rest (запущен 01:57 МСК).
