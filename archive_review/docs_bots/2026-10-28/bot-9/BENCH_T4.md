# BENCH_T4 — ядро Kaggle для замера тайлов бота 8 на T4 (KOZ3 №3)
bot-9, 28.10.2026 ~21:40 МСК. **Ядро подготовлено, НЕ запушено и НЕ запущено.** Ничего не отправлялось в GitHub/Kaggle.

## 1. Где что лежит
| Что | Путь |
|---|---|
| Папка ядра (для push) | `/workspace/koz3_shared/bot-9/kaggle/bench_t4/kernel/` |
| Slug / title | `dmitrijkv/koz3-bench-t4` / `koz3-bench-t4` (`kernel/kernel-metadata.json`) |
| Код ядра (code_file) | `kernel/koz3-bench-t4.py` (53 712 байт, собран `python3 build.py`) |
| Исходник обвязки | `bench_t4/src/bench_main.py`; сборщик `bench_t4/build.py` (по образцу `kaggle/nolidar_verifier/build.py`, та папка не менялась) |
| Скрипт бота 8, копия байт-в-байт | `kernel/bench_tile.py`, sha256 `70b37afb82e124a69212de6a377b0a35584884bdfe6a06eabbb64b33c27079ea` = оригиналу (`cmp` и `sha256sum`; `build.py` падает, если копия разошлась) |
| Вшитые HF-конфиги | `kernel/hf_configs/dfine_l/config.json`, `kernel/hf_configs/rtdetrv2_r50/config.json` |
| Логи проверок на box | `bench_t4/logs/` |

Метаданные (`kernel/kernel-metadata.json`): `enable_gpu=true`, `machine_shape="NvidiaTeslaT4"`, `enable_internet=true`, `is_private=true`, `dataset_sources=[]` (S3 и секреты не нужны, веса случайные), `docker_image` взят из шаблона nolidar_verifier (`…python@sha256:9a6bcfda…`; тот же образ стоит в уже выполненных ядрах `/workspace/kpull/s1,s3,s4,s5/kernel-metadata.json`).

Kaggle-скрипт заливает только `code_file`, поэтому `bench_tile.py` (base64 + sha256) и оба `config.json` (base64 + sha256) вшиты в `koz3-bench-t4.py`. На старте ядро пишет их в `/kaggle/working` и сверяет sha256; при расхождении останавливается.

## 2. Команда запуска (делает человек/родитель, не бот 9)
```
kaggle kernels push -p /workspace/koz3_shared/bot-9/kaggle/bench_t4/kernel
```
После завершения:
```
kaggle kernels output dmitrijkv/koz3-bench-t4 -p /workspace/kaggle_out/koz3-bench-t4
mkdir -p /workspace/koz3_shared/bot-9/bench_t4 && cp -r /workspace/kaggle_out/koz3-bench-t4/. /workspace/koz3_shared/bot-9/bench_t4/
```

## 3. Что делает ядро (`src/bench_main.py`)
1. **--require-gpu.** `nvidia-smi -L` должен показать `GPU 0`, иначе `SystemExit`. Затем в импорт-чеке `torch.cuda.is_available()` и реальная операция на `cuda:0` (`relu` + matmul + `sum`, ловит «no kernel image»). CPU-фолбэка нет: сам `bench_tile.py` на CPU тоже падает на `torch.cuda.set_device` (`logs/selftest_steps.log`).
2. **Одна GPU.** Бенч-процессы идут с `CUDA_VISIBLE_DEVICES=0`, импорт-чек требует `device_count()==1`. DataParallel не используется. `nvidia-smi -i 0` в скрипте указывает на ту же физическую GPU 0.
3. **Пины pip** (только если версия отличается): `transformers==5.16.1`, `timm==1.0.29`, `segmentation-models-pytorch==0.6.0`, флаги `--only-binary :all:` и `-c constraints_torch.txt`. Constraints-файл фиксирует *установленные* torch/torchvision Kaggle, так что pip их не заменит. Если torch всё же изменился, ядро останавливается. Лог: `logs/pip_install.log`, `logs/pip_check.log`.
4. **Жёсткий импорт-чек** в отдельном процессе того же интерпретатора: torch, timm, segmentation_models_pytorch, transformers, `DFineForObjectDetection`, `RTDetrV2ForObjectDetection`, `DFineConfig`, `RTDetrV2Config`. Любая ошибка → останов. Лог: `logs/import_check.log`.
5. **Четыре команды бота 8** (BUDGET.md §7.9) в `/kaggle/working`. Каждая идёт через `bash -c` **независимо от кода возврата предыдущей**; лог каждой пишется в `logs/<шаг>.log`:
   ```
   nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > smi_before.txt
   python bench_tile.py --models cnx_li9,cnx_li12,cnx_sp3 --dtypes fp16,fp32,bf16 --batches 1,2,4 --warmup 10 --iters 30 --out bench_t4_cnx.json
   python bench_tile.py --models dfine_l,rtdetrv2_r50 --dtypes fp16,fp32 --batches 1,4,8 --warmup 10 --iters 30 --out bench_t4_det.json --hf-config-dir hf_configs
   nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > smi_after.txt
   ```
   Единственное отличие от §7.9: `--hf-config-dir hf_configs` в det-прогоне. Hub не нужен, бенч идёт с `HF_HUB_OFFLINE=1`. `bench_tile.py` пишет JSON после каждой модели, поэтому частичные результаты сохраняются.
6. **validity.json** считается по критериям бота 8 (§5 ниже). `run_summary.json` содержит rc, время каждого шага, версии и sha256. Если какой-то шаг упал, ядро в конце выходит с ненулевым кодом, но файлы в `/kaggle/working` уже записаны.

Самопроверки на box (без GPU):
- `logs/selftest_nogpu.log`: нет nvidia-smi → `FATAL: --require-gpu`, rc=1.
- `logs/selftest_fake_smi.log`: фейковый nvidia-smi → pip-шаг ничего не ставит (версии уже совпадают), затем `FATAL: импорт-чек не пройден: torch.cuda.is_available() == False`.
- `logs/selftest_steps.log`: шаги 4–5 отдельно. Оба бенча падают (нет CUDA), smi-шаги всё равно выполнены, verdict=BROKEN, rc=1.
- Логика validity проверена на синтетическом JSON: 3 мс → BROKEN; SM 1200 МГц + 0x4 → throttled; `fatal` → BROKEN.

## 4. Проверка pip / импорта (box, CPU, scratch venv `bench_t4/_scratch/venv_dry`, Python 3.13.5)
Целевое окружение: публичные релизы Kaggle docker-python v171/v172 GPU (01.10 и 22.10.2026). По их release notes на GitHub (`api.github.com/repos/Kaggle/docker-python/releases`): `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `transformers==5.16.1`, `timm==1.0.29`, `huggingface_hub==1.29.0`, `tokenizers==0.23.1`, `safetensors==0.8.0`, `numpy==2.1.3`. Точный образ по digest `9a6bcfda…` в этих release notes не найден, поэтому torch не пинится, а берётся тот, что стоит в образе.

| Проверка | Результат | Источник |
|---|---|---|
| `pip install --dry-run` пинов + torch 2.11.0/tv 0.26.0 + constraints v171 | rc=0; transformers 5.16.1, timm 1.0.29, smp 0.6.0, все wheel (`py3-none-any`) | `logs/dry.log`, `logs/dry_report.json`, `logs/kaggle_v171_constraints.txt` |
| То же с torch 2.10.0/tv 0.25.0 (образы v167–v170) | rc=0; hf_hub→1.33.0, tokenizers→0.23.3, torch не тронут | `logs/dry_report_t210.json` |
| Реальная установка в scratch venv (torch 2.11.0+cpu) + `pip check` | «No broken requirements found» | `logs/venv_dry_freeze.txt` |
| Импорт + классы | `has_DFine=true`, `has_RTDetrV2=true`, torch 2.11.0+cpu, transformers 5.16.1, timm 1.0.29, smp 0.6.0 | `logs/cpu_check.json` |
| n_params (сборка функциями `bench_tile.py`) | dfine_l **31 090 027** (≈31 M ✓), rtdetrv2_r50 **42 737 247** (≈42 M ✓), cnx_li9 31 938 839, cnx_li12 31 943 447, cnx_sp3 31 929 623; arch=`smp.Unet(tu-convnext_tiny)` (не fallback) | `logs/cpu_check.json` |
| CPU-смоук forward | cnx_li12 256² → [1,7,256,256]; dfine_l и rtdetrv2 320² → logits [1,300,5], boxes [1,300,4], nonfinite=0; макс. RSS 608 МБ, 43 с, 2 потока | `logs/cpu_check.json`, `logs/cpu_check.err` |
| CPU bf16 autocast детекторов | оба отработали, logits bf16, nonfinite=0 (CUDA fp16 autocast на box проверить нельзя) | `logs/cpu_check2.json` |
| `torch.autocast("cuda", dtype=None, enabled=False)` (путь fp32) | ok, без предупреждений | `logs/cpu_check2.json` |
| `bench_tile.py --help` | rc=0 | `logs/help.txt` |
| FLOPs ConvNeXt-T U-Net 12×1024² (meta-тензоры, FlopCounterMode) | 289,2 GFLOP (2·MAC) ≈ 145 GMAC; сходится с оценкой §7.1 «150–250» | `logs/flops.json` |

## 5. HF-конфиги (вшиты)
| Модель | Репозиторий @ ревизия | sha256 config.json |
|---|---|---|
| dfine_l | `ustc-community/dfine-large-coco` @ `aa8f8211545e4cff5adf52ea512f9ac74a9824ec` (та же ревизия, что в `kaggle/dfine_spor`), model_type `d_fine`, backbone `hgnet_v2` | `f2c0253f…6b8d` |
| rtdetrv2_r50 | `PekingU/rtdetr_v2_r50vd` @ `282494075698cab9faa1096ae26856890030c817`, model_type `rt_detr_v2`, backbone `rt_detr_resnet` | `d754c04c…f060` |

Источник: `curl https://huggingface.co/api/models/<repo>` (sha) и `…/resolve/<rev>/config.json`. В обоих `use_timm_backbone=false` и `use_pretrained_backbone=false`, в каждом 80 классов `id2label`. Скрипт передаёт `num_labels=5`. transformers предупреждает о несовпадении (`logs/cpu_check.err`), но голова получается на 5 классов (logits `[1,300,5]`). В JSON должно быть `arch: "... config=local:hf_configs/<model>"`.

## 6. Ожидаемые выходы (`/kaggle/working` → `/workspace/kaggle_out/koz3-bench-t4` → `/workspace/koz3_shared/bot-9/bench_t4/`)
`smi_before.txt`, `smi_after.txt`, `bench_t4_cnx.json`, `bench_t4_det.json` (схема `koz3.bench_tile.v1`, BUDGET.md §7.9), `validity.json`, `run_summary.json`, `logs/{pip_install,pip_check,import_check,smi_before,bench_cnx,bench_det,smi_after}.log`, а также служебные `bench_tile.py`, `hf_configs/`, `constraints_torch.txt`.
- В `bench_t4_cnx.json` у каждой модели будет запись `bf16` с `skipped` (T4 = sm_75, `dtype_of` в `bench_tile.py`). Так задумано (§7.8 «BF16 на T4»).

## 7. Критерии годности бота 8 (BUDGET.md §7.8, п.4–5) — реализованы в `validity.json`
- **Замер ошибочный (BROKEN):** ConvNeXt-T U-Net в **fp16** `ms_per_tile_single_median` **< 5 мс или > 500 мс** на тайл 1024² (п.5: нет synchronize, не тот dtype, CPU-фолбэк). Сюда же относятся отсутствующий или нечитаемый JSON, `fatal`, `oom` и `error` в конфигурации.
- **Троттлинг (THROTTLED):** `clocks_during_timing.sm_clock_mhz.median` **< 1300 МГц** или в `throttle_reasons_active` взведены power cap / HW slowdown (п.4). Тогда брать верх диапазона k (×1590/f_med к compute-части) и пометить замер. Поле — hex-битовая маска nvidia-smi. Ядро её раскладывает: 0x4 sw_power_cap, 0x8 hw_slowdown, 0x20 sw_thermal, 0x40 hw_thermal, 0x80 hw_power_brake. Для T4 (70 Вт) бит 0x4 под нагрузкой вероятен, и по правилу бота 8 это пометка, а не брак.
- `nonfinite_outputs > 0`: время валидно, но точность fp16 надо проверять отдельно (§7.8 «BF16 на T4»).

## 8. Замечания по `bench_tile.py` (патченой копии нет: блокирующих багов не найдено)
1. **Мелочь, только после OOM или ошибки:** `del x, out` стоит внутри `try`. После исключения тензор `x` прошлой конфигурации живёт до следующего присваивания, и `peak_vram_mb` следующей конфигурации может быть завышен на размер входа (до 4×12×1024²×4 Б ≈ 192 МБ). Без OOM ни на что не влияет. На 16 ГБ при batch ≤ 4/8 OOM не ожидается [оц].
2. `clocks_throttle_reasons.active` — устаревшее имя поля (с драйвера 580 канон — `clocks_event_reasons.active`), но nvidia-smi принимает его как алиас (документация NVIDIA и NVSentinel runbook). Если поле не примут, `sm_clock_mhz` станет `null`, и `validity.json` пометит конфигурацию как «нет частоты SM». Частоты остаются в `smi_before/after.txt`.
3. `ClockSampler` сэмплирует и warmup, и замер. Медиана SM поэтому чуть консервативна.
4. CUDA fp16 autocast для HF D-FINE и RT-DETRv2 на box проверить нельзя (проверен только CPU bf16). Если упадёт, ошибка попадёт в `configs[].error`, и бенч пойдёт дальше.
5. Предупреждение transformers `num_labels=5 … id2label 80` безвредно (см. §5).

## 9. Оценка времени GPU [оц]
Число forward-проходов выведено из `bench_tile.py` при заданных аргументах:
- **CNX**, на модель и dtype: batch 1 — 8 видов × 40 + d8 9×8 + flip3 9×3 = 419 тайлов; batch 2 и 4 — (40 + 72 + 27) × B = 278 и 556 тайлов. Итого ≈1250 тайло-проходов. bf16 пропускается.
- **DET**, на модель и dtype: (10+30) + (5+30)×2 = 110 проходов × B, по batch 1+4+8 → ≈1430 чипов.
- Время прохода [оц] при 289 GFLOP на тайл (`logs/flops.json`): T4 fp16 ≈ 30–60 мс, fp32 ≈ 100–200 мс. Отсюда CNX ≈ 3 модели × (40–75 + 125–250 с) ≈ 8–16 мин. DET: D-FINE-L и RT-DETRv2 ≈ 20–40 мс/чип fp16, 60–120 мс fp32 → ≈ 4–9 мин.
- Накладные расходы: старт, pip (≈0–2 мин; при transformers 5.16.1 в образе ничего не ставится), подбор cudnn.benchmark — ≈ 2–4 мин.
- **Итого ≈ 15–30 мин GPU T4 (номинал ~20 мин), то есть ≈ 0,3–0,5 ч квоты.** Это оценка, а не замер; факт будет в `run_summary.json` (`wall_s` шагов) и в `total_wall_s` JSON.

## 10. Не сделано / ограничения
- Ядро не пушилось и не запускалось (по заданию). Тяжёлых прогонов на box не было: 2 потока, RSS ≤ 608 МБ, `nice -n 10`.
- Scratch venv `bench_t4/_scratch/venv_dry` (1,2 ГБ) оставлен для повторных проверок, его можно удалить.
