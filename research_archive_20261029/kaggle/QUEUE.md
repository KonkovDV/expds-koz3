# Очередь Kaggle (ведёт «1»)




## REQUEST [bot3] e2e-eval Fix A+B ON candidate — after calib ~23:10 MSK
- **STATUS: QUEUED-AFTER-CALIB** (slot after `koz3-e2e-eval` calibration v2)
- **from:** bot-3
- **branch / SHA:** `bots/bot3-ae-ortho-recall` / `1e7073afd29a503e2811fe5a9a9e0e52cb92643e`
- **slug:** `elenakonkova/koz3-e2e-eval` (+ cand dataset when ready)
- **what:** screen flag-ON candidate A+B (dfine_aeor Ae f4pack + ae_recall_relax). Prefer zip with models/dfine_ae + vendor + enabled.toml; commit-only path lacks weights.
- **prep:** PC Docker gate builds/validates ON first (`bot3_ae_ortho_ab_f4`); then upload `elenakonkova/koz3-e2e-cand-bot3-ae-ortho-ab` with candidate.zip + SPEC.json `{"zip":"candidate.zip","env":"kaggle-gpu","notes":"A+B ON 1e7073afd29a503e2811fe5a9a9e0e52cb92643e"}`.
- **PASS:** scores.json ALL not −>0.01 vs OFF/baseline; ae delta >0; STATUS DONE; no DFINE_AEOR_UNAVAILABLE.
- **do not:** push before calib; do not skip PC docker assert.
- **queued_msk:** 2026-10-29 20:35

## Account-2 lane (elenakonkova) — 2026-10-29 20:28 МСК

**Quota (kaggle quota):** GPU used ~0.5h / remaining **~29.5h** / total 30.00h; TPU 0.00h / **20.00h** unused; refresh **2026-10-31 00:00 UTC** (= 03:00 МСК).

**Smoke:** `elenakonkova/koz3-acc2-gpu-smoke` COMPLETE. `elenakonkova/koz3-acc2-tpu-smoke` still QUEUED.

**Private datasets / access:**
- `elenakonkova/koz3-s3-cred`, `koz3-cnx-li-prep`, `koz3-bot8-snap` — READY
- **collab reader** granted on acc1 private: `dmitrijkv/koz3-xfer-s000`…`s009` + `dmitrijkv/koz3-gt-train-v2` (metadata --update, role READER). Acc2 can list/attach; no re-upload.

**GPU sessions (2/2):**

| slug | accel | purpose | start МСК | ETA | status |
|---|---|---|---|---|---|
| elenakonkova/koz3-cnx-li-c3sc-s1f23 | GPU T4×2 | seed1 c3-sc folds 2+3 | ~20:06 | ~01:30 30.10 | RUNNING |
| elenakonkova/koz3-cnx-li-c3sc-s1f4 | GPU T4×2 | seed1 c3-sc fold 4 | ~20:10 | ~23:10 | RUNNING |

**bot8 honest10 data path FIXED (staged):**
- Slug: `elenakonkova/koz3-bot8-oof-honest10` **v5 COMPLETE** (CPU staging, enable_gpu=false): rebuild `PIXGBM_REBUILD_ALL=1` → `/kaggle/working/train-expds-3`, **regions≈210**, geo200-217 dirs=18 (labels from s000; **rasters hollow** — known), shards s000–s009 all attached, then `phase=staged_no_gpu`.
- Source: `/workspace/kaggle/push/elenakonkova-bot8-oof-honest10/` (vendored `rebuild_train_tree.py`, ensure_train_tree, dataset_sources = snap+cred+xfer-s000..s009+gt-train-v2).
- **Coverage:** xfer large blobs = geos **1–33** (gap geo14); layers li/ae/or/sp_or present (or only 1 file in s002). Geo **200–217 rasters отсутствуют** в s000–s009 (как bot6 hollow).
- **GPU OOF ещё НЕ запущен** (acc2 GPU 2/2). Когда освободится слот (~после s1f4 ≈23:10 МСК):  
  `KAGGLE_API_TOKEN=$(cat ~/.secrets/kaggle2_token) ~/.venvs/kaggle/bin/kaggle kernels push -p /workspace/kaggle/push/elenakonkova-bot8-oof-honest10`  
  (metadata уже enable_gpu=true / NvidiaTeslaT4). Ожид. **8–12 GPU-ч** → ETA старта ~23:10, финиш ~07:00–11:00 30.10 МСК.
- Не дублировать полный S3 sync (~300GB).

**Tried / blocked:**
- v3 ERROR: нет train-expds-3. v4 ERROR: sibling rebuild_train_tree.py не попал в script kernel → vendored base64 в v5.
- TPU smoke QUEUED indefinitely.

---
## [bot-6] PIXGBM cleanroom TPU-VM-as-CPU — 29.10.2026 ~16:18 МСК (обновлено: 16:44 МСК)
- **СТАТУС: SUPERSEDED → pc/QUEUE.md [bot-6] @ d173a75** (Kaggle v6 hollow: 0 geo rasters s001–s009; wall_s=8.3; ядро НЕ переделываем; retrain на ПК F:\2\train-expds-3)
- **было (не запускать снова):**
- **Slug (архив):** сохраняем текст ниже без удаления.
### [bot-6 архив] PIXGBM cleanroom TPU-VM-as-CPU — 29.10.2026 ~16:18 МСК (обновлено: 16:35
- **Slug:** `dmitrijkv/koz3-bot6-pixgbm-cleanroom`
- **Источник ядра:** ветка `bots/bot6-pixgbm-cleanroom`, commit `b40ffd4b11c7f0fdbf6994ca4896dac36a1edb44`, path `kaggle/pixgbm_cleanroom/` (`pixgbm_cleanroom_kaggle.py`, `rebuild_train_tree.py`, `kernel-metadata.json`)
- **dataset_sources:** `dmitrijkv/koz3-xfer-s000` … `koz3-xfer-s009` + `dmitrijkv/koz3-gt-train-v2` (единого train-expds-3 slug нет; дерево собирается в `/kaggle/working/train-expds-3`, только geo 200–217)
- **Скрипт обучения:** `scripts/pixgbm_retrain_cleanroom.py --enable-retrain` (тот же commit); `PIXGBM_NUM_THREADS` default=`os.cpu_count()`
- **Ресурс:** TPU-VM как CPU-хост (BACKLOG TPU-2). Оценка **2–4 ч** CPU (+ rebuild).
- **Модуль:** `expds/geo_pixgbm.py` эталон sha256 `35e4331b…` в input/рабочую копию.
- **Вернуть:** STATUS.txt DONE, retrain_log.json, models/all/{poly,finds}.txt+*.sha256, checkpoints/rebuild_report.json, logs/train.log, lightgbm version.
- **Критерий pass:** sha256 == MODELS_NOT_IN_GIT.md **или** metric_fallback.status=ok (tol=1e-4).
- **Draft PR:** https://github.com/KonkovDV/expds-koz3/pull/5 (не merge).
- **Манифесты ещё нужны offline:** `koz3-xfer-s001`…`s009` manifest.json — чтобы сузить attach; до получения — все шарды (как сейчас).

## Запуски 29.10 ~16:00 МСК (исполнитель kaggle-quota)

| slug | accel | purpose | ожид. часы | что забрать | статус на запуске |
|---|---|---|---|---|---|
| koz3-cnx-li-c3sc-f4-v2 | GPU T4×2 (NvidiaTeslaT4) | догон c3-sc_f4 после fork+filelock; jobs=`c3-sc:4`; патч ядра: spawn + CNX_WORKERS=0 + nw=0 + HF/TORCH cache в /tmp | ~1.2–1.5 ч train+infer (как c3-in_f4 ~72 мин) | `ckpt_cnx_c3-sc_f4.pt`, `oof_cnx_c3-sc_f4.geojson`, `report_cnx_full.json`, maps | RUNNING (параллельно с koz3-dfine-or; 2 GPU OK) |
| koz3-tpu-probe | TPU v5e-8 (TpuV5E8) | крошечная проба: nproc/RAM/disk/TPU/torch_xla/jax/lightgbm | ~5–15 мин | `/kaggle/working/tpu_probe.json` | QUEUED (ждёт слот TPU; рядом тоже QUEUED: koz3-cnx-li-tpu-smoke) |

Патч c3sc-f4-v2 только в скрипте ядра (репо не трогали). Источник: копия koz3-cnx-li-c3-f4 + jobs только c3-sc:4.

Блокер TPU: cancel-session API → 403; слот TPU не резервируется (timeReserved=0) пока QUEUED. После пробы — выбрать CPU-тяжёлую работу (PIXGBM/rerank) или XLA-обучение по результатам probe.

Квота на запуске: GPU remaining ~9.28h (refresh 2026-10-31 00:00 UTC = 03:00 МСК); TPU 20.00h unused.

## [bot-3] D-FINE Ae fine-tune (GPU-3) — 2026-10-29 16:14 МСК
- **Slug:** `koz3-dfine-ae-ft` (`dmitrijkv/koz3-dfine-ae-ft`)
- **Источник:** ветка `bots/bot3-dfine-ae-kernel` @ `6e42cf6` (`6e42cf605be3a6eab5bb9a92f113dd7faae98183`); путь `kaggle/2026-10-29/dfine_ae_ft/` (`koz3-dfine-ae-ft.py` + `kernel-metadata.json`).
- **База ветки:** `bots/bot3-aerial-recall` @ `475bafa` (lineage `cbd4804`).
- **kernel_sources:** `dmitrijkv/koz3-dfine-ae` (COMPLETE) — resume `dfine_f{k}`.
- **Ресурс:** GPU T4×2 (`NvidiaTeslaT4`), internet on, dataset `dmitrijkv/koz3-s3-cred`.
- **Ожид. время:** **~2.5–3.0 GPU-ч** (lean: +2000 ит/фолд + final 1500; deadline 3.0 ч). Квота ~9.4 ч до 31.10 03:00 МСК.
- **Не запускать параллельно** с другим тяжёлым T4×2, если квота < 4 ч; можно после COMPLETE `koz3-dfine-or` / рядом с лёгкими jobs.
- **Команда:**
  ```
  KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) kaggle kernels push -p /path/to/expds-koz3/kaggle/2026-10-29/dfine_ae_ft
  ```
  (checkout `bots/bot3-dfine-ae-kernel` first). Env defaults OK; optional `DFINE_ITERS`/`DFINE_FINAL_ITERS`/`DFINE_DEADLINE_H`.
- **Критерий успеха:** `STATUS.txt` → `DONE`; есть `dfine_ae/model.safetensors` + `config.json`; `thresholds_ae_oof.json` (ключи `dfine_ae`/`ae_dfine`); `dfine_report.json` без `error` в folds; OOF `oof_dfine_ae.geojson`.
- **Забрать:** `dfine_ae/`, `dfine_f{0..4}/`, `models/`, `thresholds_ae_oof.json`, `dfine_report.json`, `STATUS.*`, `oof_dfine_ae.geojson`.
- **После:** PC honest-10 с `[dfine_aeor]` **временно ON** + эти веса/пороги (флаг в main остаётся OFF); Δ ae recall vs база.
- **RULES:** без Yandex/Google/Bing/Esri, без КОЗ №2; веса obj2coco Apache-2.0 → FT на Ae KOZ3 only.

## [bot-8] GPU-5 OOF honest-10 — 2026-10-29 16:22 МСК
- **Slug:** `koz3-bot8-oof-honest10` (`dmitrijkv/koz3-bot8-oof-honest10`)
- **Источник:** ветка `bots/bot8-oof-kaggle` @ `a6f8992` (`a6f8992abfdbf300a40f624509ab9f2cd630d394`); путь `kaggle/2026-10-29/oof_honest10/` (`koz3-bot8-oof-honest10.py` + `oof_kaggle_jobs.py` + `kernel-metadata.json`).
- **Draft PR:** https://github.com/KonkovDV/expds-koz3/pull/7
- **Ресурс:** GPU T4×1 (`NvidiaTeslaT4`), internet on, dataset `dmitrijkv/koz3-s3-cred`.
- **Ожид. время:** **~8–12 GPU-ч** (все jobs из `[models]` → `*_oof.geojson`; resumable STATUS/sha256). Квота GPU — сверить перед запуском.
- **Не запускать параллельно** с другим тяжёлым T4, если remaining < 12 ч; предпочтительно ночной слот / после COMPLETE соседних GPU jobs.
- **Команда:**
  ```
  KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) kaggle kernels push -p /path/to/expds-koz3/kaggle/2026-10-29/oof_honest10
  ```
  (checkout `bots/bot8-oof-kaggle` first). Smoke: `OOF_SMOKE=1`. Optional: `OOF_JOBS=spor_focal,li_gor_mask` для урезанного прогона.
- **Критерий успеха:** `STATUS.txt` → `DONE`; есть `oof/*_oof.geojson` + `SHA256SUMS` для всех jobs каталога; нет `FAILED` без retry.
- **Забрать:** `oof/*.geojson`, `STATUS.txt`, `SHA256SUMS`, `report_oof.json` (если есть), логи.
- **После:** scoreboard `--honest 10` + density_thresholds / oof_transform (бот 7 / ПК CPU); калибровка порогов по слою.
- **RULES:** без Yandex/Google/Bing/Esri, без КОЗ №2; только веса/данные KOZ3.
- **Статус:** **QUEUED-AFTER-RESET** (31.10 03:00 МСК). Причина: ~8–12 GPU-ч не влезают в remaining≈8.5ч при reserve≥0.5ч + c3-sc_f4 (~1.2ч) + bot3 ae-ft (≤3ч). Ядро staged: `/workspace/kaggle/push/koz3-bot8-oof-honest10` (vendored oof_kaggle_jobs).


## Запуски/статусы 29.10 ~16:25 МСК (исполнитель kaggle-quota)

| slug | accel | purpose | ожид. часы | что забрать | статус |
|---|---|---|---|---|---|
| koz3-cnx-li-c3sc-f4-v2 | GPU T4 | c3-sc_f4 relaunch (fork/filelock patch) | ~1.2–1.5 | ckpt/oof/report | RUNNING |
| koz3-dfine-or | GPU T4 | D-FINE Or | (идёт) | dfine_or outputs | COMPLETE |
| koz3-bot6-pixgbm-cleanroom | **CPU** | PIXGBM cleanroom @ b40ffd4 | — | pull hollow | **SUPERSEDED→pc/QUEUE [bot-6] d173a75** (v6 hollow wall=8.3s; 0 geo rasters) ~16:44 МСК |
| koz3-dfine-ae-ft | GPU T4 | bot3 Ae FT | ≤3.0 | STATUS/dfine_ae/thresholds_ae_oof.json | RUNNING (pushed ~16:33 МСК) |
| koz3-bot8-oof-honest10 | GPU T4 | bot8 OOF honest-10 | 8–12 | STATUS/SHA256SUMS/oof/* | **QUEUED-AFTER-RESET** 31.10 03:00 МСК |
| koz3-tpu-probe / tpu-smoke | TPU | probe/smoke | — | — | ERROR (nbconvert script-as-nb); ghost держит batch TPU=1 |

Манифесты xfer: `/home/box/shared/koz3/kaggle/koz3-xfer-s000_manifest.json` … `s009` (s002/4/5/6/8 — настоящие manifest; s001/3/7/9 — полный files listing API).

Квота ~16:25 МСК: GPU remaining ≈8.5ч (refresh 2026-10-31 00:00 UTC); TPU 20.00h unused (не резервируется).


## 29.10 18:05 МСК (bot1)
- koz3-cnx-li-c3sc-f4-v2: **COMPLETE** — li@0.3=0.3126, outputs /workspace/kaggle/out/c3sc-f4-v2/.
- koz3-cnx-li-c3sc-s1f01: **RUNNING** (GPU filler, сид 1, c3-sc f0+f1, CNX_DEADLINE_H=2.6) — источник /workspace/kaggle/push/koz3-cnx-li-c3sc-s1f01.
- Квота GPU (оценка): ~5–5,5 ч до 31.10 03:00 МСК до запуска s1f01; после s1f01 и ae-ft остаётся ~1–1,5 ч резерва.

## 29.10 19:40 МСК (bot1)
- koz3-dfine-ae-ft: **COMPLETE** (10478 с). final=1 итерация (дедлайн), dfine_ae = сид f3. OOF F2 kt 0,028 / kp 0,173 / fort 0,184. Выходы /workspace/kaggle/out/dfine-ae-ft (SHA256SUMS.weights).
- 2-я GPU-сессия не заполнена: квота ≈ резерв до 31.10 03:00 МСК. Следующий после сброса: koz3-bot8-oof-honest10.

## ПОРЯДОК слотов acc2 после ~23:10 (решение «1», 29.10 20:30)
1. koz3-e2e-eval калибровка v2 (короткий прогон, ~30–60 мин) — открывает скрининг кандидатов на Kaggle.
2. elenakonkova/koz3-bot8-oof-honest10 GPU (пакет /workspace/kaggle/push/elenakonkova-bot8-oof-honest10, 8–12 GPU-ч) — во второй освободившийся слот (s1f23 ~01:30) или сразу после калибровки.
3. Далее — e2e-скрининг кандидатов bot3 (аэро/орто), bot2 (спутник), bot4 (крупные), bot8 (TTA) по мере готовности.
