# Kaggle quota max-use — 29.10 ~16:05 МСК

## Сделано
1. **koz3-dfine-or** — RUNNING (GPU). Не трогали.
2. **koz3-cnx-li-c3sc-f4-v2** — RUNNING на NvidiaTeslaT4. Новый slug. Только `c3-sc:4`. Патч в скрипте ядра: `multiprocessing spawn`, `CNX_WORKERS=0`, `nw=0` в DataLoader, кэш HF/TORCH в `/tmp`. Источник — копия `koz3-cnx-li-c3-f4`. Ожидание ~1.2–1.5 ч. Артефакты: `ckpt_cnx_c3-sc_f4.pt`, `oof_cnx_c3-sc_f4.geojson`, `report_cnx_full.json`.
3. **koz3-tpu-probe** — запушен на TpuV5E8, статус QUEUED. Печатает nproc/RAM/disk/TPU/torch_xla/jax + smoke LightGBM → `tpu_probe.json`.
4. Параллельно 2 GPU-сессии OK (dfine-or + c3sc-f4-v2).

## TPU
- Квота 20 ч не тронута (used 0).
- В очереди также **koz3-cnx-li-tpu-smoke** (QUEUED). CancelKernelSession → 403.
- Полезный TPU-job после пробы: (a) CPU-тяжёлый PIXGBM/rerank на многоядерной TPU VM если есть код+S3; (b) доработка XLA ConvNeXt из tpu-smoke; (c) TTA/infer. Пока проба не стартовала — длинный job не пушили (чтобы не плодить очередь).

## Квота
- GPU: ~9.3 ч до 2026-10-31 00:00 UTC (03:00 МСК).
- TPU: 20.00 ч, refresh тогда же.
