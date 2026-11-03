# Ae D-FINE gate (bot3 Fix A+B+C) — отчёт

**Время (Europe/Moscow):** 2026-11-01 ~19:50 MSK

## Статус
- Датасет **залит**: `elenakonkova/koz3-ae-dfine-gate` (ready)
- Ядро **собрано**, push **в очереди** (оба аккаунта 2/2 GPU). Waiter: `bash /tmp/ae_gate_wait_push.sh` (pid в `/tmp/ae_gate_wait_push.log`). Ручной push: `/workspace/kaggle/push/ae-dfine-gate/push.sh elenakonkova|dmitrijkv`
- Lidar **не трогали** (`elenakonkova/koz3-lidar-aug-f0` RUNNING). Cancel TPU-smoke через API запрещён (403).

## Коммит / артефакты
- Worktree tip: **`7b1c272`** (`bots/bot3-aerial-recall` — смесь PR#3+#16; A+B 1e7073a, Fix C 0bcb039/ffa9178 lineage)
- Base v2: `eb268d6` (LB 0.41238)
- Weights f4pack sha16: **`249b86fb9713cc1c`**
- Zip: `/workspace/kaggle/push/ae-dfine-gate/dataset/submission_ae_abc_7b1c272.zip` (~610 MB)
- Push tree: `/workspace/kaggle/push/ae-dfine-gate/{dataset,elenakonkova,dmitrijkv,dryrun,push.sh}`
- Kernel slug: **`koz3-w1-ae-dfine-gate`** (не совпадает с id датасета — иначе 409)
- Sources: `elenakonkova/koz3-ae-dfine-gate` + `dmitrijkv/koz3-gt-train-v2`
- GPU T4, internet ON (pip `transformers==5.21.0`), variants OFF / A+B / A+B+C; **mosaic_intact не включён** (не дёшево)
- Fix C обязателен: `[dfine_ae_tile]` fuse=nms, windows_mode=train (иначе WBF+flip делит скор пополам vs OOF NMS ~0.5–0.6)

## Dry-run CPU (регион 015 Демидовка, Ae crop ~659 KB)
| variant | raw Ae | written | время | классы written |
|---------|-------:|--------:|------:|----------------|
| OFF     | 0      | 0       | ~22 с | — |
| A+B+C   | 333    | 3       | ~150 с | fort 2, kurgan_tselye 1 |
Логи: `.../dryrun/out/infer_{off,abc}.log`. Флаг **живой** (Δ dets ≠ 0). Tile ON в логе: `dfine_ae_tile: ON — windows_mode=train fuse=nms`.

## ETA / pull
- Infer wall в ядре: 20 мин; 3 варианта на T4 ~несколько минут на 015 (полный xfer Ae — дольше, в датасете сейчас только 015).
- После COMPLETE:  
  `KAGGLE_API_TOKEN=… kaggle kernels output <acc>/koz3-w1-ae-dfine-gate -p /tmp/ae-gate-out`  
  Смотреть `/kaggle/working/ae_dfine_gate/summary.json` → `flag_dead`, `delta_abc_vs_off`, `results.*.ae`.

## Блокеры push (на момент отчёта)
- elena: lidar RUNNING + `koz3-acc2-tpu-smoke` QUEUED → max GPU batch 2
- dmitrijkv: `koz3-w1-lidar-aug-f2` + `koz3-w3-gordense-timing2` RUNNING
