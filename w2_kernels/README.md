# w2_kernels — исходники Kaggle-ядер W2 (код + kernel-metadata.json, без выходов)
Структура `<аккаунт>/<slug>/{<slug>.py, kernel-metadata.json}`. Копия бокса `/workspace/koz3/w2/kernels/`.
| Ядро | Что | PREREG / статус |
|---|---|---|
| `elenakonkova/koz3-w2-cnx-r50c5-f01`, `dmitrijkv/koz3-w2-cnx-r50c5-f234` | ResNet-50 U-Net, 5 каналов рельефа, фолды 0–1 / 2–4 | P-W2-9, ждёт запуска Сб 31.10 после 03:00 (HANDOFF п.1–7) |
| `koz3-w2-cnx-r50c5-f01/` (без аккаунта, с README) | черновик того же ядра | — |
| `*/koz3-w2-cnx-c5sc-f{01,23,4}` | c3-sc на 5 каналах | P-W2-1, закрыт ботом 1, не запускать |
| `elenakonkova/koz3-w2-lihm-c3sc-f01234` | heatmap-голова | P-W2-7, FAIL |
| `make_r50c5.py`, `make_c5.py` | генераторы ядер из шаблона (пути захардкожены на `/workspace/koz3/w2/kernels`; на новом боксе: `mkdir -p /workspace/koz3/w2 && cp -r w2_kernels /workspace/koz3/w2/kernels`) | `python make_r50c5.py` |
| `source_c3sc/<slug>/` | исходники уже отработавших ядер c3-sc, давших OOF/веса/карты в датасетах koz3-w2-*: s0 `koz3-cnx-li-c3sc-{f23,f4-v2}` (f01 — в `kaggle/2026-10-29/convnext_lidar/`), s1 `koz3-cnx-li-c3sc-s1{f01,f23,f4}`, s2 `koz3-w2-cnx-c3sc-s2{f01,f23,f4}` (копия `orch/coordk/` бота 1), рельеф `koz3-cnx-li-prep` | справка/перезапуск; аккаунт — поле `id` в metadata |
Запуск: `orch/lease.sh acquire slot:acc1/gpu-1 bot-W2-lidar <ч> --kernel <acc>/<slug>` → `orch/pool_kaggle.sh <acc> kernels push -p <папка>`. Входы r50c5: f01 — `dataset_sources: elenakonkova/koz3-s3-cred` + интернет, рельеф строит само ядро из S3 (~0,55 ч); f234 — `dataset_sources: dmitrijkv/koz3-s3-cred` + `kernel_sources: dmitrijkv/koz3-cnx-li-prep` (готовый рельеф из output ядра, статус COMPLETE). Все источники проверены 30.10 (ready/COMPLETE). Выходы — `kernels output` (oof_*, weights_*, maps/, report_cnx_full.json).
