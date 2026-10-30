# w2_kernels — исходники Kaggle-ядер W2 (код + kernel-metadata.json, без выходов)
Структура `<аккаунт>/<slug>/{<slug>.py, kernel-metadata.json}`. Копия бокса `/workspace/koz3/w2/kernels/`.
| Ядро | Что | PREREG / статус |
|---|---|---|
| `elenakonkova/koz3-w2-cnx-r50c5-f01`, `dmitrijkv/koz3-w2-cnx-r50c5-f234` | ResNet-50 U-Net, 5 каналов рельефа, фолды 0–1 / 2–4 | P-W2-9, ждёт запуска Сб 31.10 после 03:00 (HANDOFF п.1–7) |
| `koz3-w2-cnx-r50c5-f01/` (без аккаунта, с README) | черновик того же ядра | — |
| `*/koz3-w2-cnx-c5sc-f{01,23,4}` | c3-sc на 5 каналах | P-W2-1, закрыт ботом 1, не запускать |
| `elenakonkova/koz3-w2-lihm-c3sc-f01234` | heatmap-голова | P-W2-7, FAIL |
| `make_r50c5.py`, `make_c5.py` | генераторы ядер из шаблона c3-sc | `python make_r50c5.py` |
Запуск: `orch/lease.sh acquire slot:acc1/gpu-1 bot-W2-lidar <ч> --kernel <acc>/<slug>` → `orch/pool_kaggle.sh <acc> kernels push -p <папка>`. Входы — `dataset_sources` в metadata (koz3-cnx-li-prep и др. → `/kaggle/input/<slug>/`). Выходы — `kernels output` (oof_*, weights_*, maps/, report_cnx_full.json).
