# source_c3sc — исходники отработавших ядер c3-sc (ConvNeXt-T, 3 канала, centerhead) и рельефа
Сняты `kernels pull` с Kaggle (бокс: `/workspace/koz3/w2/src/`, s2 — `/workspace/koz3/orch/coordk/s2*`). Секретов нет: ключи S3 читаются ядром из приватного датасета `koz3-s3-cred` (в коде только имена `KOZ3_S3_ACCESS_KEY_ID`/`KOZ3_S3_SECRET_ACCESS_KEY`).
| Папка | Аккаунт/слаг (см. `id` в metadata) | Что дало |
|---|---|---|
| `koz3-cnx-li-prep` | dmitrijkv/koz3-cnx-li-prep | рельеф `cnx_relief/*.npy` (= датасет elenakonkova/koz3-cnx-li-prep) |
| `koz3-cnx-li-c3sc-f23`, `-f4-v2` | dmitrijkv | seed0 фолды 2–3, 4 (f01 — `kaggle/2026-10-29/convnext_lidar/koz3-cnx-li-c3sc-f01`) |
| `koz3-cnx-li-c3sc-s1f01`, `-s1f23`, `-s1f4` | dmitrijkv / elenakonkova | seed1. **s1f4:** ядро делит `id_no` с `koz3-w2-cnx-c3sc-s2f4` → веса s1f4 = только версия 1 (`kernels output elenakonkova/koz3-w2-cnx-c3sc-s2f4/1`, md5 efbff0f3…); последняя версия = s2f4 (P-W2-11). В koz3-w2-weights v3 исправлено. |
| `koz3-w2-cnx-c3sc-s2f01`, `-s2f23`, `-s2f4` | dmitrijkv / dmitrijkv / elenakonkova | seed2 (шаблон для r50c5: `make_r50c5.py`) |
Выходы всех этих ядер сохранены в датасетах koz3-w2-oof / -weights / -maps (см. `w2_datasets/`). Перезапускать не нужно.
