# PREP CI2-5 koz3-w3-bigobj-adacrop (бот D / W3) — 30.10 13:00 MSK, только CPU/подготовка

QUEUE CI2-5: owner W3, gpu(T4x2), account elenakonkova, status planned, prereq w1-ae-final + w1-or-final (оба COMPLETE на Kaggle).
Идея (REVIEW_SOTA:188): адаптивный кроп max(k·размер, min) → ресайз в тайл для масок gor/fort; ожидание +0.002…+0.008, GPU 2–3 ч.

## Есть
- Ядра adacrop до сегодня НЕ было. Создана заготовка (локально, не push, не в git-коммите):
  wt/w3/kernels/koz3-w3-bigobj-adacrop/{kernel-metadata.json, koz3-w3-bigobj-adacrop.py (STUB, выход без KOZ3_ADACROP_READY=1), README.md}
- Образец каркаса: wt/w3/kernels/koz3-w3-gordense-timing (dmitrijkv, T4, HARD_S, md5, xfer-шарды s000–s009, kernel_source dmitrijkv/koz3-dfine-spor-full-rest) — ready в QUEUE.
- Точка врезки: wt/w3/expds/train.py (resample_scale стр.193, augment_batch стр.378, train() стр.412); логика gor/fort: expds/gorfort.py, expds/hillfort.py, expds/gor_gate.py.

## НЕТ (блокеры)
1. Реализация adacrop в expds/train.py (флаг, default off) + тест «off ⇒ байт-в-байт».
2. Кэш тайлов для train(cache=...) с масками gor/fort — на боксе нет; pc_data/ пуст. Варианты: S3 через секреты KOZ3_S3_* (у elenakonkova) или xfer-шарды (они на dmitrijkv → QUEUE account elenakonkova расходится, решает A).
3. GT gor/fort + фолды по семьям; dataset_sources в metadata пустые.
4. PREREG W3 для CI2-5 (до запуска).
5. GPU: до 03:05 31.10 acc1/gpu-1 под r50c5; аренда слота только у A.

## Команды после 03:05 31.10
```
cd /workspace/koz3
orch/lease.sh list                       # проверить слот; аренду просит A/бот 1 (не брать самому)
# A выдаёт: orch/lease.sh acquire slot:acc1/gpu-<n> bot-W3-bigobj <ttl> --kernel elenakonkova/koz3-w3-bigobj-adacrop --note "CI2-5"
# gordense-timing (уже ready, dmitrijkv) — тоже только после аренды A:
#   orch/pool_kaggle.sh dmitrijkv kernels push -p wt/w3/kernels/koz3-w3-gordense-timing
# adacrop — только когда п.1–4 закрыты и STUB-гвард снят:
#   orch/pool_kaggle.sh elenakonkova kernels push -p wt/w3/kernels/koz3-w3-bigobj-adacrop
orch/pool_kaggle.sh elenakonkova kernels status elenakonkova/koz3-w3-bigobj-adacrop
```
Не трогать RUNNING koz3-w3-kurgan-cpu-seeds (elenakonkova, dmitrijkv).
