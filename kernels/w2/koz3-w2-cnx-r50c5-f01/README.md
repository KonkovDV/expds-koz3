# T5: второй лидарный детектор — dmitrijkv/koz3-w2-cnx-r50c5-f01 (ПОДГОТОВЛЕН, НЕ ЗАПУЩЕН)
Основа: orch/coordk/s2f01/koz3-w2-cnx-c3sc-s2f01.py (рецепт L1, centerhead, 12k ит., кроп 512, copy-paste, height/zoom-аугм., d8-TTA). Изменения (3 места + 2 строки):
- энкодер `ENC = env CNX_ENC (по умолчанию tu-resnet50)` вместо жёсткого `tu-convnext_tiny` (selftest головы, build_model, контракт экспорта);
- каналы c5 (lrm_s, lrm_l, slope, svf_r, opp_r) вместо c3; init только scratch (assert: ImageNet-init разрешён лишь для convnext → чисто для сдачи);
- CNX_JOBS_DEFAULT='c5-sc:0,c5-sc:1' (2×T4 параллельно), seed 0, deadline 11.3 ч (как у основного ядра).
Обоснование: REVIEW_SOTA #8 — «ResNet-50 U-Net … как второй член ансамбля». Разнообразие = другой энкодер (BN/ResNet vs LN/ConvNeXt) + 2 доп. канала (svf_r, opp_r: уже посчитаны prep, на диске kernel_sources koz3-cnx-li-prep). Сначала f0/f1 (=фолды s2f01) → сразу парное сравнение с cnx c3sc на тех же фолдах; правило D5 (SCHEDULE): если hit-rate курганов по f0 не выше CNX → стоп, квоту W1.
Проверено на боксе (CPU): centerhead + tu-resnet50, in=5: hm 2×128×128, wh, off, mask 3×512×512 при кропе 512, stride 4, 32,6 M параметров (convnext_tiny 32,0 M) — контракт головы совпадает.
Не проверено: скорость на T4 (ResNet-50 U-Net ≈ convnext-T по FLOPs; бюджет-логика ядра сама урежет итерации, min_iter_frac 0.25).
Запуск (после Сб 31.10 03:00 MSK; acc2/gpu-1 сейчас за coord до 30.10 03:15, acc2/gpu-2 зарезервирован W1/W3 — слот решает координатор):
  python3 orch/enqueue.py add --id w2-r50c5 --owner bot-W2-lidar --accel gpu --account dmitrijkv --kernel-dir w2/kernels/koz3-w2-cnx-r50c5-f01 --ttl 12 --pull-to w2/oof/r50c5 --hold
  (watcher сам берёт аренду acc2/gpu-N; --hold снять командой `enqueue.py set w2-r50c5 status=ready` после 03:00 31.10). В QUEUE.json НЕ добавлено.
