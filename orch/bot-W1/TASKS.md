# W1 (бот 2): Ae / Or / SpOr / Sat — задачи до 11.11
Источник: orch/MASTER_PLAN.md (29.10 23:15 MSK)
1. [30.10 до 11:00] Довести koz3-w1-dfine-ae-final и -or-final (acc1 gpu-1/2). Выложить OOF и веса в приватный датасет. Освободить lease слотов.
2. [D3, 31.10 03:00] e2e-оценка Ae all-class на honest-gate и per-modality (Ae/Or). Вывод: «в архив да/нет». Порог: Δ ≥ 0 на gate и ни один класс не ниже floor (P4). Учесть урок SpOr: gate +0.0198, e2e −0.0023.
3. [01–02.11] Or multiclass (arch/gor/povr на Or, L7) и жёсткие негативы «стога сена» (L10) для SpOr/Or/Ae.
4. [03.11] Спутник recall (#11): снизить порог и мультимасштаб; ≤ 2 GPU-ч.
5. [07.11] Дообучение финальных Ae/Or на всех данных (D6), ≈ 4 GPU-ч.
Правила: lease перед push; ветка bots/w1-*; worktree wt/w1; EN, YOLO, SAM3, DINOv3 и КОЗ-2 не использовать; LB посылает только Элена.
