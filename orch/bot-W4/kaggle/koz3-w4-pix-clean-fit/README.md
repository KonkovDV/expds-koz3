Ядро koz3-w4-pix-clean-fit — НЕ запушено (бриф: пушит бот 1). Аренда не взята: все CPU-слоты заняты (W3/W5).
Аккаунт: usertest1234 (CPU-only; слот освобождается 30.10 06:01 MSK) — или любой свободный CPU из пула.
Нужно до push: (1) аренды slot:<acc>/cpu, kernel:<acc>/koz3-w4-pix-clean-fit, dataset:<acc>/koz3-w4-pixclean-cache;
(2) приватный датасет из ./dataset (dataset-metadata.json + cache_clean/ + code/), сделан `make_ds.sh`.
Выход: /kaggle/working/models/F{1..5}_s{1,2}/ → скопировать в /workspace/koz3/w4geo/models/.
Если локальный fit_all.sh успеет раньше — ядро не нужно.
