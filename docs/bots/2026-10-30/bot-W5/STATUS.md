# bot-W5-rerank — STATUS (29.10 23:00 MSK)
- Код: analysis/2026-10-30/w5_rerank (run_all.sh), PREREG_w5_rerank.md. Ядро kaggle/koz3-w5-dino-emb (не пушено).
- Блокер: stack_arch.geojson(.gz 193 МБ) только на ПК (C:\Sloi\rerank). Нужна копия в /workspace/koz3/w5data (частями ≤95 МБ через CopyToBox или расширить local-exec root).
- Смоук на прокси (D-FINE ae OOF, 30 участков): конвейер V0–V4 работает; числа не для гейта.
- Очередь (заявка координатору): koz3-w5-dino-emb → acc1 gpu после W1/W2 (~05:00 30.10) или CPU-аккаунты, только после cands.csv.
