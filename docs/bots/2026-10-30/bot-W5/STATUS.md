# bot-W5-rerank — STATUS (29.10 23:00 MSK)
- Код: analysis/2026-10-30/w5_rerank (run_all.sh), PREREG_w5_rerank.md. Ядро kaggle/koz3-w5-dino-emb (не пушено).
- Блокер: stack_arch.geojson(.gz 193 МБ) только на ПК (C:\Sloi\rerank). Нужна копия в /workspace/koz3/w5data (частями ≤95 МБ через CopyToBox или расширить local-exec root).
- Смоук на прокси (D-FINE ae OOF, 30 участков): конвейер V0–V4 работает; числа не для гейта.
- Очередь (заявка координатору): koz3-w5-dino-emb → acc1 gpu после W1/W2 (~05:00 30.10) или CPU-аккаунты, только после cands.csv.

## 29.10 23:20 MSK — real stack run
- Real stack (1,386,709 cands): V0 reproduces A=0.314615 exactly (new34 0.29645, optical 0.14427).
- V1–V4 on real stack: all rejected (every Δ<0; best V1 mink=3 Δ=−0.0001 P=0.43). Details: analysis/2026-10-30/w5_rerank/real_policies_V0-V4.json.
- V3 copies emit a markup_type present at the site — **W4 compliance flag** ("между типами данных НЕ склеивать"); V3 rejected anyway.
- Running: nested2 V5 (X2) and V5x (X2 + W2 cnx c3-sc OOF features s0/s1/mean/agree/samecls). V6c (stack + seed-fused cnx candidates, group li_cnx) prepared, prep pending RAM.
- Kaggle: dataset elenakonkova/koz3-w5-cands (private, ready); kernel elenakonkova/koz3-w5-dino-emb v1 on CPU (lease slot:acc1/cpu до 30.10 11:18).
- Box incident ~22:54: box restored from snapshot; /workspace/koz3/venv was missing → recreated (numpy, scipy, shapely, pyproj, lightgbm, sklearn, ijson, zstandard, boto3).
