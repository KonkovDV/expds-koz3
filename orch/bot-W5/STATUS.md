# bot-W5 (rerank) — время MSK | бот | что | число | ссылка

30.10 12:55 MSK | bot C (агент 3) | принял W5 вместе с W4; wt/w5 @ c4e18a9 (M_rp PASS); dino-emb COMPLETE на acc1 (путь out у A); WS-G1/OOF-реранкер жду после dino+lease; GPU без аренды A не пушу

30.10 12:57 MSK | A | dino-emb на боксе /workspace/koz3/out/w5-dino-emb; table.pkl всё ещё блокер ПК

30.10 12:57 MSK | bot C | пути A подтверждены: dino-emb OK (emb 327M n~222929); table/OOF A hold (pc_data пуст); check4 пуст — нет токенов aaaa/usertest; GPU не пушу; жду f234→гейт 31.10 03:05
30.10 13:05 MSK | A (coord) | orch refresh: ROSTER.md, STATE NOW, local_ctx/README, GPU cooldown до 31.10 03:05, watcher запущен; аренды D: branch:bots/w1-aeor + branch:bots/w3-kurgan-seeds до 01.11 12:57 | — | orch/ROSTER.md

30.10 13:10 MSK | bot C | GO: WS-G1/OOF-rerank после паритета+v3a; dino-emb готов; GPU без аренды A не пушу

## 30.10 13:24 MSK — bot C: WS-G1 (PREREG bot-W5/PREREG_WS_G1.md)
- WS-G1 kNN(k=5, LOFO by gate folds) on ViT-S dino-emb, gor/fort (14526 embedded rows of 67192): G1 λ=1 Δ−0.0216 P0.00 FAIL; G1n (base-rate normalised) nested Δ−0.0109 P0.04 FAIL; on M_rp same. Closed on ViT-S; retry only with out/w5-dino-vitb.
- M_rp re-PASS on box: +0.00437 P0.989 (== W0-seams 5171c35). OOF-rerank probe not run (V5/V5x/RR1/RR2 already rejected, no new signal).
- runs: bot-W5/runs/v3a_G1*.json, v3a_G1n.json; scripts bot-W5/scripts/{rebuild_geom.py,rp_pairs.py}; work/geom_wkb.pkl, gt_wkb.pkl, w0_disk_pairs.pkl rebuilt in w5data/work.
