# UTILIZATION KOZ3 — NOW 30.10 13:22 MSK (A coord)
Правило: ни одного простаивающего слота с готовой задачей; GPU не трогаем до 31.10 03:05; main/LB не трогаем. Елена: max CPU/GPU/TPU.

| slot | состояние | ядро / задача | аренда |
|---|---|---|---|
| acc1/gpu-1 | COOLDOWN до 31.10 03:05 | бережём под r50c5 f01 (P-W2-9) | — |
| acc1/gpu-2 | BUSY | elenakonkova/koz3-w5-dino-emb-gpu RUNNING (остаток квоты) | coord до 19:00 |
| acc1/tpu | QUEUED (реал) | elenakonkova/koz3-w5-dino-vitb-tpu; smoke koz3-acc2-tpu-smoke ОТМЕНЁН 13:18 (v2 CPU noop → COMPLETE) | coord до 22:08 |
| acc1/cpu | BUSY | elenakonkova/koz3-w3-kurgan-cpu-seeds RUNNING (seeds 79,80) | coord до 23:00 |
| acc1/cpu-2 | BUSY (с 13:19) | elenakonkova/koz3-w3-kurgan-cpu-seeds-b RUNNING (seeds 81,82; CI2-7/PATH#9; NON-CLEANROOM — только исследование) | coord до 23:19 |
| acc2/gpu-1,2 | COOLDOWN до 31.10 03:05 | после сброса r50c5 f234 | — |
| acc2/tpu | IDLE (честно) | koz3-tpu-cnx-c3sc-train ОТМЕНЁН 13:18 (v2 CPU noop → COMPLETE); готового одобренного TPU-ядра нет (BREAKTHROUGH §4) | свободен |
| acc2/cpu | BUSY | dmitrijkv/koz3-w3-kurgan-cpu-seeds RUNNING (seeds 69,70) | coord до 17:14 |
| acc2/cpu-2 | GO WS-G2 | D push non-smoke (gorheads ERROR, no re-push) | bot-W3-kurgan до 21:23 |
| usertest/aaaa/aaaadg cpu | UNKNOWN | нет kaggle_pool_4x2_t4.json | истекают 14:08/17:50 |
| box/q1 | leased C | parity v2→v3a→xlayer→WS-G1 | bot-W4-gate до 19:11 |
| box/cpu | leased C | WS-G1/OOF-rerank | bot-W5-rerank до 19:12 |
| box/q2 | leased D | e2e Ae V7 + d3_gate | bot-W1-aeor до 19:16 |

Квоты: GPU elena ~0.35 ч / dmitrij 0 → сброс 31.10 03:00; TPU 20 ч×2 (acc2 TPU без задачи). След. сверка: vitb старт, gorheads фикс (D).
- 13:25 MSK | acc2/cpu-2 RUNNING dmitrijkv/koz3-w3-wsg2-cpu (D WS-G2).

## 30.10 13:30 MSK — orch-audit (A executor, проверено kernels status по 5 акк)
- ПРАВКА: elenakonkova/koz3-w5-dino-emb-gpu = **COMPLETE** (не RUNNING) → acc1/gpu-2 фактически свободен (GPU cooldown до 31.10 03:05), аренда slot:acc1/gpu-2 до 19:00 устарела и **без поля kernel** → watcher её не снимет и не сделает pull. Нужно: pull → out/w5-dino-emb-gpu, release.
- Пул CPU (файл есть, chmod 600, 3 username): usertest1234/aaaa0000004/aaaadg4gg4g4g koz3-w3-kurgan-cpu-seeds = RUNNING (usertest с ~08:08 → 12ч-кэп ~20:08; aaaa/aaaadg с ~10:51 → ~22:51). check4 aaaa/koz3-w4-v2cleanchk2 COMPLETE PASS (sha_ok). aaaadg/koz3-w5-cloudfeat ERROR (старый, заменён elena COMPLETE).
- IDLE сейчас: acc2/tpu (20 ч), usertest/aaaa/aaaadg **cpu-2** (×3, в SLOTS watcher-а их нет), box ≈6 из 8 ядер (load 1.2; box/cpu у B «WS-182 if ready» — без процесса), acc1/gpu-2 (cooldown, ок).
- QUEUE.dispatch: ready=0 → watcher fill_free нечего пушить; устаревшие pushed: w2-hsparity-ci2-8, w2-lihm-c3sc, w4-v2clean-check2 (все COMPLETE) — закрыть. Слоты без kernel-поля: acc1/gpu-2, acc1/cpu-2, acc2/cpu-2 (kernel-аренды отдельно) → авто-release не сработает.
- r50c5: STATE = f01 elenakonkova + f234 dmitrijkv; dispatch w2-r50c5 = только dmitrijkv/hold → свести до 02:55.
