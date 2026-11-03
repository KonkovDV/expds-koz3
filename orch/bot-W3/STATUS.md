время MSK | бот | что | число | ссылка
30.10 12:56 | bot-D (W1+W3) | online: wt/w1→bots/w1-aeor@1b457f3, wt/w3→bots/w3-kurgan-seeds@b259371; аренд веток нет (жду A) | — | /workspace/koz3/wt/{w1,w3}
30.10 12:56 | bot-D | RUNNING: elenakonkova+dmitrijkv koz3-w3-kurgan-cpu-seeds; ae-final+or-final COMPLETE; cloudfeat COMPLETE | — | pool_kaggle status
30.10 12:56 | bot-D | блокер: нет /workspace/koz3/local_ctx/ctx/kaggle_pool_4x2_t4.json → CPU-акки usertest/aaaa/aaaadg через pool_kaggle.sh недоступны | — | orch/pool_kaggle.sh
30.10 12:56 | bot-D | до 03:00 31.10 только CPU/подготовка; GPU не беру. Дальше: дождаться seeds COMPLETE, собрать входы e2e Ae (V7) и CI2-5 без push | — | TASKS W1#2, W3#2 / CI2-5
время MSK | бот | что | число | ссылка
30.10 12:56 | bot-D/W3 | wt/w3 готов; kernels koz3-w3-kurgan-cpu-seeds-* на месте; hold в QUEUE на v5 seeds (coord). 22:44 ERROR уже отрабатывался fallback-ами; сейчас RUNNING на elenakonkova+dmitrijkv | — | wt/w3/kernels/
30.10 12:56 | bot-D/W3 | CI2-5 / w3-gordense-timing: GPU только после аренды A и после 03:00; сейчас код-ревью/подготовка | ready в QUEUE | QUEUE.json
30.10 13:00 | bot-D (W3) | PREP CI2-5: ядра adacrop не было → STUB-заготовка (локально, без push); блокеры: код adacrop в train.py, кэш тайлов/данные, PREREG, аренда A после 03:05 | — | orch/bot-W3/PREP_CI2_5.md, wt/w3/kernels/koz3-w3-bigobj-adacrop/
30.10 13:05 MSK | A (coord) | orch refresh: ROSTER.md, STATE NOW, local_ctx/README, GPU cooldown до 31.10 03:05, watcher запущен; аренды D: branch:bots/w1-aeor + branch:bots/w3-kurgan-seeds до 01.11 12:57 | — | orch/ROSTER.md
30.10 13:15 MSK | bot-D/W3 | gorheads-cpu ERROR: FileNotFoundError gorheads_code.tar.gz (dataset = unpacked tree). Patched kernel → push v2 RUNNING. log→orch/bot-W3/gorheads_out/ | — | https://www.kaggle.com/code/dmitrijkv/koz3-w3-gorheads-cpu
30.10 13:15 MSK | bot-D/W3 | gorheads-cpu v2 RUNNING после фикса unpacked dataset; log orch/bot-W3/gorheads_out/; commit 0a81b30 pushed | — | https://www.kaggle.com/code/dmitrijkv/koz3-w3-gorheads-cpu
30.10 13:17 MSK | bot-D/W3 | kurgan+gorheads RUNNING держу; acc1/cpu-2 idle (нет ready CPU-ядра); D0 TPU → A; WS-G2+PREREG CI2-5 в работе | — | orch/bot-W3/
30.10 13:19 MSK | bot-D/W3 | gorheads-cpu v2 ERROR: unpack OK; SpOr tifs=4 only → validate W shape (1536,39)≠(1536,64); sha≠PC. log→gorheads_out/. Нужен полный train tree / xfer. | — | orch/bot-W3/gorheads_out/
30.10 13:22 MSK | bot-D/W3 | PREREG CI2-5 (draft) + PREREG WS-G2 записаны; WS-G2 selftest OK (closed 1.0/open 0.69/курган 0), инвентарь li gor/fort 5199 канд./190 hit/58 уч.; реальный прогон BLOCKED: нет DEM/LAS и S3-кредов на боксе | — | orch/bot-W3/PREREG_CI2_5.md, PREREG_WS_G2.md, work/wsg2_result.json
30.10 13:23 MSK | bot-D/W3 | gorheads: без полного train-tree не чиню; release acc2/cpu-2 + kernel (если ещё держал). WS-G2 CPU-ядро пишу; push после COMPLETE kurgan-seeds-b на acc1/cpu-2 + lease+GO A | — | orch/logs/gorheads_v2_err/
30.10 13:25 MSK | bot-D/W3 | WS-G2 CPU push dmitrijkv/koz3-w3-wsg2-cpu v1 RUNNING (acc2/cpu-2 lease); S3 LAS→DEM→wsg2; pull→orch/bot-W3/wsg2_out/ | — | https://www.kaggle.com/code/dmitrijkv/koz3-w3-wsg2-cpu

- 31.10 11:09 MSK | D | gordense-timing PUSHED on elenakonkova acc1/gpu-1 (lease bot-W3-gordense until 13:04): elenakonkova/koz3-w3-gordense-timing-k v1 RUNNING. Dataset elenakonkova/koz3-w3-gordense-timing created (heads from dmitrijkv ds, md5 OK; repo = git archive 86a60c5, tar md5 5f481bef… exact). Slug -k after 409 slug conflict. HARD_S=3300, orgate/dome/slope_gate off, sp_dfine OFF (no spor kernel_source). dmitrijkv/f234 untouched. No LB.

- 01.11 11:11 MSK | D | idle standby — GPU/TPU off; V7 e2e / CI2-5 / gordense only after A GO+lease; branches →04.11 11:10; hold 0.41238

- 01.11 11:39 MSK | D | ACK t225u FULL GO — STANDBY снят. Prep 2nd sat (FCOS/CenterNet) + merge; await A PROMPT. V7/CI only after lease. Hold 0.41238, no LB.

- 02.11 15:04 MSK | D/W3 | START PREREG_COMP_GORLOC (then KMASK) per 5aa57b9 on v4 zip koz3_v4c1f_07795d2_51882efd_w6dfd2e07 (sha256 f0def982…cef verified). Baseline OOF = W4 v4build/gate C1 replay (out_C1.geojson, read-only). Post-hoc on v4 final output, all τ frozen, new group gor_loc (L2) by metric rule. CPU only, no LB, no default.toml. Docs → orch/bot-W3/prereg/
- 02.11 15:22 MSK | D/W3 | DONE GORLOC+KMASK measurement on v4 out_C1 (0.361641). GORLOC nested: L1 +0.0013 P0.60; L2 +0.0018 P0.65 new34 −0.0013; L2×m +0.0039 P0.81 new34 −0.0007 — none meets bar (measurement only, W4 re-gates). KMASK oracle ceiling +0.00065 < 0.002 → no gate; nested +0.00004. Files RESULT_PREREG_COMP_{GORLOC,KMASK}.{md,json}. No LB/no default.toml
