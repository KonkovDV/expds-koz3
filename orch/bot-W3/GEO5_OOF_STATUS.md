# geo5 OOF CPU fill (bot D) — upd 30.10 23:53 MSK. MagGr E1_G1 frozen; no LB, no GPU, no main.
Work: multi-seed geo5 fold refits (F1..F5 gate folds, clean-room pix_clean_train.fit), s0 already on box. Cache = w4geo/work/feat{,_n40r8} (cache_clean/w4clean not needed).
| acc | kernel | line / seeds | dataset | status | ETA |
| usertest1234 | koz3-w3-geo5-l-s12 | n40r8L finds (127lv/mdl50/800r, maggr) s1,s2 | koz3-w3-geo5-feat-n40r8 (2.2G) | RUNNING | ~4–6h |
| aaaadg4gg4g4g | koz3-w3-geo5-l-s34 | n40r8L finds s3,s4 | koz3-w3-geo5-feat-n40r8 | RUNNING | ~4–6h |
| aaaa0000004 | koz3-w3-geo5-base-s12 | base feat finds+poly (defaults 63lv/400r) s1,s2 | koz3-w3-geo5-feat-base (548M) | RUNNING | ~1.5–3h |
| dmitrijkv (acc2) | koz3-w3-geo5-base-s34 | base finds+poly s3,s4 | koz3-w3-geo5-feat-base | RUNNING | ~1.5–3h |
| elenakonkova (acc1) | koz3-w3-geo5-l-s56 | n40r8L finds s5,s6 | koz3-w3-geo5-feat-n40r8 | RUNNING | ~4–6h |
Leases (bot-W3-kurgan, to 31.10 09:40 MSK): slot:{usertest1234,aaaadg4gg4g4g,aaaa0000004,acc2}/cpu (+acc1/cpu to 09:49) + kernel:/dataset: per row.
Output: /kaggle/working/models_{L,base}/F{1..5}_s{N}/ → pull to w4geo/work/models_n40r8L / models (s0 untouched).
Smoke on box: row counts per fold == s0 meta (L 6060186; base F1 514476/2272811). Kernel src: orch/bot-W3/kaggle_geo5/k_*.
Not done here (box step after pull): probs dump + decode + seed-ens compose/gate for E1 recipe — needs A/C decision whether to re-gate seed-ens vs frozen s0 E1_G1.
ETA is estimated (box fit ~15–20 min/L-fold @3 thr), not measured on Kaggle.
v1 of all kernels ERROR (dataset mount path is not /kaggle/input/<slug>); fixed to recursive find of code__pix_clean.py, v2 pushed ~23:47 MSK → all 5 RUNNING at 23:52 MSK; ETA counts from 23:47.
After COMPLETE+pull: seed-ens vs frozen E1_G1 → A hands C re-gate (no LB). Ping A on first COMPLETE batch.

## 31.10 00:40 MSK — first COMPLETE batch (watch routine)
- aaaa0000004/koz3-w3-geo5-base-s12 + dmitrijkv/koz3-w3-geo5-base-s34 COMPLETE. Pulled raw → orch/bot-W3/geo5_out/<kernel>/; copied F{1..5}_s{1..4} (20 dirs: finds.txt/meta.json/poly.txt) → w4geo/work/models (s0 untouched).
- Released dataset:{aaaa0000004,dmitrijkv}/koz3-w3-geo5-feat-base (base slot/kernel leases already absent).
- L kernels (usertest l-s12, aaaadg l-s34, elena l-s56) still RUNNING; leases to 09:32/09:49.
## 31.10 00:50 MSK update
- base s12 (aaaa) + s34 (acc2) COMPLETE 00:25, pulled → w4geo/work/models F*_s1..s4.
- aaaa0000004/koz3-w3-geo5-n40r8-s12: n40r8 non-L (63lv/mdl100/400r, maggr finds-only; LINE=n40r8 in run_tpl.py), seeds 1,2, dataset aaaa0000004/koz3-w3-geo5-feat-n40r8. RUNNING 00:45. ETA ~2–3h (est). Leases slot:aaaa0000004/cpu + kernel: + dataset: to 31.10 10:40.
- dmitrijkv/acc2 skipped for new push: vitb TPU still QUEUED on dmitrijkv (and elena) blocks new launches until cancel or 03:00 quota (per A).
- L s12/s34/s56 still RUNNING (L line rejected by A for ens; left running, not cancelled).
- Track 2 box seed-ens base s0..s4: orch/bot-W3/geo5_seedens/run_bens.sh (log run.log).
- 00:56–57 parent stopped+deleted all L/n40r8 geo5 kernels, leases released. Ens6 → ENS6_STATUS.md
