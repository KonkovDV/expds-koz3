# GORDENSE_STATUS (D → A) 31.10 11:09 MSK
- Kernel: elenakonkova/koz3-w3-gordense-timing-k v1 — RUNNING (pushed 11:08 MSK), slot acc1/gpu-1, lease bot-W3-gordense (until 13:04 MSK).
- Dataset: elenakonkova/koz3-w3-gordense-timing (private): models/gor_gate/gor_v3_head.npz 487c48e4…, models/gor_dense/gor_dense_head.npz d92f38f3… (from dmitrijkv ds, md5 OK), rebuild_train_tree.py, repo_86a60c5 (git archive tar md5 5f481bef… exact; Kaggle unpacked it).
- Data: dmitrijkv/koz3-xfer-s000..s009 (readable by elena). DINOv2 via HF fallback.
- sp_dfine OFF (spor kernel_source not readable by elena) → timing is without D-FINE SpOr. orgate/dome/slope_gate off.
- Slug suffix -k: original slug gave 409 SaveKernel conflict (same precedent as W4 v2cleanchk-k).
- Hard cap HARD_S=3300 (~55 min) → expect done ≤ ~12:05 MSK. Pull → orch/bot-W3/gordense_timing_out.
- dmitrijkv / r50c5-f234 untouched. No LB.
