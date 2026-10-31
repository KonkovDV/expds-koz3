# bot1 — Kaggle/PC unblock — 2026-10-29 17:10 MSK

## TASK A — TPU — DONE (queued useful job)

- Deleted disposable TPU kernels: `koz3-tpu-probe`, `koz3-cnx-li-tpu-smoke`, `koz3-tpu-probe-v3` (freed concurrency=1 slot).
- Pushed **one** useful TPU job: `dmitrijkv/koz3-tpu-cnx-c3sc-train` status **QUEUED** (TPU v5e-8 capacity wait).
  - Built-in probe writes `/kaggle/working/tpu_probe.json` first, then CNX lidar c3-sc XLA train.
  - `CNX_MODE_DEFAULT=smoke` for first verification push.
- Confirmed no other TPU kernel RUNNING/QUEUED (old probe slugs inaccessible).
- GPU left untouched: `koz3-dfine-ae-ft` RUNNING, `koz3-cnx-li-c3sc-f4-v2` RUNNING.

## TASK B — geo200 xfer — BLOCKED on transfer/auth

- PC pack ready:
  - `F:\2\koz3-xfer-geo200-upload` — 328 files, ~3.45 GB, `dataset-metadata.json` id `dmitrijkv/koz3-xfer-geo200`
  - `F:\2\koz3-xfer-geo200-parts` — 37×~90 MB zip parts (manifest earlier: sha256 425b0d5d…)
- CopyToBox from `F:\` **and** from `C:\Sloi\...` → refused ("outside the allowed local-exec root").
- Staging to C: hit disk full (17/37); staging cleaned; C: free ~1.8 GB.
- PC Kaggle CLI 2.2.5 present but **no** `KAGGLE_API_TOKEN` / `kaggle.json` on PC → cannot `datasets create` from F: yet.
- **Next (needs parent/user):** set Kaggle token on PC (User env or `%USERPROFILE%\.kaggle\kaggle.json`) without pasting into chat, then:
  `C:\Sloi\venvs\kaggle\Scripts\kaggle.exe datasets create -p F:\2\koz3-xfer-geo200-upload`
  After dataset ready: stage bot6 kernel attach `koz3-xfer-geo200` (do **not** launch; PC CPU retrain). Patch `rebuild_train_tree` for geo200 slug.

## TASK C — bot5 gor_dense heads — still BLOCKED (not UNBLOCKED)

- Documented source dataset: `dmitrijkv/koz3-b5-gordense-timing` — **never created** (phantom). Owner dataset list has no such slug; status/files → HTTP 403 (same as other missing private slugs; known-good xfer-s000 → ready). Kernel never pushed.
- Canonical files lived on prior box `/workspace/koz3_shared/bot-5/analysis/gor_patch/gor_v3_head.npz` and `.../work/gd/root/models/gor_dense/gor_dense_head.npz` — path gone; not in git (nomodels).
- PC search `C:\Sloi`, `F:\2`, `F:\expds3`: no `gor_v3_head.npz` / `gor_dense_head.npz` / matching sizes 407736 / 204984; `dense.npz` absent. dinov2-small already OK on wt.
- Full write-up: `reports/bot5_v4.md` § TASK C. `pc/QUEUE.md` BLOCKED item updated (no UNBLOCKED tag).
- Unblock needs: recover koz3_shared backup **or** re-export heads (needs dense.npz + feats), then place under `C:\Sloi\wt_bot5_50f6f27\models\{gor_gate,gor_dense}\` and record sha256.
