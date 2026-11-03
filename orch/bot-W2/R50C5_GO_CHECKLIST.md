# P-W2-9 r50c5 — GO checklist (B / bot-W2-lidar)
## Elena night 31.10 (supersedes dual-account plan)
- **ONE GPU** on `aaaa0000004` after L/n40r8 cleared (~03:05).
- Kernel: `aaaa0000004/koz3-w2-cnx-r50c5-f01` from `w2/kernels/aaaa0000004/koz3-w2-cnx-r50c5-f01/`
- **koz3-s3-cred DELETED** — `dataset_sources: []`; S3 only via Kaggle Secrets `KOZ3_S3_*` on aaaa.
- Push **only** after A says slot+cred clean (~02:45–03:05) + A frees lease. Do not self-start.
- Stop after **first honest slice** if damaged kurgans don’t rise (early_stop / honest Li). No LB.
- f234 / elena / dmitrijkv variants: hold; night = one slice on aaaa only.

## Pre-push verify
1. `kernel-metadata.json`: id=`aaaa0000004/koz3-w2-cnx-r50c5-f01`, enable_gpu=true, dataset_sources=[], no s3-cred
2. Secrets on aaaa: KOZ3_S3_ACCESS_KEY_ID / KOZ3_S3_SECRET_ACCESS_KEY (A confirms)
3. `orch/lease.sh acquire slot:aaaa0000004/gpu-1` (or slot A names) + `kernel:aaaa0000004/koz3-w2-cnx-r50c5-f01`
4. `orch/pool_kaggle.sh aaaa0000004 kernels push w2/kernels/aaaa0000004/koz3-w2-cnx-r50c5-f01`
5. After OOF: `venv/bin/python w2_tools/early_stop_r50c5.py <out>` — FAIL / no damaged-kurgan rise → stop, no f234, STATUS→A
6. No main / leaderboard / TPU. Report A ≤10 lines.

## Also stripped (standby, not night push)
- `elenakonkova/koz3-w2-cnx-r50c5-f01` dataset_sources=[]
- `dmitrijkv/koz3-w2-cnx-r50c5-f234` dataset_sources=[]
- `elenakonkova/koz3-w2-hmcp-c3sc-f01234` dataset_sources=[]

## History
- 30.10 13:35: f01=elena f234=dmitrijkv verify
- 31.10 ~00:56: Elena night → aaaa ONE GPU; strip s3-cred; wait A slot+cred clean

## S3 without User Secrets (Elena declined web login) — 31.10 01:32
1. Before push: `python3 w2/kernels/aaaa0000004/koz3-w2-cnx-r50c5-f01/_materialize_s3_env.py`
2. Kernel loads sibling `_s3_local.env` → AWS_* (gitignored)
3. Push private; then `rm -f w2/kernels/aaaa0000004/koz3-w2-cnx-r50c5-f01/_s3_local.env`
4. Verified materialize→scrub dry-run OK on box (no env left on disk)

## 31.10 03:37 — aaaa ERROR (no GPU on pool)
- aaaa0000004/koz3-w2-cnx-r50c5-f01 ERROR `--require-gpu`. **Do not re-push aaaa.**
- A releases aaaa gpu-1. Wait Elena: rerun f01 on **dmitrijkv** or **elena** (+ `_s3_local.env` + Auto-review click) OR stop night.
- f234 untouched. Prep: elena f01 + dmitrijkv f234 have materialize+sibling load ready; push only after Elena+A GO.


## 31.10 09:08 supersede
Elena via A: NOT chasing 0.61 today. Protect archive **0.41238** + honest r50c5 lidar check (gain on measure, not LB).
COORD 08:41–08:54: dual GPU OK (elena f01 + dmitrij f234). Ignore older ONE-GPU/aaaa notes above for today.
On f01 early_stop/STOP → cancel f234 + release acc2 + ping A (D gordense).

## Elena 31.10 09:24 — вектор до 11.11
- Hold 0.41238; lidar cut only; no LB; dome/slope_gate OFF.
- Damaged on f01 not growing → cancel f234.
- main 93d8cf2; archive/worked-20261031 a14428c.

## Elena 31.10 10:24 — ETA
- f01 hard end ~14:41 (DEADLINE_H=6). Fold1 only after BOTH jobs; ignore c3-sc 2h.
- f234 ~20:11 (DEADLINE_H=11.3). Keep until f01 STOP.
