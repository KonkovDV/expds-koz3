# TPU hedge access (acc2 = dmitrijkv) — 30.10.2026 15:05 MSK

## Root cause
Hedge `w5-dino-vitb-tpu-acc2-hedge` (QUEUE, HOLD) = kernel `wt/w5/kaggle/koz3-w5-dino-vitb-tpu` (dino_emb.py, TpuV5E8).
`kernel-metadata.json` dataset_sources:
- `elenakonkova/koz3-w5-cands` (cands.csv.zst, 1.9 MB)
- `elenakonkova/koz3-cnx-li-prep` (cnx_relief/*.npy+json, ~800 MB)
- `elenakonkova/koz3-s3-cred` (cred.json, 85 B — S3 keys organizer bucket train-expds-3)

All three are **private, owner elenakonkova, collaborators = none** (checked via `datasets metadata`).
Verified: elenakonkova `datasets files` → OK; dmitrijkv → **403 Forbidden** on each. Kaggle private datasets are visible only to owner + collaborators, so a kernel pushed from dmitrijkv cannot attach them. Nothing is broken — they were simply never shared.

Original vitb `elenakonkova/koz3-w5-dino-vitb-tpu` still **QUEUED** (15:04 MSK, since ~13:08).

## Fix status: PREPARED, NOT APPLIED (needs Elena's explicit OK)
Kaggle API *does* support it: `kaggle datasets metadata --update` with `collaborators` (role reader) — no UI needed.
Prepared: `orch/patches/tpu_hedge_share/<ds>/dataset-metadata.json` (isPrivate stays true, +dmitrijkv READER) and `apply_share.sh` (applies + re-checks access as dmitrijkv). Not run: sharing private data — incl. S3 creds — with another account requires the owner's explicit approval.
Note: `--update` resends title/subtitle/description/licenses from the file (copied verbatim from current metadata, so unchanged).

### Option A (recommended): Elena says «да, расшарь dmitrijkv» → run `orch/patches/tpu_hedge_share/apply_share.sh` (~1 min).
### Option B: Elena does it in UI, per dataset (3×):
kaggle.com/datasets/elenakonkova/<slug> → ⋮ / Settings → Sharing (Collaborators) → Add collaborator → `dmitrijkv` → role **Can view** → Save. Keep Private.
### Option C: copy under dmitrijkv (no cross-share): download 3 ds as elenakonkova (~800 MB, ~2–3 min) → `datasets create` as dmitrijkv private (upload + Kaggle processing ~10–20 min). Still puts S3 creds on a 2nd account; same approval needed. Then kernel_sources must point to dmitrijkv/*.

## Before hedge push (any option)
Hedge kernel copy must use `id: dmitrijkv/koz3-w5-dino-vitb-tpu` (current metadata id is elenakonkova/...). Re-check acc1 vitb status: if it has started → cancel hedge instead.

## Unhold
After share verified (dmitrijkv `datasets files` OK) AND acc1 vitb still QUEUED → hedge condition (2) met; parent/A may ACK unhold (condition (1) already met: QUEUED >1h).

## Applied 15:07 MSK
Elena: «да, расшарь» + «доступ приватный». apply_share.sh OK; dmitrijkv files list OK on all 3.
