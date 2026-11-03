# koz3-w4-li-post-full

Full-stack honest-10 for `[li_post]` **thr_nested** / **thr_nms8** vs **B′ = 0.315800**.

## B′ how
Documented Ae-ON / flags-off honest-10 baseline (`orch/HANDOFF_NOW`, tta-gate `bprime`, sat2-gate).
Artifact: `bprime_counts.pkl`. Else gate vs **A** `base_counts.pkl` (0.314615) and report `score − 0.315800`.
Pipeline: `stack_arch.geojson` → `prep.py` → `gate_v6c.py` (103 families). Per-layer OOF: `elenakonkova/koz3-w2-oof`; stack li* already production-thresholded.

## Inputs
| Source | Role |
|---|---|
| `elenakonkova/koz3-w2-gate` | `w5base/{stack_arch,gt_train_v2,base_counts}.geojson/pkl` |
| `elenakonkova/koz3-w2-oof` | attached (reference); apply is **in-place** on stack li* |
| floors | nested s0 LOFO median: povr/tselye **0.30** (train folds only) |

**Missing on box:** `stack_arch`, `base_counts.pkl`, `bprime_counts.pkl`; fam0 only `_counts.json` (no jsonl). w2-gate download 403 from box pool.

## Variants
1. `baseline` — stack identity  
2. `thr_nested` — class_tau floors  
3. `thr_nms8` — class_tau + self_nms@8  

Flags stay OFF in `configs/default.toml`; kernel applies levers explicitly.

## Push (coord only)
```bash
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/li-post-full
```

## Dry-run (no w5base → MISSING_INPUT DONE)
```bash
LI_POST_FULL_OUT=/workspace/kaggle/push/li-post-full/dryrun \
  python3 /workspace/kaggle/push/li-post-full/koz3-w4-li-post-full.py
```
