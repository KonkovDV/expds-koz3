# koz3-w1-sat2-gate — sat2 REPLACE honest-10 + sat_thr_relax sweep

**Push:** bot 1 only — `orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/sat2-gate`.  
**Do not push from bot2.** Hold LB **0.41238**.

## Root cause of v1 FAIL (fixed)
Kaggle **script** kernels upload **only** `code_file` (`koz3-w1-sat2-gate.py` → `/kaggle/src/script.py`).  
Sibling dirs (`vendor/`, `w5_rerank/`, …) are **not** uploaded → v1 `FileNotFoundError: .../vendor/materials/compute_metrics.py`.

**Fix:** all sidecars are **gzip+base64-embedded** in the single script and extracted at runtime to `/kaggle/working/_sat2_bundle`. No new dataset required.

## Contract
| | |
|---|---|
| **goal** | Full honest-10 for sat2 REPLACE; `sat_thr_relax` scale sweep on top |
| **criterion** | Δ≥+0.003 vs A counts; **P = bootstrap p_gt0 ≥ 0.9**; damaged not worse; d_new34/d_optical ≥ −0.001 |
| **P means** | paired-bootstrap P(cand>base) — **not** precision |
| **budget** | Kaggle CPU ~25–60 min |
| **expected ΔLB** | +0.003…+0.006 if pass_full; else 0 |

## Dataset
`elenakonkova/koz3-w2-gate` → `w5base/{stack_arch,gt_train_v2,base_counts}`.  
Gate vs **A** (`base_counts.pkl`); `bprime_counts.pkl` absent on box — OK.

## OOF (Drive soft-PASS)
`data/sp_det2_oof_notta.geojson` from `/workspace/drive_p4/sat2_det2_oof_replace_pack.zip`  
zip sha `c1e54f76c3d5d3d1b5296485c04370048323825424b4f277dd707dd7f41c55cb`  
OOF sha **`afaf8dcaec1fee892db96f164a7843461b194b58949be2a3a8e94c37847f921e`** (2622 feats).

## Rebuild embed after editing sidecars
```bash
cd /workspace/kaggle/push/sat2-gate && python3 build_embed.py
```

## Local dry-run (Kaggle layout)
```bash
# see dryrun/ — only script.py under kaggle/src/, no sibling vendor/
# evidence: dryrun/dryrun.log (extract → prep → gate boot=50 DONE)
```

## Expected outputs (`/kaggle/working/sat2_gate/`)
`STATUS.txt`, `gate_replace_all.json`, `gate_scale_*.json`, `sweep_summary.json`, `SHA256SUMS`

### Pass
`pass_full` ⇔ d≥0.003 ∧ p_gt0≥0.9 ∧ damaged≥0 ∧ d_new34≥−0.001 ∧ d_optical≥−0.001  
Soft-PASS ref: score 0.320516, ΔB′+0.00472, P0.838.

## Push (bot1)
```bash
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/sat2-gate
```
