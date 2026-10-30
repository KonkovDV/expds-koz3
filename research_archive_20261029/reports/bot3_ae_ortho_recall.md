# bot-3 ae recall — Fix A+B + Ae tiling audit (2026-10-29)

**MSK:** 2026-10-29 20:38  
**Branch:** `bots/bot3-ae-ortho-recall`  
**Tip:** `0bcb03985253172fae631fb7bd62f6da807d5b1f` (Fix C code; branch tip may include report follow-ups; base `eb268d6`)
**Diagnosis:** `reports/ae_ortho_diagnosis.md` (bot-1)  
**Scope (steering):** Ae D-FINE tiling/scale only. Orthophoto multi-class owned by **bot6** (OOF dfine-or f0–f2) — not duplicated here.

## Root cause (confirmed)

On LB 0.41238 @ eb268d6, aerial uses **only** fortification mask at τ≈0.998. No kurgany from ae network; **D-FINE not in submission**. Recall ae ≈0.013 is by construction.

## Shipped (default-OFF on tip)

| Fix | Flag | ON behavior | Expected |
|-----|------|-------------|----------|
| **A** | `[dfine_aeor]` | Ae-only D-FINE, weights f4pack **249b86fb**, OOF `thresholds_oof.json` | +0.02–0.05 LB |
| **B** | `[ae_recall_relax]` | ae fort→**0.995** (+ optional li_ae/fusion) | small ae F1 bump |
| **C** | `[dfine_ae_tile]` | Patches A: `windows_mode=train` + `fuse=nms` (OOF-matched) | fix train↔infer edge/fuse drift |

Code: `expds/{dfine_aeor,sp_dfine,ae_recall_relax,dfine_ae_tile}.py`; pipeline hooks; `build_submission` FLAG_MODULES for A; unit tests `test_bot3_ae_recall_relax.py` (4/4), `test_bot3_dfine_ae_tile.py` (6/6 manual).

## Ae D-FINE train↔infer tiling / scale audit

**Source of truth:** `/workspace/kaggle/push/koz3-dfine-ae-ft/koz3-dfine-ae-ft.py` + f4pack `dfine_report.json` (sha249b86fb).

| Param | Train OOF / report | Infer `dfine_aeor` DEFAULTS | Match? |
|-------|--------------------|-----------------------------|--------|
| res_m (ae) | 0.5 | 0.5 | YES |
| tile (ae) | 640 | 640 | YES |
| overlap (ae) | 160 | 160 | YES |
| min_score | 0.05 | 0.05 | YES |
| max_det | 100 | 100 | YES |
| nms_iou | 0.5 | 0.5 | YES |
| flip_tta | True | True | YES |
| scales | [1.0] (no multi-scale OOF) | [1.0] | YES (lever OFF) |
| **window grid** | `range(0, max(1,n-overlap), step)` (=sp_dfine) | **aeor** pin last to `n-tile` | **NO** |
| **cross-view fuse** | single `batched_nms` after windows×flip | default **wbf@0.55** | **NO** |
| budget_s | N/A (fold deadline) | 120 s phase → may `skip` large Ae | runtime risk, not train mismatch |

**Evidence (window last-x, tile=640 overlap=160):**

| W | train last-x | aeor last-x |
|---|-------------:|------------:|
| 1000 | 480 | 360 |
| 1471 | 960 | 831 |
| 16656 | 16320 | 16016 |

Fix C (`[dfine_ae_tile]`) sets `windows_mode=train` + `fuse=nms` when ON. Default remains aeor/wbf so A+B eval byte-path unchanged until C is flipped.

Optional levers (still default-OFF / unset): `scales=[1.0,1.5]`, `budget_s` raise, `overlap_ae` / `tile_ae`.

## Incidental ortho note (for bot6 — not owned here)

- `expds/naming.py:35` — keyword `ортофото` → code `Or` (markup `or`).
- `expds/pipeline.py:662` — `optical_feats` loops `mt in ("ae","sp_or","or")`; Or path uses segmenter weights from cfg.
- `configs/default.toml:213` — `or = {path = "models/surf_arch_r18.pt", classes = ["arkhitektury"]}` — architecture-only; no multi-class Or in v3 path.
- A+B ON candidate uses `dfine_aeor.markups=["ae"]` only — does not touch Or D-FINE. Hand off multi-class Or / OOF f0–f2 to bot6 (`reports/bot6_ortho_multiclass.md` when published).

## Queued validation (A+B only — no new GPU for C yet)

1. **PC GPU Docker** — READY-TO-LAUNCH (A+B, tile flag OFF)  
2. **Kaggle** e2e-after-calib — QUEUED (~23:10 MSK)

## Prepared (NOT queued) next eval — Fix A+B+C

After A+B gate returns positive / non-regressing:

```text
SPEC: bot3_ae_tile_trainmatch
Branch: bots/bot3-ae-ortho-recall (tip with Fix C)
Base: eb268d6
Flags ON:
  [dfine_aeor] enabled=true markups=["ae"] weights={ae=["models/dfine_ae"]}
               thresholds_file="models/dfine_aeor/thresholds_oof.json"
  [ae_recall_relax] enabled=true
  [dfine_ae_tile] enabled=true   # windows_mode=train fuse=nms
Weights: f4pack 249b86fb → models/dfine_ae + thresholds_oof.json
Runner sketch: clone A+B docker runner; inject [dfine_ae_tile] enabled=true
Assert: n_raw ae > 0 (reuse assert_dfine_aeor_ae_nonzero.py); compare LB vs A+B-only
Do NOT queue until A+B PC/Kaggle results land.
```

## How to enable A+B+C locally

```toml
[dfine_aeor]
enabled = true
markups = ["ae"]
weights = { ae = ["models/dfine_ae"] }
thresholds_file = "models/dfine_aeor/thresholds_oof.json"

[ae_recall_relax]
enabled = true

[dfine_ae_tile]
enabled = true
```
