# bot8 — TTA/MS on bot3 dfine_aeor Ae (f4pack) — EXPECTED LB

| Field | Value |
|---|---|
| **Goal** | Port U-Net TTA/MS + D-FINE Ae rot90/MS (flags OFF) onto bot3 f4pack candidate; PC gate AFTER first bot8 gate |
| **Base** | `bots/bot3-dfine-aeor-eval` @ **`2f46f20a05e0e4c4a1998016db33fe23e22e3404`** |
| **Branch** | `bots/bot8-tta-on-bot3` |
| **f4pack** | sha256 prefix **`249b86fb`**; box `/workspace/kaggle/out/dfine-ae-ft-f4pack/` → PC `C:\Sloi\pc_jobs\weights\dfine_ae_ft_f4\` |
| **Image** | `pantum/n7q:20261024` |
| **LB submit** | **No** (gate only). Defaults unchanged: `[dfine_aeor] enabled=false`, `rot90_tta=false`, `scales=[1.0]`, U-Net `tta/ms` absent = OFF |
| **Updated** | 29.10.2026 ~20:50 MSK (Europe/Moscow) |

Related: `reports/bot3_dfine_aeor_eval.md`, `reports/bot8_orsp_tta.md`, LB log `reports/lb/scoring_v2_eb268d6_0.41238.log`.

---

## What TTA covers (this branch)

| Path | OFF default | ON via overlay |
|---|---|---|
| U-Net ae/or/sp_or | `tta=1`, `ms_scales=[1.0]` | `[tta] ae=4` / `[ms] ae=[0.75,1.0,1.25]` |
| D-FINE Ae (dfine_aeor) | bot3: `flip_tta=true`, `scales=[1.0]`, **`rot90_tta=false`** | `rot90_tta=true` (90/180/270 CCW + cxcywh inverse); `scales=[0.75,1.0,1.25]`; fuse WBF/NMS unchanged; time via `budget_s` + plan_level |
| D-FINE Or | same knobs; gate uses Ae-only markups like bot3 f4 | Or markups if enabled later |

Degradation ladder unchanged: `full → no_flip → one_scale → skip` (no_flip drops flip+rot).

## PC gate (AFTER first bot8 gate @ f6fe20a)

Variants (mock + honest-10 if regions):

1. **base_ae** — bot3 candidate Ae ON f4pack; new TTA flags OFF  
2. **tta_ae** — + U-Net `tta.ae=4` + `rot90_tta=true`  
3. **tta_ms_prio** — + tta ae/or + ms ae/or + dfine rot90 + scales MS  

Runner: `pc/runners/run_bot8_tta_on_bot3_docker.ps1`  
Launch: `pc/runners/launch_bot8_tta_on_bot3_schtasks.ps1` (task `koz3_bot8_tta_on_bot3`)  
JobDir: `C:\Sloi\pc_jobs\bot8_tta_on_bot3_<sha7>`  
Compliance: Docker `scripts/check_compliance.py` on **`out_tta_ms_prio/result.geojson`** (never kit).  
Est. wall: ~15–40 min GPU for 3 mock variants (target single-run ≤20 min with budget_s/plan_level).

---

## EXPECTED LB contribution (ESTIMATE — not measured)

**Label: ESTIMATE.** No PC/LB numbers yet. Do not write EXPERIMENTS.csv from this.

### Scoring formula (from `materials/compute_metrics.py`)

```
final_score = Σ_c w_c · F1_c  /  Σ_c w_c
CLASS_WEIGHTS: kurgany_povrezhdennye=30, kurgany_tselye=20, gorodishcha=15,
               arkhitektury=10, object_poly=10, fortifikatsii=5, finds_points=10
               (Σ w = 100)
```

Data-type P/R (аэрофото / ортофото / спутник) in the LB log are **auxiliary** — they do **not** enter `final_score` directly. They only matter insofar as new TPs/FPs change class F1.

### Sensitivity at fixed precision (dtype as a proxy)

From `scoring_v2_eb268d6_0.41238.log` (`final_score ≈ 0.41238`):

| dtype | P | R | F1=2PR/(P+R) | ∂F1/∂R \|_P |
|---|---:|---:|---:|---:|
| ae (аэрофото) | 0.0909 | 0.0133 | 0.0233 | ≈1.52 |
| or (ортофото) | 0.3684 | 0.1400 | 0.2029 | ≈1.05 |
| sp (спутник) | 0.4561 | 0.2546 | 0.3268 | ≈0.82 |

At fixed P, **+0.01 dtype-R** lifts that dtype-F1 by ≈ ∂F1/∂R · 0.01 (ae ≈ +0.015 F1 points). Mapping to `final_score` requires the share of class GT that sits on that markup:

```
Δfinal ≲ (W_affected / 100) · ΔF1_class
W_affected for ae-kurgans+fort ≈ f_ae · 55,  f_ae ≪ 1 on eb268d6 (ae R=0.013 vs soft R=0.622)
```

Bot3 OOF F2 for f4pack is weak on `kurgany_tselye` (≈0.028) and modest on povr/fort — so even large relative TTA gains start from a low base.

### Realistic TTA gain range (literature + prior)

- Det/TTA literature (flip + rot90 + light MS, WBF): typically **+0.5–2% relative** AP; rarely >3% without architecture change.  
- Prior bot8 U-Net-only TTA/MS plan: small Δ unless ae detector exists (this branch adds D-FINE Ae).  
- **ESTIMATED Δfinal vs bot3 Ae-ON baseline (same f4pack), TTA/MS overlays:**
  - **Conservative:** **+0.0005 … +0.003** (few new TPs; P may dip)  
  - **Optimistic (near-miss kurgans recovered):** **+0.003 … +0.008**  
  - **Unlikely alone:** jumping Rule P by ≫0.01 without bot3 recall-relax / better Ae kernel  

Gate must confirm: no layer drop >0.01; wall ≤20 min for the chosen ON config; compliance PASS.

## Tests

Light: `pytest tests/test_dfine_aeor.py tests/test_tta.py tests/test_check_compliance.py` — new rot90/ms/kit-guard green; one pre-existing `test_not_packaged_when_off` fails on box (missing `models/pixgbm/all/poly.txt`, unrelated).
