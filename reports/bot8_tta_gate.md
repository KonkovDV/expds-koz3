# bot-8 · TTA/MS gate package (`koz3-tta-gate`)

**Когда (Europe/Moscow):** 2026-11-01 ~19:20 MSK  
**Ветка пакета:** `bots/bot8-tta-gate` @ tip (based on `f3aafa42` = tip `bots/bot8-orsp-tta`)  
**Draft PR:** https://github.com/KonkovDV/expds-koz3/pull/23 (base `bots/bot8-orsp-tta`)  
**Пуш-пакет:** `/workspace/kaggle/push/tta-gate/`  
**Slug:** `dmitrijkv/koz3-tta-gate` (**не** `koz3-ae-dfine-gate` — тот slug у bot3 Ae gate)  
**Не пушим / не гоняем Kaggle.** Bot 1. LB-submit запрещён (правило P).

## Branch state (steering: PR #19 / orsp_tta)

| ref | note |
|---|---|
| `origin/bots/bot8-orsp-tta` | tip **`f3aafa42`** — cloud agent: `[orsp_tta]` U-Net + D-FINE d4/d8/gsd WBF |
| earlier bot8 | `35ca3df` / `81b1d8d` / `82a167c` / `73b1fee` — **ancestors of f3aafa4** (continuous) |
| **PR #4** | MERGED — head `bots/bot8-orsp-tta` |
| **PR #19** | OPEN draft — same branch tip `f3aafa42` |
| this package | `bots/bot8-tta-gate` from **`f3aafa42`**; reuses `expds/orsp_tta.py` + Ae `rot90_tta` |

## Sources / Ae weights

**Kernel id:** `dmitrijkv/koz3-tta-gate` (unique).  
**Do not** push as `dmitrijkv/koz3-ae-dfine-gate` (bot3 Ae gate — overwrite risk).

| source | role |
|---|---|
| dataset `dmitrijkv/koz3-w1-ae-ft-f4pack` | Ae D-FINE weights until gate done — sha **`249b86fb`** + `thresholds_oof.json` |
| kernel `dmitrijkv/koz3-ae-dfine-gate` | **attach after bot3 Ae gate PASS** (read-only); prefer gate output if it promotes a newer pack |
| dataset `dmitrijkv/koz3-s3-cred` | S3 |
| kernel `dmitrijkv/koz3-dfine-ae` | Ae code lineage |

## Variants vs B′ = 0.315800

| id | overlays |
|---|---|
| `bprime` | `[orsp_tta] enabled=false`, `rot90_tta=false` |
| `tta_ae` | Ae `rot90_tta=true` (+ flip) |
| `tta_ms_sat` | `[orsp_tta]` `unet_tta=4`, `unet_scales=[0.75,1.0,1.25]` |
| `both` | Ae rot90 + scales `[0.75,1,1.25]` + sat TTA/MS |

Gate: Δ≥0.003, P≥0.9, layer drop ≤0.01. Resumable STATUS/SHA256.

---

## (a) Expected LB contribution — **ESTIMATE**

**Label: ESTIMATE** (not measured; do not write EXPERIMENTS.csv from this).  
Source log: `/home/box/shared/koz3/reports/lb/scoring_v2_eb268d6_0.41238.log`  
(`submission_bot7cand_v2_eb268d6.zip`, `final_score ≈ 0.4123779189`, wall **3 мин 48 с**).

### Formula

```
final_score = Σ_c (w_c · F1_c) / Σ_c w_c
```

Per-class weights (`expds/constants.py` / `materials/compute_metrics.py`), Σw = 100:

| class | w |
|---|---:|
| kurgany_povrezhdennye | 30 |
| kurgany_tselye | 20 |
| gorodishcha | 15 |
| arkhitektury | 10 |
| object_poly | 10 |
| finds_points | 10 |
| fortifikatsii | 5 |

Auxiliary dtype P/R in the LB log (**do not** enter `final_score` directly):

| dtype (log) | P | R |
|---|---:|---:|
| аэрофото (ae) | 0.0909 | 0.0133 |
| спутник (sp) | 0.4561 | 0.2546 |
| ортофото (or) | 0.3684 | 0.1400 |

At fixed P, ∂F1/∂R ≈ 2P²/(P+R)² → ae ≈ 1.52, sp ≈ 0.82 per +1.0 R.  
Mapping to `final_score`: `Δfinal ≲ (W_affected/100) · ΔF1_class` with W_affected for kurgan+fort ≈ 55, but only the **share of GT on that markup** counts; on eb268d6 ae R=0.013 vs soft R=0.622 → f_ae ≪ 1.

### Assumed recall gains (TTA/MS overlays vs Ae-ON B′, literature-scale)

| overlay | assumed relative lift on affected markup | notes |
|---|---|---|
| Ae rot90 (± light MS) | +0.5–2% rel. on ae det AP / few near-miss TPs | det-TTA lit.; bot3 f4pack kurgan OOF still weak |
| sat U-Net tta=4 × scales×3 | +0.5–2% rel. on sp mask F1 | `[orsp_tta]` avg probs |
| both | roughly additive, diminishing returns | P may dip → gate checks layer drop |

### ESTIMATED Δfinal vs Ae-ON flags-off baseline (same f4pack / gate weights)

| band | Δfinal |
|---|---|
| **Conservative** | **+0.0005 … +0.003** |
| **Optimistic** (near-miss kurgans + sat) | **+0.003 … +0.008** |
| Unlikely alone | ≫ +0.01 without bot3 recall-relax / stronger Ae |

Honest-10 gate still requires Δ≥0.003 vs B′=0.315800 before any LB story.

---

## (b) Inference time per variant on LB — **ESTIMATE**

**Base:** T₀ = **3 мин 48 с = 228 s** from the same scoring log (eb268d6).  
**Budget:** each variant ≤ **20 мин = 1200 s**.

### Derivation

Wall shares of T₀ attributed to overlay-able paths (**assumed**, labeled estimate):

| share | fraction of T₀ | affected by |
|---|---:|---|
| rest (li/geo/or/…) | 0.66 | — |
| sat U-Net (`sp_or`/`sp_focal2`) | 0.22 | `[orsp_tta]` |
| Ae D-FINE (when ON in stack) | 0.12 | `rot90_tta` / Ae scales |

Cost factors vs flags-off on that share (`orsp_tta.cost` / dfine rot90×flip×scales):

| variant | ae factor | sat factor | formula | **est. LB wall** |
|---|---:|---:|---|---|
| `bprime` | 1 | 1 | T₀ · (0.66+0.12+0.22) | **3:48 (228 s)** |
| `tta_ae` | 4 (rot90+flip vs flip) | 1 | T₀ · (0.66+0.12·4+0.22) = T₀·1.36 | **≈ 5:10 (310 s)** |
| `tta_ms_sat` | 1 | 12 (tta4×3 scales) | T₀ · (0.66+0.12+0.22·12) = T₀·3.42 | **≈ 13:00 (780 s)** |
| `both` | 12 (rot90+flip × 3 scales) | 12 | T₀ · (0.66+0.12·12+0.22·12) = T₀·4.74 | **≈ 18:01 (1081 s)** |

All four ≤ 20:00 under these share assumptions. If sat share ≳ 0.28 or Ae share ≳ 0.18, `both` can approach the cap — gate must record real wall; use `budget_s` / plan_level if needed.

---

## DRY-RUN (box CPU)

```bash
TTA_GATE_DRY=1 TTA_GATE_OUT=/tmp/tta-gate-dry nice -n 10 python3 koz3-tta-gate.py
```

Exit 0; STATUS=DONE; resume skip OK. Synthetic scores only.  
`slot:box/cpu` held by coord (nice -n 10, no exclusive lease).  
Lease `bot-8` rejected (`OWNERS` = coord|bot-W*).

## Deliverables

- `/workspace/kaggle/push/tta-gate/{koz3-tta-gate.py,kernel-metadata.json,README.md,REPORT.md}`
- wt `kaggle/2026-11-01/tta-gate/` + this report
- Draft PR #23 (package only; no Kaggle push)

## RU (кратко)

1. Уникальный slug `dmitrijkv/koz3-tta-gate` — не трогаем `koz3-ae-dfine-gate` (bot3).  
2. Веса Ae: f4pack `249b86fb` (`koz3-w1-ae-ft-f4pack`); после PASS Ae gate — kernel_source `koz3-ae-dfine-gate`.  
3. Оценка ΔLB (estimate): **+0.0005…+0.008** vs Ae-ON B′; формула Σ w F1 / 100, лог 0.41238.  
4. Оценка wall от базы 3:48: bprime 3:48 / tta_ae ≈5:10 / tta_ms_sat ≈13:00 / both ≈18:01 — все ≤20 мин.  
5. DRY-RUN OK; Kaggle/LB не пушили.
