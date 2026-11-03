# HANDOFF KOZ3 — АКТУАЛЬНО 02.11.2026 15:42 MSK (W4 agent 5)

**v4 СОБРАН + parity PASS:** `koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip` sha256 `f0def982…7202cef`. Drive [v4c1f_f0def982](https://drive.google.com/drive/folders/1h8AAWDV6xJkgxN7FwZXeLMYaAIEUx5Ia). Kaggle DS `aaaa0000004/koz3-w4-v4c1f`. Kernel `aaaa0000004/koz3-w4-v4chk` **PASS** (~15:35 MSK): D/C/A/B byte-identical smoke `2fea6d16…` ref103 `2bd9a361…`; REF_v3on differs (C1F); NEG fail-as-designed. Outs `/workspace/koz3/out/v4chk/`. v4 остаётся primary; C1FP→v5 FAIL (не формировать). Hold LB — сабмит только Елена через A. Порядок: v4 → rollback v3_on (1406c01a…) → v2clean (545ef32a…).

**W4 сейчас:** STANDBY. satdet-frcnn и dfine2-a/b COMPLETE → гейты прогнаны, все FAIL (см. ниже). Hold LB.

## UPDATE 02.11.2026 18:40 MSK — deep freeze
- satdet-frcnn **FAIL** (`bot-W4/runs/gate_satdet_frcnn.md`), dfine2-a **FAIL** (`gate_dfine2a_vs_v4.md`), dfine2 full ENS8 **FAIL** (Δ+0.00525 P0.9545; new34 −0.00116, kt −0.0030, arkh −0.0221; `gate_dfine2_full_vs_v4.md`), COMP_FL **FAIL** (`regate_COMP_FL_notes.md`), GORLOC/KMASK closed, C1FP→v5 FAIL.
- v4 остаётся primary; STANDBY; Hold LB (только Елена).
- **Полный handoff: `orch/HANDOFF_NOW_20261102.md`** (читать первым).

---
