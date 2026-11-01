# Combo gate: sat2 + FR1h (honest-10 / gate_v6c)

A=0.314615  B′=0.315800  boot=4000

| variant | score | ΔA | ΔB′ | P(gt0) | p5 | damaged | pass_full |
|---------|-------|-----|------|--------|-----|---------|-----------|
| sat2 REPLACE scale 1.0 | 0.326450 | +0.011835 | +0.010650 | 1.0 | +0.00477 | 0.03429 | yes |
| sat2+FR1h | 0.325626 | +0.011011 | +0.009826 | 1.0 | +0.00427 | 0.03178 | yes |

**Chosen: sat2_only** (sat2+FR1h < sat2 alone). FR1h dropped from pack.

Artifacts: `/workspace/koz3/out/combo/gate/COMBO_GATE.json`, `w6c_gate_sat2.json`, `w6c_gate_sat2_fr1h.json`.

Submission: `submission_sat2_on_eb268d6.zip` sha256 `573eb87251fc967eb4dffd5d610d23b0aa1226dfa92520c8635be8da0bae88e9` (~1069 MiB).
sp_det2 ON in candidate config; default.toml keeps feature OFF (commented). Release profile `sat2` enables via `apply_release_profile.py`.
