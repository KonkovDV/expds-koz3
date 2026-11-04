# Q6 (candidate GBM for finds) — W4 → W3 handoff, 02.11 13:55 MSK
Stopped by coord (re-prioritised). Clean room was kept (see orch/bot-W4/CLEANROOM_READLOG.md).

## Done
- PREREG 870433c + amendment A1 a4da8a9 (gate-side exact finds matching; committed before any variant score).
- expds/finds_cand.py (8e0df63): candidates = peaks of |highpass(robust_z)| on decimated maggr gray, 25 descriptors; NOT wired into pipeline.
- Scripts (db73cd7, `scripts/`): survey → q6_extract (per-region candidate cache) → q6_oof (LightGBM per honest-10 fold, OOF probs → ex_fgbm.geojson 26796 feats)
  → q6_prep (sanity + nested τ for maggr_fgbm, frozen others at taus_V3P; plus unfrozen REPu/UNIu) → q6_replay (W1 s2_replay copy, archive b447649) → q6_score (W1 s3 gate part, Bonf P≥0.95).
- Sanity: gate base 0.3337867 = V3P; UNI τ=∞ 0.3337867 (after A1); production base replay 0.345439 ✓.
- Gate (nested, frozen): UNI 0.335056 (+0.0013), REP worse (≤0.3155).
- **Production replay (decisive) — all FAIL** (q6_gate.json):
  | var | all Δ | P | new34 Δ | nolidar Δ | finds F1 (base 0.506) |
  |---|---|---|---|---|---|
  | REP  | −0.01637 | 0.00 | −0.0528 | −0.0164 | 0.272 |
  | UNI  | −0.00748 | 0.19 | +0.0217 | −0.0099 | 0.447 |
  | REPu | −0.01784 | 0.00 | −0.0528 | −0.0180 | 0.251 |
  | UNIu | −0.00871 | 0.11 | +0.0208 | −0.0112 | 0.434 |

## Open findings / what's left (if W3 continues)
- Gate↔production gap: in UNI, classic finds TP drops 1603→1268 in production although classic τ is unchanged — production post
  (merge_within_markup on maggr circles and/or filter_finds_points_regimes: large-survey conf≥0.60, settlement ≥0.89) interacts
  with the extra maggr finds. The gate does not model this. A fix needs either a separate markup for fgbm finds or a gate that runs post.
- Region 200 (new34) improves under UNI (F1 0.349→0.623) — the signal is there; fold 4 / fold 1 lose.
- Candidate fold models, cache, OOF: /workspace/koz3/orch/bot-W4/work/q6/{cache/,model_fold*.txt,ex_fgbm.geojson,oof_diag.json}.
- Replay outputs: /workspace/koz3/w4/q6/out/out_{base,REP,UNI,REPu,UNIu}.geojson, prep.pkl (snap.pkl deleted for disk).
- Lease slot:box/q6 released. No production hook added (no PASS).
