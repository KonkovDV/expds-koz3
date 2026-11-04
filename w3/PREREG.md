# W3 PREREG — written 29.10 22:10 MSK, BEFORE any measurement
Baseline A = v4 stack (bots/bot7-candidate-v4-nomodels), honest-10 = 0.314615.
Gate (all variants): materials/compute_metrics.py via scripts/scoreboard.py --honest 10 + scripts/honest_bootstrap.py;
Δ ≥ +0.003 and P(A>B) ≥ 0.9; no drop on 34 new LAS; no drop on no-lidar regions. W4 accepts.
Code: orch/bot-W3/code/w3_post.py (flags default off), tests test_w3_post.py (3/3 pass).

## H — hillforts written as another class (25 GT gorodishcha found as fort/arch: 15 sp_or, 4 li, 4 ae, 2 or)
New groups gor_fa_<markup> get own honest-swept thresholds (base groups untouched).
- H1: gor_from_fort_arch enabled, min_area_m2=5000, classes fort+arch, all markups, skip if same-markup gor IoM≥0.5.
- H2: as H1, markups = [sp_or, li] only.
- H3: as H1, min_area_m2=2000.
Primary = H1. H2/H3 reported, chosen only if H1 fails gate and they pass (multiplicity noted).

## K — damaged/intact kurgan confusion (489 FP povr + 591 FP tselye are GT kurgans of the other subclass)
Signal: bot-9 kurgan_subclass v2 OOF p_povr (r18+r34 imagenet mean, 10 family folds), joined by stack_idx.
- K1: bot-9 apply (relabel at thr 0.4, dedup 2 m) — reference only; prior evidence (§4.2) says plain relabel fails.
- K2 (primary): kurgan_subclass_twins enabled, groups li/li_mosaic/li_coarse, p_hi=0.6, p_lo=0.2, dedup 2 m, conf×p.
- K3: K2 with scale_conf=false.
Precondition for K: pooled OOF AUC on labeled candidates ≥ 0.70; else K not run (classifier too weak).

## Amendment A1 — 29.10 23:04 MSK, BEFORE any H/K measurement
Harness: scoreboard.py stack files (F:/expds3/work) are not on box. Measured instead on the full pre-threshold stack
/workspace/koz3/w5data/stack_arch.geojson (1,386,709 objects; bot-8 scripts/stack_scoreboard.py from origin/main 70dedf4,
same expds.metric build_curves/sweep_groups/honest_counts/paired_bootstrap_counts as scoreboard.py). Baseline A = honest-10
of stack_arch itself (NOT 0.314615, which is the v4 old stack). Variant = stack_arch + extras (add-only) or with
covered features replaced (K1, K5). Gate unchanged: honest Δ ≥ +0.003, family-bootstrap P(A>B) ≥ 0.9, no drop on 34 new / no-lidar.
Crop stack_idx verified to index stack_arch (6109/6109 match region+class).
Finding: in stack_arch every li kurgan candidate already exists as BOTH subclasses (pairs povr/tsel, same geom, own conf),
so K2/K3 dedup 2 m would suppress almost all twins. Therefore:
- K2, K3 run exactly as written (expected ~no-op; reported).
- K4: K2 with dedup_m=0 (twin added even if opposite copy exists), groups li, li_mosaic (crop cands are only there).
- K5 (replace): each crop-covered kurgan feature (both copies of the pair) moves to group <g>_sc; conf_povr×p, conf_tsel×(1−p).
Primary K stays K2; K4/K5 are secondary (multiplicity noted). K precondition (pooled AUC ≥ 0.70) unchanged.
H1–H3 unchanged; markups present in stack fort/arch: ae (ae_fort), sp_or (sp_arch), or, li (li_coarse/li_fort/li).
## Amendment A2 — 29.10 23:05 MSK, after counting H extras, BEFORE any H score
H1 = 607 twins (606 ae_fort, 1 li), H2 = 1 (li), H3 = 2312. sp_or/or arkhitektury in stack are point-like 12 m² centre boxes
(max 36 m²), so min_area excludes them all; the 15 sp_or "hillforts as arch" hits are small boxes inside big GT hillforts (IoM).
Post-hoc (exploratory, flagged) variants:
- H4: classes fort+arch, markups [sp_or, or], min_area_m2=0, skip same-markup gor IoM≥0.5, groups gor_fa_<markup>.
- H5: H1 ∪ H4.
