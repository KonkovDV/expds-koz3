# PREREG W4 P-D2-cleanroom (bot5-w4, 30.10.2026 ~16:20 MSK), written before the D2 measurements

Spec: docs/PIXGBM_SPEC_CLEANROOM.md (bbebfe3). QUEUE: "приёмка Δ≥+0.009, паритет v2 ±0.002, проверить EtGr G4 в v2".
Engine: identical to PREREG_w4_v3on.md (d3_gate TableEngine on table.pkl, honest-10 family folds, unseen=drop,
paired family bootstrap 2000 seed 0, slices all/new34/nolidar). A = stack table (classic geophysics; S1 proved its etgr rows
== A18.geojson). C = v2clean = B0 of PREREG_w4_v3on (A − etgr rows + C19etonly_orget_s0 etgr rows).
Disclosure: the number honest(C) was already printed as the B0 line of the v3 gate run (15:50 MSK), before this file.

## D2-1 acceptance (spec §8.1): C vs A
Δ_all ≥ +0.009, P(C>A) ≥ 0.95, plus spec §8.2 slices new34/nolidar ≥ −0.003 and F1 finds_points/object_poly not below A.
(Spec also asks ≥4/5 folds and seeds; only seed 0 OOF exists here → reported as not checked.)

## D2-2 parity with v2 (eb268d6, LB 0.41238) ±0.002
Needs honest(v2) on the same engine, i.e. v2's geophysics output for regions 200–217. v2 geophysics = legacy PIXGBM, which the
clean-room implementer may not run or inspect. Rule: if a non-clean-room party supplies the v2 geophysics geojson (A18 format,
black box), score |honest(C) − honest(v2)| ≤ 0.002. Otherwise D2-2 = BLOCKED (clean room), and only the proxy is reported:
Δ(C−A) against the spec reference "old line ≈ +0.012 honest-10" → proxy parity |Δ(C−A) − 0.012| ≤ 0.002.

## D2-3 EtGr G4 (layer isolation, +0.0098 LB) in v2
(a) static: post.merge_within_markup at eb268d6 (v2) and at bots/w4-gate (v2clean) must keep markup etgr groups unmerged
    (files allowed: expds/post.py only; geo_pixgbm.py not opened).
(b) dynamic, on C: C_iso (as is) vs C_merged (etgr rows of C passed through post.merge_across_sources per (region, class),
    i.e. layers merged = G4 off). Report Δ(C_iso − C_merged), P. No acceptance threshold (diagnostic).
