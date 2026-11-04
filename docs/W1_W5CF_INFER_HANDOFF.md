# W1 → W4: `expds.w5cf_infer` (RR1s-cf inference for `w5_cloudfeat`)

Branch `bots/w1-cfdeploy` (from origin/main f127db8). No EN / bots/bot3-* code. W5 feature logic, reimplemented
(feats.py, feats2.py, an1.py, prep.py of analysis/2026-10-30/w5_rerank); cloud features: `expds/cloudfeat.py`.

## Entry point
`[w5cf] module = "expds.w5cf_infer"` (the W4 default). Plug-in contract of `expds/w5_cloudfeat.py` (bots/w4-v3on 18ba8b7):
* `region_features(reg, li_feats, cfg)`: lidar pass. `reg.las` + `catalog.read_utm_json(reg.utm_json)` → 10 `cf_*` in
  `f["w5cf_feat"]` for li objects in the ranker pool (train conf threshold per (group, class)); records has_las.
* `score(feats, cfg)`: finish, all run objects → float for pool rows, None otherwise (length = len(feats)).

Config `[w5cf]`: `infer_dir` (default `models/w5cf_infer` from the archive root), `cf_deadline_s` (1300; seconds
since process start; no cf_* for a site whose LAS would not be read in time, at `cf_mb_per_s` = 60), `threads` (4).

## Features (50, order = `meta.json["names"]`)
35 stack context features over the run's own objects: `gq` comes from the train conf distribution per (grp, cls)
(`gq_ref.npz`, mid-rank on ties). 5 `cnx_*` are always NaN, because the run has no W2 ConvNeXt seeds. 10 `cf_*` come from the test LAS.
Fallback when a site has no LAS / CRS / time / region_features call, or on an error: cf_* = NaN, has_las = 0.
The model saw such rows in training (optical regions). Non-li objects never get cf_* (W4 passes only li),
which is fine: W4 only uses li scores.

## Artifacts (not committed: weights rule, ~4.5 MB of text)
Rebuild deterministically (~15 s, sha256 identical across runs) from the W5 training table on the box:
```
W5_DATA=/workspace/koz3/w1/cf/w5d python tools/rr1s_cf/train_final.py --out models/w5cf_infer --nocnx
```
Ready on the box: `/workspace/koz3/w1/cf/w5cf_infer_nocnx` (recommended) and `/workspace/koz3/w1/cf/w5cf_infer_full`
(trained with real cnx; at inference cnx = NaN → out of distribution for LAS rows). Contents: fam{0,2,3,4,5,6}.txt,
meta.json (feature order, encodings, pool thresholds, sha256, versions lightgbm 4.7.0 / numpy 2.5.4), gq_ref.npz.

## Parity (tools/w5cf_infer_parity.py; 016, 186, 190 with real LAS; 158 optical; 004 large)
* Exact mode (train gq and pool injected): all 45 non-cnx features and li scores max |diff| = 0.0 on all 5 regions.
  cf_* from LAS through region_features equal X4 (016, 186, 190).
* Default mode: the only source of difference is gq tie handling (train = stable rank in file order, inference =
  mid-rank). gq ≤ 0.057, li score max |diff| ≤ 0.047 (004), 0.012 (190), 0.0009 (016); pool mask differs only on
  conf ties at the threshold (780 of 34k on 190).
* vs outer OOF (different models by design): Spearman, li pool rows, nocnx: 016 0.80, 190 0.84, 004 0.86, 158 0.94.
* W4 `w5_cloudfeat.apply` (18ba8b7) run end-to-end with this module + ref.npz on 016 + 158: OK.

## Time / memory (tools/w5cf_infer_bench.py)
190 (heaviest, 5.6 GiB LAS, 34k objects): region_features 57.6 s, score 0.9 s, peak RSS 1018 MB. score on 149k
objects (004): 6.1 s. No need for rust.

## Re-gating
Nested OOF with cnx = NaN (`nested2.py --xset X4nc --tag w5cfnc`, all classes) gives Δ +0.0151, P 0.912, against
+0.0176 / P 0.928 with cnx. The li scale is in r_pool of ref.npz (outer OOF w5cf). For a nocnx module, bot 5 should rebuild
r_pool from nested_w5cfnc.npz (`/workspace/koz3/w1/cf/w5d/work/`).
