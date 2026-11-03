# sp_score_fusion → N groups (side patch, bot-D/W1, 01.11 MSK) — NOT shipped, default.toml untouched. Hold 0.41238.
PROMPT_SAT2 §4: mean conf within 8 m over sp_focal / sp_focal2 (where active) / sp_det2; absent member = 0.

## Code
- Side implementation: `orch/bot-W1/sat2_code/sp_score_fusion_n.py::sp_score_fusion_n(feats, spec)`.
- Self-test (`python sp_score_fusion_n.py`): **N=2 output identical** to `expds.post.sp_score_fusion` (570 synthetic feats, incl. sp_or→sp_focal alias and sp_partner_only); N=3 runs.

## Semantics
For each kurgan (povr|tselye) of group g ∈ groups: `o_h` = max kurgan conf (either class) of group h≠g within `radius_m` ground metres
on the same region, else 0. `new = (conf + Σ_h o_h) / N`.
- `denom="all"` (default): N = len(groups) → exact old behaviour for 2 groups.
- `denom="region_active"`: N = #groups with ≥1 kurgan candidate in that region (sp_focal2 is no-LAS only → avoids diluting LAS regions by 1/3).
  Note: a region where sp_det2 emits nothing then fuses like the old 2-group code — intended.
- No adds/removes (except `sp_partner_only`). Apply before thresholds on raw candidates (gen_floor), as now.

## DIFF to expds/post.py (when A approves; only after fold0 gate PASS)
```diff
-    if len(groups) != 2:
-        raise ValueError("sp_score_fusion: нужны ровно две группы")
+    denom = cfg.get("denom", "all")
+    if len(groups) < 2 or len(set(groups)) != len(groups):
+        raise ValueError("sp_score_fusion: нужны ≥2 разные группы")
 ...
+    active = defaultdict(set)
+    for reg, grp in pts: active[reg].add(grp)
     for i, reg, grp, x, y, conf in idx:
-        other = groups[1] if grp == groups[0] else groups[0]
-        o = 0.0
-        t = trees.get((reg, other))
-        if t is not None:
-            hit = t[0].query_ball_point((x * scale[reg], y * scale[reg]), radius)
-            if hit:
-                o = float(t[1][hit].max())
-        new_conf[i] = (conf + o) / 2.0
+        s = conf
+        for h in groups:
+            if h == grp: continue
+            t = trees.get((reg, h))
+            if t is not None:
+                hit = t[0].query_ball_point((x * scale[reg], y * scale[reg]), radius)
+                if hit: s += float(t[1][hit].max())
+        n = len(groups) if denom == "all" else max(1, len(active[reg]))
+        new_conf[i] = s / n
```
Callers to check before shipping: pipeline sp fusion phase (gen_floor + partner_only marking for 12 m drops) must also
emit sp_det2 raw candidates; watchdog.finish thresholds map needs `thresholds.sp_det2`; `[verify].groups` / `[post.frame_prior].groups` OPEN.
Side toml (not default.toml):
```toml
[post.sp_score_fusion]
radius_m = 8.0
groups = ["sp_focal", "sp_focal2", "sp_det2"]
denom = "region_active"   # sweep all vs region_active on OOF
```
## Kill / fallback
If mean fusion drops P on fold0/honest-10: 12 m add-only (sp_det2 circles only where no sp_focal/sp_focal2 anchor within 12 m, `pipeline.drop_near_kurgans`).
Thresholds: nested k10 sweep after OOF (one thr per class), never LB.
