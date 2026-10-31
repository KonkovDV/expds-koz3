# E1_G1 freeze backup (bot W4, 2026-10-30/31)

Box-only artifacts of the frozen best cleanroom C* (E1_G1), backed up so a box recreate cannot wipe them.
Docs/recipes only: nothing is wired into the package, no LB submission was made from here.

- Freeze note: `orch/bot-W4/FREEZE_E1_G1.md` (Elena via A, 30.10 23:04 MSK: no LB until Elena says yes)
- Recover log: `orch/PIXGBM_RECOVER_20261030.md`
- Frozen outputs:
  - raw `w4geo/work/cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1.geojson` (sha256 prefix c92b25e0)
  - ht  `w4geo/work/cstar/C_CetO3_Sr8G_noTerFl_L_E1_G1_ht.geojson` (sha256 prefix aaba16ef)
- Gates: `orch/bot-W4/runs/gate*_E1_G1*.json|log` and `w4geo/work/cstar/gate*_E1_G1*`
- Scripts/configs/logs: text files under 2 MB from `w4geo/work/{sweep,night,magp,cstar}`, plus `w4geo/{fetch,pull}_geo.py` (credentials come from env only)

Not included: caches, rasters, .pkl, model dumps, other geojsons (21 GB+ on box).
Integrity: `sha256sum -c MANIFEST.sha256`.
