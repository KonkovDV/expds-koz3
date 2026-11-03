# C→A: re-gate CetO3_Sr8G vs REF (2026-10-30 17:33 MSK)
REF re-emitted black-box (docker pantum/n7q, 254s) → w4geo/ref/out/REF_v2.geojson, 7194 feats. DOC-1 OK.
**FAIL** vs REF (raw + _ht). Surrogate honest ≥0.012 PASS.
| | geo P/R | MagGr R | honest Δ |
|---|---|---|---|
| Sr8G raw | .286/.790 | .819 (Δ−0.039 vs REF .859) | +0.0341 |
| Sr8G _ht | .604/.669 | .681 (Δ−0.177) | +0.0362 |
| REF | .664/.785 | .859 | — |
EtGr R beats REF (+0.20 raw / +0.17 _ht). MagGr still the gate miss after honest thresholds.
**TerFl pix: NOT OFF in Cet lineage.** C_CetO3_Sr8G: 252× `threshold_group=terfl_pix`. Package `v2_clean` maps=["etgr"] OFF, but `run_finds_fix.sh` Cet compose used maps=["etgr","terfl"]. Slice +0.0142 is TerFl pix. JSON: cstar/gate_CetO3_Sr8G{,_ht}.json (+ orch/bot-W4/runs/).
