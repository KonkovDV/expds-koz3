# Q4 LI→SP_OR TWINS PREREG — W3, 02.11 MSK. Committed before measurement. Spec §3 Q4 (d2fe1de). Clean-room.
Flag li_sp_twin, default off. Base: V3P.
- For each lidar kp/kt row (groups li, li_dev, li_mosaic) in a region with a satellite layer (D-FINE region), emit a copy: same class and geometry, markup sp_or, threshold group li_sp_twin, conf = α·c.
- No copy if an sp_dfine kp/kt detection of any conf lies within 8 m in the same region.
- Grid α ∈ {0.1, 0.3, 0.5, 1.0} plus off, nested per outer fold on train; ties prefer off. The li_sp_twin threshold is swept on train.
- Gate as Q3: Δ ≥ 0.003, P ≥ 1 − 0.1/3, new34 ≥ 0, nolidar ≥ 0, floors kp/kt/gor −0.002, others −0.005.
