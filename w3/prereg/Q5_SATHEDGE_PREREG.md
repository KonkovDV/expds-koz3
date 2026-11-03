# Q5 SATELLITE KURGAN HEDGE PREREG — W3, 02.11 MSK. Committed before measurement. Spec §3 Q5 (d2fe1de). Clean-room.
Flag sat_kurg_hedge, default off. Base: V3P.
- For an sp_dfine kurgan circle of class o with own score ≥ h: if a circle of the other kurgan class c with score s_c lies within 4 m in the same region, set o's conf to max(own, 0.99·max s_c).
- Geometry and group are unchanged (sp_dfine). Thresholds are re-swept on train.
- Grid h ∈ {0.2, 0.3, 0.4, 0.5} plus off, nested per outer fold on train; ties prefer off.
- Gate as Q3: Δ ≥ 0.003, P ≥ 1 − 0.1/3, new34 ≥ 0, nolidar ≥ 0, floors with kt checked strictly at −0.002 (kp/gor −0.002, others −0.005).
