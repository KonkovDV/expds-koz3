# LI-KURG-TAU PREREG — W3, 02.11 MSK. Committed before any measurement.
Task from bot 1, 09:23: items #2/#4 (lidar kurgans below threshold, oracle upper bound +0.074).
## Base
Base = V3P (v3_on b447649 in stack form, as in prereg/KURGAN_DUAL_PREREG.md), honest-10 family folds, D.honest / TableEngine unchanged. Expected 0.333787.
Li rows = kurgan rows (kurgany_povrezhdennye, kurgany_tselye) in groups li, li_dev, li_mosaic.
Variants change only the score of li rows, and in C the threshold group of li rows. All other rows and classes are unchanged.
Curves are rebuilt with the standard greedy matching in the new score order. Sanity: the rebuild with the original conf must equal V3P exactly.
Thresholds are always picked by D.honest (sweep on train regions of outer fold k, per class × group), so every parameter is chosen nested.
## Note on literal (a)
V3P already has separate per-class × per-li-group thresholds chosen by the nested sweep. Literal (a) is therefore identical to V3P (Δ ≡ 0) and is not counted as a test.
## Variants (flag li_kurg_tau=<A|B1|B2|C>, default off)
- **A (joint kurgan score, separate class thresholds).** A li row of class c gets score s = conf_c + conf_other. conf_other is the conf of the twin copy with identical geometry (key reg, cx, cy, area, nv, grp); if there is no twin, conf_other = 0. Thresholds on s are per class × li group, picked by the nested sweep.
- **B1 (per-scene quantile).**
  - Within (region, class), li rows are ranked by conf descending, rank 0..n−1.
  - s = 1 − rank/n.
  - The nested threshold on s is the kept quantile q per class × li group: the scene keeps its top q·n, i.e. the quota is proportional to its candidate count.
- **B2 (per-scene absolute top-K).** Same ranking, s = 1/(1+rank). The nested threshold is K per class × li group.
- **C (cross-modal confirmation).**
  - A li row is confirmed if, in the same region, there is a kurgan-class candidate (either kurgan class) from a non-li group with conf ≥ 0.10. Non-li groups: sp_*, ae, li_ae, D-FINE sp_dfine extras.
  - Distance between centroids must be ≤ 15 m (UTM, table cx/cy; for extras, the centroid of the collected geometry).
  - Confirmed rows move to their own group g+"|x". The nested sweep then picks their threshold separately; the conf score is unchanged.
## Gate: each variant vs V3P (honest-10, paired family bootstrap 2000)
Bonferroni m = 4 (A, B1, B2, C): P ≥ 1 − 0.1/4 = 0.975.
Accept if all hold:
- Δ_all ≥ +0.003;
- new34 Δ ≥ 0, nolidar Δ ≥ 0;
- ΔF1 damaged ≥ 0, ΔF1 intact ≥ 0, ΔF1 fort ≥ 0;
- W4 class floors hold.
Report Δ, P, new34, floors, and the per-fold thresholds for li groups.
