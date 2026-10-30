# PREREG WS-G1 (W3) — gor/fort few-shot verifier, written 30.10 14:40 MSK BEFORE any measurement
Scope: candidates in split_arch with class gorodishcha or fortifikatsii that are in the W5 DINOv2 pool (cid = stack idx, verified 165/165 sample).
Embeddings: out/w5-dino-emb-gpu-1030 (ViT-S/14, CLS+mean patch, 768-d), CPU copy out/w5-dino-emb used where gpu ok=0. Read-only.
Labels: candidate positive if its centroid lies inside a GT polygon of the same class and region (gt_train_v2), else negative.
kNN (leave-one-family-out, same class): s_knn = mean top-5 cosine to positives − mean top-5 cosine to negatives, references from other families only.
Rampart profile (li only, w2/relief/cnx_relief lrm_s, R = max(w,h)/2 ≥ 5 m): ring r∈[0.6R,1.1R] mean − centre r<0.4R mean, ring − background r∈[1.3R,1.8R], closure = share of 16 azimuth sectors with ring > background; has_ring flag (0 for non-li, features 0).
Models (per class, LOFO LogisticRegression C=1 standardized): 
 G1A conf' = p([logit conf, s_knn]); G1B conf' = p([logit conf, s_knn, ring feats, has_ring]); G1C conf' = min(1, conf·2·σ(z)), z = s_knn standardized on other families (multiplier, plan wording).
Only those candidates are replaced (drop+extras); nothing added. No new detector. Diagnostics: LOFO AUC of s_knn and of G1A/G1B vs conf.
Gate: standard w3_gate (honest-10 Δ≥+0.003, P(A>B)≥0.9, 34new and optical not below A=0.314615). Flag default off. Lease slot:box/cpu (bot-W3-bigobj).
