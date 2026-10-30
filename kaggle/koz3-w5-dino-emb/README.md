# koz3-w5-dino-emb
Frozen DINOv2 ViT-S/14 embeddings (CLS + mean patch, fp16) for the W5 reranker pool (222,929 candidates, top-15%/group).
Input dataset `elenakonkova/koz3-w5-cands` (`cands.csv.zst`, zstd-19, lossless, sha256(csv)=e0dfb7bd62c5…),
li crops from `elenakonkova/koz3-cnx-li-prep` (cnx_relief lrm_s/slope/svf_r 0.5 m), other layers from S3 via /vsis3
(keys from `elenakonkova/koz3-s3-cred`, never printed). v1 pushed 29.10 23:18 MSK on elenakonkova CPU (lease slot:acc1/cpu).
Outputs emb.npy / ok.npy / cid.npy / meta.json (partial saves every 20 min, DEADLINE_H=11.3).
