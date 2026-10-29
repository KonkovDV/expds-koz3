# koz3-w5-dino-emb (NOT pushed)
Frozen DINOv2-S/14 (CLS + mean patch, 768-d) crops for every stack candidate -> extra reranker features (W5).
Blocked on: cands.csv (needs stack_arch from the PC) -> private dataset `koz3-w5-cands` (+ xfer raw shards as dataset_sources).
Slot: elenakonkova gpu-1/gpu-2 after W1/W2 free them (~05:00 MSK 30.10), or CPU accounts (ViT-S, ~1-2 h per 100k crops).
Budget: ViT-S on a 4090 ~ 1-2 min per 50k crops -> fits the 30-min container if adopted in inference.
Lease first: orch/lease.sh acquire slot:acc1/gpu-N bot-W5-rerank <ttl> --kernel elenakonkova/koz3-w5-dino-emb
