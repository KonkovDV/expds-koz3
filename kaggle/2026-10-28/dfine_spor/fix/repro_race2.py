# После гонки: остаётся ли PreTrainedModel.tie_weights заглушкой; чинит ли явный исходный tie_weights().
import threading, torch, transformers
from transformers import DFineForObjectDetection, PreTrainedModel
ORIG = PreTrainedModel.tie_weights
CK = "hf/hub/models--ustc-community--dfine-large-coco/snapshots/aa8f8211545e4cff5adf52ea512f9ac74a9824ec"
C = ["a","b","c","d","e"]; res = {}
def load(i):
    res[i] = DFineForObjectDetection.from_pretrained(CK, num_labels=5, id2label=dict(enumerate(C)), label2id={c:i for i,c in enumerate(C)}, ignore_mismatched_sizes=True)
th = [threading.Thread(target=load, args=(i,)) for i in range(2)]; [t.start() for t in th]; [t.join() for t in th]
print("tie_weights patched after race:", PreTrainedModel.tie_weights is not ORIG, flush=True)
nmeta = lambda m: sum(p.is_meta for p in m.parameters()) + sum(b.is_meta for b in m.buffers())
for i, m in res.items():
    before = nmeta(m); ORIG(m); after = nmeta(m)
    print(f"model{i} meta before {before} after ORIG.tie_weights {after}; tied {m.class_embed[0].weight is m.model.decoder.class_embed[0].weight}", flush=True)
