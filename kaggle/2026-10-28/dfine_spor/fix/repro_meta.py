# Репро: from_pretrained D-FINE-L с 5 классами — одиночно и в 2 потоках одновременно (как gpu_worker на 2×T4).
import sys, threading, torch, transformers, huggingface_hub
from transformers import DFineForObjectDetection
CK = "hf/hub/models--ustc-community--dfine-large-coco/snapshots/aa8f8211545e4cff5adf52ea512f9ac74a9824ec"
C = ["a","b","c","d","e"]
print("versions torch", torch.__version__, "transformers", transformers.__version__, "hub", huggingface_hub.__version__, flush=True)
def load(tag, res):
    m = DFineForObjectDetection.from_pretrained(CK, num_labels=5, id2label=dict(enumerate(C)), label2id={c:i for i,c in enumerate(C)}, ignore_mismatched_sizes=True)
    meta = [n for n,p in m.named_parameters(remove_duplicate=False) if p.is_meta] + [n for n,b in m.named_buffers(remove_duplicate=False) if b.is_meta]
    tied = m.class_embed[0].weight is m.model.decoder.class_embed[0].weight
    try: m.to("cpu"); ok = "to(cpu) OK"
    except Exception as e: ok = "to(cpu) FAIL " + repr(e)[:80]
    res[tag] = (len(meta), meta[:4], tied, ok)
mode = sys.argv[1]
res = {}
if mode == "single": load("t0", res)
else:
    th = [threading.Thread(target=load, args=(f"t{i}", res)) for i in range(2)]
    [t.start() for t in th]; [t.join() for t in th]
for k, v in sorted(res.items()): print("RESULT", mode, k, "meta:", v[0], v[1], "tied:", v[2], v[3], flush=True)
