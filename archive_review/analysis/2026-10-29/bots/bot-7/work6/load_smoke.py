import sys, time, resource, torch
torch.set_num_threads(2)
sys.path.insert(0, "/workspace/koz3_shared/bot-7/repo")
from expds import sp_dfine
t = time.time()
ms = sp_dfine.load_models({"weights": ["/workspace/kaggle_out/koz3-dfine-spor-full/dfine_f1"]}, ".", torch.device("cpu"))
m = ms[0]
# эталон: from_pretrained
from transformers import DFineForObjectDetection
r = DFineForObjectDetection.from_pretrained("/workspace/kaggle_out/koz3-dfine-spor-full/dfine_f1").eval()
a, b = m.state_dict(), r.state_dict()
diff = max((a[k].float() - b[k].float()).abs().max().item() for k in b)
x = torch.rand(1, 3, 640, 640)
with torch.no_grad():
    t1 = time.time(); o1 = m(pixel_values=x); t2 = time.time(); o2 = r(pixel_values=x)
print("load+ref s", round(t1 - t, 1), "fwd s", round(t2 - t1, 2), "keys", len(a), len(b), "maxdiff sd", diff,
      "logits diff", (o1.logits - o2.logits).abs().max().item(), "maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
