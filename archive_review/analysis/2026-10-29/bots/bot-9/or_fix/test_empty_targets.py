# bot 9, 29.10: проверка пустых целей (негативные чипы) в D-FINE HF + тренере бота 5 (CPU, малый размер).
import sys, json, random, math, torch, numpy as np, transformers
from pathlib import Path
D = Path("/workspace/koz3_shared/bot-9/kaggle/dfine_spor")
sys.path.insert(0, str(D / "src")); import dfine_train_bot5 as B5
CKPT = str(D / "fix/hf/hub/models--ustc-community--dfine-large-obj2coco-e25/snapshots/c89de47153e338c07cddc81fc21a776b97cbf75e")
from transformers import DFineForObjectDetection
C = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
torch.manual_seed(0)
m = DFineForObjectDetection.from_pretrained(CKPT, num_labels=5, id2label=dict(enumerate(C)), label2id={c: i for i, c in enumerate(C)}, ignore_mismatched_sizes=True)
m.train(); print("versions torch", torch.__version__, "transformers", transformers.__version__, "num_denoising", m.config.num_denoising, flush=True)
T = 192
def empty(): return {"class_labels": torch.zeros(0, dtype=torch.int64), "boxes": torch.zeros(0, 4)}
def one(): return {"class_labels": torch.tensor([4]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]])}
res = {}
for name, lab in [("all_empty", [empty(), empty()]), ("mixed", [empty(), one()]), ("all_pos", [one(), one()])]:
    m.zero_grad(set_to_none=True)
    o = m(pixel_values=torch.rand(2, 3, T, T), labels=lab); o.loss.backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    gfin = all(torch.isfinite(x).all().item() for x in g); gn = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g)))
    ld = {k: round(float(v), 4) for k, v in (o.loss_dict or {}).items()} if getattr(o, "loss_dict", None) else {}
    print(f"TEST {name}: loss={float(o.loss):.4f} finite={bool(torch.isfinite(o.loss))} grads={len(g)} grad_finite={gfin} grad_norm={gn:.3f}", flush=True)
    print("   loss_dict", json.dumps(ld), flush=True)
    res[name] = bool(torch.isfinite(o.loss)) and gfin
# jitter бота 5 с пустыми рамками (путь CFG aug=bot5)
img = (np.random.rand(T, T, 3) * 255).astype(np.uint8); bx = np.zeros((0, 5), np.float32)
ji, jb = B5.jitter(img, np.stack([bx[:, 0], bx[:, 1], bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]], 1), random.Random(1), False)
out = np.concatenate([jb[:, :2], jb[:, :2] + jb[:, 2:4], bx[:, 4:5]], 1); print("TEST jitter_empty: shape", out.shape, flush=True)
res["jitter_empty"] = out.shape == (0, 5)
print("SUMMARY", json.dumps(res), "ALL_OK" if all(res.values()) else "SOME_FAIL", flush=True)
