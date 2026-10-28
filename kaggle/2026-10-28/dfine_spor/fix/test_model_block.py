# Тест блока MODEL из src/dfine_main.py (между маркерами) без S3/данных: гонка, ремонт, отказ.
import re, sys, time, json, threading, torch, transformers, huggingface_hub
from pathlib import Path
src = Path(__file__).parents[1].joinpath("src/dfine_main.py").read_text(encoding="utf-8")
block = src[src.index("# >>> MODEL"):src.index("# <<< MODEL")]
CKPT = str(Path(__file__).parent / "hf/hub/models--ustc-community--dfine-large-coco/snapshots/aa8f8211545e4cff5adf52ea512f9ac74a9824ec")
G = dict(torch=torch, threading=threading, json=json, time=time, CKPT=CKPT, CFG={"num_dn": -1},
         CLASSES=["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"],
         log=lambda *a: print("  log:", *a, flush=True))
exec(block, G)
from transformers import DFineForObjectDetection, PreTrainedModel
print("versions torch", torch.__version__, "transformers", transformers.__version__, "hub", huggingface_hub.__version__, flush=True)
res = {}
# D: две загрузки одновременно в потоках (как gpu_worker на 2×T4), 2 повтора
for rep in range(2):
    out = {}
    def w(i):
        try: m, info = G["make_model"](seed=100 + i); m.to("cpu"); out[i] = "OK " + json.dumps(info)
        except Exception as e: out[i] = "FAIL " + repr(e)[:150]
    th = [threading.Thread(target=w, args=(i,)) for i in range(2)]; [t.start() for t in th]; [t.join() for t in th]
    print(f"TEST threads#{rep}:", out, flush=True); res[f"threads{rep}"] = all(v.startswith("OK") for v in out.values())
# A: PreTrainedModel.tie_weights «залип» заглушкой (исход гонки) → make_model восстанавливает
orig = PreTrainedModel.__dict__["tie_weights"]; PreTrainedModel.tie_weights = lambda *a, **k: None
try: m, info = G["make_model"](seed=1); m.to("cpu"); res["stuck_patch"] = True; print("TEST stuck_patch: OK", info, flush=True)
except Exception as e: res["stuck_patch"] = False; print("TEST stuck_patch: FAIL", repr(e)[:200], flush=True)
# B: модель уже с meta (загружена при заглушке) → _repair_meta + check_model
PreTrainedModel.tie_weights = lambda *a, **k: None
C = G["CLASSES"]; kw = dict(num_labels=5, id2label=dict(enumerate(C)), label2id={c: i for i, c in enumerate(C)}, ignore_mismatched_sizes=True)
bad = DFineForObjectDetection.from_pretrained(CKPT, **kw); PreTrainedModel.tie_weights = orig
meta = G["_meta_names"](bad); print("TEST repair: meta before", len(meta), flush=True)
# C: check_model на сломанной модели — ранний понятный отказ
try: G["check_model"](bad); res["check_fails"] = False; print("TEST check_fails: НЕ упал (плохо)")
except RuntimeError as e: res["check_fails"] = True; print("TEST check_fails: OK", str(e)[:160], flush=True)
left = G["_repair_meta"](bad, meta); info = G["check_model"](bad); bad.to("cpu")
res["repair"] = not left; print("TEST repair: meta after", len(left), info, flush=True)
# E: forward после ремонта и форма logits
with torch.no_grad(): o = bad.eval()(pixel_values=torch.rand(1, 3, 128, 128))
res["forward"] = o.logits.shape[-1] == 5; print("TEST forward logits", list(o.logits.shape), flush=True)
print("SUMMARY", json.dumps(res), "ALL_OK" if all(res.values()) else "SOME_FAIL", flush=True)
