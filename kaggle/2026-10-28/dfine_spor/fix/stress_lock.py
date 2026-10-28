# Стресс: N раз по 2 потока make_model() из блока MODEL (с MODEL_LOCK) — считаем meta и to(cpu).
import sys, json, time, threading, torch
from pathlib import Path
src = Path(__file__).parents[1].joinpath("src/dfine_main.py").read_text(encoding="utf-8")
G = dict(torch=torch, threading=threading, json=json, time=time, CFG={"num_dn": -1}, log=lambda *a: print("  log:", *a),
         CKPT=str(Path(__file__).parent / "hf/hub/models--ustc-community--dfine-large-coco/snapshots/aa8f8211545e4cff5adf52ea512f9ac74a9824ec"),
         CLASSES=list("abcde"))
exec(src[src.index("# >>> MODEL"):src.index("# <<< MODEL")], G)
N = int(sys.argv[1]); ok = fail = 0
for r in range(N):
    out = {}
    def w(i):
        try: m, _ = G["make_model"](seed=i); m.to("cpu"); out[i] = len(G["_meta_names"](m))
        except Exception as e: out[i] = repr(e)[:120]
    th = [threading.Thread(target=w, args=(i,)) for i in range(2)]; [t.start() for t in th]; [t.join() for t in th]
    for v in out.values():
        if v == 0: ok += 1
        else: fail += 1; print("FAIL", v)
print(f"STRESS runs={N} loads_ok={ok} loads_fail={fail}", torch.__version__)
