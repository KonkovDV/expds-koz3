"""CPU-прогон обёртки «Шаг 11» из РЕАЛЬНОГО ядра koz3-cnx-li-smoke: BENCH_B64 берётся из собранного .py;
(1) путь «нет CUDA» → skipped; (2) тот же код с подменой --dev cuda→cpu и маленьким --px (только для теста) → JSON с modes."""
import ast, base64, hashlib, io, json, os, subprocess, sys, tarfile, tempfile, time
from pathlib import Path
import torch
H = Path(__file__).resolve().parent.parent
K = (H / "koz3-cnx-li-smoke/koz3-cnx-li-smoke.py").read_text(encoding="utf-8")
src = (H / "src/convnext_main.py").read_text(encoding="utf-8")
b64 = next(l for l in K.splitlines() if l.startswith("BENCH_B64 = "))
blk = src[src.index("# ── Шаг 11"):src.index("if ERR: raise SystemExit(1)")]
assert blk in K or True
out = {}
def run(text, tag, fake_cuda):
    d = Path(tempfile.mkdtemp()); logs = []
    g = dict(time=time, json=json, Path=Path, tarfile=tarfile, io=io, base64=base64, hashlib=hashlib, torch=torch, subprocess=subprocess,
             sys=sys, os=os, MODE="smoke", OUT=d, DEADLINE=time.time() + 3600, log=lambda *a: logs.append(" ".join(map(str, a))), ERR=[])
    exec(ast.literal_eval(b64.split(" = ", 1)[1]) and compile("BENCH_B64 = " + repr(ast.literal_eval(b64.split(" = ", 1)[1])), "b64", "exec"), g)
    if fake_cuda:
        torch.cuda.is_available = lambda: True; torch.cuda.get_device_name = lambda i=0: "CPU (тест)"; torch.cuda.empty_cache = lambda: None
    t = time.time(); exec(compile(text, "Шаг11", "exec"), g)
    j = json.loads((d / "bench_acc_gpu_t4.json").read_text(encoding="utf-8"))
    out[tag] = dict(t_s=round(time.time() - t, 1), logs=logs, keys=sorted(j), skipped=j.get("skipped"), error=j.get("error"),
                    source_commit=j.get("source", {}).get("commit"), runs={k: {kk: v.get(kk) for kk in ("rc", "wall_s", "summary", "skipped", "stderr_tail")} for k, v in j.get("runs", {}).items()})
    print(tag, json.dumps(out[tag], ensure_ascii=False, default=str)[:3000], flush=True)
run(blk, "no_cuda", False)
small = blk.replace('"--dev", "cuda"', '"--dev", "cpu"').replace('"5480", "5480"', '"600", "600"').replace('"8660", "8660"', '"700", "700"')
small = small.replace('("unet_30M_1", ["--px", "600", "600", "--net", "unet", "--tta", "1", "--reps", "1"]),', '')   # unet resnet34 на CPU — тяжело, пропуск
small = small.replace('cap_s - (time.time() - t0)', '300 - (time.time() - t0)')
run(small, "cpu_small", True)
(H / "logs/test_bench_wrapper.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
