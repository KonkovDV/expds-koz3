# KOZ3 №3 · koz3-bench-t4 (bot-9): замер тайлов ConvNeXt-T U-Net / D-FINE-L / RT-DETRv2 на ОДНОЙ T4 (cuda:0) скриптом бота 8
# (/workspace/koz3_shared/bot-8/bench/bench_tile.py, вшит байт-в-байт, sha256 сверяется). Веса случайные: S3 и секреты не нужны.
# Порядок: GPU-чек (жёстко, без CPU) → пины pip (torch Kaggle не трогаем) → жёсткий импорт-чек → 4 шага бота 8 (BUDGET.md §7.9),
# каждый шаг выполняется даже если предыдущий упал; логи — /kaggle/working/logs/. Итог — run_summary.json + validity.json.
import os, sys, subprocess, json, time, base64, hashlib, importlib.metadata as md
W = os.environ.get("BENCH_OUT", "/kaggle/working"); os.makedirs(W, exist_ok=True); os.chdir(W)
LOGS = os.path.join(W, "logs"); os.makedirs(LOGS, exist_ok=True)
T0 = time.time(); SUMMARY = {"kernel": "dmitrijkv/koz3-bench-t4", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": []}
def log(*a): print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)
def save_summary():
    SUMMARY["elapsed_s"] = round(time.time() - T0, 1)
    json.dump(SUMMARY, open(os.path.join(W, "run_summary.json"), "w"), ensure_ascii=False, indent=1)
def die(msg):
    SUMMARY["fatal"] = msg; save_summary(); log("FATAL:", msg); raise SystemExit(msg)

# ── Шаг 0. Вшитые файлы ────────────────────────────────────────────────────────────────────────────────────────
open("bench_tile.py", "wb").write(base64.b64decode(BENCH_B64))
sha = hashlib.sha256(open("bench_tile.py", "rb").read()).hexdigest()
if sha != BENCH_SHA256: die(f"bench_tile.py sha256 {sha} != {BENCH_SHA256}")
for name, cfg in HF_CONFIGS_B64.items():
    os.makedirs(os.path.join("hf_configs", name), exist_ok=True)
    raw = base64.b64decode(cfg); open(os.path.join("hf_configs", name, "config.json"), "wb").write(raw)
    if hashlib.sha256(raw).hexdigest() != HF_CONFIG_META[name]["sha256"]: die(f"hf_configs/{name} sha256 не совпал")
SUMMARY["bench_tile_sha256"] = sha; SUMMARY["hf_configs"] = HF_CONFIG_META
log("bench_tile.py sha256 OK", sha[:16], "| hf_configs:", list(HF_CONFIGS_B64))

# ── Шаг 1. --require-gpu: только CUDA, cuda:0; никакого CPU-фолбэка ──────────────────────────────────────────────
try: r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
except Exception as e: r = subprocess.CompletedProcess(["nvidia-smi"], 127, "", repr(e))
log("nvidia-smi -L:", r.stdout.strip() or r.stderr.strip())
if r.returncode != 0 or "GPU 0" not in r.stdout: die("--require-gpu: nvidia-smi не видит GPU (сессия без GPU?). Останов, без CPU-фолбэка.")
SUMMARY["nvidia_smi_L"] = r.stdout.strip()
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")

# ── Шаг 2. Пины pip; torch/torchvision Kaggle закреплены constraints-файлом (pip их не заменит) ───────────────────
PINS = {"transformers": "5.16.1", "timm": "1.0.29", "segmentation-models-pytorch": "0.6.0"}
def ver(dist):
    try: return md.version(dist)
    except md.PackageNotFoundError: return None
pre = {d: ver(d) for d in ("torch", "torchvision", *PINS)}
SUMMARY["versions_preinstalled"] = pre; log("предустановлено:", pre)
if not pre["torch"]: die("torch Kaggle не найден")
with open("constraints_torch.txt", "w") as f:
    for d in ("torch", "torchvision"):
        if pre[d]: f.write(f"{d}=={pre[d]}\n")
need = [f"{d}=={v}" for d, v in PINS.items() if pre[d] != v]
if need:
    log("pip install:", need)
    with open(os.path.join(LOGS, "pip_install.log"), "w") as lf:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary", ":all:", "-c", "constraints_torch.txt", *need],
                           stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode: die(f"pip install не удался ({need}), см. logs/pip_install.log; нужен enable_internet=true")
with open(os.path.join(LOGS, "pip_check.log"), "w") as lf:
    SUMMARY["pip_check_rc"] = subprocess.run([sys.executable, "-m", "pip", "check"], stdout=lf, stderr=subprocess.STDOUT).returncode
post = {d: ver(d) for d in ("torch", "torchvision", *PINS)}
SUMMARY["versions_after_pip"] = post; log("после pip:", post)
if post["torch"] != pre["torch"]: die(f"pip заменил torch {pre['torch']} → {post['torch']}")
bad = [f"{d}=={post[d]}≠{v}" for d, v in PINS.items() if post[d] != v]
if bad: die("версии не совпали с пинами: " + "; ".join(bad))

# ── Шаг 3. Жёсткий импорт-чек в отдельном процессе (тот же интерпретатор, что и бенч) ─────────────────────────────
CHECK = r'''
import json, torch, timm, segmentation_models_pytorch as smp, transformers as tf
from transformers import DFineForObjectDetection, RTDetrV2ForObjectDetection, DFineConfig, RTDetrV2Config
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
d = torch.device("cuda:0"); x = torch.ones(256, 256, device=d); y = (torch.relu(x) @ x).sum().item()   # ловит "no kernel image"
cap = torch.cuda.get_device_capability(d)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
  "timm": timm.__version__, "smp": smp.__version__, "transformers": tf.__version__, "device_count_visible": torch.cuda.device_count(),
  "gpu0": torch.cuda.get_device_name(d), "capability": list(cap), "smoke_sum": y}))
'''
r = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, env=ENV)
open(os.path.join(LOGS, "import_check.log"), "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
if r.returncode: die("импорт-чек не пройден: " + (r.stderr.strip().splitlines() or ["?"])[-1])
ic = json.loads(r.stdout.strip().splitlines()[-1]); SUMMARY["import_check"] = ic; log("импорт-чек OK:", ic)
if ic["device_count_visible"] != 1: die(f"видно {ic['device_count_visible']} GPU при CUDA_VISIBLE_DEVICES=0")
if "T4" not in ic["gpu0"]: log("ВНИМАНИЕ: GPU не T4:", ic["gpu0"]); SUMMARY["warning_gpu"] = ic["gpu0"]
save_summary()

# ── Шаг 4. Команды бота 8 (BUDGET.md §7.9) — каждая независимо, частичные JSON переживают падения ─────────────────
STEPS = [
 ("smi_before", "nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > smi_before.txt"),
 ("bench_cnx", "python bench_tile.py --models cnx_li9,cnx_li12,cnx_sp3 --dtypes fp16,fp32,bf16 --batches 1,2,4 --warmup 10 --iters 30 --out bench_t4_cnx.json"),
 ("bench_det", "python bench_tile.py --models dfine_l,rtdetrv2_r50 --dtypes fp16,fp32 --batches 1,4,8 --warmup 10 --iters 30 --out bench_t4_det.json --hf-config-dir hf_configs"),
 ("smi_after", "nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > smi_after.txt"),
]
BIN = "/tmp/koz3_bench_bin"; os.makedirs(BIN, exist_ok=True)   # `python` в командах = тот же интерпретатор, где прошёл импорт-чек
open(os.path.join(BIN, "python"), "w").write(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n'); os.chmod(os.path.join(BIN, "python"), 0o755)
ENV["PATH"] = BIN + os.pathsep + ENV.get("PATH", "")
for key, cmd in STEPS:
    t = time.time(); log("RUN", key, "::", cmd)
    try:
        with open(os.path.join(LOGS, f"{key}.log"), "w") as lf:
            rc = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], cwd=W, env=ENV, stdout=None if ">" in cmd else lf,
                                stderr=subprocess.STDOUT if ">" not in cmd else lf).returncode
    except Exception as e:  # noqa
        rc = f"exception: {e!r}"
    SUMMARY["steps"].append({"step": key, "cmd": cmd, "rc": rc, "wall_s": round(time.time() - t, 1), "log": f"logs/{key}.log"})
    log("DONE", key, "rc", rc, f"{time.time() - t:.0f}s"); save_summary()
    if key.startswith("bench"):
        try: print(open(os.path.join(LOGS, f"{key}.log")).read()[-6000:], flush=True)
        except Exception: pass

# ── Шаг 5. Критерии годности бота 8 (BUDGET.md §7.8 п.4–5) ────────────────────────────────────────────────────
THR_BITS = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}
def thr_decode(vals):
    out = set()
    for v in vals:
        try: n = int(v, 16)
        except Exception: continue
        out |= {nm for b, nm in THR_BITS.items() if n & b}
    return sorted(out)
VAL = {"rule": "ConvNeXt-T fp16 тайл <5 мс или >500 мс → замер ошибочный; median SM <1300 МГц или power cap/HW slowdown → троттлинг (BUDGET.md §7.8 п.4–5)",
       "configs": [], "broken": [], "throttled": []}
for fn in ("bench_t4_cnx.json", "bench_t4_det.json"):
    try: J = json.load(open(fn))
    except Exception as e: VAL["broken"].append(f"{fn}: нет/не читается ({e!r})"); continue
    for m in J.get("models", []):
        if "fatal" in m: VAL["broken"].append(f"{fn}:{m['model']}: fatal {m['fatal']}"); continue
        for c in m.get("configs", []):
            if "skipped" in c: continue
            tag = f"{m['model']}/{c['dtype']}/b{c['batch']}"
            if "oom" in c or "error" in c: VAL["broken"].append(f"{tag}: {c.get('oom') or c.get('error')}"); continue
            clk = (c.get("clocks_during_timing") or {}); sm = (clk.get("sm_clock_mhz") or {}).get("median")
            thr = thr_decode(clk.get("throttle_reasons_active") or [])
            t1 = c.get("ms_per_tile_single_median", c.get("ms_per_chip_single_median"))
            row = {"cfg": tag, "ms_single_per_item": t1, "ms_d8_or_flip2_per_item": c.get("ms_per_tile_d8_median", c.get("ms_per_chip_flip2_median")),
                   "sm_clock_median_mhz": sm, "throttle": thr, "nonfinite": c.get("nonfinite_outputs")}
            VAL["configs"].append(row)
            if m["kind"] == "unet" and c["dtype"] == "fp16" and t1 is not None and not (5 <= t1 <= 500):
                VAL["broken"].append(f"{tag}: ConvNeXt-T fp16 {t1} мс/тайл вне [5, 500]")
            if sm is None: VAL["throttled"].append(f"{tag}: нет частоты SM в логе (nvidia-smi query?)")
            elif sm < 1300 or {"sw_power_cap", "hw_slowdown", "hw_thermal", "hw_power_brake"} & set(thr):
                VAL["throttled"].append(f"{tag}: SM median {sm} МГц, {thr}")
VAL["verdict"] = "BROKEN" if VAL["broken"] else ("THROTTLED" if VAL["throttled"] else "OK")
json.dump(VAL, open("validity.json", "w"), ensure_ascii=False, indent=1)
SUMMARY["validity_verdict"] = VAL["verdict"]; save_summary()
log("validity:", VAL["verdict"], "| broken:", len(VAL["broken"]), "| throttled:", len(VAL["throttled"]))
failed = [s["step"] for s in SUMMARY["steps"] if s["rc"] != 0]
if failed: raise SystemExit(f"шаги с ошибкой: {failed} (частичные результаты в /kaggle/working)")
log("ГОТОВО")
