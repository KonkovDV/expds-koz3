#!/usr/bin/env python3
"""bench_tile.py — замер времени тайла ConvNeXt-T U-Net (1024²) и чипа D-FINE-L / RT-DETRv2 (640²) на одной GPU.
bot-8, 28.10.2026. Для T4 (Kaggle) → пересчёт на RTX 4090 по коэффициентам из bot-8/BUDGET.md §7.8.

Веса СЛУЧАЙНЫЕ: для времени и VRAM этого достаточно, потому что FLOPs, память и ядра от значений весов не зависят.
Исключение — постобработка, зависящая от числа детекций; здесь мерится только forward (как `_detect` в prev_sp_det.py:16-23).
Архитектуры повторяют EhimenNathan-EXpedition:
  ConvNeXt-T U-Net: koz3/li_train.py:253-257 (smp.Unet "tu-convnext_tiny", decoder (256,128,64,32,16), 7 выходов),
                    входы 9 каналов (li) / 12 (li_x, RELIEF_EXTRA, koz3/relief.py:9-10) / 3 (SpOr heatmap, koz3/sp_train.py:154).
  TTA: koz3/hm.py:80-121 — d8 = rot90 k=0..3 × (без/с flip(-1)); flip3 = id, flip(-1), flip(-2). Их dtype bf16 (hm.py:105).
  Детекторы: scripts/build_submission.py:56,96 — ustc-community/dfine-large-coco, PekingU/rtdetr_v2_r50vd, 5 классов;
             fp16 autocast + 2 прохода (id + flip) (koz3/prev_sp_det.py:16-23), вход x/255 без нормировки (:39), чип 640.

Запуск (одна GPU, без DataParallel):
  python bench_tile.py --out bench_t4.json                       # всё по умолчанию
  python bench_tile.py --models cnx_li9,dfine_l --batches 1,4 --dtypes fp16,fp32 --warmup 10 --iters 30
Нужно: torch, timm; segmentation_models_pytorch (если нет — встроенный эквивалент декодера, помечается в JSON);
        transformers >= 4.52 для D-FINE (DFineForObjectDetection) и RT-DETRv2 (RTDetrV2ForObjectDetection).
Конфиги детекторов: --hf-config-dir DIR (локальные config.json) → иначе Hub (нужен интернет) → иначе дефолт класса
        (помечается config_source="default_class", сверять n_params: D-FINE-L ≈31 M, RT-DETRv2-R50 ≈42 M).
"""
import argparse, json, os, platform, statistics, subprocess, sys, threading, time, traceback

import torch

SCHEMA = "koz3.bench_tile.v1"
CNX = {"cnx_li9": 9, "cnx_li12": 12, "cnx_sp3": 3}
DET = {"dfine_l": "ustc-community/dfine-large-coco", "rtdetrv2_r50": "PekingU/rtdetr_v2_r50vd"}
N_OUT = 7          # N_HM 3 + N_MK 4 (li_train.py:33)
N_CLS_DET = 5      # sp_train.py:24


# ---------------------------------------------------------------- окружение и часы GPU
def smi(idx, fields):
    try:
        r = subprocess.run(["nvidia-smi", "-i", str(idx), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return [v.strip() for v in r.stdout.strip().split(",")]
    except Exception as e:  # noqa
        return [f"ERR:{e}"]


SMI_STATIC = "name,driver_version,pstate,clocks.max.sm,clocks.max.mem,power.limit,power.max_limit,memory.total"
SMI_DYN = "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active"


class ClockSampler:
    """nvidia-smi каждые period с во время замера: частота SM, мощность, температура, причины троттлинга."""
    def __init__(self, idx, period=0.5):
        self.idx, self.period, self.rows, self._stop = idx, period, [], threading.Event()

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self

    def _run(self):
        while not self._stop.is_set():
            self.rows.append(smi(self.idx, SMI_DYN)); self._stop.wait(self.period)

    def __exit__(self, *a):
        self._stop.set(); self.t.join(timeout=5)

    def summary(self):
        def col(i):
            out = []
            for r in self.rows:
                try: out.append(float(r[i]))
                except Exception: pass
            return out
        sm, pw, tp = col(0), col(2), col(3)
        thr = sorted({r[5] for r in self.rows if len(r) > 5})
        f = lambda v: None if not v else {"min": min(v), "median": statistics.median(v), "max": max(v)}
        return {"n_samples": len(self.rows), "sm_clock_mhz": f(sm), "power_w": f(pw), "temp_c": f(tp),
                "throttle_reasons_active": thr}


def env_info(dev):
    idx = dev.index or 0
    st = smi(idx, SMI_STATIC)
    keys = SMI_STATIC.split(",")
    p = torch.cuda.get_device_properties(dev)
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(dev), "gpu_index": idx, "capability": list(torch.cuda.get_device_capability(dev)),
        "sm_count": p.multi_processor_count, "total_mem_gb": round(p.total_memory / 2**30, 2),
        "bf16_native": torch.cuda.get_device_capability(dev) >= (8, 0),
        "nvidia_smi_static": dict(zip(keys, st)) if len(st) == len(keys) else st,
        "nvidia_smi_idle": dict(zip(SMI_DYN.split(","), smi(idx, SMI_DYN))),
        "versions": {m: _ver(m) for m in ("timm", "segmentation_models_pytorch", "transformers", "numpy")},
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def _ver(m):
    try: return __import__(m).__version__
    except Exception: return None


# ---------------------------------------------------------------- модели
def build_cnx(in_ch):
    try:
        import segmentation_models_pytorch as smp
        m = smp.Unet("tu-convnext_tiny", encoder_weights=None, in_channels=in_ch, classes=N_OUT,
                     decoder_channels=(256, 128, 64, 32, 16))
        return m, "smp.Unet(tu-convnext_tiny)"
    except ImportError:
        return _FallbackUnet(in_ch), "fallback_unet(timm convnext_tiny + smp-like decoder)"


class _FallbackUnet(torch.nn.Module):
    """Эквивалент smp.Unet по структуре: timm convnext_tiny features_only (страйды 4..32) + 5 блоков декодера
    (upsample ×2 nearest, concat skip, 2×[conv3x3-BN-ReLU]) с каналами (256,128,64,32,16) + conv3x3 head."""
    def __init__(self, in_ch):
        super().__init__()
        import timm
        self.enc = timm.create_model("convnext_tiny", pretrained=False, features_only=True, in_chans=in_ch)
        enc_ch = self.enc.feature_info.channels()          # [96,192,384,768] @ /4,/8,/16,/32
        skips = [enc_ch[2], enc_ch[1], enc_ch[0], 0, 0]
        dec = (256, 128, 64, 32, 16); blocks, c_in = [], enc_ch[3]
        cbr = lambda i, o: torch.nn.Sequential(torch.nn.Conv2d(i, o, 3, padding=1, bias=False), torch.nn.BatchNorm2d(o), torch.nn.ReLU(inplace=True))
        for s, o in zip(skips, dec):
            blocks.append(torch.nn.Sequential(cbr(c_in + s, o), cbr(o, o))); c_in = o
        self.blocks = torch.nn.ModuleList(blocks); self.skips = skips
        self.head = torch.nn.Conv2d(dec[-1], N_OUT, 3, padding=1)

    def forward(self, x):
        f = self.enc(x); h = f[3]; sk = [f[2], f[1], f[0], None, None]
        for b, s in zip(self.blocks, sk):
            h = torch.nn.functional.interpolate(h, scale_factor=2, mode="nearest")
            if s is not None: h = torch.cat([h, s], 1)
            h = b(h)
        return self.head(h)


def build_det(name, cfg_dir):
    import transformers as tf
    ckpt = DET[name]
    cls = tf.DFineForObjectDetection if name == "dfine_l" else tf.RTDetrV2ForObjectDetection
    ccls = tf.DFineConfig if name == "dfine_l" else tf.RTDetrV2Config
    kw = dict(num_labels=N_CLS_DET)
    src = None
    if cfg_dir and os.path.isdir(os.path.join(cfg_dir, name)):
        cfg = ccls.from_pretrained(os.path.join(cfg_dir, name), **kw); src = f"local:{os.path.join(cfg_dir, name)}"
    else:
        try:
            cfg = tf.AutoConfig.from_pretrained(ckpt, **kw); src = f"hub:{ckpt}"
        except Exception:
            cfg = ccls(**kw); src = "default_class"
    for a in ("use_pretrained_backbone", "use_timm_backbone"):
        if hasattr(cfg, a): setattr(cfg, a, False)
    m = cls(cfg)      # случайная инициализация, без загрузки весов

    class Wrap(torch.nn.Module):   # = koz3/det_train.py:196-202
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            o = self.m(pixel_values=x); return o.logits, o.pred_boxes
    return Wrap(m), f"{cls.__name__} config={src}"


# ---------------------------------------------------------------- TTA
D8 = [(k, fl) for k in range(4) for fl in (False, True)]          # hm.py:96
FLIP3 = [(0, ()), (0, (-1,)), (0, (-2,))]                          # hm.py:98


def view_fwd(model, x, k, fl, amp_dtype):
    z = torch.rot90(x, k, dims=(-2, -1)) if k else x
    if fl is True: z = z.flip(-1)
    elif fl: z = z.flip(fl)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        o = torch.sigmoid(model(z).float())
    if fl is True: o = o.flip(-1)
    elif fl: o = o.flip(fl)
    return torch.rot90(o, -k, dims=(-2, -1)) if k else o


def view_name(k, fl):
    return f"rot{90*k}" + ("_flipW" if fl is True else ("_flip" + ("W" if fl == (-1,) else "H") if fl else ""))


# ---------------------------------------------------------------- таймер
def timeit(fn, warmup, iters, sync_dev):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(sync_dev)
    ev_ms, wall_ms = [], []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter(); s.record(); out = fn(); e.record(); torch.cuda.synchronize(sync_dev)
        wall_ms.append((time.perf_counter() - t0) * 1e3); ev_ms.append(s.elapsed_time(e))
    return stats(ev_ms), stats(wall_ms), out


def stats(v):
    v = sorted(v); n = len(v)
    q = lambda p: v[min(n - 1, int(round(p * (n - 1))))]
    return {"n": n, "median": round(statistics.median(v), 3), "p90": round(q(0.9), 3), "mean": round(statistics.mean(v), 3),
            "min": round(v[0], 3), "max": round(v[-1], 3), "std": round(statistics.pstdev(v), 3)}


def dtype_of(name, dev):
    if name == "fp32": return None, None
    if name == "fp16": return torch.float16, None
    if name == "bf16":
        if torch.cuda.get_device_capability(dev) < (8, 0):
            return None, "bf16 не поддержан аппаратно (sm<80, напр. T4 sm_75): пропуск; мерить fp16 и пересчитывать"
        return torch.bfloat16, None
    raise ValueError(name)


def nonfinite(o):
    if isinstance(o, (tuple, list)):
        return int(sum(nonfinite(t) for t in o))
    return int((~torch.isfinite(o)).sum().item())


# ---------------------------------------------------------------- прогон одной модели
def run_model(name, args, dev):
    is_det = name in DET
    rec = {"model": name, "kind": "detector" if is_det else "unet", "configs": [], "errors": []}
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    m, desc = (build_det(name, args.hf_config_dir) if is_det else build_cnx(CNX[name]))
    m = m.eval().to(dev)
    rec.update(arch=desc, n_params=sum(p.numel() for p in m.parameters()),
               weights_mb_fp16=round(sum(p.numel() for p in m.parameters()) * 2 / 2**20, 1),
               weights="random_init", weights_dtype_on_gpu="fp32 (autocast, как load_ts в models.py:47)",
               mem_after_load_mb=round(torch.cuda.memory_allocated(dev) / 2**20, 1))
    tile = args.chip if is_det else args.tile
    in_ch = 3 if is_det else CNX[name]
    for dname in args.dtypes:
        amp, skip = dtype_of(dname, dev)
        if skip:
            rec["configs"].append({"dtype": dname, "skipped": skip}); continue
        for bs in args.batches:
            cfg = {"dtype": dname, "autocast": dname != "fp32", "batch": bs, "input": [bs, in_ch, tile, tile],
                   "warmup": args.warmup, "iters": args.iters}
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
                x = torch.rand(bs, in_ch, tile, tile, device=dev)
                with torch.inference_mode():
                    if is_det:
                        def one(): 
                            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                                return m(x)
                        def flip2():                   # prev_sp_det.py:19-20: id + flip(-1)
                            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                                a = m(x); b = m(torch.flip(x, [3]))
                            return a, b
                        with ClockSampler(dev.index or 0) as cs:
                            ev, wall, out = timeit(one, args.warmup, args.iters, dev)
                            ev2, wall2, _ = timeit(flip2, args.warmup // 2, args.iters, dev)
                        cfg["views"] = {"single": {"cuda_event_ms": ev, "wall_ms": wall},
                                        "flip2_total": {"cuda_event_ms": ev2, "wall_ms": wall2}}
                        cfg["ms_per_chip_single_median"] = round(ev["median"] / bs, 3)
                        cfg["ms_per_chip_flip2_median"] = round(ev2["median"] / bs, 3)
                        cfg["ms_per_chip_flip2_p90"] = round(ev2["p90"] / bs, 3)
                    else:
                        views = {}
                        with ClockSampler(dev.index or 0) as cs:
                            for k, fl in (D8 if bs in args.per_view_batches else D8[:1]):
                                ev, wall, out = timeit(lambda: view_fwd(m, x, k, fl, amp), args.warmup, args.iters, dev)
                                views[view_name(k, fl)] = {"cuda_event_ms": ev, "wall_ms": wall}
                            def d8():
                                acc = 0
                                for k, fl in D8: acc = acc + view_fwd(m, x, k, fl, amp)
                                return acc / len(D8)
                            def f3():
                                acc = 0
                                for k, fl in FLIP3: acc = acc + view_fwd(m, x, k, fl, amp)
                                return acc / len(FLIP3)
                            ev8, wall8, out = timeit(d8, max(2, args.warmup // 4), max(5, args.iters // 4), dev)
                            ev3, wall3, _ = timeit(f3, max(2, args.warmup // 4), max(5, args.iters // 4), dev)
                        views["d8_total"] = {"cuda_event_ms": ev8, "wall_ms": wall8}
                        views["flip3_total"] = {"cuda_event_ms": ev3, "wall_ms": wall3}
                        cfg["views"] = views
                        cfg["ms_per_tile_single_median"] = round(views["rot0"]["cuda_event_ms"]["median"] / bs, 3)
                        cfg["ms_per_tile_d8_median"] = round(ev8["median"] / bs, 3)
                        cfg["ms_per_tile_d8_p90"] = round(ev8["p90"] / bs, 3)
                        cfg["ms_per_tile_flip3_median"] = round(ev3["median"] / bs, 3)
                    cfg["nonfinite_outputs"] = nonfinite(out)
                torch.cuda.synchronize(dev)
                cfg["peak_vram_mb"] = {"max_memory_allocated": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1),
                                       "max_memory_reserved": round(torch.cuda.max_memory_reserved(dev) / 2**20, 1)}
                cfg["clocks_during_timing"] = cs.summary()
                del x, out
            except torch.cuda.OutOfMemoryError as e:
                cfg["oom"] = str(e).splitlines()[0]; torch.cuda.empty_cache()
            except Exception as e:  # noqa
                cfg["error"] = f"{type(e).__name__}: {e}"; rec["errors"].append(traceback.format_exc(limit=3))
            rec["configs"].append(cfg)
            print(json.dumps({k: cfg.get(k) for k in ("dtype", "batch", "ms_per_tile_d8_median", "ms_per_chip_flip2_median",
                                                       "peak_vram_mb", "oom", "error")}, ensure_ascii=False), flush=True)
    del m; torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cnx_li9,cnx_li12,cnx_sp3,dfine_l,rtdetrv2_r50")
    ap.add_argument("--dtypes", default="fp16,fp32,bf16")
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--chip", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--per-view-batches", default="1", help="для каких batch мерить все 8 видов d8 по отдельности")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cudnn-benchmark", type=int, default=1)
    ap.add_argument("--hf-config-dir", default=None)
    ap.add_argument("--out", default="bench_tile.json")
    a = ap.parse_args()
    a.dtypes = a.dtypes.split(","); a.batches = [int(b) for b in a.batches.split(",")]
    a.per_view_batches = [int(b) for b in a.per_view_batches.split(",")]
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark = bool(a.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # как наш default.toml (tf32=false)
    res = {"schema": SCHEMA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "argv": sys.argv, "settings": {k: v for k, v in vars(a).items()}, "env": env_info(dev),
           "note": "random weights; forward only; ms = CUDA events, wall = perf_counter после synchronize", "models": []}
    t0 = time.time()
    for name in a.models.split(","):
        try:
            res["models"].append(run_model(name, a, dev))
        except Exception as e:  # noqa
            res["models"].append({"model": name, "fatal": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=5)})
        with open(a.out, "w") as f:        # пишем после каждой модели — не потерять при обрыве ядра
            json.dump(res, f, ensure_ascii=False, indent=1)
    res["env"]["nvidia_smi_end"] = dict(zip(SMI_DYN.split(","), smi(dev.index or 0, SMI_DYN)))
    res["total_wall_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
