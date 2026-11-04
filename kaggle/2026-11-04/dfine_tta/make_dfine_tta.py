#!/usr/bin/env python3
"""Собирает ядро dmitrijkv/koz3-dfine-tta (PREREG.md рядом) из проверенных кусков.

Берётся дословно:
  * из ядра D-FINE-2 (w2_kernels/.../koz3-w2-dfine2-a.py, тот же код, что дал OOF sp_dfine базы):
    пины зависимостей, фолды по семьям, входы-датасеты, проверки модели, Scene (окна 640/160 на 0,5 м UTM);
  * из expds/orsp_tta.py (прод-путь флага [orsp_tta]): _inv_point, invert_xyxy, _iou, wbf, nms, fuse_views.
Новое здесь только: загрузка весов фолда и инференс 8 видов с тремя выходами V0 / T8N / T8W.

    python kaggle/2026-11-04/dfine_tta/make_dfine_tta.py
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "w2_kernels/koz3-w2-dfine2/dmitrijkv/koz3-w2-dfine2-a/koz3-w2-dfine2-a.py"
TTA = ROOT / "expds/orsp_tta.py"
OUT = Path(__file__).resolve().parent / "dmitrijkv" / "koz3-dfine-tta"
SLUG = "dmitrijkv/koz3-dfine-tta"
DATASETS = ["dmitrijkv/koz3-w2-dfine2-raw-a", "dmitrijkv/koz3-w2-dfine2-raw-b"]
WEIGHTS = ["dmitrijkv/koz3-dfine-spor-full", "dmitrijkv/koz3-dfine-spor-full-rest"]


def block(lines, start_marker, end_marker):
    """Строки от первой, начинающейся с start_marker, до первой (после неё), начинающейся с end_marker."""
    i = next(n for n, s in enumerate(lines) if s.startswith(start_marker))
    j = next(n for n in range(i + 1, len(lines)) if lines[n].startswith(end_marker))
    return lines[i:j]


def funcs(path, names):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    seg = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = [n for n in names if n not in seg]
    assert not missing, missing
    return "\n\n".join(seg[n] for n in names)


def main():
    L = SRC.read_text(encoding="utf-8").splitlines()
    deps = block(L, "import os, sys, subprocess, importlib", "DFINE_MODE_DEFAULT")
    folds = [s for s in L if s.startswith("FOLDS = __import__('json')")]
    assert len(folds) == 1
    imports = block(L, "import json, time, re, math, glob", "MODE = os.environ")
    gpu = block(L, "# ── Шаг 1. --require-gpu", "# ── Шаг 2. Входы")
    inputs = block(L, "def _inputs():", "try:")                       # _inputs, TIF_*, _load_gt_all, GT_BY, лог
    model = block(L, "from transformers import PreTrainedModel as _PTM", "def make_model")
    inv = block(L, "# ── Шаг 3. Фолды по семьям", "# ── Шаг 4. Банк")
    tta = funcs(TTA, ["_inv_point", "invert_xyxy", "_iou", "wbf", "nms", "fuse_views"])

    head = f'''# KOZ3 · TTA D-FINE на спутнике SpOr: 8 видов D4 на весах фолдов прод-модели sp_dfine (только инференс).
# PREREG: kaggle/2026-11-04/dfine_tta/PREREG.md. Ядро собрано скриптом make_dfine_tta.py из кода ядра D-FINE-2
# (тот же путь, что дал OOF sp_dfine базы гейта) и функций expds/orsp_tta.py (прод-путь флага [orsp_tta]).
# Входы — приватные датасеты Kaggle и вывод ядер с весами; ключей и сетевых учёток в тексте нет.
# Выход: oof_{{V0,T8N,T8W}}_f{{k}}.geojson, oof_{{V}}_spor.geojson, tta_report.json.
'''
    cfg = '''DATASETS_HINT = %s
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
E = lambda k, d: type(d)(os.environ.get(k, d))
CFG = dict(res=0.5, tile=640, overlap=160, bs=E("TTA_BS", 8), min_score=0.05, nms_iou=0.5, max_det=100, wbf_iou=0.55,
           max_windows=E("TTA_MAX_WINDOWS", 0), regions=os.environ.get("TTA_REGIONS", ""), folds=os.environ.get("TTA_FOLDS", "0,1,2,3,4"),
           deadline_h=E("TTA_DEADLINE_H", 8.0), group="sp_dfine")
VIEWS = [(k, f) for k in range(4) for f in (False, True)]          # == orsp_tta._view_tuples("d8")
V0_VIEWS = [VIEWS.index((0, False)), VIEWS.index((0, True))]       # путь исходного OOF: как есть + отражение
OUT = Path(os.environ.get("TTA_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time(); DEADLINE = T0 + CFG["deadline_h"] * 3600
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
log("CFG", json.dumps(CFG, ensure_ascii=False))
AP = argparse.ArgumentParser(); AP.add_argument("--allow-cpu", action="store_true"); ARGS, _ = AP.parse_known_args()
''' % json.dumps(DATASETS)

    weights = r'''
# ── Веса фолдов прод-модели sp_dfine: вывод ядер koz3-dfine-spor-full (f1, f2, f4) и -full-rest (f0, f3) ─────────
def _weight_dirs():
    roots = [Path(p) for p in os.environ.get("TTA_WEIGHTS", "").split(":") if p] or [Path("/kaggle/input")]
    out = {}
    for r in roots:
        for p in sorted(r.rglob("model.safetensors")):
            m = re.fullmatch(r"dfine_f(\d)", p.parent.name)
            if m and (p.parent / "config.json").exists():
                out.setdefault(int(m.group(1)), p.parent)
    return out
WDIRS = _weight_dirs()
log("веса фолдов:", {k: str(v) for k, v in sorted(WDIRS.items())})

def load_fold(k):
    from transformers import DFineForObjectDetection, PreTrainedModel
    with MODEL_LOCK:
        for n, f in _PTM_ORIG.items():
            if PreTrainedModel.__dict__.get(n) is not f:
                setattr(PreTrainedModel, n, f)
        m = DFineForObjectDetection.from_pretrained(str(WDIRS[k]), device_map=None)
        meta = _meta_names(m)
        if meta:
            meta = _repair_meta(m, meta)
        info = check_model(m)
    return m.eval(), info
'''

    infer = r'''
# ── Проверка: векторное обращение видов == invert_xyxy прод-пути (до GPU-работы) ──────────────────────────────────
def inv_boxes(xyxy, rot, hflip, size):
    """Векторный аналог invert_xyxy: xyxy (N,4) torch в пикселях повёрнутого тайла → исходный тайл."""
    x0, y0, x1, y1 = xyxy.unbind(1)
    if hflip:
        x0, x1 = size - x1, size - x0
    rot %= 4
    if rot == 0:
        return torch.stack([x0, y0, x1, y1], 1)
    if rot == 1:
        return torch.stack([size - y1, x0, size - y0, x1], 1)
    if rot == 2:
        return torch.stack([size - x1, size - y1, size - x0, size - y0], 1)
    return torch.stack([y0, size - x1, y1, size - x0], 1)
_g = torch.Generator().manual_seed(0)
_a = torch.rand(500, 2, generator=_g, dtype=torch.float64) * 600; _wh = torch.rand(500, 2, generator=_g, dtype=torch.float64) * 40
_bx = torch.cat([_a, _a + _wh], 1)
for _r, _f in VIEWS:
    _ref = invert_xyxy(_bx.numpy(), _r, _f, 640.0)
    assert np.allclose(inv_boxes(_bx, _r, _f, 640.0).numpy(), _ref, atol=1e-9), f"inv_boxes != invert_xyxy для вида {(_r, _f)}"
log("обращение видов совпадает с invert_xyxy для всех 8 видов")

@torch.no_grad()
def infer_region_tta(model, inv, dev):
    """Окна сцены × 8 видов. Возвращает {V0, T8N, T8W: [features]} в формате исходного OOF."""
    from torchvision.ops import batched_nms
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as st
    from pyproj import Transformer
    T, S = CFG["tile"], CFG["tile"] - CFG["overlap"]; out = {"V0": [], "T8N": [], "T8W": []}; stat = {"windows": 0}
    for key, size in inv["tifs"]:
        sc = Scene(inv, key, size)
        xs = list(range(0, max(1, sc.W - CFG["overlap"]), S)) or [0]; ys = list(range(0, max(1, sc.H - CFG["overlap"]), S)) or [0]
        wins = [(x, y) for y in ys for x in xs]
        if CFG["max_windows"]: wins = wins[:CFG["max_windows"]]
        acc = [[[], [], []] for _ in VIEWS]; t_r = time.time()
        for b in range(0, len(wins), CFG["bs"]):
            chunk, imgs = [], []
            for x, y in wins[b:b + CFG["bs"]]:
                im = sc.read(x, y, T)
                if (im.max(2) == 0).mean() > 0.95: continue
                chunk.append((x, y)); imgs.append(im)
            if not imgs: continue
            px = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
            for vi, (rot, hflip) in enumerate(VIEWS):
                t = torch.rot90(px, rot, dims=(2, 3)) if rot else px
                if hflip: t = t.flip(3)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    o = model(pixel_values=t)
                prob = o.logits.float().sigmoid(); bx = o.pred_boxes.float()
                Q, C = prob.shape[1:]
                sc_, idx = prob.flatten(1).topk(min(CFG["max_det"], Q * C), 1)
                for i, (x, y) in enumerate(chunk):
                    keep = sc_[i] >= CFG["min_score"]; q = idx[i][keep] // C; c = idx[i][keep] % C; bb = bx[i][q] * T
                    xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2, bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2], 1)
                    xyxy = inv_boxes(xyxy.double().cpu(), rot, hflip, float(T)) + torch.tensor([x, y, x, y], dtype=torch.float64)
                    acc[vi][0].append(xyxy); acc[vi][1].append(sc_[i][keep].double().cpu()); acc[vi][2].append(c.cpu())
        stat["windows"] += len(wins)
        log(f"[infer] {inv['region']} {key.rsplit('/', 1)[-1]}: {sc.W}x{sc.H} px, окон {len(wins)}, {time.time() - t_r:.0f} с")
        def cat(vi):
            B, L, Cc = acc[vi]
            if not B: return torch.zeros((0, 4), dtype=torch.float64), torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.long)
            return torch.cat(B), torch.cat(L), torch.cat(Cc)
        res = {}
        # V0: оба вида исходного OOF вместе, один NMS (как infer_region ядра D-FINE-2)
        B = [cat(vi) for vi in V0_VIEWS]
        Bt, Lt, Ct = (torch.cat([p[j] for p in B]) for j in range(3))
        k_ = batched_nms(Bt, Lt, Ct, CFG["nms_iou"]) if len(Lt) else torch.zeros(0, dtype=torch.long)
        res["V0"] = (Bt[k_].numpy(), Lt[k_].numpy(), Ct[k_].numpy().astype(int))
        # T8*: NMS по окнам внутри вида (orsp_tta._nms_per_model), затем слияние видов (orsp_tta.fuse_views)
        parts = []
        for vi in range(len(VIEWS)):
            Bt, Lt, Ct = cat(vi)
            k_ = batched_nms(Bt, Lt, Ct, CFG["nms_iou"]) if len(Lt) else torch.zeros(0, dtype=torch.long)
            parts.append((Bt[k_].numpy(), Lt[k_].numpy(), Ct[k_].numpy().astype(int)))
        res["T8N"] = fuse_views(parts, "nms", CFG["wbf_iou"], CFG["nms_iou"])
        res["T8W"] = fuse_views(parts, "wbf", CFG["wbf_iou"], CFG["nms_iou"])
        to3857 = Transformer.from_crs(sc.crs, "EPSG:3857", always_xy=True).transform; ang = np.linspace(0, 2 * np.pi, 33)[:-1]
        for v, (BB, LL, CC) in res.items():
            for (x0, y0, x1, y1), s, c in zip(BB, LL, CC):
                cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, .5), max((y1 - y0) / 2, .5)
                ring = [sc.T * (cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in ang]
                out[v].append({"type": "Feature", "geometry": mapping(st(to3857, Polygon(ring))),
                               "properties": {"class_name": CLASSES[int(c)], "confidence": round(float(s), 5), "region_name": inv["region"],
                                              "crs": "EPSG:3857", "markup_type": "sp_or", "original_crs": sc.crs, "group": CFG["group"],
                                              "variant": v, "fold": FOLD[inv["region"]],
                                              "tif": TIF_INFO[key][1] if key in TIF_INFO else key.rsplit("/", 1)[-1]}})
        sc.close()
    return out, stat

# ── Оркестрация: фолды по 2 GPU в потоках; каждый фолд — свои тестовые участки ──────────────────────────────────
folds_run = [int(f) for f in CFG["folds"].split(",")]
miss = [k for k in folds_run if k not in WDIRS]
if miss: raise SystemExit(f"нет весов фолдов {miss}: подключите вывод ядер {WEIGHTS_HINT}")
REPORT, lock = {"cfg": CFG, "views": VIEWS, "weights": {k: str(v) for k, v in WDIRS.items()}, "folds": {}}, threading.Lock()
queue = sorted(folds_run, key=lambda k: -sum(s for i in INV if FOLD[i["region"]] == k for _, s in i["tifs"]))
def gpu_worker(gi, dev):
    while True:
        with lock:
            if not queue: return
            k = queue.pop(0)
        tag = f"[gpu{gi} f{k}]"
        try:
            model, minfo = load_fold(k); model = model.to(dev)
            test = [i for i in INV if FOLD[i["region"]] == k]
            log(tag, f"веса {WDIRS[k]}, тестовых участков {len(test)}")
            ti = time.time(); feats = {"V0": [], "T8N": [], "T8W": []}; rep = {"model_check": minfo, "windows": 0, "errors": []}
            for inv in test:
                if time.time() > DEADLINE:
                    rep["errors"].append(f"deadline:{inv['region']}"); log(tag, "потолок времени, пропуск", inv["region"]); continue
                try:
                    f, s = infer_region_tta(model, inv, dev); rep["windows"] += s["windows"]
                    for v in feats: feats[v] += f[v]
                except Exception as e:
                    rep["errors"].append(f"{inv['region']}: {e!r}"[:300]); log(tag, "ошибка", inv["region"], repr(e)[:200])
            for v, fs in feats.items():
                json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
                           "features": fs}, open(OUT / f"oof_{v}_f{k}.geojson", "w", encoding="utf-8"), ensure_ascii=False)
            rep.update({"test_regions": [i["region"] for i in test], "n_det": {v: len(fs) for v, fs in feats.items()},
                        "infer_sec": round(time.time() - ti)})
            if dev.type == "cuda": rep["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
            log(tag, "готово", json.dumps(rep, ensure_ascii=False)[:600])
            del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
        except Exception as e:
            import traceback; rep = {"error": repr(e), "tb": traceback.format_exc()[-2000:]}; log(tag, "ОШИБКА", repr(e))
        with lock:
            REPORT["folds"][k] = rep
            json.dump(REPORT, open(OUT / "tta_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ths = [threading.Thread(target=gpu_worker, args=(gi, d)) for gi, d in enumerate(DEVS)]
for th in ths: th.start()
for th in ths: th.join()
for v in ("V0", "T8N", "T8W"):
    allf = []
    for k in folds_run:
        p = OUT / f"oof_{v}_f{k}.geojson"
        if p.exists(): allf += json.load(open(p, encoding="utf-8"))["features"]
    json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": allf},
              open(OUT / f"oof_{v}_spor.geojson", "w", encoding="utf-8"), ensure_ascii=False)
    REPORT.setdefault("n_det", {})[v] = len(allf)
REPORT["total_sec"] = round(time.time() - T0)
json.dump(REPORT, open(OUT / "tta_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
errs = {k: v.get("error") or v.get("errors") for k, v in REPORT["folds"].items() if v.get("error") or v.get("errors")}
log("ГОТОВО", REPORT["n_det"], "ошибки:", errs)
if errs: raise SystemExit(f"фолды с ошибкой: {list(errs)}")
'''
    parts = [head, "\n".join(deps), folds[0], "\n".join(imports), cfg, "\n".join(gpu), "\n".join(inputs),
             "\n".join(model), "\n".join(inv), "import numpy as np\n" + tta,
             f"WEIGHTS_HINT = {json.dumps(WEIGHTS)}", weights, infer]
    text = "\n".join(parts) + "\n"
    compile(text, "koz3-dfine-tta.py", "exec")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "koz3-dfine-tta.py").write_text(text, encoding="utf-8")
    meta = {"id": SLUG, "title": "koz3-dfine-tta", "code_file": "koz3-dfine-tta.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
            "dataset_sources": DATASETS, "kernel_sources": WEIGHTS, "competition_sources": [], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("written", OUT / "koz3-dfine-tta.py", len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
