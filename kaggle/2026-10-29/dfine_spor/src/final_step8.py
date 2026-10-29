# ── Шаг 8 (final). Одна модель на ВСЕХ участках SpOr (без отложенного фолда) → dfine_final/ + замер скорости ─────────
# Рецепт, число итераций (FULL_ITERS_DEFAULT = 6000), сиды-схема и тренер — как у фолдов full/full_rest. OOF здесь
# НЕВОЗМОЖЕН (все участки в обучении): пороги sp_dfine берутся из OOF фолдов full/full_rest (5 фолдов, COCO).
assert MODE == "final", MODE
inv_train = [i for i in INV if i["marks"]]
t = time.time()
with cf.ThreadPoolExecutor(8) as ex: RECS = [r for rs in ex.map(build_chips, inv_train) for r in rs]
log(f"чипов: {len(RECS)} (объектных {sum(r['cls'] >= 0 for r in RECS)}), участков {len(inv_train)}, {time.time() - t:.0f} с;",
    "классы:", dict(collections.Counter(CLASSES[r['cls']] for r in RECS if r['cls'] >= 0)))
if not RECS: raise SystemExit("пустой банк чипов — проверьте разметку/растры")
FINAL_SEED = 105  # фолды: make_model(seed=100+k), batches(seed=1000+k), k=0..4 → final: 105 / 1005
BENCH_RESERVE_S = 1200 if MODE != "smoke" and DEVS[0].type == "cuda" else 0
REPORT = {"cfg": CFG, "hf": f"{HF_REPO}@{HF_REV}", "base": DFINE_BASE, "final": {}, "bench": {}, "skipped_regions_no_fold": skipped,
          "oof": "нет — модель обучена на всех участках SpOr; OOF невозможен, пороги sp_dfine — из OOF фолдов full/full_rest"}
def _dump(): json.dump(REPORT, open(OUT / "dfine_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
dev = DEVS[0]; tag = "[final]"
budget = DEADLINE - time.time() - 1800 - BENCH_RESERVE_S
model, minfo = make_model(seed=FINAL_SEED); model = model.to(dev)
log(tag, f"train чипов {len(RECS)} (все участки), бюджет обучения {budget / 3600:.2f} ч, итераций {CFG['iters']}")
model = TRAIN(model, batches(RECS, 1000 + FINAL_SEED - 100, dev), CFG, dev, log, budget_s=budget, tag=tag)
rep = dict(getattr(model, "train_report", {})); rep["model_check"] = minfo; rep["trainer"] = CFG["trainer"]
rep.update({"n_train_chips": len(RECS), "n_obj_chips": sum(r["cls"] >= 0 for r in RECS), "train_regions": [i["region"] for i in inv_train],
            "seed": FINAL_SEED})
if dev.type == "cuda": rep["peak_vram_train_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
FD = OUT / "dfine_final"
model.save_pretrained(FD, safe_serialization=True)  # тот же формат, что dfine_f{k}: config.json + model.safetensors (fp32)
import hashlib
rep["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(FD.iterdir())}
REPORT["final"] = rep; _dump()
log(tag, "сохранено", FD, json.dumps(rep["files"], ensure_ascii=False))
try:  # контроль: перечитать сохранённое и сравнить выход на случайном входе
    from transformers import DFineForObjectDetection
    with MODEL_LOCK: m2 = DFineForObjectDetection.from_pretrained(FD, device_map=None)
    check_model(m2); m2 = m2.to(dev).eval(); model.eval(); xx = torch.rand(1, 3, CFG["tile"], CFG["tile"], device=dev)
    with torch.no_grad(): d = (model(pixel_values=xx).logits - m2(pixel_values=xx).logits).abs().max().item()
    REPORT["final"]["reload_max_abs_diff"] = d; log(tag, "перезагрузка dfine_final OK, max|Δlogits| =", d); del m2
except Exception as e:
    REPORT["final"]["reload_error"] = repr(e)[:300]; log(tag, "ОШИБКА перезагрузки dfine_final:", repr(e)[:300])
_dump()

# ── Шаг 9 (final). Замер скорости инференса на фиксированном наборе окон (для пересчёта на RTX 4090) ─────────────────
# Окна: сетка инференса (tile 640, шаг 480) первых непустых окон фиксированных участков, в порядке infer_region.
# Режимы: fp32; fp16 autocast (как в OOF/infer_region); fp16 .half() весов. bs 1 и bs 8. Время — только forward
# (+ sigmoid/topk), с cuda.synchronize, после 3 прогревов; flip-TTA = 2 forward на окно (как infer_region).
# Плюс end-to-end infer_region (чтение+warp+2 forward+NMS) по целым участкам. Пиковая VRAM — на каждый режим.
BENCH_REGIONS = ["051_Воронеж_2", "158_Курганы_29", "163_Курганы_34"]
BENCH_N = E("DFINE_BENCH_N", 64)  # сухой прогон на CPU: DFINE_BENCH_N=4
@torch.no_grad()
def bench():
    import copy
    T, S = CFG["tile"], CFG["tile"] - CFG["overlap"]; imgs, src = [], []
    for inv in [i for r in BENCH_REGIONS for i in INV if i["region"] == r] or INV[:2]:
        key, size = inv["tifs"][0]; sc = Scene(inv, key, size)
        xs = list(range(0, max(1, sc.W - CFG["overlap"]), S)) or [0]; ys = list(range(0, max(1, sc.H - CFG["overlap"]), S)) or [0]
        n0 = len(imgs)
        for y in ys:
            for x in xs:
                if len(imgs) - n0 >= BENCH_N // 2 or len(imgs) >= BENCH_N: break
                im = sc.read(x, y, T)
                if (im.max(2) == 0).mean() <= 0.95: imgs.append(im)
        src.append({"region": inv["region"], "tif": key.rsplit("/", 1)[-1], "n": len(imgs) - n0}); sc.close()
        if len(imgs) >= BENCH_N: break
    X = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255)
    out = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", "torch": torch.__version__,
           "n_windows": len(imgs), "tile": T, "windows_from": src, "modes": {}}
    def sync(): torch.cuda.synchronize(dev) if dev.type == "cuda" else None
    def run(m, bs, ac, name, half=False):
        m.eval(); ms = []
        if dev.type == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        for rep_ in range(2):
            for b in range(0, len(X), bs):
                xb = X[b:b + bs].to(dev); xb = xb.half() if half else xb
                for flip in (False, True):
                    sync(); t0 = time.perf_counter()
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=ac and dev.type == "cuda"):
                        o = m(pixel_values=xb.flip(3) if flip else xb)
                    p = o.logits.float().sigmoid(); p.flatten(1).topk(min(CFG["max_det"], p.shape[1] * p.shape[2]), 1); sync()
                    if rep_ == 1 or b // bs >= 3: ms.append((time.perf_counter() - t0) * 1000 / xb.shape[0])  # прогрев: 3 первых батча 1-го прохода
        r = {"bs": bs, "ms_per_window_forward_median": round(float(np.median(ms)), 2), "ms_per_window_forward_mean": round(float(np.mean(ms)), 2),
             "ms_per_window_with_flipTTA": round(2 * float(np.median(ms)), 2), "n_meas": len(ms)}
        if dev.type == "cuda": r["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 3)
        out["modes"][name] = r; log("[bench]", name, json.dumps(r))
    def safe(*a, **k):  # сбой одного режима (напр. .half()) не должен терять остальные замеры
        try: run(*a, **k)
        except Exception as e: out["modes"][a[3]] = {"error": repr(e)[:300]}; log("[bench]", a[3], "ОШИБКА", repr(e)[:300])
    for bs in (1, 8):
        safe(model.float(), bs, False, f"fp32_bs{bs}")
        safe(model, bs, True, f"fp16autocast_bs{bs}")
    if dev.type == "cuda":
        mh = copy.deepcopy(model).half()
        for bs in (1, 8): safe(mh, bs, False, f"fp16half_bs{bs}", half=True)
        del mh
    e2e = []
    for inv in [i for r in BENCH_REGIONS for i in INV if i["region"] == r]:
        if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time(); f = infer_region(model, inv, dev); dt = time.time() - t0
        T_, S_ = CFG["tile"], CFG["tile"] - CFG["overlap"]; nw = 0
        for key, size in inv["tifs"]:
            sc = Scene(inv, key, size); nw += len(list(range(0, max(1, sc.W - CFG["overlap"]), S_)) or [0]) * len(list(range(0, max(1, sc.H - CFG["overlap"]), S_)) or [0]); sc.close()
        if CFG["max_windows"]: nw = min(nw, CFG["max_windows"] * len(inv["tifs"]))
        e2e.append({"region": inv["region"], "windows": nw, "sec": round(dt, 2), "ms_per_window": round(dt * 1000 / max(nw, 1), 1), "n_det": len(f),
                    **({"peak_vram_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 3)} if dev.type == "cuda" else {})})
        log("[bench e2e]", json.dumps(e2e[-1], ensure_ascii=False))
    out["e2e_infer_region_fp16autocast_bs%d_flipTTA" % CFG["bs"]] = e2e
    return out
try:
    if time.time() < DEADLINE - 600: REPORT["bench"] = bench()
    else: REPORT["bench"] = {"skipped": "нет времени до DEADLINE"}
except Exception as e:
    import traceback; REPORT["bench"] = {"error": repr(e)[:300], "tb": traceback.format_exc()[-1500:]}; log("[bench] ОШИБКА", repr(e)[:300])
REPORT["total_sec"] = round(time.time() - T0); _dump()
log("ГОТОВО final:", json.dumps({k: REPORT["final"].get(k) for k in ("iters", "planned", "s_it_median", "sec", "n_train_chips")}, ensure_ascii=False))
if "reload_error" in REPORT["final"]: raise SystemExit("dfine_final не перечитался — см. dfine_report.json")
