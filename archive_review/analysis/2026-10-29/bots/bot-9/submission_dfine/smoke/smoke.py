# CPU-смоук: dfine_infer на 4 окнах малого tif (bs 1, OMP 2), fp16-копия весов, сверка с портом бота 7 (только чтение его кода).
import sys, os, json, time, resource, gc, importlib.util
sys.path.insert(0, "/workspace/koz3_shared/bot-9/submission_dfine")
import numpy as np, torch
torch.set_num_threads(2)
import dfine_infer as D
D.CFG["bs"] = 1                     # на CPU в 2 ГБ; на результат окна не влияет (eval, без батч-статистик)
TIF = "/workspace/data/train-expds-3/055_Курменту/04_Курменту_SpOr_спутник/01_Курменту_SpOr_спутник.tif"
UTM = "/workspace/data/train-expds-3/055_Курменту/UTM.json"
F0 = "/workspace/kaggle_out/koz3-dfine-spor-full-rest/dfine_f0"; F0H = sys.argv[1]; NW = 4
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
rep = {}
def run(ck, tag, output="circles"):
    t = time.time(); det = D.DFineSpOr([ck], device="cpu", read_threads=2)
    g, B, L, C = det.detect_tif(TIF, UTM, max_windows=NW)
    feats = det.run_regions([("055_Курменту", [TIF], UTM)], output=output, max_windows=NW)
    rep[tag] = dict(sec=round(time.time() - t, 1), n_boxes=int(len(L)), n_feats=len(feats), info=det.info[0], crs=g.crs, W=g.W, H=g.H,
                    windows_total=len(D.windows(g)), maxrss_MB=rss())
    del det; gc.collect(); return (B, L, C), feats
(b32, f32c) = run(F0, "fp32_circles")
_, f32e = run(F0, "fp32_ellipses", "ellipses")
(b16, _) = run(F0H, "fp16w_circles")
# валидность выхода
from shapely.geometry import shape
def chk(fs):
    gs = [shape(f["geometry"]) for f in fs]
    return dict(n=len(fs), types=sorted({f["geometry"]["type"] for f in fs}), invalid=sum(not g.is_valid for g in gs),
                groups=sorted({f["properties"]["threshold_group"] for f in fs}), markup=sorted({f["properties"]["markup_type"] for f in fs}),
                classes={c: sum(f["properties"]["class_name"] == c for f in fs) for c in D.CLASSES},
                conf_max=max((f["properties"]["confidence"] for f in fs), default=None),
                x_range=[min(g.bounds[0] for g in gs), max(g.bounds[2] for g in gs)] if gs else None)
rep["valid_circles"] = chk(f32c); rep["valid_ellipses"] = chk(f32e)
# fp16-веса против fp32: совпадение боксов (одинаковые пары класс/IoU>0,9) и разница оценок
def match(a, b):
    from torchvision.ops import box_iou
    (Ba, La, Ca), (Bb, Lb, Cb) = a, b
    if not len(La) or not len(Lb): return {}
    iou = box_iou(torch.as_tensor(Ba), torch.as_tensor(Bb)).numpy(); iou[Ca[:, None] != Cb[None]] = 0
    j = iou.argmax(1); ok = iou[np.arange(len(La)), j] > 0.9
    d = np.abs(La[ok] - Lb[j[ok]])
    return dict(n_a=int(len(La)), n_b=int(len(Lb)), matched=int(ok.sum()), score_absdiff_max=float(d.max()) if d.size else None,
                score_absdiff_mean=float(d.mean()) if d.size else None, top_a=[round(float(x), 5) for x in np.sort(La)[::-1][:5]],
                top_b=[round(float(x), 5) for x in np.sort(Lb)[::-1][:5]])
rep["fp16w_vs_fp32"] = match(b32, b16)
# порт бота 7 (expds/sp_dfine.py, его working tree, только чтение) на тех же окнах
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("b7_sp_dfine", "/workspace/koz3_shared/bot-7/repo/expds/sp_dfine.py")
B7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(B7)
st = B7.settings({"sp_dfine": {"weights": [F0], "batch": 1, "device": "cpu"}})
t = time.time(); ms = B7.load_models(st, "/", torch.device("cpu"))
crs = D.target_crs(UTM, __import__("rasterio").open(TIF)); sc = B7.Scene(TIF, crs, 0.5)
W = B7.windows; B7.windows = lambda *a: W(*a)[:NW]
res, done, total = B7.detect_scene(ms, sc, st, torch.device("cpu")); sc.close()
B, L, C = res[0]
rep["bot7_port"] = dict(sec=round(time.time() - t, 1), n_boxes=int(len(L)), maxrss_MB=rss(), vs_mine=match(b32, (B, L, C)))
print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
