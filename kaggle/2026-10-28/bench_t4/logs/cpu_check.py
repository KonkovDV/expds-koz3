import sys, json, time, importlib.util, torch, warnings
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("bt", "/workspace/koz3_shared/bot-8/bench/bench_tile.py"); bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
import transformers, timm, segmentation_models_pytorch as smp
res = {"torch": torch.__version__, "transformers": transformers.__version__, "timm": timm.__version__, "smp": smp.__version__,
       "has_DFine": hasattr(transformers, "DFineForObjectDetection"), "has_RTDetrV2": hasattr(transformers, "RTDetrV2ForObjectDetection"),
       "allow_tf32_attr": hasattr(torch.backends.cuda.matmul, "allow_tf32"), "OOMError": hasattr(torch.cuda, "OutOfMemoryError")}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always"); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    _ = torch.backends.cuda.matmul.allow_tf32
    res["tf32_warnings"] = [str(x.message)[:160] for x in w]
for name, ch in bt.CNX.items():
    m, desc = bt.build_cnx(ch); res[name] = {"arch": desc, "n_params": sum(p.numel() for p in m.parameters())}
    if name == "cnx_li12":
        t = time.time()
        with torch.inference_mode(): o = m(torch.rand(1, ch, 256, 256))
        res[name]["fwd256_shape"] = list(o.shape); res[name]["fwd256_s"] = round(time.time() - t, 2)
    del m
for name in bt.DET:
    m, desc = bt.build_det(name, sys.argv[1]); m.eval()
    r = {"arch": desc, "n_params": sum(p.numel() for p in m.parameters())}
    t = time.time()
    with torch.inference_mode(): lg, bx = m(torch.rand(1, 3, 320, 320))
    r.update(fwd320_logits=list(lg.shape), fwd320_boxes=list(bx.shape), fwd320_s=round(time.time() - t, 2), nonfinite=bt.nonfinite((lg, bx)))
    res[name] = r; del m
print(json.dumps(res, indent=1, ensure_ascii=False))
