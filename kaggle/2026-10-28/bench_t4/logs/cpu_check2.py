import importlib.util, torch, json, warnings
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("bt", "/workspace/koz3_shared/bot-8/bench/bench_tile.py"); bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
out = {}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        with torch.autocast("cuda", dtype=None, enabled=False): pass
        out["autocast_cuda_dtype_None_disabled"] = "ok"
    except Exception as e: out["autocast_cuda_dtype_None_disabled"] = repr(e)
    out["warn"] = [str(x.message)[:150] for x in w]
for name in bt.DET:
    m, _ = bt.build_det(name, "../kernel/hf_configs"); m.eval()
    try:
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            lg, bx = m(torch.rand(1, 3, 320, 320))
        out[name + "_cpu_bf16_autocast"] = [list(lg.shape), str(lg.dtype), bt.nonfinite((lg, bx))]
    except Exception as e: out[name + "_cpu_bf16_autocast"] = repr(e)[:300]
    del m
print(json.dumps(out, indent=1))
