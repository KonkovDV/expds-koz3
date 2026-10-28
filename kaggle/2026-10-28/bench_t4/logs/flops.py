import importlib.util, torch, json
from torch.utils.flop_counter import FlopCounterMode
spec = importlib.util.spec_from_file_location("bt", "/workspace/koz3_shared/bot-8/bench/bench_tile.py"); bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
out = {}
for name, mk, shape in (("cnx_li12", lambda: bt.build_cnx(12)[0], (1, 12, 1024, 1024)), ("dfine_l", lambda: bt.build_det("dfine_l", "../kernel/hf_configs")[0], (1, 3, 640, 640)),
                        ("rtdetrv2_r50", lambda: bt.build_det("rtdetrv2_r50", "../kernel/hf_configs")[0], (1, 3, 640, 640))):
    m = mk().eval().to("meta")
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc: m(torch.empty(shape, device="meta"))
        out[name] = round(fc.get_total_flops() / 1e9, 1)
    except Exception as e: out[name] = repr(e)[:200]
print(json.dumps({"GFLOP_per_forward_(2*MAC)": out}))
